//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "cache/clock_cache.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <numeric>
#include <thread>
#include <type_traits>

#include "cache/cache_key.h"
#include "cache/secondary_cache_adapter.h"
#include "logging/logging.h"
#include "monitoring/perf_context_imp.h"
#include "monitoring/statistics_impl.h"
#include "port/lang.h"
#include "rocksdb/env.h"
#include "util/hash.h"
#include "util/math.h"
#include "util/random.h"

namespace ROCKSDB_NAMESPACE {

namespace clock_cache {

namespace {
inline uint64_t GetRefcount(uint64_t meta) {
  return ((meta >> ClockHandle::kAcquireCounterShift) -
          (meta >> ClockHandle::kReleaseCounterShift)) &
         ClockHandle::kCounterMask;
}

inline uint64_t GetInitialCountdown(Cache::Priority priority) {
  // Set initial clock data from priority
  // TODO: configuration parameters for priority handling and clock cycle
  // count?
  switch (priority) {
    case Cache::Priority::HIGH:
      return ClockHandle::kHighCountdown;
    default:
      assert(false);
      FALLTHROUGH_INTENDED;
    case Cache::Priority::LOW:
      return ClockHandle::kLowCountdown;
    case Cache::Priority::BOTTOM:
      return ClockHandle::kBottomCountdown;
  }
}

inline void MarkEmpty(ClockHandle& h) {
#ifndef NDEBUG
  // Mark slot as empty, with assertion
  uint64_t meta = h.meta.exchange(0, std::memory_order_release);
  assert(meta >> ClockHandle::kStateShift == ClockHandle::kStateConstruction);
#else
  // Mark slot as empty
  h.meta.store(0, std::memory_order_release);
#endif
}

inline void FreeDataMarkEmpty(ClockHandle& h, MemoryAllocator* allocator) {
  // NOTE: in theory there's more room for parallelism if we copy the handle
  // data and delay actions like this until after marking the entry as empty,
  // but performance tests only show a regression by copying the few words
  // of data.
  h.FreeData(allocator);

  MarkEmpty(h);
}

// Called to undo the effect of referencing an entry for internal purposes,
// so it should not be marked as having been used.
inline void Unref(const ClockHandle& h, uint64_t count = 1) {
  // Pretend we never took the reference
  // WART: there's a tiny chance we release last ref to invisible
  // entry here. If that happens, we let eviction take care of it.
  uint64_t old_meta = h.meta.fetch_sub(ClockHandle::kAcquireIncrement * count,
                                       std::memory_order_release);
  assert(GetRefcount(old_meta) != 0);
  (void)old_meta;
}

inline bool ClockUpdate(ClockHandle& h) {
  uint64_t meta = h.meta.load(std::memory_order_relaxed);

  uint64_t acquire_count =
      (meta >> ClockHandle::kAcquireCounterShift) & ClockHandle::kCounterMask;
  uint64_t release_count =
      (meta >> ClockHandle::kReleaseCounterShift) & ClockHandle::kCounterMask;
  // fprintf(stderr, "ClockUpdate @ %p: %lu %lu %u\n", &h, acquire_count,
  // release_count, (unsigned)(meta >> ClockHandle::kStateShift));
  if (acquire_count != release_count) {
    // Only clock update entries with no outstanding refs
    return false;
  }
  if (!((meta >> ClockHandle::kStateShift) & ClockHandle::kStateShareableBit)) {
    // Only clock update Shareable entries
    return false;
  }
  if ((meta >> ClockHandle::kStateShift == ClockHandle::kStateVisible) &&
      acquire_count > 0) {
    // Decrement clock
    uint64_t new_count =
        std::min(acquire_count - 1, uint64_t{ClockHandle::kMaxCountdown} - 1);
    // Compare-exchange in the decremented clock info, but
    // not aggressively
    uint64_t new_meta =
        (uint64_t{ClockHandle::kStateVisible} << ClockHandle::kStateShift) |
        (new_count << ClockHandle::kReleaseCounterShift) |
        (new_count << ClockHandle::kAcquireCounterShift);
    h.meta.compare_exchange_strong(meta, new_meta, std::memory_order_relaxed);
    return false;
  }
  // Otherwise, remove entry (either unreferenced invisible or
  // unreferenced and expired visible).
  if (h.meta.compare_exchange_strong(
          meta,
          uint64_t{ClockHandle::kStateConstruction} << ClockHandle::kStateShift,
          std::memory_order_acquire)) {
    // Took ownership.
    return true;
  } else {
    // Compare-exchange failing probably
    // indicates the entry was used, so skip it in that case.
    return false;
  }
}

// If an entry doesn't receive clock updates but is repeatedly referenced &
// released, the acquire and release counters could overflow without some
// intervention. This is that intervention, which should be inexpensive
// because it only incurs a simple, very predictable check. (Applying a bit
// mask in addition to an increment to every Release likely would be
// relatively expensive, because it's an extra atomic update.)
//
// We do have to assume that we never have many millions of simultaneous
// references to a cache handle, because we cannot represent so many
// references with the difference in counters, masked to the number of
// counter bits. Similarly, we assume there aren't millions of threads
// holding transient references (which might be "undone" rather than
// released by the way).
//
// Consider these possible states for each counter:
// low: less than kMaxCountdown
// medium: kMaxCountdown to half way to overflow + kMaxCountdown
// high: half way to overflow + kMaxCountdown, or greater
//
// And these possible states for the combination of counters:
// acquire / release
// -------   -------
// low       low       - Normal / common, with caveats (see below)
// medium    low       - Can happen while holding some refs
// high      low       - Violates assumptions (too many refs)
// low       medium    - Violates assumptions (refs underflow, etc.)
// medium    medium    - Normal (very read heavy cache)
// high      medium    - Can happen while holding some refs
// low       high      - This function is supposed to prevent
// medium    high      - Violates assumptions (refs underflow, etc.)
// high      high      - Needs CorrectNearOverflow
//
// Basically, this function detects (high, high) state (inferred from
// release alone being high) and bumps it back down to (medium, medium)
// state with the same refcount and the same logical countdown counter
// (everything > kMaxCountdown is logically the same). Note that bumping
// down to (low, low) would modify the countdown counter, so is "reserved"
// in a sense.
//
// If near-overflow correction is triggered here, there's no guarantee
// that another thread hasn't freed the entry and replaced it with another.
// Therefore, it must be the case that the correction does not affect
// entries unless they are very old (many millions of acquire-release cycles).
// (Our bit manipulation is indeed idempotent and only affects entries in
// exceptional cases.) We assume a pre-empted thread will not stall that long.
// If it did, the state could be corrupted in the (unlikely) case that the top
// bit of the acquire counter is set but not the release counter, and thus
// we only clear the top bit of the acquire counter on resumption. It would
// then appear that there are too many refs and the entry would be permanently
// pinned (which is not terrible for an exceptionally rare occurrence), unless
// it is referenced enough (at least kMaxCountdown more times) for the release
// counter to reach "high" state again and bumped back to "medium." (This
// motivates only checking for release counter in high state, not both in high
// state.)
inline void CorrectNearOverflow(uint64_t old_meta,
                                std::atomic<uint64_t>& meta) {
  // We clear both top-most counter bits at the same time.
  constexpr uint64_t kCounterTopBit = uint64_t{1}
                                      << (ClockHandle::kCounterNumBits - 1);
  constexpr uint64_t kClearBits =
      (kCounterTopBit << ClockHandle::kAcquireCounterShift) |
      (kCounterTopBit << ClockHandle::kReleaseCounterShift);
  // A simple check that allows us to initiate clearing the top bits for
  // a large portion of the "high" state space on release counter.
  constexpr uint64_t kCheckBits =
      (kCounterTopBit | (ClockHandle::kMaxCountdown + 1))
      << ClockHandle::kReleaseCounterShift;

  if (UNLIKELY(old_meta & kCheckBits)) {
    meta.fetch_and(~kClearBits, std::memory_order_relaxed);
  }
}

inline bool BeginSlotInsert(const ClockHandleBasicData& proto, ClockHandle& h,
                            uint64_t initial_countdown, bool* already_matches) {
  assert(*already_matches == false);
  // Optimistically transition the slot from "empty" to
  // "under construction" (no effect on other states)
  uint64_t old_meta = h.meta.fetch_or(
      uint64_t{ClockHandle::kStateOccupiedBit} << ClockHandle::kStateShift,
      std::memory_order_acq_rel);
  uint64_t old_state = old_meta >> ClockHandle::kStateShift;

  if (old_state == ClockHandle::kStateEmpty) {
    // We've started inserting into an available slot, and taken
    // ownership.
    return true;
  } else if (old_state != ClockHandle::kStateVisible) {
    // Slot not usable / touchable now
    return false;
  }
  // Existing, visible entry, which might be a match.
  // But first, we need to acquire a ref to read it. In fact, number of
  // refs for initial countdown, so that we boost the clock state if
  // this is a match.
  old_meta =
      h.meta.fetch_add(ClockHandle::kAcquireIncrement * initial_countdown,
                       std::memory_order_acq_rel);
  // Like Lookup
  if ((old_meta >> ClockHandle::kStateShift) == ClockHandle::kStateVisible) {
    // Acquired a read reference
    if (h.hashed_key == proto.hashed_key) {
      // Match. Release in a way that boosts the clock state
      old_meta =
          h.meta.fetch_add(ClockHandle::kReleaseIncrement * initial_countdown,
                           std::memory_order_acq_rel);
      // Correct for possible (but rare) overflow
      CorrectNearOverflow(old_meta, h.meta);
      // Insert detached instead (only if return handle needed)
      *already_matches = true;
      return false;
    } else {
      // Mismatch.
      Unref(h, initial_countdown);
    }
  } else if (UNLIKELY((old_meta >> ClockHandle::kStateShift) ==
                      ClockHandle::kStateInvisible)) {
    // Pretend we never took the reference
    Unref(h, initial_countdown);
  } else {
    // For other states, incrementing the acquire counter has no effect
    // so we don't need to undo it.
    // Slot not usable / touchable now.
  }
  return false;
}

inline void FinishSlotInsert(const ClockHandleBasicData& proto, ClockHandle& h,
                             uint64_t initial_countdown, bool keep_ref) {
  // Save data fields
  ClockHandleBasicData* h_alias = &h;
  *h_alias = proto;

  // Transition from "under construction" state to "visible" state
  uint64_t new_meta = uint64_t{ClockHandle::kStateVisible}
                      << ClockHandle::kStateShift;

  // Maybe with an outstanding reference
  new_meta |= initial_countdown << ClockHandle::kAcquireCounterShift;
  new_meta |= (initial_countdown - keep_ref)
              << ClockHandle::kReleaseCounterShift;

#ifndef NDEBUG
  // Save the state transition, with assertion
  uint64_t old_meta = h.meta.exchange(new_meta, std::memory_order_release);
  assert(old_meta >> ClockHandle::kStateShift ==
         ClockHandle::kStateConstruction);
#else
  // Save the state transition
  h.meta.store(new_meta, std::memory_order_release);
#endif
}

bool TryInsert(const ClockHandleBasicData& proto, ClockHandle& h,
               uint64_t initial_countdown, bool keep_ref,
               bool* already_matches) {
  bool b = BeginSlotInsert(proto, h, initial_countdown, already_matches);
  if (b) {
    FinishSlotInsert(proto, h, initial_countdown, keep_ref);
  }
  return b;
}

// Func must be const HandleImpl& -> void callable
template <class HandleImpl, class Func>
void ConstApplyToEntriesRange(const Func& func, const HandleImpl* begin,
                              const HandleImpl* end,
                              bool apply_if_will_be_deleted) {
  uint64_t check_state_mask = ClockHandle::kStateShareableBit;
  if (!apply_if_will_be_deleted) {
    check_state_mask |= ClockHandle::kStateVisibleBit;
  }

  for (const HandleImpl* h = begin; h < end; ++h) {
    // Note: to avoid using compare_exchange, we have to be extra careful.
    uint64_t old_meta = h->meta.load(std::memory_order_relaxed);
    // Check if it's an entry visible to lookups
    if ((old_meta >> ClockHandle::kStateShift) & check_state_mask) {
      // Increment acquire counter. Note: it's possible that the entry has
      // completely changed since we loaded old_meta, but incrementing acquire
      // count is always safe. (Similar to optimistic Lookup here.)
      old_meta = h->meta.fetch_add(ClockHandle::kAcquireIncrement,
                                   std::memory_order_acquire);
      // Check whether we actually acquired a reference.
      if ((old_meta >> ClockHandle::kStateShift) &
          ClockHandle::kStateShareableBit) {
        // Apply func if appropriate
        if ((old_meta >> ClockHandle::kStateShift) & check_state_mask) {
          func(*h);
        }
        // Pretend we never took the reference
        Unref(*h);
        // No net change, so don't need to check for overflow
      } else {
        // For other states, incrementing the acquire counter has no effect
        // so we don't need to undo it. Furthermore, we cannot safely undo
        // it because we did not acquire a read reference to lock the
        // entry in a Shareable state.
      }
    }
  }
}

}  // namespace

void ClockHandleBasicData::FreeData(MemoryAllocator* allocator) const {
  if (helper->del_cb) {
    helper->del_cb(value, allocator);
  }
}

template <class HandleImpl>
HandleImpl* BaseClockTable::StandaloneInsert(
    const ClockHandleBasicData& proto) {
  // Heap allocated separate from table
  HandleImpl* h = new HandleImpl();
  ClockHandleBasicData* h_alias = h;
  *h_alias = proto;
  h->SetStandalone();
  // Single reference (standalone entries only created if returning a refed
  // Handle back to user)
  uint64_t meta = uint64_t{ClockHandle::kStateInvisible}
                  << ClockHandle::kStateShift;
  meta |= uint64_t{1} << ClockHandle::kAcquireCounterShift;
  h->meta.store(meta, std::memory_order_release);
  // Keep track of how much of usage is standalone
  standalone_usage_.fetch_add(proto.GetTotalCharge(),
                              std::memory_order_relaxed);
  return h;
}

template <class Table>
typename Table::HandleImpl* BaseClockTable::CreateStandalone(
    ClockHandleBasicData& proto, size_t capacity, bool strict_capacity_limit,
    bool allow_uncharged) {
  Table& derived = static_cast<Table&>(*this);
  typename Table::InsertState state;
  derived.StartInsert(state);

  const size_t total_charge = proto.GetTotalCharge();
  if (strict_capacity_limit) {
    Status s = ChargeUsageMaybeEvictStrict<Table>(
        total_charge, capacity,
        /*need_evict_for_occupancy=*/false, state);
    if (!s.ok()) {
      if (allow_uncharged) {
        proto.total_charge = 0;
      } else {
        return nullptr;
      }
    }
  } else {
    // Case strict_capacity_limit == false
    bool success = ChargeUsageMaybeEvictNonStrict<Table>(
        total_charge, capacity,
        /*need_evict_for_occupancy=*/false, state);
    if (!success) {
      // Force the issue
      usage_.fetch_add(total_charge, std::memory_order_relaxed);
    }
  }

  return StandaloneInsert<typename Table::HandleImpl>(proto);
}

template <class Table>
Status BaseClockTable::ChargeUsageMaybeEvictStrict(
    size_t total_charge, size_t capacity, bool need_evict_for_occupancy,
    typename Table::InsertState& state) {
  if (total_charge > capacity) {
    return Status::MemoryLimit(
        "Cache entry too large for a single cache shard: " +
        std::to_string(total_charge) + " > " + std::to_string(capacity));
  }
  // Grab any available capacity, and free up any more required.
  size_t old_usage = usage_.load(std::memory_order_relaxed);
  size_t new_usage;
    do {
      new_usage = std::min(capacity, old_usage + total_charge);
      if (new_usage == old_usage) {
        // No change needed
        break;
      }
    } while (!usage_.compare_exchange_weak(old_usage, new_usage,
                                           std::memory_order_relaxed));
  // How much do we need to evict then?
  size_t need_evict_charge = old_usage + total_charge - new_usage;
  size_t request_evict_charge = need_evict_charge;
  if (UNLIKELY(need_evict_for_occupancy) && request_evict_charge == 0) {
    // Require at least 1 eviction.
    request_evict_charge = 1;
  }
  if (request_evict_charge > 0) {
    EvictionData data;
    static_cast<Table*>(this)->Evict(request_evict_charge, state, &data);
    occupancy_.fetch_sub(data.freed_count, std::memory_order_release);
    if (LIKELY(data.freed_charge > need_evict_charge)) {
      assert(data.freed_count > 0);
      // Evicted more than enough
      usage_.fetch_sub(data.freed_charge - need_evict_charge,
                       std::memory_order_relaxed);
    } else if (data.freed_charge < need_evict_charge ||
               (UNLIKELY(need_evict_for_occupancy) && data.freed_count == 0)) {
      // Roll back to old usage minus evicted
      usage_.fetch_sub(data.freed_charge + (new_usage - old_usage),
                       std::memory_order_relaxed);
      if (data.freed_charge < need_evict_charge) {
        return Status::MemoryLimit(
            "Insert failed because unable to evict entries to stay within "
            "capacity limit.");
      } else {
        return Status::MemoryLimit(
            "Insert failed because unable to evict entries to stay within "
            "table occupancy limit.");
      }
    }
    // If we needed to evict something and we are proceeding, we must have
    // evicted something.
    assert(data.freed_count > 0);
  }
  return Status::OK();
}

template <class Table>
inline bool BaseClockTable::ChargeUsageMaybeEvictNonStrict(
    size_t total_charge, size_t capacity, bool need_evict_for_occupancy,
    typename Table::InsertState& state) {
  // For simplicity, we consider that either the cache can accept the insert
  // with no evictions, or we must evict enough to make (at least) enough
  // space. It could lead to unnecessary failures or excessive evictions in
  // some extreme cases, but allows a fast, simple protocol. If we allow a
  // race to get us over capacity, then we might never get back to capacity
  // limit if the sizes of entries allow each insertion to evict the minimum
  // charge. Thus, we should evict some extra if it's not a signifcant
  // portion of the shard capacity. This can have the side benefit of
  // involving fewer threads in eviction.
  size_t old_usage = usage_.load(std::memory_order_relaxed);
  size_t need_evict_charge;
  // NOTE: if total_charge > old_usage, there isn't yet enough to evict
  // `total_charge` amount. Even if we only try to evict `old_usage` amount,
  // there's likely something referenced and we would eat CPU looking for
  // enough to evict.
  if (old_usage + total_charge <= capacity || total_charge > old_usage) {
    // Good enough for me (might run over with a race)
    need_evict_charge = 0;
  } else {
    // Try to evict enough space, and maybe some extra
    need_evict_charge = total_charge;
    if (old_usage > capacity) {
      // Not too much to avoid thundering herd while avoiding strict
      // synchronization, such as the compare_exchange used with strict
      // capacity limit.
      need_evict_charge += std::min(capacity / 1024, total_charge) + 1;
    }
  }
  if (UNLIKELY(need_evict_for_occupancy) && need_evict_charge == 0) {
    // Special case: require at least 1 eviction if we only have to
    // deal with occupancy
    need_evict_charge = 1;
  }
  EvictionData data;
  if (need_evict_charge > 0) {
    static_cast<Table*>(this)->Evict(need_evict_charge, state, &data);
    // Deal with potential occupancy deficit
    if (UNLIKELY(need_evict_for_occupancy) && data.freed_count == 0) {
      assert(data.freed_charge == 0);
      // Can't meet occupancy requirement
      return false;
    } else {
      // Update occupancy for evictions
      occupancy_.fetch_sub(data.freed_count, std::memory_order_release);
    }
  }
  // Track new usage even if we weren't able to evict enough
  usage_.fetch_add(total_charge - data.freed_charge, std::memory_order_relaxed);
  // No underflow
  assert(usage_.load(std::memory_order_relaxed) < SIZE_MAX / 2);
  // Success
  return true;
}

void BaseClockTable::TrackAndReleaseEvictedEntry(
    ClockHandle* h, BaseClockTable::EvictionData* data) {
  data->freed_charge += h->GetTotalCharge();
  data->freed_count += 1;

  bool took_value_ownership = false;
  if (eviction_callback_) {
    // For key reconstructed from hash
    UniqueId64x2 unhashed;
    took_value_ownership =
        eviction_callback_(ClockCacheShard<HyperClockTable>::ReverseHash(
                               h->GetHash(), &unhashed, hash_seed_),
                           reinterpret_cast<Cache::Handle*>(h));
  }
  if (!took_value_ownership) {
    h->FreeData(allocator_);
  }
  MarkEmpty(*h);
}

template <class Table>
Status BaseClockTable::Insert(const ClockHandleBasicData& proto,
                              typename Table::HandleImpl** handle,
                              Cache::Priority priority, size_t capacity,
                              bool strict_capacity_limit) {
  using HandleImpl = typename Table::HandleImpl;
  Table& derived = static_cast<Table&>(*this);

  typename Table::InsertState state;
  derived.StartInsert(state);

  // Do we have the available occupancy? Optimistically assume we do
  // and deal with it if we don't.
  size_t old_occupancy = occupancy_.fetch_add(1, std::memory_order_acquire);
  // Whether we over-committed and need an eviction to make up for it
  bool need_evict_for_occupancy =
      !derived.GrowIfNeeded(old_occupancy + 1, state);

  // Usage/capacity handling is somewhat different depending on
  // strict_capacity_limit, but mostly pessimistic.
  bool use_standalone_insert = false;
  const size_t total_charge = proto.GetTotalCharge();
  if (strict_capacity_limit) {
    Status s = ChargeUsageMaybeEvictStrict<Table>(
        total_charge, capacity, need_evict_for_occupancy, state);
    if (!s.ok()) {
      // Revert occupancy
      occupancy_.fetch_sub(1, std::memory_order_relaxed);
      return s;
    }
  } else {
    // Case strict_capacity_limit == false
    bool success = ChargeUsageMaybeEvictNonStrict<Table>(
        total_charge, capacity, need_evict_for_occupancy, state);
    if (!success) {
      // Revert occupancy
      occupancy_.fetch_sub(1, std::memory_order_relaxed);
      if (handle == nullptr) {
        // Don't insert the entry but still return ok, as if the entry
        // inserted into cache and evicted immediately.
        proto.FreeData(allocator_);
        return Status::OK();
      } else {
        // Need to track usage of fallback standalone insert
        usage_.fetch_add(total_charge, std::memory_order_relaxed);
        use_standalone_insert = true;
      }
    }
  }

  if (!use_standalone_insert) {
    // Attempt a table insert, but abort if we find an existing entry for the
    // key. If we were to overwrite old entries, we would either
    // * Have to gain ownership over an existing entry to overwrite it, which
    // would only work if there are no outstanding (read) references and would
    // create a small gap in availability of the entry (old or new) to lookups.
    // * Have to insert into a suboptimal location (more probes) so that the
    // old entry can be kept around as well.

    uint64_t initial_countdown = GetInitialCountdown(priority);
    assert(initial_countdown > 0);

    HandleImpl* e =
        derived.DoInsert(proto, initial_countdown, handle != nullptr, state);

    if (e) {
      // Successfully inserted
      if (handle) {
        *handle = e;
      }
      return Status::OK();
    }
    // Not inserted
    // Revert occupancy
    occupancy_.fetch_sub(1, std::memory_order_relaxed);
    // Maybe fall back on standalone insert
    if (handle == nullptr) {
      // Revert usage
      usage_.fetch_sub(total_charge, std::memory_order_relaxed);
      // No underflow
      assert(usage_.load(std::memory_order_relaxed) < SIZE_MAX / 2);
      // As if unrefed entry immdiately evicted
      proto.FreeData(allocator_);
      return Status::OK();
    }

    use_standalone_insert = true;
  }

  // Run standalone insert
  assert(use_standalone_insert);

  *handle = StandaloneInsert<HandleImpl>(proto);

  // The OkOverwritten status is used to count "redundant" insertions into
  // block cache. This implementation doesn't strictly check for redundant
  // insertions, but we instead are probably interested in how many insertions
  // didn't go into the table (instead "standalone"), which could be redundant
  // Insert or some other reason (use_standalone_insert reasons above).
  return Status::OkOverwritten();
}

void BaseClockTable::Ref(ClockHandle& h) {
  // Increment acquire counter
  uint64_t old_meta = h.meta.fetch_add(ClockHandle::kAcquireIncrement,
                                       std::memory_order_acquire);

  assert((old_meta >> ClockHandle::kStateShift) &
         ClockHandle::kStateShareableBit);
  // Must have already had a reference
  assert(GetRefcount(old_meta) > 0);
  (void)old_meta;
}

#ifndef NDEBUG
void BaseClockTable::TEST_RefN(ClockHandle& h, size_t n) {
  // Increment acquire counter
  uint64_t old_meta = h.meta.fetch_add(n * ClockHandle::kAcquireIncrement,
                                       std::memory_order_acquire);

  assert((old_meta >> ClockHandle::kStateShift) &
         ClockHandle::kStateShareableBit);
  (void)old_meta;
}

void BaseClockTable::TEST_ReleaseNMinus1(ClockHandle* h, size_t n) {
  assert(n > 0);

  // Like n-1 Releases, but assumes one more will happen in the caller to take
  // care of anything like erasing an unreferenced, invisible entry.
  uint64_t old_meta = h->meta.fetch_add(
      (n - 1) * ClockHandle::kReleaseIncrement, std::memory_order_acquire);
  assert((old_meta >> ClockHandle::kStateShift) &
         ClockHandle::kStateShareableBit);
  (void)old_meta;
}
#endif

HyperClockTable::HyperClockTable(
    size_t capacity, bool /*strict_capacity_limit*/,
    CacheMetadataChargePolicy metadata_charge_policy,
    MemoryAllocator* allocator,
    const Cache::EvictionCallback* eviction_callback, const uint32_t* hash_seed,
    const Opts& opts)
    : BaseClockTable(metadata_charge_policy, allocator, eviction_callback,
                     hash_seed),
      length_bits_(CalcHashBits(capacity, opts.estimated_value_size,
                                metadata_charge_policy)),
      length_bits_mask_((size_t{1} << length_bits_) - 1),
      occupancy_limit_(static_cast<size_t>((uint64_t{1} << length_bits_) *
                                           kStrictLoadFactor)),
      array_(new HandleImpl[size_t{1} << length_bits_]) {
  if (metadata_charge_policy ==
      CacheMetadataChargePolicy::kFullChargeCacheMetadata) {
    usage_ += size_t{GetTableSize()} * sizeof(HandleImpl);
  }

  static_assert(sizeof(HandleImpl) == 64U,
                "Expecting size / alignment with common cache line size");
}

HyperClockTable::~HyperClockTable() {
  // Assumes there are no references or active operations on any slot/element
  // in the table.
  for (size_t i = 0; i < GetTableSize(); i++) {
    HandleImpl& h = array_[i];
    switch (h.meta >> ClockHandle::kStateShift) {
      case ClockHandle::kStateEmpty:
        // noop
        break;
      case ClockHandle::kStateInvisible:  // rare but possible
      case ClockHandle::kStateVisible:
        assert(GetRefcount(h.meta) == 0);
        h.FreeData(allocator_);
#ifndef NDEBUG
        Rollback(h.hashed_key, &h);
        ReclaimEntryUsage(h.GetTotalCharge());
#endif
        break;
      // otherwise
      default:
        assert(false);
        break;
    }
  }

#ifndef NDEBUG
  for (size_t i = 0; i < GetTableSize(); i++) {
    assert(array_[i].displacements.load() == 0);
  }
#endif

  assert(usage_.load() == 0 ||
         usage_.load() == size_t{GetTableSize()} * sizeof(HandleImpl));
  assert(occupancy_ == 0);
}

void HyperClockTable::StartInsert(InsertState&) {}

bool HyperClockTable::GrowIfNeeded(size_t new_occupancy, InsertState&) {
  return new_occupancy <= occupancy_limit_;
}

HyperClockTable::HandleImpl* HyperClockTable::DoInsert(
    const ClockHandleBasicData& proto, uint64_t initial_countdown,
    bool keep_ref, InsertState&) {
  bool already_matches = false;
  HandleImpl* e = FindSlot(
      proto.hashed_key,
      [&](HandleImpl* h) {
        return TryInsert(proto, *h, initial_countdown, keep_ref,
                         &already_matches);
      },
      [&](HandleImpl* h) {
        if (already_matches) {
          // Stop searching & roll back displacements
          Rollback(proto.hashed_key, h);
          return true;
        } else {
          // Keep going
          return false;
        }
      },
      [&](HandleImpl* h, bool is_last) {
        if (is_last) {
          // Search is ending. Roll back displacements
          Rollback(proto.hashed_key, h);
        } else {
          h->displacements.fetch_add(1, std::memory_order_relaxed);
        }
      });
  if (already_matches) {
    // Insertion skipped
    return nullptr;
  }
  if (e != nullptr) {
    // Successfully inserted
    return e;
  }
  // Else, no available slot found. Occupancy check should generally prevent
  // this, except it's theoretically possible for other threads to evict and
  // replace entries in the right order to hit every slot when it is populated.
  // Assuming random hashing, the chance of that should be no higher than
  // pow(kStrictLoadFactor, n) for n slots. That should be infeasible for
  // roughly n >= 256, so if this assertion fails, that suggests something is
  // going wrong.
  assert(GetTableSize() < 256);
  return nullptr;
}

HyperClockTable::HandleImpl* HyperClockTable::Lookup(
    const UniqueId64x2& hashed_key) {
  HandleImpl* e = FindSlot(
      hashed_key,
      [&](HandleImpl* h) {
        // Mostly branch-free version (similar performance)
        /*
        uint64_t old_meta = h->meta.fetch_add(ClockHandle::kAcquireIncrement,
                                     std::memory_order_acquire);
        bool Shareable = (old_meta >> (ClockHandle::kStateShift + 1)) & 1U;
        bool visible = (old_meta >> ClockHandle::kStateShift) & 1U;
        bool match = (h->key == key) & visible;
        h->meta.fetch_sub(static_cast<uint64_t>(Shareable & !match) <<
        ClockHandle::kAcquireCounterShift, std::memory_order_release); return
        match;
        */
        // Optimistic lookup should pay off when the table is relatively
        // sparse.
        constexpr bool kOptimisticLookup = true;
        uint64_t old_meta;
        if (!kOptimisticLookup) {
          old_meta = h->meta.load(std::memory_order_acquire);
          if ((old_meta >> ClockHandle::kStateShift) !=
              ClockHandle::kStateVisible) {
            return false;
          }
        }
        // (Optimistically) increment acquire counter
        old_meta = h->meta.fetch_add(ClockHandle::kAcquireIncrement,
                                     std::memory_order_acquire);
        // Check if it's an entry visible to lookups
        if ((old_meta >> ClockHandle::kStateShift) ==
            ClockHandle::kStateVisible) {
          // Acquired a read reference
          if (h->hashed_key == hashed_key) {
            // Match
            return true;
          } else {
            // Mismatch. Pretend we never took the reference
            Unref(*h);
          }
        } else if (UNLIKELY((old_meta >> ClockHandle::kStateShift) ==
                            ClockHandle::kStateInvisible)) {
          // Pretend we never took the reference
          Unref(*h);
        } else {
          // For other states, incrementing the acquire counter has no effect
          // so we don't need to undo it. Furthermore, we cannot safely undo
          // it because we did not acquire a read reference to lock the
          // entry in a Shareable state.
        }
        return false;
      },
      [&](HandleImpl* h) {
        return h->displacements.load(std::memory_order_relaxed) == 0;
      },
      [&](HandleImpl* /*h*/, bool /*is_last*/) {});

  return e;
}

bool HyperClockTable::Release(HandleImpl* h, bool useful,
                              bool erase_if_last_ref) {
  // In contrast with LRUCache's Release, this function won't delete the handle
  // when the cache is above capacity and the reference is the last one. Space
  // is only freed up by EvictFromClock (called by Insert when space is needed)
  // and Erase. We do this to avoid an extra atomic read of the variable usage_.

  uint64_t old_meta;
  if (useful) {
    // Increment release counter to indicate was used
    old_meta = h->meta.fetch_add(ClockHandle::kReleaseIncrement,
                                 std::memory_order_release);
  } else {
    // Decrement acquire counter to pretend it never happened
    old_meta = h->meta.fetch_sub(ClockHandle::kAcquireIncrement,
                                 std::memory_order_release);
  }

  assert((old_meta >> ClockHandle::kStateShift) &
         ClockHandle::kStateShareableBit);
  // No underflow
  assert(((old_meta >> ClockHandle::kAcquireCounterShift) &
          ClockHandle::kCounterMask) !=
         ((old_meta >> ClockHandle::kReleaseCounterShift) &
          ClockHandle::kCounterMask));

  if (erase_if_last_ref || UNLIKELY(old_meta >> ClockHandle::kStateShift ==
                                    ClockHandle::kStateInvisible)) {
    // Update for last fetch_add op
    if (useful) {
      old_meta += ClockHandle::kReleaseIncrement;
    } else {
      old_meta -= ClockHandle::kAcquireIncrement;
    }
    // Take ownership if no refs
    do {
      if (GetRefcount(old_meta) != 0) {
        // Not last ref at some point in time during this Release call
        // Correct for possible (but rare) overflow
        CorrectNearOverflow(old_meta, h->meta);
        return false;
      }
      if ((old_meta & (uint64_t{ClockHandle::kStateShareableBit}
                       << ClockHandle::kStateShift)) == 0) {
        // Someone else took ownership
        return false;
      }
      // Note that there's a small chance that we release, another thread
      // replaces this entry with another, reaches zero refs, and then we end
      // up erasing that other entry. That's an acceptable risk / imprecision.
    } while (!h->meta.compare_exchange_weak(
        old_meta,
        uint64_t{ClockHandle::kStateConstruction} << ClockHandle::kStateShift,
        std::memory_order_acquire));
    // Took ownership
    size_t total_charge = h->GetTotalCharge();
    if (UNLIKELY(h->IsStandalone())) {
      h->FreeData(allocator_);
      // Delete standalone handle
      delete h;
      standalone_usage_.fetch_sub(total_charge, std::memory_order_relaxed);
      usage_.fetch_sub(total_charge, std::memory_order_relaxed);
    } else {
      Rollback(h->hashed_key, h);
      FreeDataMarkEmpty(*h, allocator_);
      ReclaimEntryUsage(total_charge);
    }
    return true;
  } else {
    // Correct for possible (but rare) overflow
    CorrectNearOverflow(old_meta, h->meta);
    return false;
  }
}

#ifndef NDEBUG
void HyperClockTable::TEST_ReleaseN(HandleImpl* h, size_t n) {
  if (n > 0) {
    // Do n-1 simple releases first
    TEST_ReleaseNMinus1(h, n);

    // Then the last release might be more involved
    Release(h, /*useful*/ true, /*erase_if_last_ref*/ false);
  }
}
#endif

void HyperClockTable::Erase(const UniqueId64x2& hashed_key) {
  (void)FindSlot(
      hashed_key,
      [&](HandleImpl* h) {
        // Could be multiple entries in rare cases. Erase them all.
        // Optimistically increment acquire counter
        uint64_t old_meta = h->meta.fetch_add(ClockHandle::kAcquireIncrement,
                                              std::memory_order_acquire);
        // Check if it's an entry visible to lookups
        if ((old_meta >> ClockHandle::kStateShift) ==
            ClockHandle::kStateVisible) {
          // Acquired a read reference
          if (h->hashed_key == hashed_key) {
            // Match. Set invisible.
            old_meta =
                h->meta.fetch_and(~(uint64_t{ClockHandle::kStateVisibleBit}
                                    << ClockHandle::kStateShift),
                                  std::memory_order_acq_rel);
            // Apply update to local copy
            old_meta &= ~(uint64_t{ClockHandle::kStateVisibleBit}
                          << ClockHandle::kStateShift);
            for (;;) {
              uint64_t refcount = GetRefcount(old_meta);
              assert(refcount > 0);
              if (refcount > 1) {
                // Not last ref at some point in time during this Erase call
                // Pretend we never took the reference
                Unref(*h);
                break;
              } else if (h->meta.compare_exchange_weak(
                             old_meta,
                             uint64_t{ClockHandle::kStateConstruction}
                                 << ClockHandle::kStateShift,
                             std::memory_order_acq_rel)) {
                // Took ownership
                assert(hashed_key == h->hashed_key);
                size_t total_charge = h->GetTotalCharge();
                FreeDataMarkEmpty(*h, allocator_);
                ReclaimEntryUsage(total_charge);
                // We already have a copy of hashed_key in this case, so OK to
                // delay Rollback until after releasing the entry
                Rollback(hashed_key, h);
                break;
              }
            }
          } else {
            // Mismatch. Pretend we never took the reference
            Unref(*h);
          }
        } else if (UNLIKELY((old_meta >> ClockHandle::kStateShift) ==
                            ClockHandle::kStateInvisible)) {
          // Pretend we never took the reference
          Unref(*h);
        } else {
          // For other states, incrementing the acquire counter has no effect
          // so we don't need to undo it.
        }
        return false;
      },
      [&](HandleImpl* h) {
        return h->displacements.load(std::memory_order_relaxed) == 0;
      },
      [&](HandleImpl* /*h*/, bool /*is_last*/) {});
}

void HyperClockTable::EraseUnRefEntries() {
  for (size_t i = 0; i <= this->length_bits_mask_; i++) {
    HandleImpl& h = array_[i];

    uint64_t old_meta = h.meta.load(std::memory_order_relaxed);
    if (old_meta & (uint64_t{ClockHandle::kStateShareableBit}
                    << ClockHandle::kStateShift) &&
        GetRefcount(old_meta) == 0 &&
        h.meta.compare_exchange_strong(old_meta,
                                       uint64_t{ClockHandle::kStateConstruction}
                                           << ClockHandle::kStateShift,
                                       std::memory_order_acquire)) {
      // Took ownership
      size_t total_charge = h.GetTotalCharge();
      Rollback(h.hashed_key, &h);
      FreeDataMarkEmpty(h, allocator_);
      ReclaimEntryUsage(total_charge);
    }
  }
}

template <typename MatchFn, typename AbortFn, typename UpdateFn>
inline HyperClockTable::HandleImpl* HyperClockTable::FindSlot(
    const UniqueId64x2& hashed_key, const MatchFn& match_fn,
    const AbortFn& abort_fn, const UpdateFn& update_fn) {
  // NOTE: upper 32 bits of hashed_key[0] is used for sharding
  //
  // We use double-hashing probing. Every probe in the sequence is a
  // pseudorandom integer, computed as a linear function of two random hashes,
  // which we call base and increment. Specifically, the i-th probe is base + i
  // * increment modulo the table size.
  size_t base = static_cast<size_t>(hashed_key[1]);
  // We use an odd increment, which is relatively prime with the power-of-two
  // table size. This implies that we cycle back to the first probe only
  // after probing every slot exactly once.
  // TODO: we could also reconsider linear probing, though locality benefits
  // are limited because each slot is a full cache line
  size_t increment = static_cast<size_t>(hashed_key[0]) | 1U;
  size_t first = ModTableSize(base);
  size_t current = first;
  bool is_last;
  do {
    HandleImpl* h = &array_[current];
    if (match_fn(h)) {
      return h;
    }
    if (abort_fn(h)) {
      return nullptr;
    }
    current = ModTableSize(current + increment);
    is_last = current == first;
    update_fn(h, is_last);
  } while (!is_last);
  // We looped back.
  return nullptr;
}

inline void HyperClockTable::Rollback(const UniqueId64x2& hashed_key,
                                      const HandleImpl* h) {
  size_t current = ModTableSize(hashed_key[1]);
  size_t increment = static_cast<size_t>(hashed_key[0]) | 1U;
  while (&array_[current] != h) {
    array_[current].displacements.fetch_sub(1, std::memory_order_relaxed);
    current = ModTableSize(current + increment);
  }
}

inline void HyperClockTable::ReclaimEntryUsage(size_t total_charge) {
  auto old_occupancy = occupancy_.fetch_sub(1U, std::memory_order_release);
  (void)old_occupancy;
  // No underflow
  assert(old_occupancy > 0);
  auto old_usage = usage_.fetch_sub(total_charge, std::memory_order_relaxed);
  (void)old_usage;
  // No underflow
  assert(old_usage >= total_charge);
}

inline void HyperClockTable::Evict(size_t requested_charge, InsertState&,
                                   EvictionData* data) {
  // precondition
  assert(requested_charge > 0);

  // TODO: make a tuning parameter?
  constexpr size_t step_size = 4;

  // First (concurrent) increment clock pointer
  uint64_t old_clock_pointer =
      clock_pointer_.fetch_add(step_size, std::memory_order_relaxed);

  // Cap the eviction effort at this thread (along with those operating in
  // parallel) circling through the whole structure kMaxCountdown times.
  // In other words, this eviction run must find something/anything that is
  // unreferenced at start of and during the eviction run that isn't reclaimed
  // by a concurrent eviction run.
  uint64_t max_clock_pointer =
      old_clock_pointer + (ClockHandle::kMaxCountdown << length_bits_);

  for (;;) {
    for (size_t i = 0; i < step_size; i++) {
      HandleImpl& h = array_[ModTableSize(Lower32of64(old_clock_pointer + i))];
      bool evicting = ClockUpdate(h);
      if (evicting) {
        Rollback(h.hashed_key, &h);
        TrackAndReleaseEvictedEntry(&h, data);
      }
    }

    // Loop exit condition
    if (data->freed_charge >= requested_charge) {
      return;
    }
    if (old_clock_pointer >= max_clock_pointer) {
      return;
    }

    // Advance clock pointer (concurrently)
    old_clock_pointer =
        clock_pointer_.fetch_add(step_size, std::memory_order_relaxed);
  }
}

template <class Table>
ClockCacheShard<Table>::ClockCacheShard(
    size_t capacity, bool strict_capacity_limit,
    CacheMetadataChargePolicy metadata_charge_policy,
    MemoryAllocator* allocator,
    const Cache::EvictionCallback* eviction_callback, const uint32_t* hash_seed,
    const typename Table::Opts& opts)
    : CacheShardBase(metadata_charge_policy),
      table_(capacity, strict_capacity_limit, metadata_charge_policy, allocator,
             eviction_callback, hash_seed, opts),
      capacity_(capacity),
      strict_capacity_limit_(strict_capacity_limit) {
  // Initial charge metadata should not exceed capacity
  assert(table_.GetUsage() <= capacity_ || capacity_ < sizeof(HandleImpl));
}

template <class Table>
void ClockCacheShard<Table>::EraseUnRefEntries() {
  table_.EraseUnRefEntries();
}

template <class Table>
void ClockCacheShard<Table>::ApplyToSomeEntries(
    const std::function<void(const Slice& key, Cache::ObjectPtr value,
                             size_t charge,
                             const Cache::CacheItemHelper* helper)>& callback,
    size_t average_entries_per_lock, size_t* state) {
  // The state will be a simple index into the table. Even with a dynamic
  // hyper clock cache, entries will generally stay in their existing
  // slots, so we don't need to be aware of the high-level organization
  // that makes lookup efficient.
  size_t length = table_.GetTableSize();

  assert(average_entries_per_lock > 0);

  size_t index_begin = *state;
  size_t index_end = index_begin + average_entries_per_lock;
  if (index_end >= length) {
    // Going to end.
    index_end = length;
    *state = SIZE_MAX;
  } else {
    *state = index_end;
  }

  auto hash_seed = table_.GetHashSeed();
  ConstApplyToEntriesRange(
      [callback, hash_seed](const HandleImpl& h) {
        UniqueId64x2 unhashed;
        callback(ReverseHash(h.hashed_key, &unhashed, hash_seed), h.value,
                 h.GetTotalCharge(), h.helper);
      },
      table_.HandlePtr(index_begin), table_.HandlePtr(index_end), false);
}

int HyperClockTable::CalcHashBits(
    size_t capacity, size_t estimated_value_size,
    CacheMetadataChargePolicy metadata_charge_policy) {
  double average_slot_charge = estimated_value_size * kLoadFactor;
  if (metadata_charge_policy == kFullChargeCacheMetadata) {
    average_slot_charge += sizeof(HandleImpl);
  }
  assert(average_slot_charge > 0.0);
  uint64_t num_slots =
      static_cast<uint64_t>(capacity / average_slot_charge + 0.999999);

  int hash_bits = FloorLog2((num_slots << 1) - 1);
  if (metadata_charge_policy == kFullChargeCacheMetadata) {
    // For very small estimated value sizes, it's possible to overshoot
    while (hash_bits > 0 &&
           uint64_t{sizeof(HandleImpl)} << hash_bits > capacity) {
      hash_bits--;
    }
  }
  return hash_bits;
}

template <class Table>
void ClockCacheShard<Table>::SetCapacity(size_t capacity) {
  capacity_.store(capacity, std::memory_order_relaxed);
  // next Insert will take care of any necessary evictions
}

template <class Table>
void ClockCacheShard<Table>::SetStrictCapacityLimit(
    bool strict_capacity_limit) {
  strict_capacity_limit_.store(strict_capacity_limit,
                               std::memory_order_relaxed);
  // next Insert will take care of any necessary evictions
}

template <class Table>
Status ClockCacheShard<Table>::Insert(const Slice& key,
                                      const UniqueId64x2& hashed_key,
                                      Cache::ObjectPtr value,
                                      const Cache::CacheItemHelper* helper,
                                      size_t charge, HandleImpl** handle,
                                      Cache::Priority priority) {
  if (UNLIKELY(key.size() != kCacheKeySize)) {
    return Status::NotSupported("ClockCache only supports key size " +
                                std::to_string(kCacheKeySize) + "B");
  }
  ClockHandleBasicData proto;
  proto.hashed_key = hashed_key;
  proto.value = value;
  proto.helper = helper;
  proto.total_charge = charge;
  return table_.template Insert<Table>(
      proto, handle, priority, capacity_.load(std::memory_order_relaxed),
      strict_capacity_limit_.load(std::memory_order_relaxed));
}

template <class Table>
typename Table::HandleImpl* ClockCacheShard<Table>::CreateStandalone(
    const Slice& key, const UniqueId64x2& hashed_key, Cache::ObjectPtr obj,
    const Cache::CacheItemHelper* helper, size_t charge, bool allow_uncharged) {
  if (UNLIKELY(key.size() != kCacheKeySize)) {
    return nullptr;
  }
  ClockHandleBasicData proto;
  proto.hashed_key = hashed_key;
  proto.value = obj;
  proto.helper = helper;
  proto.total_charge = charge;
  return table_.template CreateStandalone<Table>(
      proto, capacity_.load(std::memory_order_relaxed),
      strict_capacity_limit_.load(std::memory_order_relaxed), allow_uncharged);
}

template <class Table>
typename ClockCacheShard<Table>::HandleImpl* ClockCacheShard<Table>::Lookup(
    const Slice& key, const UniqueId64x2& hashed_key) {
  if (UNLIKELY(key.size() != kCacheKeySize)) {
    return nullptr;
  }
  return table_.Lookup(hashed_key);
}

template <class Table>
bool ClockCacheShard<Table>::Ref(HandleImpl* h) {
  if (h == nullptr) {
    return false;
  }
  table_.Ref(*h);
  return true;
}

template <class Table>
bool ClockCacheShard<Table>::Release(HandleImpl* handle, bool useful,
                                     bool erase_if_last_ref) {
  if (handle == nullptr) {
    return false;
  }
  return table_.Release(handle, useful, erase_if_last_ref);
}

#ifndef NDEBUG
template <class Table>
void ClockCacheShard<Table>::TEST_RefN(HandleImpl* h, size_t n) {
  table_.TEST_RefN(*h, n);
}

template <class Table>
void ClockCacheShard<Table>::TEST_ReleaseN(HandleImpl* h, size_t n) {
  table_.TEST_ReleaseN(h, n);
}
#endif

template <class Table>
bool ClockCacheShard<Table>::Release(HandleImpl* handle,
                                     bool erase_if_last_ref) {
  return Release(handle, /*useful=*/true, erase_if_last_ref);
}

template <class Table>
void ClockCacheShard<Table>::Erase(const Slice& key,
                                   const UniqueId64x2& hashed_key) {
  if (UNLIKELY(key.size() != kCacheKeySize)) {
    return;
  }
  table_.Erase(hashed_key);
}

template <class Table>
size_t ClockCacheShard<Table>::GetUsage() const {
  return table_.GetUsage();
}

template <class Table>
size_t ClockCacheShard<Table>::GetStandaloneUsage() const {
  return table_.GetStandaloneUsage();
}

template <class Table>
size_t ClockCacheShard<Table>::GetCapacity() const {
  return capacity_;
}

template <class Table>
size_t ClockCacheShard<Table>::GetPinnedUsage() const {
  // Computes the pinned usage by scanning the whole hash table. This
  // is slow, but avoids keeping an exact counter on the clock usage,
  // i.e., the number of not externally referenced elements.
  // Why avoid this counter? Because Lookup removes elements from the clock
  // list, so it would need to update the pinned usage every time,
  // which creates additional synchronization costs.
  size_t table_pinned_usage = 0;
  const bool charge_metadata =
      metadata_charge_policy_ == kFullChargeCacheMetadata;
  ConstApplyToEntriesRange(
      [&table_pinned_usage, charge_metadata](const HandleImpl& h) {
        uint64_t meta = h.meta.load(std::memory_order_relaxed);
        uint64_t refcount = GetRefcount(meta);
        // Holding one ref for ConstApplyToEntriesRange
        assert(refcount > 0);
        if (refcount > 1) {
          table_pinned_usage += h.GetTotalCharge();
          if (charge_metadata) {
            table_pinned_usage += sizeof(HandleImpl);
          }
        }
      },
      table_.HandlePtr(0), table_.HandlePtr(table_.GetTableSize()), true);

  return table_pinned_usage + table_.GetStandaloneUsage();
}

template <class Table>
size_t ClockCacheShard<Table>::GetOccupancyCount() const {
  return table_.GetOccupancy();
}

template <class Table>
size_t ClockCacheShard<Table>::GetOccupancyLimit() const {
  return table_.GetOccupancyLimit();
}

template <class Table>
size_t ClockCacheShard<Table>::GetTableAddressCount() const {
  return table_.GetTableSize();
}

// Explicit instantiation
template class ClockCacheShard<HyperClockTable>;

HyperClockCache::HyperClockCache(const HyperClockCacheOptions& opts)
    : ShardedCache(opts) {
  assert(opts.estimated_entry_charge > 0 ||
         opts.metadata_charge_policy != kDontChargeCacheMetadata);
  // TODO: should not need to go through two levels of pointer indirection to
  // get to table entries
  size_t per_shard = GetPerShardCapacity();
  MemoryAllocator* alloc = this->memory_allocator();
  InitShards([&](Shard* cs) {
    HyperClockTable::Opts table_opts;
    table_opts.estimated_value_size = opts.estimated_entry_charge;
    new (cs) Shard(per_shard, opts.strict_capacity_limit,
                   opts.metadata_charge_policy, alloc, &eviction_callback_,
                   &hash_seed_, table_opts);
  });
}

Cache::ObjectPtr HyperClockCache::Value(Handle* handle) {
  return reinterpret_cast<const HandleImpl*>(handle)->value;
}

size_t HyperClockCache::GetCharge(Handle* handle) const {
  return reinterpret_cast<const HandleImpl*>(handle)->GetTotalCharge();
}

const Cache::CacheItemHelper* HyperClockCache::GetCacheItemHelper(
    Handle* handle) const {
  auto h = reinterpret_cast<const HandleImpl*>(handle);
  return h->helper;
}

namespace {

// For each cache shard, estimate what the table load factor would be if
// cache filled to capacity with average entries. This is considered
// indicative of a potential problem if the shard is essentially operating
// "at limit", which we define as high actual usage (>80% of capacity)
// or actual occupancy very close to limit (>95% of limit).
// Also, for each shard compute the recommended estimated_entry_charge,
// and keep the minimum one for use as overall recommendation.
void AddShardEvaluation(const HyperClockCache::Shard& shard,
                        std::vector<double>& predicted_load_factors,
                        size_t& min_recommendation) {
  size_t usage = shard.GetUsage() - shard.GetStandaloneUsage();
  size_t capacity = shard.GetCapacity();
  double usage_ratio = 1.0 * usage / capacity;

  size_t occupancy = shard.GetOccupancyCount();
  size_t occ_limit = shard.GetOccupancyLimit();
  double occ_ratio = 1.0 * occupancy / occ_limit;
  if (usage == 0 || occupancy == 0 || (usage_ratio < 0.8 && occ_ratio < 0.95)) {
    // Skip as described above
    return;
  }

  // If filled to capacity, what would the occupancy ratio be?
  double ratio = occ_ratio / usage_ratio;
  // Given max load factor, what that load factor be?
  double lf = ratio * kStrictLoadFactor;
  predicted_load_factors.push_back(lf);

  // Update min_recommendation also
  size_t recommendation = usage / occupancy;
  min_recommendation = std::min(min_recommendation, recommendation);
}

}  // namespace

void HyperClockCache::ReportProblems(
    const std::shared_ptr<Logger>& info_log) const {
  uint32_t shard_count = GetNumShards();
  std::vector<double> predicted_load_factors;
  size_t min_recommendation = SIZE_MAX;
  const_cast<HyperClockCache*>(this)->ForEachShard(
      [&](HyperClockCache::Shard* shard) {
        AddShardEvaluation(*shard, predicted_load_factors, min_recommendation);
      });

  if (predicted_load_factors.empty()) {
    // None operating "at limit" -> nothing to report
    return;
  }
  std::sort(predicted_load_factors.begin(), predicted_load_factors.end());

  // First, if the average load factor is within spec, we aren't going to
  // complain about a few shards being out of spec.
  // NOTE: this is only the average among cache shards operating "at limit,"
  // which should be representative of what we care about. It it normal, even
  // desirable, for a cache to operate "at limit" so this should not create
  // selection bias. See AddShardEvaluation().
  // TODO: Consider detecting cases where decreasing the number of shards
  // would be good, e.g. serious imbalance among shards.
  double average_load_factor =
      std::accumulate(predicted_load_factors.begin(),
                      predicted_load_factors.end(), 0.0) /
      shard_count;

  constexpr double kLowSpecLoadFactor = kLoadFactor / 2;
  constexpr double kMidSpecLoadFactor = kLoadFactor / 1.414;
  if (average_load_factor > kLoadFactor) {
    // Out of spec => Consider reporting load factor too high
    // Estimate effective overall capacity loss due to enforcing occupancy limit
    double lost_portion = 0.0;
    int over_count = 0;
    for (double lf : predicted_load_factors) {
      if (lf > kStrictLoadFactor) {
        ++over_count;
        lost_portion += (lf - kStrictLoadFactor) / lf / shard_count;
      }
    }
    // >= 20% loss -> error
    // >= 10% loss -> consistent warning
    // >= 1% loss -> intermittent warning
    InfoLogLevel level = InfoLogLevel::INFO_LEVEL;
    bool report = true;
    if (lost_portion > 0.2) {
      level = InfoLogLevel::ERROR_LEVEL;
    } else if (lost_portion > 0.1) {
      level = InfoLogLevel::WARN_LEVEL;
    } else if (lost_portion > 0.01) {
      int report_percent = static_cast<int>(lost_portion * 100.0);
      if (Random::GetTLSInstance()->PercentTrue(report_percent)) {
        level = InfoLogLevel::WARN_LEVEL;
      }
    } else {
      // don't report
      report = false;
    }
    if (report) {
      ROCKS_LOG_AT_LEVEL(
          info_log, level,
          "HyperClockCache@%p unable to use estimated %.1f%% capacity because "
          "of "
          "full occupancy in %d/%u cache shards (estimated_entry_charge too "
          "high). Recommend estimated_entry_charge=%zu",
          this, lost_portion * 100.0, over_count, (unsigned)shard_count,
          min_recommendation);
    }
  } else if (average_load_factor < kLowSpecLoadFactor) {
    // Out of spec => Consider reporting load factor too low
    // But cautiously because low is not as big of a problem.

    // Only report if highest occupancy shard is also below
    // spec and only if average is substantially out of spec
    if (predicted_load_factors.back() < kLowSpecLoadFactor &&
        average_load_factor < kLowSpecLoadFactor / 1.414) {
      InfoLogLevel level = InfoLogLevel::INFO_LEVEL;
      if (average_load_factor < kLowSpecLoadFactor / 2) {
        level = InfoLogLevel::WARN_LEVEL;
      }
      ROCKS_LOG_AT_LEVEL(
          info_log, level,
          "HyperClockCache@%p table has low occupancy at full capacity. Higher "
          "estimated_entry_charge (about %.1fx) would likely improve "
          "performance. Recommend estimated_entry_charge=%zu",
          this, kMidSpecLoadFactor / average_load_factor, min_recommendation);
    }
  }
}

// =======================================================================
//                             FastClockCache
// =======================================================================

// Used length  | min shift  | threshold  | max shift
// 2            | 1          | 0          | 1
// 3            | 1          | 1          | 2
// 4            | 2          | 0          | 2
// 5            | 2          | 1          | 3
// 6            | 2          | 2          | 3
// 7            | 2          | 3          | 3
// 8            | 3          | 0          | 3
// 9            | 3          | 1          | 4
// ...
// Note:
// * min shift = floor(log2(used length))
// * max shift = ceil(log2(used length))
// * used length == (1 << shift) + threshold
// Also, shift=0 is never used in practice, so is reserved for "unset"

namespace {

inline int LengthInfoToMinShift(uint64_t length_info) {
  int mask_shift = BitwiseAnd(length_info, int{255});
  assert(mask_shift <= 63);
  assert(mask_shift > 0);
  return mask_shift;
}

inline size_t LengthInfoToThreshold(uint64_t length_info) {
  return static_cast<size_t>(length_info >> 8);
}

inline size_t LengthInfoToUsedLength(uint64_t length_info) {
  size_t threshold = LengthInfoToThreshold(length_info);
  int shift = LengthInfoToMinShift(length_info);
  assert(threshold < (size_t{1} << shift));
  size_t used_length = (size_t{1} << shift) + threshold;
  assert(used_length >= 2);
  return used_length;
}

inline uint64_t UsedLengthToLengthInfo(size_t used_length) {
  assert(used_length >= 2);
  int shift = FloorLog2(used_length);
  uint64_t threshold = BottomNBits(used_length, shift);
  uint64_t length_info =
      (uint64_t{threshold} << 8) + static_cast<uint64_t>(shift);
  assert(LengthInfoToUsedLength(length_info) == used_length);
  assert(LengthInfoToMinShift(length_info) == shift);
  assert(LengthInfoToThreshold(length_info) == threshold);
  return length_info;
}

inline size_t GetStartingLength() {
  return port::kPageSize / sizeof(FastClockTable::HandleImpl);
}

inline size_t GetHomeIndex(uint64_t hash, int shift) {
  return static_cast<size_t>(BottomNBits(hash, shift));
}

inline void GetHomeIndexAndShift(uint64_t length_info, uint64_t hash,
                                 size_t* home, int* shift) {
  int min_shift = LengthInfoToMinShift(length_info);
  size_t threshold = LengthInfoToThreshold(length_info);
  bool extra_shift = GetHomeIndex(hash, min_shift) < threshold;
  *home = GetHomeIndex(hash, min_shift + extra_shift);
  *shift = min_shift + extra_shift;
  assert(*home < LengthInfoToUsedLength(length_info));
}

inline int GetShiftFromNextWithShift(uint64_t next_with_shift) {
  return BitwiseAnd(next_with_shift, FastClockTable::HandleImpl::kShiftMask);
}

inline size_t GetNextFromNextWithShift(uint64_t next_with_shift) {
  return static_cast<size_t>(next_with_shift >>
                             FastClockTable::HandleImpl::kNextShift);
}

inline uint64_t MakeNextWithShift(size_t next, int shift, bool end) {
  return (end ? FastClockTable::HandleImpl::kNextEndFlag : 0U) |
         (uint64_t{next} << FastClockTable::HandleImpl::kNextShift) |
         static_cast<uint64_t>(shift);
}

inline bool MatchAndRef(const UniqueId64x2* hashed_key, ClockHandle& h,
                        int shift = 0, size_t home = 0,
                        bool* full_match_or_unknown = nullptr,
                        bool* evicting = nullptr) {
  // Must be at least something to match
  assert(hashed_key || shift > 0);

  uint64_t old_meta;
  // (Optimistically) increment acquire counter.
  old_meta = h.meta.fetch_add(ClockHandle::kAcquireIncrement,
                              std::memory_order_acquire);
  // Check if it's a referencable (sharable) entry
  if ((old_meta & (uint64_t{ClockHandle::kStateShareableBit}
                   << ClockHandle::kStateShift)) == 0) {
    // For non-sharable states, incrementing the acquire counter has no effect
    // so we don't need to undo it. Furthermore, we cannot safely undo
    // it because we did not acquire a read reference to lock the
    // entry in a Shareable state.
    if (full_match_or_unknown) {
      *full_match_or_unknown = true;
    }
    return false;
  }
  // Else acquired a read reference
  uint64_t meta = old_meta + ClockHandle::kAcquireIncrement;
  assert(GetRefcount(meta) > 0);
  if (hashed_key && h.hashed_key == *hashed_key &&
      LIKELY(old_meta & (uint64_t{ClockHandle::kStateVisibleBit}
                         << ClockHandle::kStateShift))) {
    // Match on full key, visible
    if (full_match_or_unknown) {
      *full_match_or_unknown = true;
    }
    return true;
  } else if (shift > 0 && home == BottomNBits(h.hashed_key[0], shift)) {
    // Match on home address, possibly invisible
    if (evicting) {
      // Perform clock update, and/or purge invisible, while holding
      // one read ref ourselves
      // WART: this is sufficiently different from ClockUpdate that it's
      // difficult to share code with it
      assert(*evicting == false);  // default as precondition
      for (;;) {
        uint64_t acquire_count = (meta >> ClockHandle::kAcquireCounterShift) &
                                 ClockHandle::kCounterMask;
        uint64_t release_count = (meta >> ClockHandle::kReleaseCounterShift) &
                                 ClockHandle::kCounterMask;

        // We are holding a ref. It should not appear unrefed.
        assert(acquire_count != release_count);
        if (acquire_count != release_count + 1) {
          // Saw another reference on the entrty. No clock update or eviction.
          break;
        }
        // Else, otherwise unreferenced, eligible for clock update and possible
        // eviction
        if (release_count > 0 &&
            LIKELY(meta & (uint64_t{ClockHandle::kStateVisibleBit}
                           << ClockHandle::kStateShift))) {
          // Clock update (decrement counter)
          uint64_t new_release_count = std::min(
              release_count - 1, uint64_t{ClockHandle::kMaxCountdown} - 1);
          uint64_t new_meta =
              (uint64_t{ClockHandle::kStateVisible}
               << ClockHandle::kStateShift) |
              (new_release_count << ClockHandle::kReleaseCounterShift) |
              ((new_release_count + 1) << ClockHandle::kAcquireCounterShift);
          if (h.meta.compare_exchange_weak(meta, new_meta,
                                           std::memory_order_acq_rel)) {
            // Success, keeping a read ref but no eviction
            break;
          }
          // Else retry
        } else {
          // Invisible or expired. Take ownership for eviction.
          uint64_t new_meta = uint64_t{ClockHandle::kStateConstruction}
                              << ClockHandle::kStateShift;
          if (h.meta.compare_exchange_weak(meta, new_meta,
                                           std::memory_order_acq_rel)) {
            // Success, but took ownership for eviction
            *evicting = true;
            // Like other "under construction" cases
            if (full_match_or_unknown) {
              *full_match_or_unknown = true;
            }
            return false;
          }
        }
      }
    }
    if (full_match_or_unknown) {
      *full_match_or_unknown = false;
    }
    return true;
  } else {
    // Mismatch. Pretend we never took the reference
    Unref(h);
    if (full_match_or_unknown) {
      *full_match_or_unknown = false;
    }
    return false;
  }
}

}  // namespace

FastClockTable::FastClockTable(size_t capacity, bool /*strict_capacity_limit*/,
                               CacheMetadataChargePolicy metadata_charge_policy,
                               MemoryAllocator* allocator,
                               const Cache::EvictionCallback* eviction_callback,
                               const uint32_t* hash_seed, const Opts& opts)
    : BaseClockTable(metadata_charge_policy, allocator, eviction_callback,
                     hash_seed),
      max_usable_length_(CalcMaxUsableLength(capacity, opts.min_avg_value_size,
                                             metadata_charge_policy)),
      array_mem_(MemMapping::AllocateLazyZeroed(sizeof(HandleImpl) *
                                                max_usable_length_)),
      array_(static_cast<HandleImpl*>(array_mem_.Get())),
      length_info_(UsedLengthToLengthInfo(GetStartingLength())),
      clock_pointer_mask_(
          BottomNBits(UINT64_MAX, LengthInfoToMinShift(length_info_.load()))) {
  if (metadata_charge_policy ==
      CacheMetadataChargePolicy::kFullChargeCacheMetadata) {
    // NOTE: ignoring page boundaries for simplicity
    usage_ += size_t{GetTableSize()} * sizeof(HandleImpl);
  }

  static_assert(sizeof(HandleImpl) == 64U,
                "Expecting size / alignment with common cache line size");

  // Populate head pointers
  uint64_t length_info = length_info_.load();
  int min_shift = LengthInfoToMinShift(length_info);
  int max_shift = min_shift + 1;
  size_t major = uint64_t{1} << min_shift;
  size_t used_length = GetTableSize();

  assert(major <= used_length);
  assert(used_length <= major * 2);

  for (size_t i = 0; i < major; ++i) {
#ifndef NDEBUG
    int shift;
    size_t home;
#endif
    if (major + i < used_length) {
      array_[i].head_next_with_shift =
          MakeNextWithShift(i, max_shift, /*end=*/true);
      array_[major + i].head_next_with_shift =
          MakeNextWithShift(major + i, max_shift, /*end=*/true);
#ifndef NDEBUG
      GetHomeIndexAndShift(length_info, i, &home, &shift);
      assert(home == i);
      assert(shift == max_shift);
      GetHomeIndexAndShift(length_info, major + i, &home, &shift);
      assert(home == major + i);
      assert(shift == max_shift);
#endif
    } else {
      array_[i].head_next_with_shift =
          MakeNextWithShift(i, min_shift, /*end=*/true);
#ifndef NDEBUG
      GetHomeIndexAndShift(length_info, i, &home, &shift);
      assert(home == i);
      assert(shift == min_shift);
      GetHomeIndexAndShift(length_info, major + i, &home, &shift);
      assert(home == i);
      assert(shift == min_shift);
#endif
    }
  }
}

FastClockTable::~FastClockTable() {
  // Assumes there are no references or active operations on any slot/element
  // in the table.
  size_t end = GetTableSize();
#ifndef NDEBUG
  std::vector<bool> was_populated(end);
  std::vector<bool> was_pointed_to(end);
#endif
  for (size_t i = 0; i < end; i++) {
    HandleImpl& h = array_[i];
    switch (h.meta >> ClockHandle::kStateShift) {
      case ClockHandle::kStateEmpty:
        // noop
        break;
      case ClockHandle::kStateInvisible:  // rare but possible
      case ClockHandle::kStateVisible:
        assert(GetRefcount(h.meta) == 0);
        h.FreeData(allocator_);
#ifndef NDEBUG
        usage_.fetch_sub(h.total_charge, std::memory_order_relaxed);
        occupancy_.fetch_sub(1U, std::memory_order_relaxed);
        was_populated[i] = true;
        if ((h.chain_next_with_shift & HandleImpl::kNextEndFlag) == 0) {
          size_t next = GetNextFromNextWithShift(h.chain_next_with_shift);
          assert(!was_pointed_to[next]);
          was_pointed_to[next] = true;
        }
#endif
        break;
      // otherwise
      default:
        assert(false);
        break;
    }
#ifndef NDEBUG
    if ((h.head_next_with_shift & HandleImpl::kNextEndFlag) == 0) {
      size_t next = GetNextFromNextWithShift(h.head_next_with_shift);
      assert(!was_pointed_to[next]);
      was_pointed_to[next] = true;
    }
#endif
  }
#ifndef NDEBUG
  // This check is not perfect, but should detect most reasonable cases
  // of abandonned or floating entries, etc.  (A floating cycle would not
  // be reported as bad.)
  for (size_t i = 0; i < end; i++) {
    if (was_populated[i]) {
      assert(was_pointed_to[i]);
    } else {
      assert(!was_pointed_to[i]);
    }
  }
#endif

  assert(usage_.load() == 0 ||
         usage_.load() == size_t{GetTableSize()} * sizeof(HandleImpl));
  // TODO: metadata usage
  assert(occupancy_ == 0);
}

size_t FastClockTable::GetTableSize() const {
  return LengthInfoToUsedLength(length_info_.load(std::memory_order_acquire));
}

size_t FastClockTable::GetOccupancyLimit() const {
  // Assume 50% occupancy max
  return LengthInfoToUsedLength(length_info_.load(std::memory_order_acquire)) /
         2;
}

void FastClockTable::StartInsert(InsertState& state) {
  state.saved_length_info = length_info_.load(std::memory_order_acquire);
}

bool FastClockTable::GrowIfNeeded(size_t new_occupancy, InsertState& state) {
  size_t used_length = LengthInfoToUsedLength(state.saved_length_info);
  // Assume 50% occupancy max
  if (new_occupancy <= used_length / 2) {
    // Don't need to grow
    return true;
  }

  // Without a mutex, it's not easy to exactly coordinate on when growth
  // is strictly required between threads, but given that number of threads
  // tends to be reasonably bounded, approximate is fine.
  // At this point we commit the thread to growing unless we've reached the
  // limit. Try to take ownership of a grow slot as the first thread to
  // set its head_next_with_shift to non-zero, using kNextEndFlag as a
  // placeholder. (We don't need to be super efficient here.)
  size_t grow_home = used_length;
  for (;; ++grow_home) {
    // !!!!! FIXME !!!!!: Needs a unit test
    if (grow_home >= max_usable_length_) {
      // Can't grow any more
      return false;
    }
    uint64_t expected_zero = 0;

    bool own = array_[grow_home].head_next_with_shift.compare_exchange_strong(
        expected_zero, HandleImpl::kNextEndFlag, std::memory_order_acq_rel);
    if (own) {
      break;
    } else {
      // Taken by another thread. Try next slot.
      assert(expected_zero != 0);
    }
  }

  // TOOD: polish/etc.
  // AHome -Old-> A1 -Old-> B1 -Old-> A2 -Old-> B2 -Old-> AHome(End)
  // BHome -Old-/
  // ===>
  // AHome -Old-> A1 -Old-> B1 -Old-> A2 -Old-> B2 -Old-> AHome(End)
  // BHome -New-> PH -Old-/
  // ===> (disables B writes to AHome)
  // AHome -New-> A1 -Old-> B1 -Old-> A2 -Old-> B2 -Old-> AHome(End)
  // BHome -New-> PH -Old-/
  // ===> (upgrade one side)
  // AHome -New-> A1 -Old-> B1 -Old-> A2 -Old-> B2 -Old-> AHome(End)
  // BHome -New-> PH -New-/
  // ===> (pin next and jump the other side to it)
  // AHome -New-> A1 -----------Old-> A2 -Old-> B2 -Old-> AHome(End)
  // BHome -New-> PH -New-> B1 -Old/
  // ===> (upgrade one side)
  // AHome -New-> A1 -----------New-> A2 -Old-> B2 -Old-> AHome(End)
  // BHome -New-> PH -New-> B1 -Old/
  // ===> (pin next and jump the other side to it)
  // AHome -New-> A1 -New-> A2 -Old-> B2 -Old-> AHome(End)
  // BHome -New-> PH -New-> B1 -Old/
  // ===> (upgrade one side)
  // AHome -New-> A1 -New-> A2 -Old-> B2 -Old-> AHome(End)
  // BHome -New-> PH -New-> B1 -New/
  // ===> (no next to pin, but jump the other side to end)
  // AHome -New-> A1 -New-> A2 -----------Old-> AHome(End)
  // BHome -New-> PH -New-> B1 -New-> B2 -Old-/
  // ===> (upgrade A end)
  // AHome -New-> A1 -New-> A2 -New-> AHome(End)
  // BHome -New-> PH -New-> B1 -New-> B2 -Old-> AHome(End)
  // ===> (upgrade B end)
  // AHome -New-> A1 -New-> A2 -New-> AHome(End)
  // BHome -New-> PH -New-> B1 -New-> B2 -New-> BHome(End)
  // Then erase PH

  size_t new_used_length = grow_home + 1;
  uint64_t new_length_info = UsedLengthToLengthInfo(new_used_length);
  // Ceiling log2
  int new_shift = 1 + FloorLog2(new_used_length - 1);
  assert(new_shift >= 2);
  int old_shift = new_shift - 1;
  size_t old_home = BottomNBits(grow_home, old_shift);
  assert(old_home + (size_t{1} << old_shift) == grow_home);

  // For simplicity and to avoid waiting with entries pinned in the chain,
  // let's wait here on any grow operations that would feed into this one.
  while (LengthInfoToUsedLength(state.saved_length_info) < old_home) {
    std::this_thread::yield();
    StartInsert(/*mutable*/ state);
  }

  // TODO: re-review this
  // Next we want the new (grow) home and old home to share chains, so that
  // Lookup and (non-growing) Insert can proceed wait-free during this Grow
  // operation. Ideally, we would transactionally update all three of
  // (a) old home head, (b) new (grow) home head, and (c) length_info_ to
  // switch over to the new home mappings, but that's not possible. So we
  // need some way of dealing with intermediate states, and the way we do that
  // is to allow length_info_ to be behind if the shift amount on the heads
  // are updated. And we can essentially update both old home head and new home
  // head transactionally, because new home head is not used for lookup or
  // insertion until either old home head or length_info_ are updated. So a CAS
  // loop on old head, with updating new/grow head, suffices.

  // Any time we have two pointers to a chain entry, we need to prevent it
  // from being deleted, by taking a read reference. If we are trying to share
  // an entry that is already "under construction," we must wait for a
  // referencable entry to take its place (deletion to finish, or whatever).

  // Also, we can't naively share whatever is the first entry in the chain,
  // because with the head pointers using the new shift, the first entry has
  // to be a match for the appropriate home under the new shift. TODO: diagram
  // However, we would like to do this in a way such that each thread operating
  // on the cache has in the worst case only O(1) entries pinned for internal
  // handling purposes, to help avoid rare mishaps that would make eviction
  // more difficult. To do this, we can mark the new slot as "under
  // construction" and use it as a placeholder for a hypothetical entry with
  // the correct home. This entry will get mixed in with others that we hold a
  // read reference on, but we know that this one is special by its index and
  // that we need to call Remove on it.

  // Also also, each time we read another pointer in the chain, including
  // the head, we need to wait (spin/yield) if it's still not up-to-date with
  // the old shift (Grow from last generation still finishing)

  // For each of these, when present (!= SIZE_MAX), we either hold a read ref
  // or it's the placeholder at grow_home.
  size_t zero_chain_from = SIZE_MAX;
  size_t one_chain_from = SIZE_MAX;
  size_t shared_entry = SIZE_MAX;
  bool using_placeholder = false;

  for (;;) {
    // Need to do initial processing of head pointers
    assert(shared_entry == SIZE_MAX);
    assert(zero_chain_from == SIZE_MAX);
    assert(one_chain_from == SIZE_MAX);

    uint64_t next_with_shift =
        array_[old_home].head_next_with_shift.load(std::memory_order_acquire);
    assert(GetShiftFromNextWithShift(next_with_shift) == old_shift);

    // Easy case: next is effectively null
    if (next_with_shift & HandleImpl::kNextEndFlag) {
      assert(GetNextFromNextWithShift(next_with_shift) == old_home);
      array_[grow_home].head_next_with_shift.store(
          MakeNextWithShift(grow_home, new_shift, /*end=*/true),
          std::memory_order_release);
      if (array_[old_home].head_next_with_shift.compare_exchange_strong(
              next_with_shift,
              MakeNextWithShift(old_home, new_shift, /*end=*/true),
              std::memory_order_acq_rel)) {
        // Both heads successfully updated to (empty) with new shift
        break;
      } else {
        // next_with_shift modified by another thread; retry
        continue;
      }
    }

    // Try to pin first entry
    size_t first_entry = GetNextFromNextWithShift(next_with_shift);
    if (!MatchAndRef(/*hashed_key=*/nullptr, array_[first_entry], old_shift,
                     old_home)) {
      // FIXME: pro-active removal
      // Couldn't pin first entry; possibly being deleted which we need to
      // wait for. Yield then retry.
      std::this_thread::yield();
      continue;
    }

    // Move to next
    uint64_t next_next_with_shift =
        array_[first_entry].chain_next_with_shift.load(
            std::memory_order_acquire);
    assert(GetShiftFromNextWithShift(next_next_with_shift) == old_shift);

    size_t first_entry_new_home =
        GetHomeIndex(array_[first_entry].hashed_key[0], new_shift);
    assert(first_entry_new_home == old_home ||
           first_entry_new_home == grow_home);

    // Another easy case: just that one entry in chain
    if (next_next_with_shift & HandleImpl::kNextEndFlag) {
      assert(GetNextFromNextWithShift(next_next_with_shift) == old_home);
      uint64_t zero_head_next_with_shift;
      uint64_t one_head_next_with_shift;
      if (first_entry_new_home == grow_home) {
        zero_head_next_with_shift =
            MakeNextWithShift(old_home, new_shift, /*end=*/true);
        one_head_next_with_shift =
            MakeNextWithShift(first_entry, new_shift, /*end=*/false);
        one_chain_from = first_entry;
      } else {
        zero_head_next_with_shift =
            MakeNextWithShift(first_entry, new_shift, /*end=*/false);
        zero_chain_from = first_entry;
        one_head_next_with_shift =
            MakeNextWithShift(grow_home, new_shift, /*end=*/true);
      }

      array_[grow_home].head_next_with_shift.store(one_head_next_with_shift,
                                                   std::memory_order_release);
      bool cas_success =
          array_[old_home].head_next_with_shift.compare_exchange_strong(
              next_with_shift, zero_head_next_with_shift,
              std::memory_order_acq_rel);

      if (cas_success) {
        // Both heads successfully updated with new shift and properly placed
        // single entry
        break;
      } else {
        // next_with_shift modified by another thread; retry

        // Pretend we never took the reference & reset
        Unref(array_[first_entry]);
        zero_chain_from = SIZE_MAX;
        one_chain_from = SIZE_MAX;

        continue;
      }
    }

    // Try to pin second entry, which will be the first shared part of the chain
    shared_entry = GetNextFromNextWithShift(next_next_with_shift);
    if (!MatchAndRef(/*hashed_key=*/nullptr, array_[shared_entry], old_shift,
                     old_home)) {
      // Pretend we never took the first reference
      Unref(array_[first_entry]);

      // FIXME: pro-active removal
      // Couldn't pin second entry; possibly being deleted which we need to
      // wait for. Yield then retry.
      std::this_thread::yield();
      continue;
    }

    // OK, with at least two entries, with first two pinned, we can start
    // our non-trivial chain migration. The old chain is being split into
    // the "zero chain" starting at old home and the "one chain" starting at
    // the grow home. first_entry will be the new first entry for one of
    // those two and the placeholder at grow_home will be the new first entry
    // for the other.

    if (first_entry_new_home == grow_home) {
      zero_chain_from = grow_home;
      one_chain_from = first_entry;
      // When we Remove, it will need to be able to find the chain using
      // this info
      array_[grow_home].hashed_key[0] = old_home;
    } else {
      zero_chain_from = first_entry;
      one_chain_from = grow_home;
      // When we Remove, it will need to be able to find the chain using
      // this info
      array_[grow_home].hashed_key[0] = grow_home;
    }

    // Populate the placeholder
    using_placeholder = true;
    assert(array_[grow_home].meta.load() == 0);
    array_[grow_home].meta.store(
        uint64_t{ClockHandle::kStateConstruction} << ClockHandle::kStateShift,
        std::memory_order_release);
    array_[grow_home].chain_next_with_shift.store(next_next_with_shift,
                                                  std::memory_order_release);
    array_[grow_home].head_next_with_shift.store(
        MakeNextWithShift(one_chain_from, new_shift, /*end=*/false),
        std::memory_order_release);
    if (array_[old_home].head_next_with_shift.compare_exchange_strong(
            next_with_shift,
            MakeNextWithShift(zero_chain_from, new_shift, /*end=*/false),
            std::memory_order_acq_rel)) {
      // Successful beginning of chain migration.
      break;
    } else {
      // Head at old_home changed from another thread, almost certainly
      // an insertion. Unfortunately, we now have to pin different entries
      // for starting the migration, and it's easiest to just roll back all
      // the way and retry.

      // Might not need placeholder entry next time
      array_[grow_home].meta.store(0, std::memory_order_release);
      array_[grow_home].chain_next_with_shift.store(0,
                                                    std::memory_order_release);
      // Pretend we never took the references
      Unref(array_[first_entry]);
      Unref(array_[shared_entry]);

      zero_chain_from = SIZE_MAX;
      one_chain_from = SIZE_MAX;
      shared_entry = SIZE_MAX;
      continue;
    }
  }

  // Step-wise processing of the rest of the chain (if not a trivial case)
  while (shared_entry != SIZE_MAX) {
    assert(zero_chain_from != SIZE_MAX);
    assert(one_chain_from != SIZE_MAX);
    assert(shared_entry != SIZE_MAX);

    size_t shared_new_home =
        GetHomeIndex(array_[shared_entry].hashed_key[0], new_shift);
    assert(shared_new_home == old_home || shared_new_home == grow_home);
    size_t& from_var =
        shared_new_home == old_home ? zero_chain_from : one_chain_from;
    // Upgrade one side
    assert(array_[from_var].chain_next_with_shift.load(
               std::memory_order_acquire) ==
           MakeNextWithShift(shared_entry, old_shift, /*end=*/false));
    array_[from_var].chain_next_with_shift.store(
        MakeNextWithShift(shared_entry, new_shift, /*end=*/false),
        std::memory_order_release);

    // Prep for updating other side as well
    size_t& other_var =
        shared_new_home == old_home ? one_chain_from : zero_chain_from;
    assert(array_[other_var].chain_next_with_shift.load(
               std::memory_order_acquire) ==
           MakeNextWithShift(shared_entry, old_shift, /*end=*/false));

    // Un-pin entry just upgraded. Pretend we never took the reference,
    // unless it's the placeholder
    if (from_var != grow_home) {
      Unref(array_[from_var]);
    }
    // Advance our variables. Modifies zero_chain_from or one_chain_from
    from_var = shared_entry;
    // pessimistic about there being a next entry
    shared_entry = SIZE_MAX;

    // Pin next entry, if it exists
    uint64_t next_with_shift;
    for (;;) {
      next_with_shift = array_[from_var].chain_next_with_shift.load(
          std::memory_order_acquire);
      assert(GetShiftFromNextWithShift(next_with_shift) == old_shift);

      if (next_with_shift & HandleImpl::kNextEndFlag) {
        // Effectively null. Nothing to pin.
        break;
      }

      // Try to pin next entry
      size_t next = GetNextFromNextWithShift(next_with_shift);
      if (MatchAndRef(/*hashed_key=*/nullptr, array_[next], old_shift,
                      old_home)) {
        // Pinned
        shared_entry = next;
        break;
      }

      // FIXME: pro-active removal
      // Couldn't pin; possibly being deleted which we need to wait for.
      // Yield then retry.
      std::this_thread::yield();
    }

    // Jump the other chain over the entry that doesn't belong there.
    // (No CAS needed with both ends of the link pinned.)
    array_[other_var].chain_next_with_shift.store(next_with_shift,
                                                  std::memory_order_release);
  }

  if (zero_chain_from != SIZE_MAX) {
    // Upgrade tail of the zero chain
    assert(array_[zero_chain_from].chain_next_with_shift.load(
               std::memory_order_acquire) ==
           MakeNextWithShift(old_home, old_shift, /*end=*/true));
    array_[zero_chain_from].chain_next_with_shift.store(
        MakeNextWithShift(old_home, new_shift, /*end=*/true),
        std::memory_order_release);
    if (zero_chain_from != grow_home) {
      Unref(array_[zero_chain_from]);
    }
    zero_chain_from = SIZE_MAX;
  }

  if (one_chain_from != SIZE_MAX) {
    // Upgrade tail of the one chain
    assert(array_[one_chain_from].chain_next_with_shift.load(
               std::memory_order_acquire) ==
           MakeNextWithShift(old_home, old_shift, /*end=*/true));
    array_[one_chain_from].chain_next_with_shift.store(
        MakeNextWithShift(grow_home, new_shift, /*end=*/true),
        std::memory_order_release);
    if (one_chain_from != grow_home) {
      Unref(array_[one_chain_from]);
    }
    one_chain_from = SIZE_MAX;
  }

  if (using_placeholder) {
    // Remove+clear the placeholder
    Remove(&array_[grow_home]);
    MarkEmpty(array_[grow_home]);
  } else {
    // Already empty
    assert(array_[grow_home].meta.load() == 0);
  }

  // Update usage_
  if (metadata_charge_policy_ ==
      CacheMetadataChargePolicy::kFullChargeCacheMetadata) {
    // NOTE: ignoring page boundaries for simplicity
    usage_.fetch_add(sizeof(HandleImpl), std::memory_order_relaxed);
  }

  // We have waited until the end to update length_info_, so that we don't
  // keep any entries pinned while waiting for other Grow operations
  for (;;) {
    uint64_t prev_length_info = UsedLengthToLengthInfo(grow_home);
    if (length_info_.compare_exchange_weak(prev_length_info, new_length_info,
                                           std::memory_order_acq_rel)) {
      // Success
      break;
    }
    // Need to wait for other Grow ops
    std::this_thread::yield();
    // TODO: how to detect livelock?
  }

  // Success
  return true;
}

FastClockTable::HandleImpl* FastClockTable::DoInsert(
    const ClockHandleBasicData& proto, uint64_t initial_countdown,
    bool take_ref, InsertState& state) {
  size_t home;
  int orig_home_shift;
  GetHomeIndexAndShift(state.saved_length_info, proto.hashed_key[0], &home,
                       &orig_home_shift);

  // We could go searching through the chain for any duplicate, but that's
  // not typically helpful. (Inferior duplicates will age out with eviction.)
  // Except we do skip insertion if the home slot already has a match
  // (already_matches below).

  // Find an available slot and insert it there. Starting with home slot
  // (same cache line as head pointer).
  // TODO: then slots in the same memory page, then random places
  size_t used_length = LengthInfoToUsedLength(state.saved_length_info);
  assert(home < used_length);

  size_t idx = home;
  for (int cycles = 0;;) {
    bool already_matches = false;
    if (TryInsert(proto, array_[idx], initial_countdown, take_ref,
                  &already_matches)) {
      break;
    }
    if (already_matches) {
      return nullptr;
    }
    ++idx;
    if (idx >= used_length) {
      idx -= used_length;
    }
    if (idx == home) {
      // Cycling back should not happen unless there is enough random churn
      // in parallel that we happen to hit each slot at a time that it's
      // occupied, which is really only feasible for small structures
      assert(used_length <= 256);
      ++cycles;
      assert(cycles < 100);
    }
  }

  // Now insert into chain using head pointer
  uint64_t next_with_shift;
  int home_shift = orig_home_shift;

  // Might need to retry
  for (;;) {
    next_with_shift =
        array_[home].head_next_with_shift.load(std::memory_order_acquire);
    int shift = GetShiftFromNextWithShift(next_with_shift);

    if (UNLIKELY(shift != home_shift)) {
      // NOTE: shift increases with table growth
      if (shift > home_shift) {
        // Must be grow in progress or completed since reading length_info.
        // Pull out one more hash bit. (See OmnibusWalkChain() for why we can't
        // safely jump to the shift that was read.)
        home_shift++;
        uint64_t hash_bit_mask = uint64_t{1} << (home_shift - 1);
        assert((home & hash_bit_mask) == 0);
        home += proto.hashed_key[0] & hash_bit_mask;
        continue;
      } else {
        // Should not happen because length_info_ is only updated after both
        // old and new home heads are marked with new shift
        assert(false);
      }
    }

    uint64_t new_next_with_shift =
        MakeNextWithShift(idx, home_shift, /*end=*/false);

    array_[idx].chain_next_with_shift.store(next_with_shift,
                                            std::memory_order_release);
    if (array_[home].head_next_with_shift.compare_exchange_weak(
            next_with_shift, new_next_with_shift, std::memory_order_acq_rel)) {
      // Success
      return array_ + idx;
    }
  }
}

using LookupOpData = const UniqueId64x2;
using RemoveOpData = FastClockTable::HandleImpl;
using EvictOpData = BaseClockTable::EvictionData;

template <class OpData>
FastClockTable::HandleImpl* FastClockTable::OmnibusWalkChain(size_t home,
                                                             int home_shift,
                                                             OpData* op_data) {
  // Lookup is wait-free with low occurrence of retries, back-tracking, and
  // fallback based on these strategies:
  // * Keep a known good read ref in the chain for "island hopping." When
  // we observe that a concurrent write takes us off to another chain, we
  // only need to fall back to our last known good read ref (most recent
  // entry on the chain that is not "under construction," which is a transient
  // state). We don't want to compound the CPU toil of a long chain with
  // operations that might need to retry from scratch, with probability
  // in proportion to chain length.
  // * Only detect a chain is potentially incomplete because of a Grow in
  // progress by looking at shift in the next pointer tags (rather than
  // re-checking length_info_).

  // FIXME: see main loop code below
  // Note that just because we see a newer shift amount on the chain doesn't
  // necessarily mean a Grow happened. It could be from a re-assigned slot
  // (that we're not holding a reference on) with a different home. But if it's
  // from a slot that we are holding a read reference on, we can be sure that
  // home has been (or still is) the target of a Grow. If our new home is
  // different, we should restart from new home immediately. (Re-check
  // length_info_ for extra checks.) If new home is the same as old home,
  // we only need to restart if we get caught with a read ref to something that
  // belongs in the alternate new home.

  constexpr bool kIsLookup = std::is_same_v<OpData, LookupOpData>;
  const UniqueId64x2* lookup_key = nullptr;
  if constexpr (kIsLookup) {
    lookup_key = op_data;
  }

  constexpr bool kIsRemove = std::is_same_v<OpData, RemoveOpData>;
  constexpr bool kIsEvict = std::is_same_v<OpData, EvictOpData>;

  // Exactly one op specified
  static_assert(kIsLookup + kIsRemove + kIsEvict == 1);

  if constexpr (kIsLookup) {
    assert(home == GetHomeIndex(lookup_key->at(0), home_shift));
  }

  HandleImpl* const arr = array_;

  HandleImpl* h = nullptr;
  HandleImpl* read_ref_on_chain = nullptr;
  uint64_t read_ref_on_chain_next_with_shift = 0;

  // i is our counter for non-progress or backward progress moves, and
  // it should be statisitcally infeasible to make too many of those, but
  // we want to check for infinite or wait loop bugs in Lookup.
  for (size_t i = 0;;) {
    // Read head or chain pointer
    uint64_t next_with_shift =
        h ? h->chain_next_with_shift : arr[home].head_next_with_shift;
    int shift = GetShiftFromNextWithShift(next_with_shift);

    // Make sure it's usable
    size_t effective_home = home;
    if (kIsEvict && home_shift == 0) {
      // Allow Evict to leave home_shift unspecified and pick it up from
      // the head.
      home_shift = shift;
    } else if (UNLIKELY(shift != home_shift)) {
      // We have potentially gone awry somehow, but it's possible we're just
      // hitting old data that is not yet completed Grow.
      // NOTE: shift bits goes up with table growth.
      if (shift < home_shift) {
        // To avoid waiting, an old shift amount needs to be processed as if
        // we were still using it and (potentially different or the same) the
        // old home.
        // We can assert it's not too old, because each generation of Grow
        // waits on its ancestor in the previous generation.
        assert(shift + 1 == home_shift);
        effective_home = GetHomeIndex(home, shift);
      } else if (h == nullptr) {
        assert(shift > home_shift);
        assert(read_ref_on_chain == nullptr);
        // Newer head might not yet be reflected in length_info_ (an atomicity
        // gap in Grow), so operate as if it is. Note that other insertions
        // could happen using this shift before length_info_ is updated, and
        // it's possible (though unlikely) that multiple generations of Grow
        // have occurred. If shift is more than one generation ahead of
        // home_shift, it's possible that not all descendent homes have
        // reached the `shift` generation. Thus, we need to advance only one
        // shift at a time looking for a home+head with a matching shift
        // amount.
        home_shift++;
        if constexpr (kIsLookup) {
          home = GetHomeIndex((*lookup_key)[0], home_shift);
        } else if constexpr (kIsRemove) {
          home = GetHomeIndex(op_data->hashed_key[0], home_shift);
        } else {
          // Eviction is allowed to be incomplete on data currently being
          // migrated to a new home. It's just simpler this way.
        }
        // Didn't make progress & retry
        ++i;
        assert(i < 1000);
        continue;
      } else {
        assert(shift > home_shift);
        // We have either gotten off our chain or our home shift is out of
        // date. The simplest resolution is to restart, as we will see updated
        // info if we do.
        h = nullptr;
        if (read_ref_on_chain) {
          Unref(*read_ref_on_chain);
          read_ref_on_chain = nullptr;
        }
        // Didn't make progress & retry
        ++i;
        assert(i < 1000);
        continue;
      }
    }

    // Check for end marker
    if (next_with_shift & HandleImpl::kNextEndFlag) {
      // To ensure we didn't miss anything in the chain, the end marker must
      // point back to the correct home.
      if (LIKELY(GetNextFromNextWithShift(next_with_shift) == effective_home)) {
        // Complete, clean iteration of the chain, not found.
        // Even for remove op, the entry might have been removed from the
        // chain opportunistically by an adjacent entry removal.
        // Clean up.
        if (read_ref_on_chain) {
          Unref(*read_ref_on_chain);
        }
        return nullptr;
      } else {
        // Something went awry. Revert back to a safe point (if we have it)
        h = read_ref_on_chain;
        // Didn't make progress & retry
        ++i;
        assert(i < 1000);
        continue;
      }
    }

    if (h == read_ref_on_chain) {
      read_ref_on_chain_next_with_shift = next_with_shift;
    }

    // Follow the next and check for full key match, home match, or neither
    h = &arr[GetNextFromNextWithShift(next_with_shift)];
    bool full_match_or_unknown = false;

    if constexpr (kIsRemove) {
      if (h == op_data) {
        // Found the entry we need to remove from the chain. We can safely
        // also remove any other entries between here and read_ref_on_chain
        // (or head) because they are only "under construction" if being
        // erased. This allows us to avoid waiting on another thread to
        // remove them from the chain.

        // Based on the assumption that the current thread owns the logical
        // write lock on the entry to erase, op_data, we can assume its chain
        // next will not be overwritten during this function, so we can safely
        // move that pointer.
        uint64_t new_next_with_shift =
            h->chain_next_with_shift.load(std::memory_order_acquire);
        auto& next_with_shift_var =
            read_ref_on_chain ? read_ref_on_chain->chain_next_with_shift
                              : arr[home].head_next_with_shift;
        if (next_with_shift_var.compare_exchange_strong(
                read_ref_on_chain_next_with_shift, new_next_with_shift,
                std::memory_order_acq_rel)) {
          // Successful removal from chain
          // Clean up any read ref
          if (read_ref_on_chain) {
            Unref(*read_ref_on_chain);
          }
          // NOTE: caller must call MarkEmpty when ready
          // NOTE: not clearing the chain_next as the stale value could be
          // useful in getting some other operation safely back on the correct
          // chain, even though that sounds dubious
          return nullptr;
        } else {
          // Another thread must have intervened with an adjacent removal,
          // grow in progress, or head insertion. Revert back to a safe point
          // (if we have it)
          h = read_ref_on_chain;
          // Didn't make progress & retry
          ++i;
          assert(i < 1000);
          continue;
        }
      }
    }

    bool evicting = false;
    if (MatchAndRef(kIsLookup ? lookup_key : nullptr, *h, home_shift, home,
                    &full_match_or_unknown, kIsEvict ? &evicting : nullptr)) {
      // Got a useful next ref; can release old one
      if (read_ref_on_chain) {
        // Pretend we never took the reference.
        Unref(*read_ref_on_chain);
      }
      if (full_match_or_unknown) {
        // Only for Lookup case
        assert(kIsLookup);
        // Full match. All done.
        return h;
      } else {
        // Correct home location, so we are on the right chain
        read_ref_on_chain = h;
      }
    } else {
      if (full_match_or_unknown) {
        // Must have been an "under construction" entry
        if constexpr (kIsEvict) {
          if (evicting) {
            // Took ownership for eviction
            // Remove the entry from the chain
            // FIXME: do this more efficiently! The tricky part about rolling
            // the finish erase into eviction is that eviction must then follow
            // all grown-into chains
            OmnibusWalkChain(home, home_shift, h);
            TrackAndReleaseEvictedEntry(h, op_data);
            // Revert back to a safe point (if we have it). This will not
            // repeat any clock updates, as read_ref_on_chain must be the
            // last entry on the chain to have been clock updated without
            // being evicted.
            h = read_ref_on_chain;
          }
        }
        // Otherwise, owned by another thread, so we can safely skip it, but
        // there's a chance we'll have to backtrack later
      } else {
        // Home mismatch! Revert back to a safe point (if we have it)
        h = read_ref_on_chain;
        // Didn't make progress
        ++i;
      }
    }
  }
}

FastClockTable::HandleImpl* FastClockTable::Lookup(
    const UniqueId64x2& hashed_key) {
  // Reading length_info_ is not strictly required for Lookup, if we were
  // to increment shift sizes until we see a shift size match on the
  // relevant head pointer. Thus, reading with relaxed memory order gives
  // us a safe and almost always up-to-date jump into finding the correct
  // home and head.
  size_t home;
  int home_shift;
  GetHomeIndexAndShift(length_info_.load(std::memory_order_relaxed),
                       hashed_key[0], &home, &home_shift);
  assert(home_shift > 0);

  // TODO: for efficiency, consider probing home slot without checking head
  // pointer.

  return OmnibusWalkChain(home, home_shift, &hashed_key);
}

void FastClockTable::Remove(HandleImpl* h) {
  assert((h->meta.load() >> ClockHandle::kStateShift) ==
         ClockHandle::kStateConstruction);

  size_t home;
  int home_shift;
  GetHomeIndexAndShift(length_info_.load(std::memory_order_relaxed),
                       h->hashed_key[0], &home, &home_shift);
  assert(home_shift > 0);

  OmnibusWalkChain(home, home_shift, h);
}

bool FastClockTable::TryEraseHandle(HandleImpl* h, bool holding_ref,
                                    bool mark_invisible) {
  uint64_t meta;
  if (mark_invisible) {
    // Set invisible
    meta = h->meta.fetch_and(
        ~(uint64_t{ClockHandle::kStateVisibleBit} << ClockHandle::kStateShift),
        std::memory_order_acq_rel);
    // To local variable also
    meta &=
        ~(uint64_t{ClockHandle::kStateVisibleBit} << ClockHandle::kStateShift);
  } else {
    meta = h->meta.load(std::memory_order_acquire);
  }

  // Take ownership if no other refs
  do {
    if (GetRefcount(meta) != holding_ref) {
      // Not last ref at some point in time during this call
      return false;
    }
    if ((meta & (uint64_t{ClockHandle::kStateShareableBit}
                 << ClockHandle::kStateShift)) == 0) {
      // Someone else took ownership
      return false;
    }
    // Note that if !holding_ref, there's a small chance that we release,
    // another thread replaces this entry with another, reaches zero refs, and
    // then we end up erasing that other entry. That's an acceptable risk /
    // imprecision.
  } while (!h->meta.compare_exchange_weak(
      meta,
      uint64_t{ClockHandle::kStateConstruction} << ClockHandle::kStateShift,
      std::memory_order_acquire));
  // Took ownership
  // TODO? Delay freeing?
  h->FreeData(allocator_);
  size_t total_charge = h->total_charge;
  if (UNLIKELY(h->IsStandalone())) {
    // Delete detached handle
    delete h;
    standalone_usage_.fetch_sub(total_charge, std::memory_order_relaxed);
  } else {
    Remove(h);
    MarkEmpty(*h);
    occupancy_.fetch_sub(1U, std::memory_order_release);
  }
  usage_.fetch_sub(total_charge, std::memory_order_relaxed);
  assert(usage_.load(std::memory_order_relaxed) < SIZE_MAX / 2);
  return true;
}

bool FastClockTable::Release(HandleImpl* h, bool useful,
                             bool erase_if_last_ref) {
  // In contrast with LRUCache's Release, this function won't delete the handle
  // when the cache is above capacity and the reference is the last one. Space
  // is only freed up by EvictFromClock (called by Insert when space is needed)
  // and Erase. We do this to avoid an extra atomic read of the variable usage_.

  uint64_t old_meta;
  if (useful) {
    // Increment release counter to indicate was used
    old_meta = h->meta.fetch_add(ClockHandle::kReleaseIncrement,
                                 std::memory_order_release);
    // Correct for possible (but rare) overflow
    CorrectNearOverflow(old_meta, h->meta);
  } else {
    // Decrement acquire counter to pretend it never happened
    old_meta = h->meta.fetch_sub(ClockHandle::kAcquireIncrement,
                                 std::memory_order_release);
  }

  assert((old_meta >> ClockHandle::kStateShift) &
         ClockHandle::kStateShareableBit);
  // No underflow
  assert(((old_meta >> ClockHandle::kAcquireCounterShift) &
          ClockHandle::kCounterMask) !=
         ((old_meta >> ClockHandle::kReleaseCounterShift) &
          ClockHandle::kCounterMask));

  if ((erase_if_last_ref || UNLIKELY(old_meta >> ClockHandle::kStateShift ==
                                     ClockHandle::kStateInvisible))) {
    return TryEraseHandle(h, /*holding_ref=*/false, /*mark_invisible=*/false);
  } else {
    return false;
  }
}

#ifndef NDEBUG
void FastClockTable::TEST_ReleaseN(HandleImpl* h, size_t n) {
  if (n > 0) {
    // Do n-1 simple releases first
    TEST_ReleaseNMinus1(h, n);

    // Then the last release might be more involved
    Release(h, /*useful*/ true, /*erase_if_last_ref*/ false);
  }
}
#endif

void FastClockTable::Erase(const UniqueId64x2& hashed_key) {
  // Don't need to be efficient.
  // Might be one match masking another, so loop.
  while (HandleImpl* h = Lookup(hashed_key)) {
    bool gone =
        TryEraseHandle(h, /*holding_ref=*/true, /*mark_invisible=*/true);
    if (!gone) {
      // Only marked invisible, which is ok.
      // Pretend we never took the reference from Lookup.
      Unref(*h);
    }
  }
}

void FastClockTable::EraseUnRefEntries() {
  size_t frontier = GetTableSize();
  for (size_t i = 0; i < frontier; i++) {
    HandleImpl& h = array_[i];

    uint64_t old_meta = h.meta.load(std::memory_order_relaxed);
    if (old_meta & (uint64_t{ClockHandle::kStateShareableBit}
                    << ClockHandle::kStateShift) &&
        GetRefcount(old_meta) == 0 &&
        h.meta.compare_exchange_strong(old_meta,
                                       uint64_t{ClockHandle::kStateConstruction}
                                           << ClockHandle::kStateShift,
                                       std::memory_order_acquire)) {
      // Took ownership
      h.FreeData(allocator_);
      usage_.fetch_sub(h.total_charge, std::memory_order_relaxed);
      // NOTE: could be more efficient with a dedicated variant of
      // OmnibusWalkChain, but this is not a common operation
      Remove(&h);
      MarkEmpty(h);
      occupancy_.fetch_sub(1U, std::memory_order_release);
    }
  }
}

void FastClockTable::Evict(size_t requested_charge, InsertState& state,
                           EvictionData* data) {
  // precondition
  assert(requested_charge > 0);

  // We need the clock pointer to seemlessly "wrap around" at the end of the
  // table, and to be reasonably stable under Grow operations. This is
  // challenging when the linear hashing progressively opens additional
  // most-significant-hash-bits in determining home locations.

  // TODO: make a tuning parameter?
  // Up to 2x this number of homes will be evicted per step. In very rare
  // cases, possibly more, as homes of an out-of-date generation will be
  // resolved to multiple in a newer generation.
  constexpr size_t step_size = 4;

  // A clock_pointer_mask_ field separate from length_info_ enables us to use
  // the same mask (way of dividing up the space among evicting threads) for
  // iterating over the whole structure before considering changing the mask
  // at the beginning of each pass. This ensures we do not have a large portion
  // of the space that receives redundant or missed clock updates. However,
  // with two variables, for each update to clock_pointer_mask (< 64 ever in
  // the life of the cache), there will be a brief period where concurrent
  // eviction threads could use the old mask value, possibly causing redundant
  // or missed clock updates for a *small* portion of the table.
  size_t clock_pointer_mask =
      clock_pointer_mask_.load(std::memory_order_relaxed);

  uint64_t max_clock_pointer = 0;  // unset

  // TODO: consider updating during a long eviction
  size_t used_length = LengthInfoToUsedLength(state.saved_length_info);

  // Loop until enough freed, or limit reached (see bottom of loop)
  for (;;) {
    // First (concurrent) increment clock pointer
    uint64_t old_clock_pointer =
        clock_pointer_.fetch_add(step_size, std::memory_order_relaxed);

    if (UNLIKELY((old_clock_pointer & clock_pointer_mask) == 0)) {
      // Back at the beginning. See if clock_pointer_mask should be updated.
      uint64_t mask = BottomNBits(
          UINT64_MAX, LengthInfoToMinShift(state.saved_length_info));
      if (clock_pointer_mask != mask) {
        clock_pointer_mask = static_cast<size_t>(mask);
        clock_pointer_mask_.store(clock_pointer_mask,
                                  std::memory_order_relaxed);
      }
    }

    size_t major_step = clock_pointer_mask + 1;
    assert((major_step & clock_pointer_mask) == 0);

    for (size_t base_home = old_clock_pointer & clock_pointer_mask;
         base_home < used_length; base_home += major_step) {
      for (size_t i = 0; i < step_size; i++) {
        size_t home = base_home + i;
        if (home >= used_length) {
          break;
        }
        OmnibusWalkChain(home, /*unspecified home_shift*/ 0, data);
      }
    }

    // Loop exit conditions
    if (data->freed_charge >= requested_charge) {
      return;
    }

    if (max_clock_pointer == 0) {
      // Cap the eviction effort at this thread (along with those operating in
      // parallel) circling through the whole structure kMaxCountdown times.
      // In other words, this eviction run must find something/anything that is
      // unreferenced at start of and during the eviction run that isn't
      // reclaimed by a concurrent eviction run.
      // TODO: Does HyperClockCache need kMaxCountdown + 1?
      max_clock_pointer =
          old_clock_pointer +
          (uint64_t{ClockHandle::kMaxCountdown + 1} * major_step);
    }

    if (old_clock_pointer + step_size >= max_clock_pointer) {
      return;
    }
  }
}

size_t FastClockTable::CalcMaxUsableLength(
    size_t capacity, size_t min_avg_value_size,
    CacheMetadataChargePolicy metadata_charge_policy) {
  // Assume 50% load factor
  double min_avg_slot_charge = min_avg_value_size / 2.0;
  if (metadata_charge_policy == kFullChargeCacheMetadata) {
    min_avg_slot_charge += sizeof(HandleImpl);
  }
  assert(min_avg_slot_charge > 0.0);
  size_t num_slots =
      static_cast<size_t>(capacity / min_avg_slot_charge + 0.999999);

  const size_t slots_per_page = port::kPageSize / sizeof(HandleImpl);

  // Round up to page size
  return ((num_slots + slots_per_page - 1) / slots_per_page) * slots_per_page;
}

// Explicit instantiation
template class ClockCacheShard<FastClockTable>;

FastClockCache::FastClockCache(const FastClockCacheOptions& opts)
    : ShardedCache(opts) {
  assert(opts.min_avg_entry_charge > 0 ||
         opts.metadata_charge_policy != kDontChargeCacheMetadata);
  // TODO: should not need to go through two levels of pointer indirection to
  // get to table entries
  size_t per_shard = GetPerShardCapacity();
  MemoryAllocator* alloc = this->memory_allocator();
  InitShards([=](Shard* cs) {
    FastClockTable::Opts table_opts;
    table_opts.min_avg_value_size = opts.min_avg_entry_charge;
    new (cs) Shard(per_shard, opts.strict_capacity_limit,
                   opts.metadata_charge_policy, alloc, &eviction_callback_,
                   &hash_seed_, table_opts);
  });
}

void* FastClockCache::Value(Handle* handle) {
  return reinterpret_cast<const HandleImpl*>(handle)->value;
}

size_t FastClockCache::GetCharge(Handle* handle) const {
  return reinterpret_cast<const HandleImpl*>(handle)->total_charge;
}

const Cache::CacheItemHelper* FastClockCache::GetCacheItemHelper(
    Handle* handle) const {
  auto h = reinterpret_cast<const HandleImpl*>(handle);
  return h->helper;
}

void FastClockCache::ReportProblems(
    const std::shared_ptr<Logger>& /*info_log*/) const {
  // TODO
}

}  // namespace clock_cache

// DEPRECATED (see public API)
std::shared_ptr<Cache> NewClockCache(
    size_t capacity, int num_shard_bits, bool strict_capacity_limit,
    CacheMetadataChargePolicy metadata_charge_policy) {
  return NewLRUCache(capacity, num_shard_bits, strict_capacity_limit,
                     /* high_pri_pool_ratio */ 0.5, nullptr,
                     kDefaultToAdaptiveMutex, metadata_charge_policy,
                     /* low_pri_pool_ratio */ 0.0);
}

std::shared_ptr<Cache> HyperClockCacheOptions::MakeSharedCache() const {
  // For sanitized options
  HyperClockCacheOptions opts = *this;
  if (opts.num_shard_bits >= 20) {
    return nullptr;  // The cache cannot be sharded into too many fine pieces.
  }
  if (opts.num_shard_bits < 0) {
    // Use larger shard size to reduce risk of large entries clustering
    // or skewing individual shards.
    constexpr size_t min_shard_size = 32U * 1024U * 1024U;
    opts.num_shard_bits =
        GetDefaultCacheShardBits(opts.capacity, min_shard_size);
  }
  std::shared_ptr<Cache> cache =
      std::make_shared<clock_cache::HyperClockCache>(opts);
  if (opts.secondary_cache) {
    cache = std::make_shared<CacheWithSecondaryAdapter>(cache,
                                                        opts.secondary_cache);
  }
  return cache;
}

std::shared_ptr<Cache> FastClockCacheOptions::MakeSharedCache() const {
  // For sanitized options
  FastClockCacheOptions opts = *this;
  if (opts.num_shard_bits >= 20) {
    return nullptr;  // The cache cannot be sharded into too many fine pieces.
  }
  if (opts.num_shard_bits < 0) {
    // Use larger shard size to reduce risk of large entries clustering
    // or skewing individual shards.
    constexpr size_t min_shard_size = 32U * 1024U * 1024U;
    opts.num_shard_bits =
        GetDefaultCacheShardBits(opts.capacity, min_shard_size);
  }
  std::shared_ptr<Cache> cache =
      std::make_shared<clock_cache::FastClockCache>(opts);
  if (opts.secondary_cache) {
    cache = std::make_shared<CacheWithSecondaryAdapter>(cache,
                                                        opts.secondary_cache);
  }
  return cache;
}

}  // namespace ROCKSDB_NAMESPACE
