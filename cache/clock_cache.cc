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
#include <exception>
#include <functional>
#include <numeric>
#include <thread>

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
  if (LIKELY(old_usage != capacity)) {
    do {
      new_usage = std::min(capacity, old_usage + total_charge);
    } while (!usage_.compare_exchange_weak(old_usage, new_usage,
                                           std::memory_order_relaxed));
  } else {
    new_usage = old_usage;
  }
  // How much do we need to evict then?
  size_t need_evict_charge = old_usage + total_charge - new_usage;
  size_t request_evict_charge = need_evict_charge;
  if (UNLIKELY(need_evict_for_occupancy) && request_evict_charge == 0) {
    // Require at least 1 eviction.
    request_evict_charge = 1;
  }
  if (request_evict_charge > 0) {
    size_t evicted_charge = 0;
    size_t evicted_count = 0;
    static_cast<Table*>(this)->Evict(request_evict_charge, &evicted_charge,
                                     &evicted_count, state);
    occupancy_.fetch_sub(evicted_count, std::memory_order_release);
    if (LIKELY(evicted_charge > need_evict_charge)) {
      assert(evicted_count > 0);
      // Evicted more than enough
      usage_.fetch_sub(evicted_charge - need_evict_charge,
                       std::memory_order_relaxed);
    } else if (evicted_charge < need_evict_charge ||
               (UNLIKELY(need_evict_for_occupancy) && evicted_count == 0)) {
      // Roll back to old usage minus evicted
      usage_.fetch_sub(evicted_charge + (new_usage - old_usage),
                       std::memory_order_relaxed);
      if (evicted_charge < need_evict_charge) {
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
    assert(evicted_count > 0);
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
  size_t evicted_charge = 0;
  size_t evicted_count = 0;
  if (need_evict_charge > 0) {
    static_cast<Table*>(this)->Evict(need_evict_charge, &evicted_charge,
                                     &evicted_count, state);
    // Deal with potential occupancy deficit
    if (UNLIKELY(need_evict_for_occupancy) && evicted_count == 0) {
      assert(evicted_charge == 0);
      // Can't meet occupancy requirement
      return false;
    } else {
      // Update occupancy for evictions
      occupancy_.fetch_sub(evicted_count, std::memory_order_release);
    }
  }
  // Track new usage even if we weren't able to evict enough
  usage_.fetch_add(total_charge - evicted_charge, std::memory_order_relaxed);
  // No underflow
  assert(usage_.load(std::memory_order_relaxed) < SIZE_MAX / 2);
  // Success
  return true;
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
  auto revert_occupancy_fn = [&]() {
    occupancy_.fetch_sub(1, std::memory_order_relaxed);
  };
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
      revert_occupancy_fn();
      return s;
    }
  } else {
    // Case strict_capacity_limit == false
    bool success = ChargeUsageMaybeEvictNonStrict<Table>(
        total_charge, capacity, need_evict_for_occupancy, state);
    if (!success) {
      revert_occupancy_fn();
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
  auto revert_usage_fn = [&]() {
    usage_.fetch_sub(total_charge, std::memory_order_relaxed);
    // No underflow
    assert(usage_.load(std::memory_order_relaxed) < SIZE_MAX / 2);
  };

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
    revert_occupancy_fn();
    // Maybe fall back on standalone insert
    if (handle == nullptr) {
      revert_usage_fn();
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

void HyperClockTable::StartInsert(InsertState&) {}

bool HyperClockTable::GrowIfNeeded(size_t new_occupancy, InsertState&) {
  return new_occupancy <= occupancy_limit_;
}

HyperClockTable::HandleImpl* HyperClockTable::DoInsert(
    const ClockHandleBasicData& proto, uint64_t initial_countdown,
    bool take_ref, InsertState&) {
  size_t probe = 0;
  bool need_rollback = false;
  HandleImpl* e = FindSlot(
      proto.hashed_key,
      [&](HandleImpl* h) {
        // Optimistically transition the slot from "empty" to
        // "under construction" (no effect on other states)
        uint64_t old_meta =
            h->meta.fetch_or(uint64_t{ClockHandle::kStateOccupiedBit}
                                 << ClockHandle::kStateShift,
                             std::memory_order_acq_rel);
        uint64_t old_state = old_meta >> ClockHandle::kStateShift;

        if (old_state == ClockHandle::kStateEmpty) {
          // We've started inserting into an available slot, and taken
          // ownership Save data fields
          ClockHandleBasicData* h_alias = h;
          *h_alias = proto;

          // Transition from "under construction" state to "visible" state
          uint64_t new_meta = uint64_t{ClockHandle::kStateVisible}
                              << ClockHandle::kStateShift;

          // Maybe with an outstanding reference
          new_meta |= initial_countdown << ClockHandle::kAcquireCounterShift;
          new_meta |= (initial_countdown - take_ref)
                      << ClockHandle::kReleaseCounterShift;

#ifndef NDEBUG
          // Save the state transition, with assertion
          old_meta = h->meta.exchange(new_meta, std::memory_order_release);
          assert(old_meta >> ClockHandle::kStateShift ==
                 ClockHandle::kStateConstruction);
#else
          // Save the state transition
          h->meta.store(new_meta, std::memory_order_release);
#endif
          return true;
        } else if (old_state != ClockHandle::kStateVisible) {
          // Slot not usable / touchable now
          return false;
        }
        // Existing, visible entry, which might be a match.
        // But first, we need to acquire a ref to read it. In fact, number of
        // refs for initial countdown, so that we boost the clock state if
        // this is a match.
        old_meta = h->meta.fetch_add(
            ClockHandle::kAcquireIncrement * initial_countdown,
            std::memory_order_acq_rel);
        // Like Lookup
        if ((old_meta >> ClockHandle::kStateShift) ==
            ClockHandle::kStateVisible) {
          // Acquired a read reference
          if (h->hashed_key == proto.hashed_key) {
            // Match. Release in a way that boosts the clock state
            old_meta = h->meta.fetch_add(
                ClockHandle::kReleaseIncrement * initial_countdown,
                std::memory_order_acq_rel);
            // Correct for possible (but rare) overflow
            CorrectNearOverflow(old_meta, h->meta);
            // Insert standalone instead (only if return handle needed)
            need_rollback = true;
            return true;
          } else {
            // Mismatch. Pretend we never took the reference
            old_meta = h->meta.fetch_sub(
                ClockHandle::kAcquireIncrement * initial_countdown,
                std::memory_order_acq_rel);
          }
        } else if (UNLIKELY((old_meta >> ClockHandle::kStateShift) ==
                            ClockHandle::kStateInvisible)) {
          // Pretend we never took the reference
          // WART: there's a tiny chance we release last ref to invisible
          // entry here. If that happens, we let eviction take care of it.
          old_meta = h->meta.fetch_sub(
              ClockHandle::kAcquireIncrement * initial_countdown,
              std::memory_order_acq_rel);
        } else {
          // For other states, incrementing the acquire counter has no effect
          // so we don't need to undo it.
          // Slot not usable / touchable now.
        }
        (void)old_meta;
        return false;
      },
      [&](HandleImpl* /*h*/) { return false; },
      [&](HandleImpl* h) {
        h->displacements.fetch_add(1, std::memory_order_relaxed);
      },
      probe);
  if (e == nullptr) {
    // Occupancy check and never abort FindSlot above should generally
    // prevent this, except it's theoretically possible for other threads
    // to evict and replace entries in the right order to hit every slot
    // when it is populated. Assuming random hashing, the chance of that
    // should be no higher than pow(kStrictLoadFactor, n) for n slots.
    // That should be infeasible for roughly n >= 256, so if this assertion
    // fails, that suggests something is going wrong.
    assert(GetTableSize() < 256);
    return nullptr;
  }
  if (need_rollback) {
    Rollback(proto.hashed_key, e);
    return nullptr;
  }
  // Successfully inserted
  return e;
}

HyperClockTable::HandleImpl* HyperClockTable::Lookup(
    const UniqueId64x2& hashed_key) {
  size_t probe = 0;
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
            old_meta = h->meta.fetch_sub(ClockHandle::kAcquireIncrement,
                                         std::memory_order_release);
          }
        } else if (UNLIKELY((old_meta >> ClockHandle::kStateShift) ==
                            ClockHandle::kStateInvisible)) {
          // Pretend we never took the reference
          // WART: there's a tiny chance we release last ref to invisible
          // entry here. If that happens, we let eviction take care of it.
          old_meta = h->meta.fetch_sub(ClockHandle::kAcquireIncrement,
                                       std::memory_order_release);
        } else {
          // For other states, incrementing the acquire counter has no effect
          // so we don't need to undo it. Furthermore, we cannot safely undo
          // it because we did not acquire a read reference to lock the
          // entry in a Shareable state.
        }
        (void)old_meta;
        return false;
      },
      [&](HandleImpl* h) {
        return h->displacements.load(std::memory_order_relaxed) == 0;
      },
      [&](HandleImpl* /*h*/) {}, probe);

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

void HyperClockTable::Ref(HandleImpl& h) {
  // Increment acquire counter
  uint64_t old_meta = h.meta.fetch_add(ClockHandle::kAcquireIncrement,
                                       std::memory_order_acquire);

  assert((old_meta >> ClockHandle::kStateShift) &
         ClockHandle::kStateShareableBit);
  // Must have already had a reference
  assert(GetRefcount(old_meta) > 0);
  (void)old_meta;
}

void HyperClockTable::TEST_RefN(HandleImpl& h, size_t n) {
  // Increment acquire counter
  uint64_t old_meta = h.meta.fetch_add(n * ClockHandle::kAcquireIncrement,
                                       std::memory_order_acquire);

  assert((old_meta >> ClockHandle::kStateShift) &
         ClockHandle::kStateShareableBit);
  (void)old_meta;
}

void HyperClockTable::TEST_ReleaseN(HandleImpl* h, size_t n) {
  if (n > 0) {
    // Split into n - 1 and 1 steps.
    uint64_t old_meta = h->meta.fetch_add(
        (n - 1) * ClockHandle::kReleaseIncrement, std::memory_order_acquire);
    assert((old_meta >> ClockHandle::kStateShift) &
           ClockHandle::kStateShareableBit);
    (void)old_meta;

    Release(h, /*useful*/ true, /*erase_if_last_ref*/ false);
  }
}

void HyperClockTable::Erase(const UniqueId64x2& hashed_key) {
  size_t probe = 0;
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
                h->meta.fetch_sub(ClockHandle::kAcquireIncrement,
                                  std::memory_order_release);
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
            h->meta.fetch_sub(ClockHandle::kAcquireIncrement,
                              std::memory_order_release);
          }
        } else if (UNLIKELY((old_meta >> ClockHandle::kStateShift) ==
                            ClockHandle::kStateInvisible)) {
          // Pretend we never took the reference
          // WART: there's a tiny chance we release last ref to invisible
          // entry here. If that happens, we let eviction take care of it.
          h->meta.fetch_sub(ClockHandle::kAcquireIncrement,
                            std::memory_order_release);
        } else {
          // For other states, incrementing the acquire counter has no effect
          // so we don't need to undo it.
        }
        return false;
      },
      [&](HandleImpl* h) {
        return h->displacements.load(std::memory_order_relaxed) == 0;
      },
      [&](HandleImpl* /*h*/) {}, probe);
}

void HyperClockTable::ConstApplyToEntriesRange(
    std::function<void(const HandleImpl&)> func, size_t index_begin,
    size_t index_end, bool apply_if_will_be_deleted) const {
  uint64_t check_state_mask = ClockHandle::kStateShareableBit;
  if (!apply_if_will_be_deleted) {
    check_state_mask |= ClockHandle::kStateVisibleBit;
  }

  for (size_t i = index_begin; i < index_end; i++) {
    HandleImpl& h = array_[i];

    // Note: to avoid using compare_exchange, we have to be extra careful.
    uint64_t old_meta = h.meta.load(std::memory_order_relaxed);
    // Check if it's an entry visible to lookups
    if ((old_meta >> ClockHandle::kStateShift) & check_state_mask) {
      // Increment acquire counter. Note: it's possible that the entry has
      // completely changed since we loaded old_meta, but incrementing acquire
      // count is always safe. (Similar to optimistic Lookup here.)
      old_meta = h.meta.fetch_add(ClockHandle::kAcquireIncrement,
                                  std::memory_order_acquire);
      // Check whether we actually acquired a reference.
      if ((old_meta >> ClockHandle::kStateShift) &
          ClockHandle::kStateShareableBit) {
        // Apply func if appropriate
        if ((old_meta >> ClockHandle::kStateShift) & check_state_mask) {
          func(h);
        }
        // Pretend we never took the reference
        h.meta.fetch_sub(ClockHandle::kAcquireIncrement,
                         std::memory_order_release);
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

inline HyperClockTable::HandleImpl* HyperClockTable::FindSlot(
    const UniqueId64x2& hashed_key, std::function<bool(HandleImpl*)> match_fn,
    std::function<bool(HandleImpl*)> abort_fn,
    std::function<void(HandleImpl*)> update_fn, size_t& probe) {
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
  size_t current = ModTableSize(base + probe * increment);
  while (probe <= length_bits_mask_) {
    HandleImpl* h = &array_[current];
    if (match_fn(h)) {
      probe++;
      return h;
    }
    if (abort_fn(h)) {
      return nullptr;
    }
    probe++;
    update_fn(h);
    current = ModTableSize(current + increment);
  }
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

inline void HyperClockTable::Evict(size_t requested_charge,
                                   size_t* freed_charge, size_t* freed_count,
                                   InsertState&) {
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

  // For key reconstructed from hash
  UniqueId64x2 unhashed;

  for (;;) {
    for (size_t i = 0; i < step_size; i++) {
      HandleImpl& h = array_[ModTableSize(Lower32of64(old_clock_pointer + i))];
      bool evicting = ClockUpdate(h);
      if (evicting) {
        Rollback(h.hashed_key, &h);
        *freed_charge += h.GetTotalCharge();
        *freed_count += 1;
        bool took_ownership = false;
        if (eviction_callback_) {
          took_ownership =
              eviction_callback_(ClockCacheShard<HyperClockTable>::ReverseHash(
                                     h.GetHash(), &unhashed, hash_seed_),
                                 reinterpret_cast<Cache::Handle*>(&h));
        }
        if (!took_ownership) {
          h.FreeData(allocator_);
        }
        MarkEmpty(h);
      }
    }

    // Loop exit condition
    if (*freed_charge >= requested_charge) {
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
  // The state is essentially going to be the starting hash, which works
  // nicely even if we resize between calls because we use upper-most
  // hash bits for table indexes.
  size_t length_bits = table_.GetLengthBits();
  size_t length = table_.GetTableSize();

  assert(average_entries_per_lock > 0);
  // Assuming we are called with same average_entries_per_lock repeatedly,
  // this simplifies some logic (index_end will not overflow).
  assert(average_entries_per_lock < length || *state == 0);

  size_t index_begin = *state >> (sizeof(size_t) * 8u - length_bits);
  size_t index_end = index_begin + average_entries_per_lock;
  if (index_end >= length) {
    // Going to end.
    index_end = length;
    *state = SIZE_MAX;
  } else {
    *state = index_end << (sizeof(size_t) * 8u - length_bits);
  }

  auto hash_seed = table_.GetHashSeed();
  table_.ConstApplyToEntriesRange(
      [callback, hash_seed](const HandleImpl& h) {
        UniqueId64x2 unhashed;
        callback(ReverseHash(h.hashed_key, &unhashed, hash_seed), h.value,
                 h.GetTotalCharge(), h.helper);
      },
      index_begin, index_end, false);
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

template <class Table>
void ClockCacheShard<Table>::TEST_RefN(HandleImpl* h, size_t n) {
  table_.TEST_RefN(*h, n);
}

template <class Table>
void ClockCacheShard<Table>::TEST_ReleaseN(HandleImpl* h, size_t n) {
  table_.TEST_ReleaseN(h, n);
}

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
  table_.ConstApplyToEntriesRange(
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
      0, table_.GetTableSize(), true);

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

// Frontier  | Threshold (bin)  | (min) shift
// 2         | 1                | 62
// 4         | 10               | 61
// 6         | 11               | 61
// 8         | 100              | 60
// 10        | 101              | 60
// ...

inline uint64_t FrontierToThreshold(size_t frontier) {
  assert(frontier >= 2);
  int lshift = 63 - FloorLog2(frontier);
  return (uint64_t{frontier} << lshift) | lshift;
}

inline size_t ThresholdToFrontier(uint64_t threshold) {
  uint64_t hash_threshold = threshold & ~uint64_t{255};
  assert(hash_threshold >> 63);
  int min_shift = static_cast<int>(threshold) & int{255};
  assert(min_shift > 0);
  size_t f = static_cast<size_t>(hash_threshold >> min_shift);
  assert(f > 0 && (f & 1) == 0);
  return f;
}

inline size_t GetMinHomeIndex(uint64_t threshold) {
  return ThresholdToFrontier(threshold) >> 1;
}

inline size_t GetHomeIndex(uint64_t threshold, uint64_t hash) {
  hash |= uint64_t{1} << 63;
  uint64_t hash_threshold = threshold & ~uint64_t{255};
  int min_shift = static_cast<int>(threshold) & int{255};
  int maybe_one_more = hash >= hash_threshold;
  size_t home = static_cast<size_t>(hash >> (min_shift + maybe_one_more));
  assert(home >= GetMinHomeIndex(threshold));
  assert(home < ThresholdToFrontier(threshold));
  return home;
}

inline size_t GetStartingFrontier() {
  return port::kPageSize / sizeof(FastClockTable::HandleImpl);
}

FastClockTable::FastClockTable(size_t capacity, bool /*strict_capacity_limit*/,
                               CacheMetadataChargePolicy metadata_charge_policy,
                               MemoryAllocator* allocator,
                               const Cache::EvictionCallback* eviction_callback,
                               const uint32_t* hash_seed, const Opts& opts)
    : BaseClockTable(metadata_charge_policy, allocator, eviction_callback,
                     hash_seed),
      max_length_bits_(CalcHashBits(capacity, opts.min_avg_value_size,
                                    metadata_charge_policy)),
      array_mem_(MemMapping::AllocateLazyZeroed(sizeof(HandleImpl)
                                                << (max_length_bits_ + 1))),
      array_(static_cast<HandleImpl*>(array_mem_.Get())),
      threshold_(FrontierToThreshold(GetStartingFrontier())) {
  if (metadata_charge_policy ==
      CacheMetadataChargePolicy::kFullChargeCacheMetadata) {
    // NOTE: ignoring page boundaries for simplicity
    usage_ += size_t{GetTableSize()} * sizeof(HandleImpl);
  }

  static_assert(sizeof(HandleImpl) == 64U,
                "Expecting size / alignment with common cache line size");

  // Mark next pointers as "null" and home addresses as insertable
  size_t min_home = GetMinHomeIndex(threshold_.load());
  size_t end = ThresholdToFrontier(threshold_.load());
  size_t i = 0;
  for (; i < min_home; ++i) {
    array_[i].next = HandleImpl::kNextNull;
  }
  for (; i < end; ++i) {
    array_[i].next = HandleImpl::kNextNull | HandleImpl::kNextInsertableFlag;
  }
}

FastClockTable::~FastClockTable() {
  // Assumes there are no references or active operations on any slot/element
  // in the table.
  size_t frontier = ThresholdToFrontier(threshold_.load());
  for (size_t i = 0; i < frontier; i++) {
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
#endif
        break;
      // otherwise
      default:
        assert(false);
        break;
    }
  }

  assert(usage_.load() == 0 ||
         usage_.load() == size_t{GetTableSize()} * sizeof(HandleImpl));
  // TODO: metadata usage
  assert(occupancy_ == 0);
}

size_t FastClockTable::GetTableSize() const {
  return ThresholdToFrontier(threshold_.load());
}

size_t FastClockTable::GetOccupancyLimit() const {
  return GetMinHomeIndex(threshold_.load());
}

void FastClockTable::StartInsert(InsertState& state) {
  state.saved_threshold = threshold_.load(std::memory_order_acquire);
}

inline bool MatchAndRef(const UniqueId64x2* hashed_key, ClockHandle& h,
                        size_t home = 0, uint64_t threshold = 0,
                        bool* full_match = nullptr) {
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
  assert(hashed_key || home > 0);
  constexpr bool kOptimisticLookup = true;
  uint64_t old_meta;
  if (!kOptimisticLookup) {
    old_meta = h.meta.load(std::memory_order_acquire);
    if ((old_meta >> ClockHandle::kStateShift) != ClockHandle::kStateVisible) {
      return false;
    }
  }
  // (Optimistically) increment acquire counter
  old_meta = h.meta.fetch_add(ClockHandle::kAcquireIncrement,
                              std::memory_order_acquire);
  // Check if it's an entry visible to lookups
  if ((old_meta >> ClockHandle::kStateShift) == ClockHandle::kStateVisible) {
    // Acquired a read reference
    if (hashed_key && h.hashed_key == *hashed_key) {
      // Match on full key
      if (full_match) {
        *full_match = true;
      }
      return true;
    } else if (home > 0 && home == GetHomeIndex(threshold, h.hashed_key[0])) {
      // Match on home address
      if (full_match) {
        *full_match = false;
      }
      return true;
    } else {
      // Mismatch. Pretend we never took the reference
      old_meta = h.meta.fetch_sub(ClockHandle::kAcquireIncrement,
                                  std::memory_order_release);
    }
  } else if (UNLIKELY((old_meta >> ClockHandle::kStateShift) ==
                      ClockHandle::kStateInvisible)) {
    // Pretend we never took the reference
    // WART: there's a tiny chance we release last ref to invisible
    // entry here. If that happens, we let eviction take care of it.
    old_meta = h.meta.fetch_sub(ClockHandle::kAcquireIncrement,
                                std::memory_order_release);
  } else {
    // For other states, incrementing the acquire counter has no effect
    // so we don't need to undo it. Furthermore, we cannot safely undo
    // it because we did not acquire a read reference to lock the
    // entry in a Shareable state.
  }
  (void)old_meta;
  return false;
}

bool FastClockTable::GrowIfNeeded(size_t new_occupancy, InsertState& state) {
  size_t old_min_home = GetMinHomeIndex(state.saved_threshold);
  if (new_occupancy <= old_min_home) {
    // Don't need to grow
    return true;
  }
  // !!!!! FIXME !!!!!: Needs a unit test
  if ((old_min_home >> max_length_bits_) > 0) {
    // Can't grow any more
    return false;
  }
  // Might need to grow, depending on whether evictions suffice while waiting
  // for mutex
  MutexLock l(&grow_mutex_);
  // Double-check
  new_occupancy = occupancy_.load();
  state.saved_threshold = threshold_.load();
  old_min_home = GetMinHomeIndex(state.saved_threshold);
  if (new_occupancy <= old_min_home) {
    // Don't need to grow
    return true;
  }
  // !!!!! FIXME !!!!!: Verify against null next etc.
  // TODO: make sure max_length_bits doesn't let us overflow next refs
  if ((old_min_home >> max_length_bits_) > 0) {
    // Can't grow any more
    return false;
  }

  // OK we are growing one old address into two new ones.
  HandleImpl& old_home = array_[old_min_home];
  size_t old_frontier = old_min_home * 2;

  HandleImpl* new_homes = array_ + old_frontier;
  uint64_t new_threshold = FrontierToThreshold(old_frontier + 2);

  // We start by redirecting insertions to the new homes, by marking them
  // insertable and then the old no longer insertable (which will cause both
  // to be read until threshold_ is updated).

  new_homes[0].next.fetch_or(HandleImpl::kNextInsertableFlag, std::memory_order_acquire);
  new_homes[1].next.fetch_or(HandleImpl::kNextInsertableFlag, std::memory_order_release);

  old_home.next.fetch_and(~HandleImpl::kNextInsertableFlag, std::memory_order_acq_rel);

  // Next we need to move entries from old to appropriate new in tail to head
  // order, so that we can safely, temporarily have the chains converge on
  // entries shared between them. And we take read refs on each entry in the
  // chain to ensure they are pinned (not removed). (Note: neither under
  // construction nor empty entries are allowed in the chain.) For example,
  // updating one atomic at a time, marked with *, in sequence:
  //
  // Old -> A1 -> B1 -> A2 -> B2 ->||
  // NewA ->||
  // NewB ->||
  // ===>
  // Old -> A1 -> B1 -> A2 -\
  // NewA ->||               \
  // NewB ---------*----------> B2 ->||
  // ===>
  // Old -> A1 -> B1 -> A2 *>||
  // NewA ->||
  // NewB -> B2 ->||
  // ===>
  // Old -> A1 -> B1 -\
  // NewA ------*------> A2 ->||
  // NewB -> B2 ->||
  // ===>
  // Old -> A1 -> B1 *>||
  // NewA -> A2 ->||
  // NewB -> B2 ->||
  // ===>
  // Old -> A1 -> B1 *>\
  // NewA -> A2 ->||    \
  // NewB ---------------> B2 ->||
  // ===>
  // Old -> A1 ----->\
  // NewA -> A2 ->||  \
  // NewB ------*------> B1 -> B2 ->||
  // ===>
  // Old -> A1 *>||
  // NewA -> A2 ->||
  // NewB -> B1 -> B2 ->||
  // ===>
  // Old -> A1 *\
  // NewA -------> A2 ->||
  // NewB -> B1 -> B2 ->||
  // == Now suppose another thread adds A9 to NewA during this =>
  // Old -> A1 -\
  // NewA -> A9 -> A2 ->||
  // NewB -> B1 -> B2 ->||
  // == We get a CAS failure on NewA next and have to adjust =>
  // Old -> A1 *\
  // NewA -------> A9 -> A2 ->||
  // NewB -> B1 -> B2 ->||
  // ===>
  // Old -\
  // NewA *> A1 -> A9 -> A2 ->||
  // NewB -> B1 -> B2 ->||
  // ===>
  // Old *>||
  // NewA -> A1 -> A9 -> A2 ->||
  // NewB -> B1 -> B2 ->||
  //
  // We will deal with any "at home" entries later (not depicted above) before
  // updating threshold_.

  autovector<uint32_t> pinned_refs_in_chain;

  // Similar to island hopping in Lookup, with local retries, except we keep
  // all the refs.
  HandleImpl *h = &old_home;
  for (size_t i = 0;;) {
    uint32_t next = h->next.load(std::memory_order_acquire);
    assert(next == (next & HandleImpl::kNextNoFlagsMask));
    if (next == HandleImpl::kNextNull) {
      // End of chain
      break;
    }

    HandleImpl* h_next = array_ + next;
    if (MatchAndRef(/*hashed_key=*/nullptr, *h_next, old_min_home, state.saved_threshold)) {
      pinned_refs_in_chain.push_back(next);
      h = h_next;
      i = 0;
    } else {
      // Need to try again with same h
      ++i;
      assert(i < 100);
    }
  }

  // Now carefully transfer each entry to its new chain, without
  // interrupting read access nor other insertions into the new chains.
  // This doesn't need to be highly optimized, just safe and correct.
  // At the end of each step, we reset the tail of the old chain to null,
  // to end sharing so that the new chain can take full ownership of the
  // migrated entrty.
  while (!pinned_refs_in_chain.empty()) {
    uint32_t entry_idx = pinned_refs_in_chain.back();
    h = array_ + entry_idx;
    HandleImpl* new_home = array_ + GetHomeIndex(new_threshold, h->hashed_key[0]);
    assert(new_home == new_homes + 0 || new_home == new_homes + 1);
    uint32_t next = new_home->next.load(std::memory_order_acquire);
    assert(h->next.load(std::memory_order_acquire) == HandleImpl::kNextNull);
    h->next.store(next & HandleImpl::kNextNoFlagsMask, std::memory_order_release);
    if (new_home->next.compare_exchange_weak(next, entry_idx | HandleImpl::kNextInsertableFlag, std::memory_order_acq_rel)) {
      // Set old home chain tail to null so that we can safely un-ref the
      // migrated entry
      pinned_refs_in_chain.pop_back();
      HandleImpl& last = pinned_refs_in_chain.empty() ? old_home : array_[pinned_refs_in_chain.back()];
      last.next.store(HandleImpl::kNextNull, std::memory_order_release);
      Release(h, /*useful=*/false, /*erase_if_last_ref=*/false);
    } else {
      // Reset tail back to null and try again
      h->next.store(HandleImpl::kNextNull, std::memory_order_release);
      // TODO: assert retry count
    }
  }

  // Now dealing with any "at home" entry is a pain because it might be under
  // construction and we don't know where it belongs. And without adding
  // another metadata bit, we can't agree on whether this thread or the owning
  // thread should migrate.
  // Also, in case it's empty, we need to get a write lock to block any other
  // writes to it. But first we try taking a read ref, and alternate until
  // something works. Unfortunately, this might be a true spin+yield waiting
  // on another thread.
  for (size_t i = 0;; ++i) {
    uint64_t old_meta = old_home.meta.fetch_add(ClockHandle::kAcquireIncrement,
                              std::memory_order_acquire);
    if ((old_meta >> ClockHandle::kStateShift) == ClockHandle::kStateVisible) {

  }


  // pinning the home entry, (a) so that other operations can
  // safely erase a home to empty slot without worrying that Grow will turn the
  // slot into a chained entry concurrently (which is not allowed to be empty
  // and part of a chain), and (b) so that we can determine whether it is
  // definitively in or out of one of the new chains. If it's already empty,
  // we need to acquire a write lock instead of a read reference.

  bool old_home_write_locked;  // else read refed
  // Might need to retry/spin
  for (size_t i = 0;; ++i) {
    // Optimistically transition the slot from "empty" to
    // "under construction" (no effect on other states)
    uint64_t old_meta = old_home.meta.fetch_or(
        uint64_t{ClockHandle::kStateOccupiedBit} << ClockHandle::kStateShift,
        std::memory_order_acq_rel);
    uint64_t old_state = old_meta >> ClockHandle::kStateShift;
    if (old_state == ClockHandle::kStateEmpty) {
      old_home_write_locked = true;
      break;
    }

    // Acquire a ref if possible
    old_meta = old_home.meta.fetch_add(ClockHandle::kAcquireIncrement,
                                       std::memory_order_acq_rel);
    // Check if it's a refcounted entry
    if ((old_meta >> ClockHandle::kStateShift) &
        ClockHandle::kStateShareableBit) {
      old_home_write_locked = false;
      break;
    }
    if (i >= 200) {
      std::this_thread::yield();
    }
  }

  // Next we mark the "next" ref as NOT insertable, so that we can ensure
  // nothing more is inserted while attempting to "pin" the entire row.
  // Other Inserts here (beyond the home) will spin until the new home is
  // ready (based on update to threshold_), but removals might still happen.
  // This comes after pinning 'home' entry so that FinishErasure can get a
  // definitive determination on whether an entry is "at home" or not.
  old_home.next.fetch_and(~HandleImpl::kNextInsertableFlag,
                          std::memory_order_acq_rel);

  // Might need to retry pinning entire chain (due to intervening erase / evict)
  autovector<uint32_t> pinned_refs_in_chain;
  for (size_t i = 0;; ++i) {
    uint32_t next = old_home.next.load(std::memory_order_acquire);
    bool complete = true;
    while (next & HandleImpl::kNextFollowFlag) {
      uint32_t ref = next & HandleImpl::kNextNoFlagsMask;
      HandleImpl& h = array_[ref];

      // Acquire a ref if possible
      uint64_t old_meta = h.meta.fetch_add(ClockHandle::kAcquireIncrement,
                                           std::memory_order_acq_rel);
      // Check if it's a refcounted entry
      if ((old_meta >> ClockHandle::kStateShift) &
          ClockHandle::kStateShareableBit) {
        // Acquired a read reference (to pin the entry)
        pinned_refs_in_chain.push_back(ref);
        next = h.next.load(std::memory_order_acquire);
      } else {
        // Must be under construction for removal
        // TODO: spin here instead?
        complete = false;
        break;
      }
    }
    if (complete && (next & HandleImpl::kNextNoFlagsMask) == old_min_home) {
      // Pinned entire chain
      // Make sure we didn't get a stale next from home.
      next = old_home.next.load(std::memory_order_acquire);
      if ((next & HandleImpl::kNextFollowFlag) == 0 ||
          (!pinned_refs_in_chain.empty() &&
           pinned_refs_in_chain.front() ==
               (next & HandleImpl::kNextNoFlagsMask))) {
        break;
      }
    }
    // Else roll back and retry
    for (uint32_t rel : pinned_refs_in_chain) {
      // Pretend we never took the reference
      array_[rel].meta.fetch_sub(ClockHandle::kAcquireIncrement,
                                 std::memory_order_release);
    }
    pinned_refs_in_chain.clear();
    if (i >= 200) {
      std::this_thread::yield();
    }
  }

  // Now we are going to block reads (Lookup) from the chain on this home
  old_home.next.store(0);

  // Now we collate the entries into the two new rows.
  // Start with successful loop-back markers
  std::array<uint32_t, 2> new_row_heads = {
      static_cast<uint32_t>(old_frontier),
      static_cast<uint32_t>(old_frontier + 1)};
  for (uint32_t ref : pinned_refs_in_chain) {
    HandleImpl& h = array_[ref];
    size_t new_home = GetHomeIndex(new_threshold, h.hashed_key[0]);
    assert(new_home == old_frontier || new_home == old_frontier + 1);
    h.next.store(new_row_heads[new_home & 1], std::memory_order_acq_rel);
    new_row_heads[new_home & 1] = ref | HandleImpl::kNextFollowFlag;
  }

  // Finally before we publish the new rows we need to handle old_home itself,
  // by including it only if it wasn't empty.
  if (!old_home_write_locked) {
    // Add it to the appropriate new row
    size_t new_home = GetHomeIndex(new_threshold, old_home.hashed_key[0]);
    assert(new_home == old_frontier || new_home == old_frontier + 1);
    old_home.next.store(new_row_heads[new_home & 1], std::memory_order_acq_rel);
    new_row_heads[new_home & 1] =
        (old_frontier / 2) | HandleImpl::kNextFollowFlag;
  }

  // Publish the new rows while everything still pinned
  new_homes[0].next.store(new_row_heads[0] | HandleImpl::kNextInsertableFlag,
                          std::memory_order_release);
  new_homes[1].next.store(new_row_heads[1] | HandleImpl::kNextInsertableFlag,
                          std::memory_order_release);
  // And readable
  threshold_.store(new_threshold, std::memory_order_acq_rel);
  state.saved_threshold = new_threshold;

  // Maybe charge metadata (NOTE: ignoring page boundaries for simplicity)
  if (metadata_charge_policy_ ==
      CacheMetadataChargePolicy::kFullChargeCacheMetadata) {
    usage_.fetch_add(2 * sizeof(HandleImpl), std::memory_order_relaxed);
  }

#ifndef NDEBUG
  // debug_history.append("Grew from " + std::to_string(old_min_home) + "\n");
  if (true) {
    // Verify all the entries can be found with Lookup
    if (!old_home_write_locked) {
      auto hh = Lookup(old_home.hashed_key);
      assert(hh != nullptr);
      assert(hh->hashed_key == old_home.hashed_key);
      Release(hh, /*useful*/ false, /*erase_if_last_ref*/ false);
      // debug_history.append("Verified " + std::to_string(hh - array_) + "\n");
    }
    for (uint32_t ref : pinned_refs_in_chain) {
      HandleImpl* h = array_ + ref;
      auto hh = Lookup(h->hashed_key);
      assert(hh != nullptr);
      assert(hh->hashed_key == h->hashed_key);
      Release(hh, /*useful*/ false, /*erase_if_last_ref*/ false);
      // debug_history.append("Verified " + std::to_string(hh - array_) + "\n");
    }
  }
#endif

  // Release everything, starting with old_home
  if (old_home_write_locked) {
    // Re-mark empty
    old_home.meta.store(0, std::memory_order_release);
  } else {
    // Pretend we never took the reference
    // WART: there's a chance we release last ref to invisible
    // entry here. If that happens, we let eviction take care of it.
    old_home.meta.fetch_sub(ClockHandle::kAcquireIncrement,
                            std::memory_order_release);
  }

  // Now release chain entries
  for (uint32_t ref : pinned_refs_in_chain) {
    // Pretend we never took the reference
    // WART: there's a chance we release last ref to invisible
    // entry here. If that happens, we let eviction take care of it.
    array_[ref].meta.fetch_sub(ClockHandle::kAcquireIncrement,
                               std::memory_order_release);
  }

  // Success
  return true;
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
      // Mismatch. Pretend we never took the reference
      old_meta =
          h.meta.fetch_sub(ClockHandle::kAcquireIncrement * initial_countdown,
                           std::memory_order_acq_rel);
    }
  } else if (UNLIKELY((old_meta >> ClockHandle::kStateShift) ==
                      ClockHandle::kStateInvisible)) {
    // Pretend we never took the reference
    // WART/FIXME?: there's a tiny chance we release last ref to invisible
    // entry here. If that happens, we let eviction take care of it.
    old_meta =
        h.meta.fetch_sub(ClockHandle::kAcquireIncrement * initial_countdown,
                         std::memory_order_acq_rel);
  } else {
    // For other states, incrementing the acquire counter has no effect
    // so we don't need to undo it.
    // Slot not usable / touchable now.
  }
  (void)old_meta;
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

FastClockTable::HandleImpl* FastClockTable::DoInsert(
    const ClockHandleBasicData& proto, uint64_t initial_countdown,
    bool take_ref, InsertState& state) {
  size_t home = GetHomeIndex(state.saved_threshold, proto.hashed_key[0]);

  // Try to insert at home slot
  {
    bool already_matches = false;
    HandleImpl& h = array_[home];
    if (BeginSlotInsert(proto, h, initial_countdown, &already_matches)) {
      // Verify this is still a home address
      if (LIKELY(h.next.load(std::memory_order_acquire) &
                 HandleImpl::kNextInsertableFlag)) {
        FinishSlotInsert(proto, h, initial_countdown, take_ref);
        return array_ + home;
        // debug_history.append("Insert @home " + std::to_string(home) +
        // "\n");
      } else {
        // Concurrent change to non-home.
        MarkEmpty(h);
        // Continue on to chain insertion (even though we already know of an
        // empty slot, it's not easy to wedge into that code)
      }
    }
    if (already_matches) {
      // Prefer to keep existing entry than to insert one further down the
      // chain. (Can't return the existing entry, because that wouldn't take
      // ownership of the cache object under a Handle. We need a standalone
      // entry for that.)
      return nullptr;
    }
  }

  // We could also go searching through the chain for any duplicate, but
  // that's not typically helpful. (Inferior duplicates will age out
  // eviction.)
  //
  // Also, even though we might need to retry and new home might have open
  // slot, it's simpler to commit to chained insertion by going ahead and
  // adding to an available slot before ensuring home is ready for chained
  // insertion. (It might be momentarily tied up in Grow operation.)

  // Find available slot
  // NOTE: because we checked occupancy above based on frontier, we didn't
  // fully guarantee available chainable slot right now
  size_t chainable_count = GetMinHomeIndex(state.saved_threshold);
  // (More uniform selection than e.g. home / 2)
  size_t starting_idx = FastRange64(proto.hashed_key[1], chainable_count);
  size_t idx = starting_idx;
  for (int cycles = 0;;) {
    bool already_matches = false;
    if (TryInsert(proto, array_[idx], initial_countdown, take_ref,
                  &already_matches)) {
      assert(array_[idx].next.load(std::memory_order_acquire) ==
             HandleImpl::kNextNull);
      break;
    }
    // Else keep searching for empty slot
    idx += 1;
    if (idx >= chainable_count) {
      idx = 0;
    }
    // NOTE: it's possible we could cycle around due to unlucky temporary
    // arrangements of entries in slots, but that should not be indefinite
    if (idx == starting_idx) {
      ++cycles;
      assert(cycles < 100);
      std::this_thread::yield();
      chainable_count =
          GetMinHomeIndex(threshold_.load(std::memory_order_acquire));
    }
  }

  // Now insert into chain starting at home address, though
  // might need to retry. During Grow, threshold_ might not be up to date
  // with where we should insert, but we should only need to check one
  // level beyond the home given by threshold_. However, saved_threshold can
  // be out of date with threshold_.
  for (size_t home_changes = 0;;) {
    uint32_t next = array_[home].next.load(std::memory_order_acquire);
    // Home might change with another thread executing Grow
    if (UNLIKELY((next & HandleImpl::kNextInsertableFlag) == 0)) {
      ++home_changes;
      if (home_changes > 50) {
        std::terminate();
      }
      // Try next level
      size_t forward_home =
          GetHomeIndex(state.saved_threshold - 1, proto.hashed_key[0]);
      next = array_[forward_home].next.load(std::memory_order_acquire);
      if (UNLIKELY((next & HandleImpl::kNextInsertableFlag) == 0)) {
        // There should always be an insertable home for entries, so this
        // should only happen if saved_threshold is way behind.
        assert(GetHomeIndex(threshold_.load(std::memory_order_acquire),
                            proto.hashed_key[0]) > home);
        state.saved_threshold = threshold_.load(std::memory_order_acquire);
        home = GetHomeIndex(state.saved_threshold, proto.hashed_key[0]);
        continue;
      }
      home = forward_home;
    }
    array_[idx].next.store(next & ~HandleImpl::kNextInsertableFlag,
                           std::memory_order_acq_rel);
    assert(idx <= HandleImpl::kNextNoFlagsMask);
    if (array_[home].next.compare_exchange_weak(
            next, static_cast<uint32_t>(idx) | HandleImpl::kNextInsertableFlag,
            std::memory_order_acq_rel)) {
      // debug_history.append("Insert @ " + std::to_string(idx) + " for
      // home " + std::to_string(home) + "\n");

      // Successful insertion. All done.
      return array_ + idx;
    }
  }
}

FastClockTable::HandleImpl* FastClockTable::Lookup(
    const UniqueId64x2& hashed_key) {
  // Overall, Lookup works by "island hopping."" While holding a read reference
  // to one entry in a chain, we follow the next link, retrying as needed,
  // until we get a read reference to an entry that belongs in the chain (thus
  // is the next entry, or in extremely rare cases, moves backward in the chain
  // due to reusing a slot) and only then release the read reference on the
  // previous entry. Note that we never skip over entries in the chain, as
  // entries in the chain must be read-refable (thus, removed from the chain
  // before being marked "under construction" or empty).
  //
  // Two tricky parts:
  // * Bootstrapping the island hopping: ...
  // * Fall foward: ...

  uint64_t threshold = threshold_.load(std::memory_order_acquire);
  size_t home = GetHomeIndex(threshold, hashed_key[0]);

  // In rare cases, might have to check more than one home.
  size_t home_changes = 0;
// C++ still doesn't have labelled break/continue
home_changed : {
  // Probe home slot. Only need to keep a ref on full match, because
  // we need special handling for initial jump from a home slot into the
  // chain.
  HandleImpl* h = array_ + home;
  if (MatchAndRef(&hashed_key, *h)) {
    return h;
  }

  // Now, what was our home slot could now be in one of three states:
  // * Still exclusively the correct home slot (kNextInsertableFlag)
  // * A Grow is in the process of migrating from this slot to a new home
  // slot (this one no longer insertable).
  // * In the background, a Grow has completed and the slot has been
  // freed up and possibly re-used in another chain.

  bool full_match = false;
  bool grow_in_progress = false;

  // Make the initial jump into the chain (if present)
  for (size_t i = 0;; ++i) {
    uint32_t next = h->next.load(std::memory_order_acquire);
    if (UNLIKELY((next & HandleImpl::kNextInsertableFlag) == 0)) {
      // Re-read threshold_ to decide whether a Grow is still
      // in progress for this old home slot.
      threshold = threshold_.load(std::memory_order_acquire);
      size_t new_home = GetHomeIndex(threshold, hashed_key[0]);
      if (new_home != home) {
        // Reading this chain is no longer necessary. Start over at new
        // home.
        home = new_home;
        ++home_changes;
        if (home_changes > 50) {
          std::terminate();
        }
        goto home_changed;
      }
      // We will have to check the next home also
      grow_in_progress = true;
    }
    // Correct home slot, or old one whose chain must be read
    next &= HandleImpl::kNextNoFlagsMask;
    if (next == HandleImpl::kNextNull) {
      // No chain to search
      h = nullptr;
      break;
    }

    HandleImpl* h_next = array_ + next;
    if (MatchAndRef(&hashed_key, *h_next, home, threshold, &full_match)) {
      if (full_match) {
        return h_next;
      } else {
        // Correct home location, so we are on the right chain
        h = h_next;
        break;
      }
    }
    // Else, need to try again
    assert(i < 100);
  }

  if (h) {
    // Continue island hopping down the chain.
    // Loop invariant: we have a read ref on h which ensures it is part of
    // the home chain (though could be migrated to next home chain during
    // this operation - ok).
    for (size_t i = 0;;) {
      uint32_t next = h->next.load(std::memory_order_acquire);
      assert(next == (next & HandleImpl::kNextNoFlagsMask));
      if (next == HandleImpl::kNextNull) {
        // End of chain
        Release(h, /*useful=*/false, /*erase_if_last_ref=*/false);
        h = nullptr;
        break;
      }

      HandleImpl* h_next = array_ + next;
      if (MatchAndRef(&hashed_key, *h_next, home, threshold, &full_match)) {
        // Release this island and hop to the next
        Release(h, /*useful=*/false, /*erase_if_last_ref=*/false);
        if (full_match) {
          // Return full match.
          return h_next;
        } else {
          // Correct home location, so we can continue down the correct
          // chain.
          h = h_next;
          i = 0;
        }
      } else {
        // Need to try again with same h
        ++i;
        assert(i < 100);
      }
    }
    // Determine whether a relevant Grow initiated while walking the chain
    if (!grow_in_progress) {
      uint32_t next = array_[home].next.load(std::memory_order_acquire);
      if (UNLIKELY((next & HandleImpl::kNextInsertableFlag) == 0)) {
        grow_in_progress = true;
      }
    }
  }

  if (grow_in_progress) {
    // Also search next new home (reduce the min shift by 1)
    assert((threshold & 255) > 0);
    home = GetHomeIndex(threshold - 1, hashed_key[0]);
    ++home_changes;
    if (home_changes > 50) {
      std::terminate();
    }
    goto home_changed;
  }
}
  // Not found
  return nullptr;
}

// IAMHERE

uint32_t FastClockTable::FinishErasure(HandleImpl* h,
                                       HandleImpl* possible_prev) {
  assert((h->meta.load() >> ClockHandle::kStateShift) ==
         ClockHandle::kStateConstruction);

  // First check for the easy case.
  // If this is a home address, we can simply mark empty and move on.
  uint32_t next = h->next.load(std::memory_order_acquire);
  if (next & HandleImpl::kNextInsertableFlag) {
    // NOTE: holding a write lock (under construction) on h guarantees
    // no concurrent Grow on this home.
    MarkEmpty(*h);
    return next;
  }

  // Must acquire after h marked under construction, but don't need to
  // update after that (because holding the write lock blocks relevant Grow
  // ops)
  const uint64_t threshold = threshold_.load(std::memory_order_acquire);
  const size_t home = GetHomeIndex(threshold, h->hashed_key[0]);

  // Earlier check on h->next must have taken care of "at home" entry case
  assert(h != array_ + home);

  // Otherwise, we need a stable next (pinned or !follow) in order to
  // guarantee next is not unsafely removed concurrently, but might need
  // to retry in case of concurrent write
  HandleImpl* to_release_read_ref = nullptr;
  for (size_t i = 0;; ++i) {
    if ((next & HandleImpl::kNextFollowFlag) == 0) {
      // !follow means stable end of list.
      break;
    }
    // Else need to pin next with a read ref to establish stable next
    HandleImpl& next_h = array_[next & HandleImpl::kNextNoFlagsMask];
    uint64_t old_meta = next_h.meta.fetch_add(ClockHandle::kAcquireIncrement,
                                              std::memory_order_acq_rel);
    // Check if it's a refcounted entry
    if ((old_meta >> ClockHandle::kStateShift) &
        ClockHandle::kStateShareableBit) {
      // Pinned the entry with a read ref
      // Double-check it's still the right 'next'.
      uint32_t updated_next = h->next.load(std::memory_order_acquire);
      assert((updated_next & HandleImpl::kNextInsertableFlag) == 0);
      if (updated_next == next) {
        // Good
        to_release_read_ref = &next_h;
        break;
      } else {
        // Need to retry, releasing this ref
        // Pretend we never took the reference
        next_h.meta.fetch_sub(ClockHandle::kAcquireIncrement,
                              std::memory_order_release);
        next = updated_next;
      }
    } else {
      // Must be under construction or something
      // Make sure we're up to date
      next = h->next.load(std::memory_order_acquire);
    }
    if (i >= 200) {
      std::this_thread::yield();
      // Make sure we're up to date
      next = h->next.load(std::memory_order_acquire);
    }
  }

  // Now find the prev node that needs to have its 'next' updated.
  // Like Lookup, we might have to retry to find the entry we are looking
  // for, but invariants should guarantee it is found if we get a clean
  // search path from home.
  // NOTE: the write lock on h ensures no concurrent Grow rewrites the
  // row and ensures the prev is not removed (because it can't get a
  // stable next). However, if prev is a home, there might be concurrent
  // inserts between prev and h.
  for (size_t i = 0;; ++i) {
    HandleImpl* prev_h = nullptr;
    uint32_t h_ref = 0;
    HandleImpl* search_h = array_ + home;
    if (possible_prev) {
      search_h = possible_prev;
      possible_prev = nullptr;
    }
    assert(search_h != h);
    uint32_t search_next = search_h->next.load(std::memory_order_acquire);
    while (search_next & HandleImpl::kNextFollowFlag) {
      HandleImpl* next_h =
          array_ + (search_next & HandleImpl::kNextNoFlagsMask);
      if (next_h == h) {
        prev_h = search_h;
        h_ref = search_next;
        break;
      }
      search_h = next_h;
      search_next = search_h->next.load(std::memory_order_acquire);
    }
    if (prev_h) {
      assert(h_ref > 0);
      uint32_t next_with_insertable =
          next | (h_ref & HandleImpl::kNextInsertableFlag);
      if (prev_h->next.compare_exchange_strong(h_ref, next_with_insertable,
                                               std::memory_order_acq_rel)) {
        // Success
        break;
      }
    }
    // If we didn't find h it must have been an unclean query
    assert(GetHomeIndex(threshold_.load(std::memory_order_acquire),
                        h->hashed_key[0]) == home);
    assert((search_next & HandleImpl::kNextNoFlagsMask) != home);
    // Retry
    if (i >= 200) {
      std::this_thread::yield();
    }
  }

  // Wrap up
  // Clear h->next before marking empty so that we don't end up with something
  // weird like a cycle in stale entries.
  h->next.store(0, std::memory_order_release);
  MarkEmpty(*h);
  if (to_release_read_ref) {
    // Pretend we never took the reference
    to_release_read_ref->meta.fetch_sub(ClockHandle::kAcquireIncrement,
                                        std::memory_order_release);
  }
  return next;
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
    FinishErasure(h, /*possible_prev unknown*/ nullptr);
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

void FastClockTable::Ref(HandleImpl& h) {
  // Increment acquire counter
  uint64_t old_meta = h.meta.fetch_add(ClockHandle::kAcquireIncrement,
                                       std::memory_order_acquire);

  assert((old_meta >> ClockHandle::kStateShift) &
         ClockHandle::kStateShareableBit);
  // Must have already had a reference
  assert(GetRefcount(old_meta) > 0);
  (void)old_meta;
}

void FastClockTable::TEST_RefN(HandleImpl& h, size_t n) {
  // Increment acquire counter
  uint64_t old_meta = h.meta.fetch_add(n * ClockHandle::kAcquireIncrement,
                                       std::memory_order_acquire);

  assert((old_meta >> ClockHandle::kStateShift) &
         ClockHandle::kStateShareableBit);
  (void)old_meta;
}

void FastClockTable::TEST_ReleaseN(HandleImpl* h, size_t n) {
  if (n > 0) {
    // Split into n - 1 and 1 steps.
    uint64_t old_meta = h->meta.fetch_add(
        (n - 1) * ClockHandle::kReleaseIncrement, std::memory_order_acquire);
    assert((old_meta >> ClockHandle::kStateShift) &
           ClockHandle::kStateShareableBit);
    (void)old_meta;

    Release(h, /*useful*/ true, /*erase_if_last_ref*/ false);
  }
}

void FastClockTable::Erase(const UniqueId64x2& hashed_key) {
  // Don't need to be efficient.
  // Might be one match masking another, so loop.
  while (HandleImpl* h = Lookup(hashed_key)) {
    bool gone =
        TryEraseHandle(h, /*holding_ref=*/true, /*mark_invisible=*/true);
    if (!gone) {
      // Only marked invisible, which is ok.
      // Pretend we never took the reference from Lookup.
      h->meta.fetch_sub(ClockHandle::kAcquireIncrement,
                        std::memory_order_release);
    }
  }
}

void FastClockTable::ConstApplyToEntriesRange(
    std::function<void(const HandleImpl&)> func, size_t index_begin,
    size_t index_end, bool apply_if_will_be_deleted) const {
  uint64_t check_state_mask = ClockHandle::kStateShareableBit;
  if (!apply_if_will_be_deleted) {
    check_state_mask |= ClockHandle::kStateVisibleBit;
  }

  for (size_t i = index_begin; i < index_end; i++) {
    HandleImpl& h = array_[i];

    // Note: to avoid using compare_exchange, we have to be extra careful.
    uint64_t old_meta = h.meta.load(std::memory_order_relaxed);
    // Check if it's an entry visible to lookups
    if ((old_meta >> ClockHandle::kStateShift) & check_state_mask) {
      // Increment acquire counter. Note: it's possible that the entry has
      // completely changed since we loaded old_meta, but incrementing acquire
      // count is always safe. (Similar to optimistic Lookup here.)
      old_meta = h.meta.fetch_add(ClockHandle::kAcquireIncrement,
                                  std::memory_order_acquire);
      // Check whether we actually acquired a reference.
      if ((old_meta >> ClockHandle::kStateShift) &
          ClockHandle::kStateShareableBit) {
        // Apply func if appropriate
        if ((old_meta >> ClockHandle::kStateShift) & check_state_mask) {
          func(h);
        }
        // Pretend we never took the reference
        h.meta.fetch_sub(ClockHandle::kAcquireIncrement,
                         std::memory_order_release);
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
      FinishErasure(&h, /*possible_prev unknown*/ nullptr);
      occupancy_.fetch_sub(1U, std::memory_order_release);
    }
  }
}

void FastClockTable::Evict(size_t requested_charge, size_t* freed_charge,
                           size_t* freed_count, InsertState& state) {
  // precondition
  assert(requested_charge > 0);

  // Evict 2-4 homes at a time (+ 2) rather than 1-2 (+ 1) or 0-1 (+ 0)
  // And (- 31) for using only bottom 32 bits of clock pointer field for
  // non-wrap-around count, rather than bottom 63 bits of threshold (except
  // bottom 8).
  // TODO: make a tuning parameter?
  const int step_shift =
      static_cast<int>(state.saved_threshold & 255U) + 2 - 31;
  const uint64_t step = uint64_t{1} << step_shift;

  // First (concurrent) increment clock pointer
  uint64_t clock_pointer =
      clock_pointer_.fetch_add(step, std::memory_order_relaxed);

  // Cap the eviction effort at this thread (along with those operating in
  // parallel) circling through the whole structure kMaxCountdown times.
  // In other words, this eviction run must find something/anything that is
  // unreferenced at start of and during the eviction run that isn't reclaimed
  // by a concurrent eviction run.
  // TODO: Does HyperClockCache need kMaxCountdown + 1?
  const uint64_t max_clock_pointer =
      clock_pointer + (uint64_t{ClockHandle::kMaxCountdown + 1} << 32);

  const size_t frontier = ThresholdToFrontier(state.saved_threshold);

  // Loop until enough freed, or limit reached (see bottom of loop)
  for (;;) {
    // Compute range to apply clock eviction to
    size_t home_begin =
        GetHomeIndex(state.saved_threshold, clock_pointer << 31);
    // Apply last atomic update to local copy
    clock_pointer += step;
    size_t home_end = GetHomeIndex(state.saved_threshold, clock_pointer << 31);

    // fprintf(stderr, "Evicting %lu through %lu\n", home_begin, home_end - 1);
    for (size_t home = home_begin; home != home_end; ++home) {
      // Wrap around
      if (UNLIKELY(home >= frontier)) {
        assert(home == frontier);
        home = home / 2;
        assert(home == GetMinHomeIndex(state.saved_threshold));
        if (home == home_end) {
          break;
        }
      }
      // fprintf(stderr, "Evicting %lu\n", home);

      // Iterate over the entries with this home (though not necessarily
      // a consistent set) and ClockUpdate them. All the crazy corner
      // cases for deletion are dealt with in FinishErasure().
      HandleImpl* prev_h = nullptr;
      HandleImpl* h = array_ + home;
      for (;;) {
        bool evicting = ClockUpdate(*h);
        uint32_t next;
        if (evicting) {
          *freed_charge += h->GetTotalCharge();
          *freed_count += 1;
          next = FinishErasure(h, prev_h);
        } else {
          next = h->next.load(std::memory_order_acquire);
        }
        if ((next & HandleImpl::kNextFollowFlag) == 0) {
          // Reached an end
          // Even if it wasn't *the* end we hoped for, we can tolerate
          // the small chance of clock updating some wrong entries.
          break;
        }
        prev_h = h;
        h = array_ + (next & HandleImpl::kNextNoFlagsMask);
      }
    }

    // Loop exit conditions
    if (*freed_charge >= requested_charge) {
      return;
    }
    if (clock_pointer >= max_clock_pointer) {
      return;
    }

    // Advance clock pointer (concurrently)
    clock_pointer = clock_pointer_.fetch_add(step, std::memory_order_relaxed);
  }
}

int FastClockTable::CalcHashBits(
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
  // Use at least one page for handle table
  hash_bits =
      std::max(hash_bits, FloorLog2(port::kPageSize / sizeof(HandleImpl)));
  return hash_bits;
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
