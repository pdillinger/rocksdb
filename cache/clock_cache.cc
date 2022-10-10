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
#include <cassert>
#include <functional>
#include <thread>

#include "cache/cache_key.h"
#include "monitoring/perf_context_imp.h"
#include "monitoring/statistics.h"
#include "port/lang.h"
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

inline void FreeDataMarkEmpty(ClockHandle& h) {
  // NOTE: in theory there's more room for parallelism if we copy the handle
  // data and delay actions like this until after marking the entry as empty,
  // but performance tests only show a regression by copying the few words
  // of data.
  h.FreeData();

#ifndef NDEBUG
  // Mark slot as empty, with assertion
  uint64_t meta = h.meta.exchange(0, std::memory_order_release);
  assert(meta >> ClockHandle::kStateShift == ClockHandle::kStateConstruction);
#else
  // Mark slot as empty
  h.meta.store(0, std::memory_order_release);
#endif
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

void ClockHandleBasicData::FreeData() const {
  if (deleter) {
    UniqueId64x2 unhashed;
    (*deleter)(
        ClockCacheShard<HyperClockTable>::ReverseHash(hashed_key, &unhashed),
        value);
  }
}

HyperClockTable::HyperClockTable(
    size_t capacity, bool /*strict_capacity_limit*/,
    CacheMetadataChargePolicy metadata_charge_policy, const Opts& opts)
    : length_bits_(CalcHashBits(capacity, opts.estimated_value_size,
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
        h.FreeData();
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

inline Status HyperClockTable::ChargeUsageMaybeEvictStrict(
    size_t total_charge, size_t capacity, bool need_evict_for_occupancy) {
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
    Evict(request_evict_charge, &evicted_charge, &evicted_count);
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

inline bool HyperClockTable::ChargeUsageMaybeEvictNonStrict(
    size_t total_charge, size_t capacity, bool need_evict_for_occupancy) {
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
    Evict(need_evict_charge, &evicted_charge, &evicted_count);
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

inline HyperClockTable::HandleImpl* HyperClockTable::DetachedInsert(
    const ClockHandleBasicData& proto) {
  // Heap allocated separate from table
  HandleImpl* h = new HandleImpl();
  ClockHandleBasicData* h_alias = h;
  *h_alias = proto;
  h->SetDetached();
  // Single reference (detached entries only created if returning a refed
  // Handle back to user)
  uint64_t meta = uint64_t{ClockHandle::kStateInvisible}
                  << ClockHandle::kStateShift;
  meta |= uint64_t{1} << ClockHandle::kAcquireCounterShift;
  h->meta.store(meta, std::memory_order_release);
  // Keep track of how much of usage is detached
  detached_usage_.fetch_add(proto.GetTotalCharge(), std::memory_order_relaxed);
  return h;
}

Status HyperClockTable::Insert(const ClockHandleBasicData& proto,
                               HandleImpl** handle, Cache::Priority priority,
                               size_t capacity, bool strict_capacity_limit) {
  // Do we have the available occupancy? Optimistically assume we do
  // and deal with it if we don't.
  size_t old_occupancy = occupancy_.fetch_add(1, std::memory_order_acquire);
  auto revert_occupancy_fn = [&]() {
    occupancy_.fetch_sub(1, std::memory_order_relaxed);
  };
  // Whether we over-committed and need an eviction to make up for it
  bool need_evict_for_occupancy = old_occupancy >= occupancy_limit_;

  // Usage/capacity handling is somewhat different depending on
  // strict_capacity_limit, but mostly pessimistic.
  bool use_detached_insert = false;
  const size_t total_charge = proto.GetTotalCharge();
  if (strict_capacity_limit) {
    Status s = ChargeUsageMaybeEvictStrict(total_charge, capacity,
                                           need_evict_for_occupancy);
    if (!s.ok()) {
      revert_occupancy_fn();
      return s;
    }
  } else {
    // Case strict_capacity_limit == false
    bool success = ChargeUsageMaybeEvictNonStrict(total_charge, capacity,
                                                  need_evict_for_occupancy);
    if (!success) {
      revert_occupancy_fn();
      if (handle == nullptr) {
        // Don't insert the entry but still return ok, as if the entry
        // inserted into cache and evicted immediately.
        proto.FreeData();
        return Status::OK();
      } else {
        // Need to track usage of fallback detached insert
        usage_.fetch_add(total_charge, std::memory_order_relaxed);
        use_detached_insert = true;
      }
    }
  }
  auto revert_usage_fn = [&]() {
    usage_.fetch_sub(total_charge, std::memory_order_relaxed);
    // No underflow
    assert(usage_.load(std::memory_order_relaxed) < SIZE_MAX / 2);
  };

  if (!use_detached_insert) {
    // Attempt a table insert, but abort if we find an existing entry for the
    // key. If we were to overwrite old entries, we would either
    // * Have to gain ownership over an existing entry to overwrite it, which
    // would only work if there are no outstanding (read) references and would
    // create a small gap in availability of the entry (old or new) to lookups.
    // * Have to insert into a suboptimal location (more probes) so that the
    // old entry can be kept around as well.

    uint64_t initial_countdown = GetInitialCountdown(priority);
    assert(initial_countdown > 0);

    size_t probe = 0;
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
            new_meta |= (initial_countdown - (handle != nullptr))
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
              // Insert detached instead (only if return handle needed)
              use_detached_insert = true;
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
      use_detached_insert = true;
    }
    if (!use_detached_insert) {
      // Successfully inserted
      if (handle) {
        *handle = e;
      }
      return Status::OK();
    }
    // Roll back table insertion
    Rollback(proto.hashed_key, e);
    revert_occupancy_fn();
    // Maybe fall back on detached insert
    if (handle == nullptr) {
      revert_usage_fn();
      // As if unrefed entry immdiately evicted
      proto.FreeData();
      return Status::OK();
    }
  }

  // Run detached insert
  assert(use_detached_insert);

  *handle = DetachedInsert(proto);

  // The OkOverwritten status is used to count "redundant" insertions into
  // block cache. This implementation doesn't strictly check for redundant
  // insertions, but we instead are probably interested in how many insertions
  // didn't go into the table (instead "detached"), which could be redundant
  // Insert or some other reason (use_detached_insert reasons above).
  return Status::OkOverwritten();
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
    if (UNLIKELY(h->IsDetached())) {
      h->FreeData();
      // Delete detached handle
      delete h;
      detached_usage_.fetch_sub(total_charge, std::memory_order_relaxed);
      usage_.fetch_sub(total_charge, std::memory_order_relaxed);
    } else {
      Rollback(h->hashed_key, h);
      FreeDataMarkEmpty(*h);
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
                FreeDataMarkEmpty(*h);
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
      FreeDataMarkEmpty(h);
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
                                   size_t* freed_charge, size_t* freed_count) {
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
        *freed_charge += h.GetTotalCharge();
        *freed_count += 1;
        FreeDataMarkEmpty(h);
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
    const typename Table::Opts& opts)
    : CacheShardBase(metadata_charge_policy),
      table_(capacity, strict_capacity_limit, metadata_charge_policy, opts),
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
    const std::function<void(const Slice& key, void* value, size_t charge,
                             DeleterFn deleter)>& callback,
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

  table_.ConstApplyToEntriesRange(
      [callback](const HandleImpl& h) {
        UniqueId64x2 unhashed;
        callback(ReverseHash(h.hashed_key, &unhashed), h.value,
                 h.GetTotalCharge(), h.deleter);
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
                                      void* value, size_t charge,
                                      Cache::DeleterFn deleter,
                                      HandleImpl** handle,
                                      Cache::Priority priority) {
  if (UNLIKELY(key.size() != kCacheKeySize)) {
    return Status::NotSupported("ClockCache only supports key size " +
                                std::to_string(kCacheKeySize) + "B");
  }
  ClockHandleBasicData proto;
  proto.hashed_key = hashed_key;
  proto.value = value;
  proto.deleter = deleter;
  proto.total_charge = charge;
  Status s = table_.Insert(
      proto, handle, priority, capacity_.load(std::memory_order_relaxed),
      strict_capacity_limit_.load(std::memory_order_relaxed));
  return s;
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
  return table_pinned_usage + table_.GetDetachedUsage();
}

template <class Table>
size_t ClockCacheShard<Table>::GetOccupancyCount() const {
  return table_.GetOccupancy();
}

template <class Table>
size_t ClockCacheShard<Table>::GetTableAddressCount() const {
  return table_.GetTableSize();
}

// Explicit instantiation
template class ClockCacheShard<HyperClockTable>;

HyperClockCache::HyperClockCache(
    size_t capacity, size_t estimated_value_size, int num_shard_bits,
    bool strict_capacity_limit,
    CacheMetadataChargePolicy metadata_charge_policy,
    std::shared_ptr<MemoryAllocator> memory_allocator)
    : ShardedCache(capacity, num_shard_bits, strict_capacity_limit,
                   std::move(memory_allocator)) {
  assert(estimated_value_size > 0 ||
         metadata_charge_policy != kDontChargeCacheMetadata);
  // TODO: should not need to go through two levels of pointer indirection to
  // get to table entries
  size_t per_shard = GetPerShardCapacity();
  InitShards([=](Shard* cs) {
    HyperClockTable::Opts opts;
    opts.estimated_value_size = estimated_value_size;
    new (cs)
        Shard(per_shard, strict_capacity_limit, metadata_charge_policy, opts);
  });
}

void* HyperClockCache::Value(Handle* handle) {
  return reinterpret_cast<const HandleImpl*>(handle)->value;
}

size_t HyperClockCache::GetCharge(Handle* handle) const {
  return reinterpret_cast<const HandleImpl*>(handle)->GetTotalCharge();
}

Cache::DeleterFn HyperClockCache::GetDeleter(Handle* handle) const {
  auto h = reinterpret_cast<const HandleImpl*>(handle);
  return h->deleter;
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
                               const Opts& opts)
    : max_length_bits_(CalcHashBits(capacity, opts.min_avg_value_size,
                                    metadata_charge_policy)),
      metadata_charge_policy_(metadata_charge_policy),
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

  // Mark home addresses as insertable
  size_t end = ThresholdToFrontier(threshold_.load());
  for (size_t i = GetMinHomeIndex(threshold_.load()); i < end; ++i) {
    array_[i].next = static_cast<uint32_t>(i) | HandleImpl::kNextInsertableFlag;
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
        h.FreeData();
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

bool FastClockTable::GrowIfNeeded(size_t new_occupancy,
                                  uint64_t& known_threshold) {
  size_t old_min_home = GetMinHomeIndex(known_threshold);
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
  known_threshold = threshold_.load();
  old_min_home = GetMinHomeIndex(known_threshold);
  if (new_occupancy <= old_min_home) {
    // Don't need to grow
    return true;
  }
  if ((old_min_home >> max_length_bits_) > 0) {
    // Can't grow any more
    return false;
  }

  // TODO: make sure max_length_bits doesn't let us overflow next refs

  // OK we are growing one old address into two new ones.
  HandleImpl& old_home = array_[old_min_home];
  size_t old_frontier = old_min_home * 2;

  HandleImpl* new_homes = array_ + old_frontier;
  uint64_t new_threshold = FrontierToThreshold(old_frontier + 2);

  // We start by pinning the home entry, (a) so that other operations can
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
  known_threshold = new_threshold;

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

inline bool BeginInsert(const ClockHandleBasicData& proto, ClockHandle& h,
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
    // WART: there's a tiny chance we release last ref to invisible
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

inline void FinishInsert(const ClockHandleBasicData& proto, ClockHandle& h,
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
  bool b = BeginInsert(proto, h, initial_countdown, already_matches);
  if (b) {
    FinishInsert(proto, h, initial_countdown, keep_ref);
  }
  return b;
}

Status FastClockTable::Insert(const ClockHandleBasicData& proto,
                              HandleImpl** handle, Cache::Priority priority,
                              size_t capacity, bool strict_capacity_limit) {
  // *********** ALMOST UNCHANGED ************** //
  // Do we have the available occupancy? Optimistically assume we do
  // and deal with it if we don't.
  size_t old_occupancy = occupancy_.fetch_add(1, std::memory_order_acquire);
  auto revert_occupancy_fn = [&]() {
    occupancy_.fetch_sub(1, std::memory_order_relaxed);
  };
  // Whether we over-committed and need an eviction to make up for it
  bool need_evict_for_occupancy = false;
  uint64_t threshold = threshold_.load(std::memory_order_acquire);
  if (!GrowIfNeeded(old_occupancy + 1, /*in/out*/ threshold)) {
    need_evict_for_occupancy = true;
  }

  // Usage/capacity handling is somewhat different depending on
  // strict_capacity_limit, but mostly pessimistic.
  bool use_detached_insert = false;
  const size_t total_charge = proto.total_charge;
  if (strict_capacity_limit) {
    if (total_charge > capacity) {
      assert(!use_detached_insert);
      revert_occupancy_fn();
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
      Evict(request_evict_charge, threshold, &evicted_charge, &evicted_count);
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
        assert(!use_detached_insert);
        revert_occupancy_fn();
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
  } else {
    // Case strict_capacity_limit == false

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
        // synchronization
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
      // fprintf(stderr, "%s", debug_history.c_str());
      Evict(need_evict_charge, threshold, &evicted_charge, &evicted_count);
      // fprintf(stderr, "Evict %lu to %lu %lu\n", need_evict_charge,
      // evicted_charge, evicted_count);
      // Deal with potential occupancy deficit
      if (UNLIKELY(need_evict_for_occupancy) && evicted_count == 0) {
        assert(evicted_charge == 0);
        revert_occupancy_fn();
        if (handle == nullptr) {
          // Don't insert the entry but still return ok, as if the entry
          // inserted into cache and evicted immediately.
          proto.FreeData();
          return Status::OK();
        } else {
          use_detached_insert = true;
        }
      } else {
        // Update occupancy for evictions
        occupancy_.fetch_sub(evicted_count, std::memory_order_release);
      }
    }
    // Track new usage even if we weren't able to evict enough
    usage_.fetch_add(total_charge - evicted_charge, std::memory_order_relaxed);
    // No underflow
    assert(usage_.load(std::memory_order_relaxed) < SIZE_MAX / 2);
  }
  auto revert_usage_fn = [&]() {
    usage_.fetch_sub(total_charge, std::memory_order_relaxed);
    // No underflow
    assert(usage_.load(std::memory_order_relaxed) < SIZE_MAX / 2);
  };

  if (!use_detached_insert) {
    // Attempt a table insert, but abort if we find an existing entry for the
    // key. If we were to overwrite old entries, we would either
    // * Have to gain ownership over an existing entry to overwrite it, which
    // would only work if there are no outstanding (read) references and would
    // create a small gap in availability of the entry (old or new) to lookups.
    // * Have to insert into a suboptimal location (more probes) so that the
    // old entry can be kept around as well.

    // Set initial clock data from priority
    // TODO: configuration parameters for priority handling and clock cycle
    // count?
    uint64_t initial_countdown;
    switch (priority) {
      case Cache::Priority::HIGH:
        initial_countdown = ClockHandle::kHighCountdown;
        break;
      default:
        assert(false);
        FALLTHROUGH_INTENDED;
      case Cache::Priority::LOW:
        initial_countdown = ClockHandle::kLowCountdown;
        break;
      case Cache::Priority::BOTTOM:
        initial_countdown = ClockHandle::kBottomCountdown;
        break;
    }
    assert(initial_countdown > 0);

    // *********** END ALMOST UNCHANGED ************** //

    size_t home = GetHomeIndex(threshold, proto.hashed_key[0]);

    // Try to insert at home slot
    {
      bool already_matches = false;
      HandleImpl& h = array_[home];
      if (BeginInsert(proto, h, initial_countdown, &already_matches)) {
        // Verify this is still a home address
        if (LIKELY(h.next.load(std::memory_order_acquire) &
                   HandleImpl::kNextInsertableFlag)) {
          FinishInsert(proto, h, initial_countdown,
                       /*keep_ref*/ handle != nullptr);
          if (handle) {
            *handle = array_ + home;
          }
          // debug_history.append("Insert @home " + std::to_string(home) +
          // "\n");
          return Status::OK();
        } else {
          // Concurrent change to non-home.
          MarkEmpty(h);
          // Continue on to chain insertion (even though we already know of an
          // empty slot, it's not easy to wedge into that code)
        }
      }
      if (already_matches) {
        revert_occupancy_fn();
        if (handle == nullptr) {
          revert_usage_fn();
          // As if unrefed entry immdiately evicted
          proto.FreeData();
          return Status::OK();
        } else {
          use_detached_insert = true;
        }
      }
    }

    if (!use_detached_insert) {
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
      size_t chainable_count = GetMinHomeIndex(threshold);
      // (More uniform selection than e.g. home / 2)
      size_t starting_idx = FastRange64(proto.hashed_key[1], chainable_count);
      size_t idx = starting_idx;
      for (int cycles = 0;;) {
        bool already_matches = false;
        if (TryInsert(proto, array_[idx], initial_countdown,
                      /*keep_ref*/ handle != nullptr, &already_matches)) {
          if (handle) {
            *handle = array_ + idx;
          }
          break;
        }
        // Else keep searching for empty slot
        idx += 1;
        if (idx >= chainable_count) {
          idx = 0;
        }
        // NOTE: it's possible we could cycle around due to unlucky temporary
        // arrangements of entries in slots, but should not be indefinite
        if (idx == starting_idx) {
          ++cycles;
          assert(cycles < 100);
          std::this_thread::yield();
          chainable_count =
              GetMinHomeIndex(threshold_.load(std::memory_order_acquire));
        }
      }

      // Now insert into chain starting at home address, though
      // might need to spin / retry
      for (size_t i = 0;; ++i) {
        uint32_t next = array_[home].next.load(std::memory_order_acquire);
        if (next & HandleImpl::kNextInsertableFlag) {
          array_[idx].next.store(next & ~HandleImpl::kNextInsertableFlag,
                                 std::memory_order_acq_rel);
          if (array_[home].next.compare_exchange_weak(
                  next,
                  static_cast<uint32_t>(idx) | HandleImpl::kNextInsertableFlag |
                      HandleImpl::kNextFollowFlag,
                  std::memory_order_acq_rel)) {
            // debug_history.append("Insert @ " + std::to_string(idx) + " for
            // home " + std::to_string(home) + "\n");

            // Successful insertion. All done.
            return Status::OK();
          }
        }
        if (i >= 200) {
          std::this_thread::yield();
        }
        // Home might change with another thread executing Grow
        threshold = threshold_.load(std::memory_order_acquire);
        home = GetHomeIndex(threshold, proto.hashed_key[0]);
      }
    }
  }
  // Else
  // *********** ALMOST UNCHANGED ************** //
  assert(use_detached_insert);

  HandleImpl* h = new HandleImpl();
  ClockHandleBasicData* h_alias = h;
  *h_alias = proto;
  h->detached = true;
  // Single reference (detached entries only created if returning a refed
  // Handle back to user)
  uint64_t meta = uint64_t{ClockHandle::kStateInvisible}
                  << ClockHandle::kStateShift;
  meta |= uint64_t{1} << ClockHandle::kAcquireCounterShift;
  h->meta.store(meta, std::memory_order_release);
  // Keep track of usage
  detached_usage_.fetch_add(total_charge, std::memory_order_relaxed);

  *handle = h;
  // The OkOverwritten status is used to count "redundant" insertions into
  // block cache. This implementation doesn't strictly check for redundant
  // insertions, but we instead are probably interested in how many insertions
  // didn't go into the table (instead "detached"), which could be redundant
  // Insert or some other reason (use_detached_insert reasons above).
  return Status::OkOverwritten();
  // *********** END ALMOST UNCHANGED ************** //
}

inline bool MatchAndRef(const UniqueId64x2& hashed_key, ClockHandle& h) {
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
    if (h.hashed_key == hashed_key) {
      // Match
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

FastClockTable::HandleImpl* FastClockTable::Lookup(
    const UniqueId64x2& hashed_key) {
  // Might have to retry in rare cases
  for (size_t i = 0;; ++i) {
    uint64_t threshold = threshold_.load(std::memory_order_acquire);
    size_t home = GetHomeIndex(threshold, hashed_key[0]);

    HandleImpl* h = array_ + home;
    if (MatchAndRef(hashed_key, *h)) {
      return h;
    }

    uint32_t next = h->next.load(std::memory_order_acquire);
    while (next & HandleImpl::kNextFollowFlag) {
      h = array_ + (next & HandleImpl::kNextNoFlagsMask);
      if (MatchAndRef(hashed_key, *h)) {
        return h;
      }
      next = h->next.load(std::memory_order_acquire);
    }

    if ((next & HandleImpl::kNextNoFlagsMask) == home) {
      // Clean query, no match found
      return nullptr;
    }
    // Else, some other action prevented us from walking the list cleanly.
    // Try again, though yield after a spin limit
    if (i >= 200) {
      std::this_thread::yield();
    }
  }
}

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
  h->FreeData();
  size_t total_charge = h->total_charge;
  if (UNLIKELY(h->detached)) {
    // Delete detached handle
    delete h;
    detached_usage_.fetch_sub(total_charge, std::memory_order_relaxed);
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
      h.FreeData();
      usage_.fetch_sub(h.total_charge, std::memory_order_relaxed);
      FinishErasure(&h, /*possible_prev unknown*/ nullptr);
      occupancy_.fetch_sub(1U, std::memory_order_release);
    }
  }
}

void FastClockTable::Evict(size_t requested_charge, uint64_t known_threshold,
                           size_t* freed_charge, size_t* freed_count) {
  // precondition
  assert(requested_charge > 0);

  // Evict 2-4 homes at a time (+ 2) rather than 1-2 (+ 1) or 0-1 (+ 0)
  // And (- 31) for using only bottom 32 bits of clock pointer field for
  // non-wrap-around count, rather than bottom 63 bits of threshold (except
  // bottom 8).
  // TODO: make a tuning parameter?
  const int step_shift = static_cast<int>(known_threshold & 255U) + 2 - 31;
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

  const size_t frontier = ThresholdToFrontier(known_threshold);

  // Loop until enough freed, or limit reached (see bottom of loop)
  for (;;) {
    // Compute range to apply clock eviction to
    size_t home_begin = GetHomeIndex(known_threshold, clock_pointer << 31);
    // Apply last atomic update to local copy
    clock_pointer += step;
    size_t home_end = GetHomeIndex(known_threshold, clock_pointer << 31);

    // fprintf(stderr, "Evicting %lu through %lu\n", home_begin, home_end - 1);
    for (size_t home = home_begin; home != home_end; ++home) {
      // Wrap around
      if (UNLIKELY(home >= frontier)) {
        assert(home == frontier);
        home = home / 2;
        assert(home == GetMinHomeIndex(known_threshold));
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
        bool evicting = ClockUpdate(*h, freed_charge, freed_count);
        uint32_t next;
        if (evicting) {
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

FastClockCache::FastClockCache(
    size_t capacity, int num_shard_bits, bool strict_capacity_limit,
    CacheMetadataChargePolicy metadata_charge_policy,
    std::shared_ptr<MemoryAllocator> memory_allocator,
    size_t min_avg_value_size)
    : ShardedCache(capacity, num_shard_bits, strict_capacity_limit,
                   std::move(memory_allocator)) {
  assert(min_avg_value_size > 0 ||
         metadata_charge_policy != kDontChargeCacheMetadata);
  // TODO: should not need to go through two levels of pointer indirection to
  // get to table entries
  size_t per_shard = GetPerShardCapacity();
  InitShards([=](Shard* cs) {
    FastClockTable::Opts opts;
    opts.min_avg_value_size = min_avg_value_size;
    new (cs)
        Shard(per_shard, strict_capacity_limit, metadata_charge_policy, opts);
  });
}

void* FastClockCache::Value(Handle* handle) {
  return reinterpret_cast<const HandleImpl*>(handle)->value;
}

size_t FastClockCache::GetCharge(Handle* handle) const {
  return reinterpret_cast<const HandleImpl*>(handle)->total_charge;
}

Cache::DeleterFn FastClockCache::GetDeleter(Handle* handle) const {
  auto h = reinterpret_cast<const HandleImpl*>(handle);
  return h->deleter;
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
  auto my_num_shard_bits = num_shard_bits;
  if (my_num_shard_bits >= 20) {
    return nullptr;  // The cache cannot be sharded into too many fine pieces.
  }
  if (my_num_shard_bits < 0) {
    // Use larger shard size to reduce risk of large entries clustering
    // or skewing individual shards.
    constexpr size_t min_shard_size = 32U * 1024U * 1024U;
    my_num_shard_bits = GetDefaultCacheShardBits(capacity, min_shard_size);
  }
  return std::make_shared<clock_cache::HyperClockCache>(
      capacity, estimated_entry_charge, my_num_shard_bits,
      strict_capacity_limit, metadata_charge_policy, memory_allocator);
}

std::shared_ptr<Cache> FastClockCacheOptions::MakeSharedCache() const {
  auto my_num_shard_bits = num_shard_bits;
  if (my_num_shard_bits >= 20) {
    return nullptr;  // The cache cannot be sharded into too many fine pieces.
  }
  if (my_num_shard_bits < 0) {
    // Use larger shard size to reduce risk of large entries clustering
    // or skewing individual shards.
    constexpr size_t min_shard_size = 32U * 1024U * 1024U;
    my_num_shard_bits = GetDefaultCacheShardBits(capacity, min_shard_size);
  }
  return std::make_shared<clock_cache::FastClockCache>(
      capacity, my_num_shard_bits, strict_capacity_limit,
      metadata_charge_policy, memory_allocator, min_avg_entry_charge);
}

}  // namespace ROCKSDB_NAMESPACE
