//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "cache/clock_cache.h"

#include <atomic>
#include <condition_variable>
#include <mutex>

#include "port/port.h"
#include "rocksdb/cache.h"
#include "util/autovector.h"
#include "util/fastrange.h"
#include "util/hash.h"

namespace ROCKSDB_NAMESPACE {

namespace clock_cache {

static constexpr uint32_t kMaxProbes = 100;

struct ProbingInfo {
  ProbingInfo() {}

  ProbingInfo(const Slice& key) {
    Hash2x64(key.data(), key.size(), &hi_data, &lo_data);
    probe_depth = 0;
  }

  uint64_t hi_data;
  uint64_t lo_data;
  uint32_t probe_depth;

  inline bool UnderLimit() { return probe_depth <= kMaxProbes; }

  inline void Incr() {
    hi_data += lo_data;
    lo_data += 0x9E3779B185EBCA87U;
    probe_depth++;
  }

  inline bool Decr() {
    if (probe_depth == 0) {
      return false;
    }
    lo_data -= 0x9E3779B185EBCA87U;
    hi_data -= lo_data;
    probe_depth--;
    return true;
  }

  inline size_t GetPosition(size_t range) {
    return FastRange64(hi_data, range);
  }

  inline uint32_t GetIdHash() { return Lower32of64(lo_data); }

  bool operator==(const ProbingInfo& other) {
    return hi_data == other.hi_data & lo_data == other.lo_data &
           probe_depth == other.probe_depth;
  }
};

// Not "visible" means the handle has been erased (not available for lookup)
// but there are still references, or that the handle is reserved and being
// populated.
static constexpr uint64_t kVisible = uint64_t{1} << 31;
// Pending evict means the handle would have been evicted except for
// references holding it. If it is actually in use or happens to be
// used before no more refs, then it is rescued from eviction.
// This flag is only meaningful if !kRecentUse && kVisible
static constexpr uint64_t kPendingEvict = uint64_t{1} << 30;
static constexpr uint64_t kRecentUse = uint64_t{1} << 29;
static constexpr uint64_t kRefCountMask = kRecentUse - 1;

struct ALIGN_AS(CACHE_LINE_SIZE) ClockHandle : public Cache::Handle {
  // Top 32: hash
  // then meta bits
  // Bottom 30: ref count
  std::atomic<uint64_t> meta{};  // 8b

  void* value;                             // 8b
  Cache::DeleterFn deleter;                // 8b
  ProbingInfo probing;                     // 20b
  std::atomic<uint32_t> overflow_count{};  // 4b
  size_t charge;                           // 8b

  void* unused_padding;  // 8b
};

/*
struct ALIGN_AS(CACHE_LINE_SIZE) ClockUsage {
  std::atomic<size_t> usage{};
  char unused_padding[CACHE_LINE_SIZE - sizeof(size_t)];
};
*/

class ClockCache : public Cache {
 public:
  ClockCache(size_t capacity, size_t num_entries)
      : capacity_(capacity),
        num_entries_(num_entries),
        capacity_low_(capacity - capacity / 256),
        capacity_high_(capacity + capacity / 256) {
    table_.reset(new ClockHandle[num_entries]);
  }

  const char* Name() const override { return "ClockCacheV2"; };

  Status Insert(const Slice& key, void* value, size_t charge, DeleterFn deleter,
                Handle** handle_out, Priority /*priority*/) override {
    autovector<ClockHandle*> to_release_erase;
    ProbingInfo probing(key);
    usage_.fetch_add(charge, std::memory_order_relaxed);
    for (;; probing.Incr()) {
      ClockHandle* handle = table_.get() + probing.GetPosition(num_entries_);
      uint32_t id_hash = probing.GetIdHash();
      uint64_t seen_meta = handle->meta.load(std::memory_order_acquire);
      if (Lower32of64(seen_meta) == 0) {
        // Available to use. Add a ref to reserve it.
        if (handle->meta.compare_exchange_strong(/*&*/ seen_meta, 1)) {
          // Success
          handle->probing = probing;
          handle->value = value;
          handle->deleter = deleter;
          handle->charge = charge;

          uint64_t new_meta_or =
              (uint64_t{id_hash} << 32) | kVisible | kRecentUse;
          seen_meta = handle->meta.fetch_or(new_meta_or);
          assert(seen_meta & kRefCountMask);
          for (ClockHandle* h : to_release_erase) {
            UnrefEraseImpl(h);
          }
          return Status::OK();
        } else {
          // Must have been taken from us. Keep looking.
        }
      } else if (Upper32of64(seen_meta) == id_hash && (seen_meta & kVisible)) {
        // Visible entry and potential match
        // Ref to check fully. (No cmpxchg because no need to interfere with
        // someone else taking or releasing a ref.)
        seen_meta = ++handle->meta;
        if (Upper32of64(seen_meta) == id_hash && (seen_meta & kVisible)) {
          if (handle->probing == probing) {
            // Full match. Remember to erase this one so that our new one
            // replaces it
            to_release_erase.push_back(handle);
          } else {
            // Not a full match. Oh well.
            UnrefImpl(handle);
          }
        } else {
          // It changed. Oh well.
          UnrefImpl(handle);
        }
      }
      if (!probing.UnderLimit()) {
        // Undo usage
        usage_.fetch_sub(charge, std::memory_order_relaxed);
        // Reached probe limit. Reverse course
        BackOutProbingImpl(&probing);
        // Don't erase other matches after all
        for (ClockHandle* h : to_release_erase) {
          UnrefImpl(h);
        }
        // Failed
        *handle_out = nullptr;
        // TODO: maybe forced eviction instead?
        return Status::Incomplete("Probe limit exceeded");
      }
      // Optimistically record our need to probe further for insert
      // TODO: relaxed?
      ++handle->overflow_count;
    }
  }

  Handle* Lookup(const Slice& key, Statistics* /*stats*/) override {
    return LookupImpl(key);
  }

  ClockHandle* LookupImpl(const Slice& key) {
    ProbingInfo probing(key);
    for (; probing.UnderLimit(); probing.Incr()) {
      ClockHandle* handle = table_.get() + probing.GetPosition(num_entries_);
      uint32_t id_hash = probing.GetIdHash();
      uint64_t seen_meta = handle->meta.load(std::memory_order_acquire);
      if (Upper32of64(seen_meta) == id_hash && (seen_meta & kVisible)) {
        // Visible entry and potential match
        // Ref to check fully. (No cmpxchg because no need to interfere with
        // someone else taking or releasing a ref.)
        seen_meta = ++handle->meta;
        // Re-check with ref
        if (Upper32of64(seen_meta) == id_hash && (seen_meta & kVisible)) {
          if (handle->probing == probing) {
            // Full match!
            return handle;
          } else {
            // Not a full match. Oh well.
          }
        } else {
          // It changed. Oh well.
        }
        UnrefImpl(handle);
      }
      if (handle->overflow_count.load(std::memory_order_relaxed) == 0) {
        // No need to probe further
        return nullptr;
      }
    }
    return nullptr;
  }

  void Erase(const Slice& key) override {
    auto handle = LookupImpl(key);
    if (handle) {
      UnrefEraseImpl(handle);
    }
  }

  bool Ref(Handle* handle_base) override {
    if (handle_base == nullptr) {
      return false;
    }
    ClockHandle* handle = static_cast<ClockHandle*>(handle_base);
    // TODO: just acquire?
    ++handle->meta;
    return true;
  }

  bool Release(Handle* handle_base, bool force_erase) override {
    if (handle_base == nullptr) {
      return false;
    }
    ClockHandle* handle = static_cast<ClockHandle*>(handle_base);
    if (force_erase) {
      UnrefEraseImpl(handle);
    } else {
      handle->meta.fetch_or(kRecentUse);
      UnrefImpl(handle);
    }
    return true;
  }

  void UnrefEraseImpl(ClockHandle* handle) {
    // Set not visible (and not used)
    handle->meta.fetch_and(~(kVisible | kRecentUse));

    UnrefImpl(handle);
  }

  inline void UnrefImpl(ClockHandle* handle) {
    // Save info for freeing in case we release last reference on an
    // invisible entry
    void* saved_value = handle->value;
    Cache::DeleterFn saved_deleter = handle->deleter;
    ProbingInfo probing = handle->probing;
    size_t charge = handle->charge;

    // Decrement ref count (atomic with sequential consistency)
    uint64_t new_meta = --handle->meta;

    // Check if we were last
    if ((new_meta & kRefCountMask) == 0) {
      // Unreferenced but still occupying handle.
      // Check if still available for lookup
      if (!(new_meta & kVisible)) {
        // Invisible and unreferenced means we released the entry.
        // Now actually free it.
        // FIXME? actual key by reverse Hash2x64?
        saved_deleter(Slice(), saved_value);
        // Note: we could do this step on setting invisible, but that's
        // not expected to improve much because erase should be rare
        BackOutProbingImpl(&probing);
        // Reclaim usage
        usage_.fetch_sub(charge, std::memory_order_relaxed);
      }
    }
  }

  inline void BackOutProbingImpl(ProbingInfo* probing) {
    while (probing->Decr()) {
      ClockHandle* handle = table_.get() + probing->GetPosition(num_entries_);
      handle->overflow_count.fetch_sub(1, std::memory_order_relaxed);
    }
  }

  void* Value(Handle* handle_base) override {
    ClockHandle* handle = static_cast<ClockHandle*>(handle_base);
    return handle->value;
  }

  size_t GetCharge(Handle* handle_base) const override {
    ClockHandle* handle = static_cast<ClockHandle*>(handle_base);
    return handle->charge;
  }

  DeleterFn GetDeleter(Handle* handle_base) const override {
    ClockHandle* handle = static_cast<ClockHandle*>(handle_base);
    return handle->deleter;
  }

  size_t GetUsage(Handle* handle_base) const override {
    // TODO: metadata charge
    return GetCharge(handle_base);
  }

  uint64_t NewId() override {
    return last_id_.fetch_add(1, std::memory_order_relaxed);
  }

  void SetCapacity(size_t capacity) override {
    // TODO
  }

  void SetStrictCapacityLimit(bool strict_capacity_limit) override {
    // TODO
  }

  bool HasStrictCapacityLimit() const override {
    // TODO
    return false;
  }

  size_t GetCapacity() const override { return capacity_; }

  size_t GetUsage() const override {
    // TODO
    return 0;
  }

  size_t GetPinnedUsage() const override {
    // TODO
    return 0;
  }

  void DisownData() override {
    // TODO
  }

  void ApplyToAllEntries(
      const std::function<void(const Slice& key, void* value, size_t charge,
                               DeleterFn deleter)>& callback,
      const ApplyToAllEntriesOptions& opts) override {
    // TODO
    (void)callback;
    (void)opts;
  }

  void EraseUnRefEntries() override {
    // TODO
  }

 private:
  void BgEvictThreadFn() {
    size_t next_pos = 0;
    std::unique_lock<std::mutex> l(evict_mutex_);
    while (!stop_evict_bg_thread_) {
      size_t usage_copy = usage_;
      if (usage_copy > capacity_) {
        size_t start_pos = next_pos;
        size_t to_free = usage_copy - capacity_low_;
        size_t freed = 0;
        do {
          ClockHandle *handle = table_.get() + next_pos;
          uint64_t seen_meta = handle->meta.load(std::memory_order_acquire);

          if (!(seen_meta & kVisible)) {
            // Nothing to do (invisible entries are removed after last ref)
          } else if (seen_meta & kRecentUse) {
            // Clear recent use and pending evict, even if referenced (e.g. by
            // non-matching lookup)
            handle->meta.fetch_and(~(kRecentUse | kPendingEvict), std::memory_order_release);
          } else {
            // TODO: can we avoid ref on already referenced entry?
            seen_meta = ++handle->meta;
            // Small chance we put this on the wrong thing; oh well
            seen_meta = handle->meta.fetch_or(kPendingEvict);

          }
      if (Upper32of64(seen_meta) == id_hash && (seen_meta & kVisible)) {
        // Visible entry and potential match
        // Ref to check fully. (No cmpxchg because no need to interfere with
        // someone else taking or releasing a ref.)
        seen_meta = ++handle->meta;
        // Re-check with ref
        if (Upper32of64(seen_meta) == id_hash && (seen_meta & kVisible)) {
          if (handle->probing == probing) {
            // Full match!
            return handle;
          } else {
            // Not a full match. Oh well.
          }
        } else {
          // It changed. Oh well.
        }
        UnrefImpl(handle);
      }
          ++next_pos;
          if (next_pos == num_entries_) {next_pos = 0;}
        } while (freed < to_free && next_pos != start_pos);

      } else {
        evict_cv_.wait(/*&*/l);
      }
    }
  }

  const size_t capacity_;
  const size_t num_entries_;
  // FIXME: inline
  std::unique_ptr<ClockHandle[]> table_;

  std::atomic<size_t> usage_{};
  const size_t capacity_low_;
  const size_t capacity_high_;
  /*
  // TOOD: use core local instead (somehow?)
  std::unique_ptr<ClockUsage[]> usage_shards_;
  const size_t usage_shard_mask_;
  const size_t shard_usage_high_;
  const size_t shard_usage_med_;
  const size_t shard_usage_low_;
  */

  std::thread evict_bg_thread_;
  std::mutex evict_mutex_;
  std::condition_variable evict_cv_;
  bool stop_evict_bg_thread_ = false;

  std::atomic<uint64_t> last_id_{};
};

}  // namespace clock_cache

std::shared_ptr<Cache> NewClockCache(
    size_t capacity, int num_shard_bits, bool /*strict_capacity_limit*/,
    CacheMetadataChargePolicy /*metadata_charge_policy*/) {
  return std::make_shared<clock_cache::ClockCache>(capacity,
                                                   /*entries*/ capacity / 3072,
                                                   num_shard_bits);
}

}  // namespace ROCKSDB_NAMESPACE
