//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#pragma once

#include <atomic>
#include <cstdint>
#include <string>

#include "port/lang.h"
#include "port/port.h"
#include "rocksdb/advanced_cache.h"
#include "rocksdb/secondary_cache.h"
#include "util/hash.h"
#include "util/mutexlock.h"

namespace ROCKSDB_NAMESPACE {

struct CacheHandleBase : public Cache::Handle {
  // If not the thread that owns populating the value and this handle is
  // potentially pending, the value can spontaneously go from nullptr to
  // non-nullptr (no longer pending), but never changes once non-nullptr.
  // (If some thread owns populating the value, it will be keeping a ref
  // to the handle.)
  std::atomic<Cache::ObjectPtr> value;

  // Bit 0 (& 1): "standalone" marker
  // Bit 1 (& 2): "in secondary cache" marker
  // ^^^ The above bits are immutable after initial handle population.
  // Bit 2 (& 4): whether some thread may be waiting on a result from
  // this handle (optimization to minimize interaction with mutex+cv)
  // Bit 3 (& 8): whether this handle is "pending". Right before the "total
  // charge" so that we can use arithmetic to change from pending to ready
  // with a particular charge.
  //
  // If this handle is pending, then the value with the bottom bits masked
  // to 0 is a special SecondaryCacheResultHandle*.
  // == marker value: some thread owns producing a value for this handle.
  // == 0: no current owner for producing a value and no associated
  // SecondaryCacheResultHandle. The handle might be about to be abandonned/
  // erased due to an aborted lookup so it looks like total_charge==0 to the
  // Release() function. This state is also used for "dummy" entries in primary
  // cache, to mark a recent access without full promotion.
  // Else: owner pointer to a SecondaryCacheResultHandle* that no thread
  // yet owns waiting on, but at least one thread holds a reference to this
  // pending handle.
  //
  // If this handle is not initially pending, then the value is set to
  // `total_charge << 4`.
  //
  // For async Lookups, threads not populating the value+charge might see a
  // wrong charge even if the value is non-nullptr. (value is updated before
  // this field to ensure that !IsPending() implies value != nullptr if
  // is a secondary cache compatible Lookup and no failure).
  //
  // NOTE: if the thread that waits on the SecondaryCacheResultHandle finds
  // no result is returned, that same thread can own producing a value through
  // normal means (I/O).
  //
  // In case of needing to abort, mark the handle as needing an owner with
  // the special 0 SecondaryCacheResultHandle* (under mutex and signal cv, if
  // maybe another waiter) and call Release(erase_if_last_ref=true) which will
  // release the handle if no other thread is waiting for a result.
  std::atomic<uintptr_t> total_charge_or_pending_meta;
  static constexpr int kFlagStandaloneShift = 0U;
  static constexpr uintptr_t kFlagStandaloneBit = 1U << kFlagStandaloneShift;
  static constexpr int kFlagInSecondaryShift = 1U;
  static constexpr uintptr_t kFlagInSecondaryBit = 1U << kFlagInSecondaryShift;
  static constexpr int kFlagOtherWaitingShift = 2U;
  static constexpr uintptr_t kFlagOtherWaitingBit = 1U << kFlagOtherWaitingShift;
  static constexpr int kFlagPendingShift = 3U;
  static constexpr uintptr_t kFlagPendingBit = 1U << kFlagPendingShift;
  static constexpr int kTotalChargeShift = 4U;

  static constexpr uintptr_t kNoFlagsMask = ~(kFlagStandaloneBit | kFlagInSecondaryBit | kFlagOtherWaitingBit | kFlagPendingBit);

  // We assume no legit object will have such a low address
  static constexpr uintptr_t kOwnedPendingMarker = 1U << kTotalChargeShift;


  bool IsPending() const {
    return (total_charge_or_pending_meta.load(std::memory_order_relaxed) & kFlagPendingBit) != 0;
  }

  // Immutable after inital handle population
  const Cache::CacheItemHelper* helper;
};

// Optional base class for classes implementing the CacheShard concept
class CacheShardBase {
 public:
  // Expected by concept CacheShard (TODO with C++20 support)
  // Some Defaults
  std::string GetPrintableOptions() const { return ""; }
  using HashVal = uint64_t;
  using HashCref = uint64_t;
  static inline HashVal ComputeHash(const Slice& key) {
    return GetSliceNPHash64(key);
  }
  static inline uint32_t HashPieceForSharding(HashCref hash) {
    return Lower32of64(hash);
  }
  static size_t CalcMetadataCharge() {
    // E.g. for HyperClockCache which doesn't charge metadata on insertion
    return 0;
  }

  void AppendPrintableOptions(std::string& /*str*/) const {}

  // Must be provided for concept CacheShard (TODO with C++20 support)
  /*
  struct HandleImpl : public CacheHandleBase {  // for concept HandleImpl
    HashVal hash;
    HashCref GetHash() const;
    ...
  };


  Status Insert(const Slice& key, HashCref hash, Cache::ObjectPtr value,
                const Cache::CacheItemHelper* helper, size_t charge,
                HandleImpl** handle, Cache::Priority priority) = 0;
  HandleImpl* CreateStandalone(const Slice& key, HashCref hash, Cache::ObjectPtr value,
                               const Cache::CacheItemHelper* helper, uintpt_t total_charge_or_pending_meta) = 0;
  HandleImpl* Lookup(const Slice& key, HashCref hash) = 0;
  HandleImpl* LookupOrInsert(const Slice& key, HashCref hash,
                        Cache::ObjectPtr value,
                        const Cache::CacheItemHelper* helper,
                        uintpt_t total_charge_or_pending_meta,
                        Cache::CreateContext* create_context,
                        Cache::Priority priority,
                        bool* insertion_owner) = 0;
  Status ChargeForFinishInsert(size_t charge, size_t* total_charge) = 0;

  bool Release(HandleImpl* handle, bool useful, bool erase_if_last_ref) = 0;
  bool Ref(HandleImpl* handle) = 0;
  void Erase(const Slice& key, HashCref hash) = 0;
  void SetCapacity(size_t capacity) = 0;
  void SetStrictCapacityLimit(bool strict_capacity_limit) = 0;
  size_t GetUsage() const = 0;
  size_t GetPinnedUsage() const = 0;
  size_t GetOccupancyCount() const = 0;
  size_t GetTableAddressCount() const = 0;
  // Handles iterating over roughly `average_entries_per_lock` entries, using
  // `state` to somehow record where it last ended up. Caller initially uses
  // *state == 0 and implementation sets *state = SIZE_MAX to indicate
  // completion.
  void ApplyToSomeEntries(
      const std::function<void(const Slice& key, ObjectPtr value,
                               size_t charge,
                               const Cache::CacheItemHelper* helper)>& callback,
      size_t average_entries_per_lock, size_t* state) = 0;
  void EraseUnRefEntries() = 0;
  */
};

// Portions of ShardedCache that do not depend on the template parameter
class ShardedCacheBase : public Cache {
 public:
  ShardedCacheBase(size_t capacity, int num_shard_bits,
                   bool strict_capacity_limit,
                   std::shared_ptr<MemoryAllocator> memory_allocator,
                   CacheMetadataChargePolicy metadata_charge_policy);
  virtual ~ShardedCacheBase() = default;

  int GetNumShardBits() const;
  uint32_t GetNumShards() const;

  uint64_t NewId() override;

  bool HasStrictCapacityLimit() const override;
  size_t GetCapacity() const override;

  using Cache::GetUsage;
  size_t GetUsage(Handle* handle) const override;
  std::string GetPrintableOptions() const override;

  ObjectPtr Value(Handle* handle) override;
  const CacheItemHelper* GetCacheItemHelper(Handle* handle) const override;

 protected:  // fns
  virtual void AppendPrintableOptions(std::string& str) const = 0;
  size_t GetPerShardCapacity() const;
  size_t ComputePerShardCapacity(size_t capacity) const;
  size_t GetChargeFromTotalCharge(size_t total_charge) const;
  size_t GetTotalChargeFromCharge(size_t total_charge) const;

 protected:                        // data
  std::atomic<uint64_t> last_id_;  // For NewId
  const uint32_t shard_mask_;

  // Dynamic configuration parameters, guarded by config_mutex_
  bool strict_capacity_limit_;
  size_t capacity_;
  mutable port::Mutex config_mutex_;

  const CacheMetadataChargePolicy metadata_charge_policy_;
};

// Generic cache interface that shards cache by hash of keys. 2^num_shard_bits
// shards will be created, with capacity split evenly to each of the shards.
// Keys are typically sharded by the lowest num_shard_bits bits of hash value
// so that the upper bits of the hash value can keep a stable ordering of
// table entries even as the table grows (using more upper hash bits).
// See CacheShardBase above for what is expected of the CacheShard parameter.
template <class CacheShard>
class ShardedCache : public ShardedCacheBase {
 public:
  using typename CacheShard::HashVal;
  using typename CacheShard::HashCref;
  using typename CacheShard::HandleImpl;

  ShardedCache(size_t capacity, int num_shard_bits, bool strict_capacity_limit,
               std::shared_ptr<MemoryAllocator> allocator, CacheMetadataChargePolicy metadata_charge_policy)
      : ShardedCacheBase(capacity, num_shard_bits, strict_capacity_limit,
                         std::move(allocator), metadata_charge_policy),
        shards_(reinterpret_cast<CacheShard*>(port::cacheline_aligned_alloc(
            sizeof(CacheShard) * GetNumShards()))),
        destroy_shards_in_dtor_(false) {}

  virtual ~ShardedCache() {
    if (destroy_shards_in_dtor_) {
      ForEachShard([](CacheShard* cs) { cs->~CacheShard(); });
    }
    port::cacheline_aligned_free(shards_);
  }

  CacheShard& GetShard(HashCref hash) {
    return shards_[CacheShard::HashPieceForSharding(hash) & shard_mask_];
  }

  const CacheShard& GetShard(HashCref hash) const {
    return shards_[CacheShard::HashPieceForSharding(hash) & shard_mask_];
  }

  size_t GetCharge(Handle* handle) const override {
    auto h = static_cast<const CacheHandleBase*>(handle);
    auto total_charge_or_pending_meta = h->total_charge_or_pending_meta.load(std::memory_order_relaxed);
    assert((total_charge_or_pending_meta & CacheHandleBase::kFlagPendingBit) == 0);
    return GetShard(0).GetChargeFromTotalCharge(total_charge_or_pending_meta >> CacheHandleBase::kTotalChargeShift);
  }

  void SetCapacity(size_t capacity) override {
    MutexLock l(&config_mutex_);
    capacity_ = capacity;
    auto per_shard = ComputePerShardCapacity(capacity);
    ForEachShard([=](CacheShard* cs) { cs->SetCapacity(per_shard); });
  }

  void SetStrictCapacityLimit(bool s_c_l) override {
    MutexLock l(&config_mutex_);
    strict_capacity_limit_ = s_c_l;
    ForEachShard(
        [s_c_l](CacheShard* cs) { cs->SetStrictCapacityLimit(s_c_l); });
  }

  Status Insert(const Slice& key, ObjectPtr value,
                const CacheItemHelper* helper, size_t charge,
                Handle** handle = nullptr,
                Priority priority = Priority::LOW) override {
    assert(helper);
    HashVal hash = CacheShard::ComputeHash(key);
    auto h_out = static_cast<HandleImpl**>(handle);
    return GetShard(hash).Insert(key, hash, value, helper, charge, h_out,
                                 priority);
  }

  Handle* Lookup(const Slice& key, const CacheItemHelper* helper = nullptr,
                 CreateContext* create_context = nullptr,
                 Priority priority = Priority::LOW, bool wait = true,
                 Statistics* stats = nullptr) override {
    // No secondary support
    (void)helper;
    (void)create_context;
    (void)priority;
    (void)wait;
    (void)stats;

    HashVal hash = CacheShard::ComputeHash(key);
    HandleImpl* result = GetShard(hash).Lookup(
        key, hash);
    return result;
  }

  void Erase(const Slice& key) override {
    HashVal hash = CacheShard::ComputeHash(key);
    GetShard(hash).Erase(key, hash);
  }

  bool Release(Handle* handle, bool useful,
               bool erase_if_last_ref = false) override {
    auto h = static_cast<HandleImpl*>(handle);
    return GetShard(h->GetHash()).Release(h, useful, erase_if_last_ref);
  }
  bool IsReady(Handle* handle) override {
    // No secondary support
    return true;
  }
  void Wait(Handle* handle) override {
    // No secondary support
  }
  bool Ref(Handle* handle) override {
    auto h = static_cast<HandleImpl*>(handle);
    return GetShard(h->GetHash()).Ref(h);
  }
  bool Release(Handle* handle, bool erase_if_last_ref = false) override {
    return Release(handle, true /*useful*/, erase_if_last_ref);
  }
  using ShardedCacheBase::GetUsage;
  size_t GetUsage() const override {
    return SumOverShards2(&CacheShard::GetUsage);
  }
  size_t GetPinnedUsage() const override {
    return SumOverShards2(&CacheShard::GetPinnedUsage);
  }
  size_t GetOccupancyCount() const override {
    return SumOverShards2(&CacheShard::GetPinnedUsage);
  }
  size_t GetTableAddressCount() const override {
    return SumOverShards2(&CacheShard::GetTableAddressCount);
  }
  void ApplyToAllEntries(
      const std::function<void(const Slice& key, ObjectPtr value, size_t charge,
                               const CacheItemHelper* helper)>& callback,
      const ApplyToAllEntriesOptions& opts) override {
    uint32_t num_shards = GetNumShards();
    // Iterate over part of each shard, rotating between shards, to
    // minimize impact on latency of concurrent operations.
    std::unique_ptr<size_t[]> states(new size_t[num_shards]{});

    size_t aepl = opts.average_entries_per_lock;
    aepl = std::min(aepl, size_t{1});

    bool remaining_work;
    do {
      remaining_work = false;
      for (uint32_t i = 0; i < num_shards; i++) {
        if (states[i] != SIZE_MAX) {
          shards_[i].ApplyToSomeEntries(callback, aepl, &states[i]);
          remaining_work |= states[i] != SIZE_MAX;
        }
      }
    } while (remaining_work);
  }

  virtual void EraseUnRefEntries() override {
    ForEachShard([](CacheShard* cs) { cs->EraseUnRefEntries(); });
  }

  void DisownData() override {
    // Leak data only if that won't generate an ASAN/valgrind warning.
    if (!kMustFreeHeapAllocations) {
      destroy_shards_in_dtor_ = false;
    }
  }

 protected:
  inline void ForEachShard(const std::function<void(CacheShard*)>& fn) {
    uint32_t num_shards = GetNumShards();
    for (uint32_t i = 0; i < num_shards; i++) {
      fn(shards_ + i);
    }
  }

  inline size_t SumOverShards(
      const std::function<size_t(CacheShard&)>& fn) const {
    uint32_t num_shards = GetNumShards();
    size_t result = 0;
    for (uint32_t i = 0; i < num_shards; i++) {
      result += fn(shards_[i]);
    }
    return result;
  }

  inline size_t SumOverShards2(size_t (CacheShard::*fn)() const) const {
    return SumOverShards([fn](CacheShard& cs) { return (cs.*fn)(); });
  }

  // Must be called exactly once by derived class constructor
  void InitShards(const std::function<void(CacheShard*)>& placement_new) {
    ForEachShard(placement_new);
    destroy_shards_in_dtor_ = true;
  }

  void AppendPrintableOptions(std::string& str) const override {
    shards_[0].AppendPrintableOptions(str);
  }

 private:
  CacheShard* const shards_;
  bool destroy_shards_in_dtor_;
};

template <class CacheShard>
class ShardedCacheWithSecondary : public ShardedCache<CacheShard> {
 public:
  using typename CacheShard::HashVal;
  using typename CacheShard::HashCref;
  using typename CacheShard::HandleImpl;
  using typename Cache::Handle;
  using typename Cache::CacheItemHelper;
  using typename Cache::CreateContext;
  using typename Cache::Priority;

  ShardedCacheWithSecondary(size_t capacity, int num_shard_bits, bool strict_capacity_limit,
               std::shared_ptr<MemoryAllocator> allocator, std::shared_ptr<SecondaryCache> secondary_cache)
      : ShardedCache<CacheShard>(capacity, num_shard_bits, strict_capacity_limit,
                         std::move(allocator)),
                         secondary_cache_(std::move(secondary_cache)) {
    assert(secondary_cache_);
  }

  Handle* Lookup(const Slice& key, const CacheItemHelper* helper = nullptr,
                 CreateContext* create_context = nullptr,
                 Priority priority = Priority::LOW, bool wait = true,
                 Statistics* stats = nullptr) override {
    HashVal hash = CacheShard::ComputeHash(key);
    if (helper == nullptr) {
      // Not SecondaryCache compatible
      return GetShard(hash).Lookup(key, hash);
    }

    bool insertion_owner = false;
    uintptr_t pending_meta = CacheHandleBase::kOwnedPendingMarker | CacheHandleBase::kFlagPendingBit;
    HandleImpl* handle = GetShard(hash).LookupOrStartInsert(
        key, hash, helper, pending_meta, create_context, priority, &insertion_owner);

    if (UNLIKELY(handle == nullptr)) {
      // Unlikely: not found and can't add a handle for secondary cache lookup
      return nullptr;
    }

    if (wait) {
      if (handle->IsPending()) {
        if (!insertion_owner) {
          // TODO: mark shared?
          // TODO: attempt take ownership if unowned, maybe getting an existing secondary handle
        }
        if (insertion_owner) {
          // TODO: synchronous lookup from secondary cache
          // TODO: store result and notify waiters if shared
        } else {
          // or maybe waitimpl because we don't need to check ispending before acquiring mutex
          Wait(handle);
        }
      }
      assert(!handle->IsPending());
      if (Value(handle) == nullptr) {
        // If secondary cache lookup fails with wait=true, we do not return
        // a handle to the caller.
        Release(handle, /*erase_if_last_ref*/true);
        return nullptr;
      }
    } else if (insertion_owner) {
      // TODO: initiate async lookup from secondary cache
    }

    return handle;
  }

  bool IsReady(Handle* handle) override {
    // TODO: complicated. Here's a rough approximation
    auto h = static_cast<HandleImpl*>(handle);
    return !h->IsPending();
  }

  void Wait(Handle* handle) override {
    auto h = static_cast<HandleImpl*>(handle);
    GetShard(h->GetHash()).Wait(h);
  }

 protected:
  std::shared_ptr<SecondaryCache> secondary_cache_;
};

// 512KB is traditional minimum shard size.
int GetDefaultCacheShardBits(size_t capacity,
                             size_t min_shard_size = 512U * 1024U);

}  // namespace ROCKSDB_NAMESPACE
