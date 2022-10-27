//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#pragma once

#include "rocksdb/cache.h"

namespace ROCKSDB_NAMESPACE {

class Cache {
 public:
  using Priority = cache::Priority;
  using SizeCallback = cache::SizeCallback;
  using SaveToCallback = cache::SaveToCallback;
  using DeleterFn = cache::DeleterFn;
  using CacheItemHelper = cache::CacheItemHelper;
  using CreateCallback = cache::CreateCallback;

  Cache(std::shared_ptr<MemoryAllocator> allocator = nullptr)
      : memory_allocator_(std::move(allocator)) {}
  // No copying allowed
  Cache(const Cache&) = delete;
  Cache& operator=(const Cache&) = delete;

  // Creates a new Cache based on the input value string and returns the result.
  // Currently, this method can be used to create LRUCaches only
  // @param config_options
  // @param value  The value might be:
  //   - an old-style cache ("1M") -- equivalent to NewLRUCache(1024*102(
  //   - Name-value option pairs -- "capacity=1M; num_shard_bits=4;
  //     For the LRUCache, the values are defined in LRUCacheOptions.
  // @param result The new Cache object
  // @return OK if the cache was successfully created
  // @return NotFound if an invalid name was specified in the value
  // @return InvalidArgument if either the options were not valid
  static Status CreateFromString(const ConfigOptions& config_options,
                                 const std::string& value,
                                 std::shared_ptr<Cache>* result);

  // Destroys all existing entries by calling the "deleter"
  // function that was passed via the Insert() function.
  //
  // @See Insert
  virtual ~Cache() {}

  // Opaque handle to an entry stored in the cache.
  struct Handle {};

  // The type of the Cache
  virtual const char* Name() const = 0;

  // Insert a mapping from key->value into the volatile cache only
  // and assign it with the specified charge against the total cache capacity.
  // If strict_capacity_limit is true and cache reaches its full capacity,
  // return Status::MemoryLimit.
  //
  // If handle is not nullptr, returns a handle that corresponds to the
  // mapping. The caller must call this->Release(handle) when the returned
  // mapping is no longer needed. In case of error caller is responsible to
  // cleanup the value (i.e. calling "deleter").
  //
  // If handle is nullptr, it is as if Release is called immediately after
  // insert. In case of error value will be cleanup.
  //
  // When the inserted entry is no longer needed, the key and
  // value will be passed to "deleter" which must delete the value.
  // (The Cache is responsible for copying and reclaiming space for
  // the key.)
  virtual Status Insert(const Slice& key, void* value, size_t charge,
                        DeleterFn deleter, Handle** handle = nullptr,
                        Priority priority = Priority::LOW) = 0;

  // If the cache has no mapping for "key", returns nullptr.
  //
  // Else return a handle that corresponds to the mapping.  The caller
  // must call this->Release(handle) when the returned mapping is no
  // longer needed.
  // If stats is not nullptr, relative tickers could be used inside the
  // function.
  virtual Handle* Lookup(const Slice& key, Statistics* stats = nullptr) = 0;

  // Increments the reference count for the handle if it refers to an entry in
  // the cache. Returns true if refcount was incremented; otherwise, returns
  // false.
  // REQUIRES: handle must have been returned by a method on *this.
  virtual bool Ref(Handle* handle) = 0;

  /**
   * Release a mapping returned by a previous Lookup(). A released entry might
   * still remain in cache in case it is later looked up by others. If
   * erase_if_last_ref is set then it also erases it from the cache if there is
   * no other reference to  it. Erasing it should call the deleter function that
   * was provided when the entry was inserted.
   *
   * Returns true if the entry was also erased.
   */
  // REQUIRES: handle must not have been released yet.
  // REQUIRES: handle must have been returned by a method on *this.
  virtual bool Release(Handle* handle, bool erase_if_last_ref = false) = 0;

  // Return the value encapsulated in a handle returned by a
  // successful Lookup().
  // REQUIRES: handle must not have been released yet.
  // REQUIRES: handle must have been returned by a method on *this.
  virtual void* Value(Handle* handle) = 0;

  // If the cache contains the entry for the key, erase it.  Note that the
  // underlying entry will be kept around until all existing handles
  // to it have been released.
  virtual void Erase(const Slice& key) = 0;
  // Return a new numeric id.  May be used by multiple clients who are
  // sharding the same cache to partition the key space.  Typically the
  // client will allocate a new id at startup and prepend the id to
  // its cache keys.
  virtual uint64_t NewId() = 0;

  // sets the maximum configured capacity of the cache. When the new
  // capacity is less than the old capacity and the existing usage is
  // greater than new capacity, the implementation will do its best job to
  // purge the released entries from the cache in order to lower the usage
  virtual void SetCapacity(size_t capacity) = 0;

  // Set whether to return error on insertion when cache reaches its full
  // capacity.
  virtual void SetStrictCapacityLimit(bool strict_capacity_limit) = 0;

  // Get the flag whether to return error on insertion when cache reaches its
  // full capacity.
  virtual bool HasStrictCapacityLimit() const = 0;

  // Returns the maximum configured capacity of the cache
  virtual size_t GetCapacity() const = 0;

  // Returns the memory size for the entries residing in the cache.
  virtual size_t GetUsage() const = 0;

  // Returns the number of entries currently tracked in the table. SIZE_MAX
  // means "not supported." This is used for inspecting the load factor, along
  // with GetTableAddressCount().
  virtual size_t GetOccupancyCount() const { return SIZE_MAX; }

  // Returns the number of ways the hash function is divided for addressing
  // entries. Zero means "not supported." This is used for inspecting the load
  // factor, along with GetOccupancyCount().
  virtual size_t GetTableAddressCount() const { return 0; }

  // Returns the memory size for a specific entry in the cache.
  virtual size_t GetUsage(Handle* handle) const = 0;

  // Returns the memory size for the entries in use by the system
  virtual size_t GetPinnedUsage() const = 0;

  // Returns the charge for the specific entry in the cache.
  virtual size_t GetCharge(Handle* handle) const = 0;

  // Returns the deleter for the specified entry. This might seem useless
  // as the Cache itself is responsible for calling the deleter, but
  // the deleter can essentially verify that a cache entry is of an
  // expected type from an expected code source.
  virtual DeleterFn GetDeleter(Handle* handle) const = 0;

  // Call this on shutdown if you want to speed it up. Cache will disown
  // any underlying data and will not free it on delete. This call will leak
  // memory - call this only if you're shutting down the process.
  // Any attempts of using cache after this call will fail terribly.
  // Always delete the DB object before calling this method!
  virtual void DisownData() {
    // default implementation is noop
  }

  struct ApplyToAllEntriesOptions {
    // If the Cache uses locks, setting `average_entries_per_lock` to
    // a higher value suggests iterating over more entries each time a lock
    // is acquired, likely reducing the time for ApplyToAllEntries but
    // increasing latency for concurrent users of the Cache. Setting
    // `average_entries_per_lock` to a smaller value could be helpful if
    // callback is relatively expensive, such as using large data structures.
    size_t average_entries_per_lock = 256;
  };

  // Apply a callback to all entries in the cache. The Cache must ensure
  // thread safety but does not guarantee that a consistent snapshot of all
  // entries is iterated over if other threads are operating on the Cache
  // also.
  virtual void ApplyToAllEntries(
      const std::function<void(const Slice& key, void* value, size_t charge,
                               DeleterFn deleter)>& callback,
      const ApplyToAllEntriesOptions& opts) = 0;

  // DEPRECATED version of above. (Default implementation uses above.)
  virtual void ApplyToAllCacheEntries(void (*callback)(void* value,
                                                       size_t charge),
                                      bool /*thread_safe*/) {
    ApplyToAllEntries([callback](const Slice&, void* value, size_t charge,
                                 DeleterFn) { callback(value, charge); },
                      {});
  }

  // Remove all entries.
  // Prerequisite: no entry is referenced.
  virtual void EraseUnRefEntries() = 0;

  virtual std::string GetPrintableOptions() const { return ""; }

  MemoryAllocator* memory_allocator() const { return memory_allocator_.get(); }

  // EXPERIMENTAL
  // The following APIs are experimental and might change in the future.
  // The Insert and Lookup APIs below are intended to allow cached objects
  // to be demoted/promoted between the primary block cache and a secondary
  // cache. The secondary cache could be a non-volatile cache, and will
  // likely store the object in a different representation. They rely on a
  // per object CacheItemHelper to do the conversions.
  // The secondary cache may persist across process and system restarts,
  // and may even be moved between hosts. Therefore, the cache key must
  // be repeatable across restarts/reboots, and globally unique if
  // multiple DBs share the same cache and the set of DBs can change
  // over time.

  // Insert a mapping from key->value into the cache and assign it
  // the specified charge against the total cache capacity. If
  // strict_capacity_limit is true and cache reaches its full capacity,
  // return Status::MemoryLimit. `value` must be non-nullptr for this
  // Insert() because Value() == nullptr is reserved for indicating failure
  // with secondary-cache-compatible mappings.
  //
  // The helper argument is saved by the cache and will be used when the
  // inserted object is evicted or promoted to the secondary cache. It,
  // therefore, must outlive the cache.
  //
  // If handle is not nullptr, returns a handle that corresponds to the
  // mapping. The caller must call this->Release(handle) when the returned
  // mapping is no longer needed. In case of error caller is responsible to
  // cleanup the value (i.e. calling "deleter").
  //
  // If handle is nullptr, it is as if Release is called immediately after
  // insert. In case of error value will be cleanup.
  //
  // Regardless of whether the item was inserted into the cache,
  // it will attempt to insert it into the secondary cache if one is
  // configured, and the helper supports it.
  // The cache implementation must support a secondary cache, otherwise
  // the item is only inserted into the primary cache. It may
  // defer the insertion to the secondary cache as it sees fit.
  //
  // When the inserted entry is no longer needed, the key and
  // value will be passed to "deleter".
  virtual Status Insert(const Slice& key, void* value,
                        const CacheItemHelper* helper, size_t charge,
                        Handle** handle = nullptr,
                        Priority priority = Priority::LOW) {
    if (!helper) {
      return Status::InvalidArgument();
    }
    return Insert(key, value, charge, helper->del_cb, handle, priority);
  }

  // Lookup the key in the primary and secondary caches (if one is configured).
  // The create_cb callback function object will be used to contruct the
  // cached object.
  // If none of the caches have the mapping for the key, returns nullptr.
  // Else, returns a handle that corresponds to the mapping.
  //
  // This call may promote the object from the secondary cache (if one is
  // configured, and has the given key) to the primary cache.
  //
  // The helper argument should be provided if the caller wants the lookup
  // to include the secondary cache (if one is configured) and the object,
  // if it exists, to be promoted to the primary cache. The helper may be
  // saved and used later when the object is evicted. Therefore, it must
  // outlive the cache.
  //
  // ======================== Async Lookup (wait=false) ======================
  // When wait=false, the handle returned might be in any of three states:
  // * Present - If Value() != nullptr, then the result is present and
  // the handle can be used just as if wait=true.
  // * Pending, not ready (IsReady() == false) - secondary cache is still
  // working to retrieve the value. Might become ready any time.
  // * Pending, ready (IsReady() == true) - secondary cache has the value
  // but it has not been loaded into primary cache. Call to Wait()/WaitAll()
  // will not block.
  //
  // IMPORTANT: Pending handles are not thread-safe, and only these functions
  // are allowed on them: Value(), IsReady(), Wait(), WaitAll(). Even Release()
  // can only come after Wait() or WaitAll() even though a reference is held.
  //
  // Only Wait()/WaitAll() gets a Handle out of a Pending state. (Waiting is
  // safe and has no effect on other handle states.) After waiting on a Handle,
  // it is in one of two states:
  // * Present - if Value() != nullptr
  // * Failed - if Value() == nullptr, such as if the secondary cache
  // initially thought it had the value but actually did not.
  //
  // Note that given an arbitrary Handle, the only way to distinguish the
  // Pending+ready state from the Failed state is to Wait() on it. A cache
  // entry not compatible with secondary cache can also have Value()==nullptr
  // like the Failed state, but this is not generally a concern.
  virtual Handle* Lookup(const Slice& key, const CacheItemHelper* /*helper_cb*/,
                         const CreateCallback& /*create_cb*/,
                         Priority /*priority*/, bool /*wait*/,
                         Statistics* stats = nullptr) {
    return Lookup(key, stats);
  }

  // Release a mapping returned by a previous Lookup(). The "useful"
  // parameter specifies whether the data was actually used or not,
  // which may be used by the cache implementation to decide whether
  // to consider it as a hit for retention purposes. As noted elsewhere,
  // "pending" handles require Wait()/WaitAll() before Release().
  virtual bool Release(Handle* handle, bool /*useful*/,
                       bool erase_if_last_ref) {
    return Release(handle, erase_if_last_ref);
  }

  // Determines if the handle returned by Lookup() can give a value without
  // blocking, though Wait()/WaitAll() might be required to publish it to
  // Value(). See secondary cache compatible Lookup() above for details.
  // This call is not thread safe on "pending" handles.
  virtual bool IsReady(Handle* /*handle*/) { return true; }

  // Convert a "pending" handle into a full thread-shareable handle by
  // * If necessary, wait until secondary cache finishes loading the value.
  // * Construct the value for primary cache and set it in the handle.
  // Even after Wait() on a pending handle, the caller must check for
  // Value() == nullptr in case of failure. This call is not thread-safe
  // on pending handles. This call has no effect on non-pending handles.
  // See secondary cache compatible Lookup() above for details.
  virtual void Wait(Handle* /*handle*/) {}

  // Wait for a vector of handles to become ready. As with Wait(), the user
  // should check the Value() of each handle for nullptr. This call is not
  // thread-safe on pending handles.
  virtual void WaitAll(std::vector<Handle*>& /*handles*/) {}

 private:
  std::shared_ptr<MemoryAllocator> memory_allocator_;
};

using CacheItemHelper = Cache::CacheItemHelper;
using CacheCreateCallback = Cache::CreateCallback;

}  // namespace ROCKSDB_NAMESPACE
