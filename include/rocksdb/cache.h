// Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
//
// A Cache is an interface that maps keys to values.  It has internal
// synchronization and may be safely accessed concurrently from
// multiple threads.  It may automatically evict entries to make room
// for new entries.  Values have a specified charge against the cache
// capacity.  For example, a cache where the values are variable
// length strings, may use the length of the string as the charge for
// the string.
//
// A builtin cache implementation with a least-recently-used eviction
// policy is provided.  Clients may use their own implementations if
// they want something more sophisticated (like scan-resistance, a
// custom eviction policy, variable cache sizing, etc.)

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "rocksdb/compression_type.h"
#include "rocksdb/memory_allocator.h"
#include "rocksdb/slice.h"
#include "rocksdb/statistics.h"
#include "rocksdb/status.h"

namespace ROCKSDB_NAMESPACE {

class Cache;
struct ConfigOptions;
class SecondaryCache;

extern const bool kDefaultToAdaptiveMutex;

enum CacheMetadataChargePolicy {
  // Only the `charge` of each entry inserted into a Cache counts against
  // the `capacity`
  kDontChargeCacheMetadata,
  // In addition to the `charge`, the approximate space overheads in the
  // Cache (in bytes) also count against `capacity`. These space overheads
  // are for supporting fast Lookup and managing the lifetime of entries.
  kFullChargeCacheMetadata
};
const CacheMetadataChargePolicy kDefaultCacheMetadataChargePolicy =
    kFullChargeCacheMetadata;

// Options shared betweeen various cache implementations that
// divide the key space into shards using hashing.
struct ShardedCacheOptions {
  // Capacity of the cache, in the same units as the `charge` of each entry.
  // This is typically measured in bytes, but can be a different unit if using
  // kDontChargeCacheMetadata.
  size_t capacity = 0;

  // Cache is sharded into 2^num_shard_bits shards, by hash of key.
  // If < 0, a good default is chosen based on the capacity and the
  // implementation. (Mutex-based implementations are much more reliant
  // on many shards for parallel scalability.)
  int num_shard_bits = -1;

  // If strict_capacity_limit is set, Insert() will fail if there is not
  // enough capacity for the new entry along with all the existing referenced
  // (pinned) cache entries. (Unreferenced cache entries are evicted as
  // needed, sometimes immediately.) If strict_capacity_limit == false
  // (default), Insert() never fails.
  bool strict_capacity_limit = false;

  // If non-nullptr, RocksDB will use this allocator instead of system
  // allocator when allocating memory for cache blocks.
  //
  // Caveat: when the cache is used as block cache, the memory allocator is
  // ignored when dealing with compression libraries that allocate memory
  // internally (currently only XPRESS).
  std::shared_ptr<MemoryAllocator> memory_allocator;

  // See CacheMetadataChargePolicy
  CacheMetadataChargePolicy metadata_charge_policy =
      kDefaultCacheMetadataChargePolicy;

  ShardedCacheOptions() {}
  ShardedCacheOptions(
      size_t _capacity, int _num_shard_bits, bool _strict_capacity_limit,
      std::shared_ptr<MemoryAllocator> _memory_allocator = nullptr,
      CacheMetadataChargePolicy _metadata_charge_policy =
          kDefaultCacheMetadataChargePolicy)
      : capacity(_capacity),
        num_shard_bits(_num_shard_bits),
        strict_capacity_limit(_strict_capacity_limit),
        memory_allocator(std::move(_memory_allocator)),
        metadata_charge_policy(_metadata_charge_policy) {}
};

struct LRUCacheOptions : public ShardedCacheOptions {
  // Ratio of cache reserved for high-priority and low-priority entries,
  // respectively. (See Cache::Priority below more information on the levels.)
  // Valid values are between 0 and 1 (inclusive), and the sum of the two
  // values cannot exceed 1.
  //
  // If high_pri_pool_ratio is greater than zero, a dedicated high-priority LRU
  // list is maintained by the cache. Similarly, if low_pri_pool_ratio is
  // greater than zero, a dedicated low-priority LRU list is maintained.
  // There is also a bottom-priority LRU list, which is always enabled and not
  // explicitly configurable. Entries are spilled over to the next available
  // lower-priority pool if a certain pool's capacity is exceeded.
  //
  // Entries with cache hits are inserted into the highest priority LRU list
  // available regardless of the entry's priority. Entries without hits
  // are inserted into highest priority LRU list available whose priority
  // does not exceed the entry's priority. (For example, high-priority items
  // with no hits are placed in the high-priority pool if available;
  // otherwise, they are placed in the low-priority pool if available;
  // otherwise, they are placed in the bottom-priority pool.) This results
  // in lower-priority entries without hits getting evicted from the cache
  // sooner.
  //
  // Default values: high_pri_pool_ratio = 0.5 (which is referred to as
  // "midpoint insertion"), low_pri_pool_ratio = 0
  double high_pri_pool_ratio = 0.5;
  double low_pri_pool_ratio = 0.0;

  // Whether to use adaptive mutexes for cache shards. Note that adaptive
  // mutexes need to be supported by the platform in order for this to have any
  // effect. The default value is true if RocksDB is compiled with
  // -DROCKSDB_DEFAULT_TO_ADAPTIVE_MUTEX, false otherwise.
  bool use_adaptive_mutex = kDefaultToAdaptiveMutex;

  // A SecondaryCache instance to use a the non-volatile tier.
  std::shared_ptr<SecondaryCache> secondary_cache;

  LRUCacheOptions() {}
  LRUCacheOptions(size_t _capacity, int _num_shard_bits,
                  bool _strict_capacity_limit, double _high_pri_pool_ratio,
                  std::shared_ptr<MemoryAllocator> _memory_allocator = nullptr,
                  bool _use_adaptive_mutex = kDefaultToAdaptiveMutex,
                  CacheMetadataChargePolicy _metadata_charge_policy =
                      kDefaultCacheMetadataChargePolicy,
                  double _low_pri_pool_ratio = 0.0)
      : ShardedCacheOptions(_capacity, _num_shard_bits, _strict_capacity_limit,
                            std::move(_memory_allocator),
                            _metadata_charge_policy),
        high_pri_pool_ratio(_high_pri_pool_ratio),
        low_pri_pool_ratio(_low_pri_pool_ratio),
        use_adaptive_mutex(_use_adaptive_mutex) {}
};

// Create a new cache with a fixed size capacity. The cache is sharded
// to 2^num_shard_bits shards, by hash of the key. The total capacity
// is divided and evenly assigned to each shard. If strict_capacity_limit
// is set, insert to the cache will fail when cache is full. User can also
// set percentage of the cache reserves for high priority entries via
// high_pri_pool_pct.
// num_shard_bits = -1 means it is automatically determined: every shard
// will be at least 512KB and number of shard bits will not exceed 6.
extern std::shared_ptr<Cache> NewLRUCache(
    size_t capacity, int num_shard_bits = -1,
    bool strict_capacity_limit = false, double high_pri_pool_ratio = 0.5,
    std::shared_ptr<MemoryAllocator> memory_allocator = nullptr,
    bool use_adaptive_mutex = kDefaultToAdaptiveMutex,
    CacheMetadataChargePolicy metadata_charge_policy =
        kDefaultCacheMetadataChargePolicy,
    double low_pri_pool_ratio = 0.0);

extern std::shared_ptr<Cache> NewLRUCache(const LRUCacheOptions& cache_opts);

// EXPERIMENTAL
// Options structure for configuring a SecondaryCache instance based on
// LRUCache. The LRUCacheOptions.secondary_cache is not used and
// should not be set.
struct CompressedSecondaryCacheOptions : LRUCacheOptions {
  // The compression method (if any) that is used to compress data.
  CompressionType compression_type = CompressionType::kLZ4Compression;

  // compress_format_version can have two values:
  // compress_format_version == 1 -- decompressed size is not included in the
  // block header.
  // compress_format_version == 2 -- decompressed size is included in the block
  // header in varint32 format.
  uint32_t compress_format_version = 2;

  // Enable the custom split and merge feature, which split the compressed value
  // into chunks so that they may better fit jemalloc bins.
  bool enable_custom_split_merge = false;

  CompressedSecondaryCacheOptions() {}
  CompressedSecondaryCacheOptions(
      size_t _capacity, int _num_shard_bits, bool _strict_capacity_limit,
      double _high_pri_pool_ratio, double _low_pri_pool_ratio = 0.0,
      std::shared_ptr<MemoryAllocator> _memory_allocator = nullptr,
      bool _use_adaptive_mutex = kDefaultToAdaptiveMutex,
      CacheMetadataChargePolicy _metadata_charge_policy =
          kDefaultCacheMetadataChargePolicy,
      CompressionType _compression_type = CompressionType::kLZ4Compression,
      uint32_t _compress_format_version = 2,
      bool _enable_custom_split_merge = false)
      : LRUCacheOptions(_capacity, _num_shard_bits, _strict_capacity_limit,
                        _high_pri_pool_ratio, std::move(_memory_allocator),
                        _use_adaptive_mutex, _metadata_charge_policy,
                        _low_pri_pool_ratio),
        compression_type(_compression_type),
        compress_format_version(_compress_format_version),
        enable_custom_split_merge(_enable_custom_split_merge) {}
};

// EXPERIMENTAL
// Create a new Secondary Cache that is implemented on top of LRUCache.
extern std::shared_ptr<SecondaryCache> NewCompressedSecondaryCache(
    size_t capacity, int num_shard_bits = -1,
    bool strict_capacity_limit = false, double high_pri_pool_ratio = 0.5,
    double low_pri_pool_ratio = 0.0,
    std::shared_ptr<MemoryAllocator> memory_allocator = nullptr,
    bool use_adaptive_mutex = kDefaultToAdaptiveMutex,
    CacheMetadataChargePolicy metadata_charge_policy =
        kDefaultCacheMetadataChargePolicy,
    CompressionType compression_type = CompressionType::kLZ4Compression,
    uint32_t compress_format_version = 2,
    bool enable_custom_split_merge = false);

extern std::shared_ptr<SecondaryCache> NewCompressedSecondaryCache(
    const CompressedSecondaryCacheOptions& opts);

// HyperClockCache - EXPERIMENTAL
//
// A lock-free Cache alternative for RocksDB block cache that offers much
// improved CPU efficiency under high parallel load or high contention, with
// some caveats.
//
// See internal cache/clock_cache.h for full description.
struct HyperClockCacheOptions : public ShardedCacheOptions {
  // The estimated average `charge` associated with cache entries. This is a
  // critical configuration parameter for good performance from the hyper
  // cache, because having a table size that is fixed at creation time greatly
  // reduces the required synchronization between threads.
  // * If the estimate is substantially too low (e.g. less than half the true
  // average) then metadata space overhead with be substantially higher (e.g.
  // 200 bytes per entry rather than 100). With kFullChargeCacheMetadata, this
  // can slightly reduce cache hit rates, and slightly reduce access times due
  // to the larger working memory size.
  // * If the estimate is substantially too high (e.g. 25% higher than the true
  // average) then there might not be sufficient slots in the hash table for
  // both efficient operation and capacity utilization (hit rate). The hyper
  // cache will evict entries to prevent load factors that could dramatically
  // affect lookup times, instead letting the hit rate suffer by not utilizing
  // the full capacity.
  //
  // A reasonable choice is the larger of block_size and metadata_block_size.
  // When WriteBufferManager (and similar) charge memory usage to the block
  // cache, this can lead to the same effect as estimate being too low, which
  // is better than the opposite. Therefore, the general recommendation is to
  // assume that other memory charged to block cache could be negligible, and
  // ignore it in making the estimate.
  //
  // The best parameter choice based on a cache in use is given by
  // GetUsage() / GetOccupancyCount(), ignoring metadata overheads such as
  // with kDontChargeCacheMetadata. More precisely with
  // kFullChargeCacheMetadata is (GetUsage() - 64 * GetTableAddressCount()) /
  // GetOccupancyCount(). However, when the average value size might vary
  // (e.g. balance between metadata and data blocks in cache), it is better
  // to estimate toward the lower side than the higher side.
  size_t estimated_entry_charge;

  HyperClockCacheOptions(
      size_t _capacity, size_t _estimated_entry_charge,
      int _num_shard_bits = -1, bool _strict_capacity_limit = false,
      std::shared_ptr<MemoryAllocator> _memory_allocator = nullptr,
      CacheMetadataChargePolicy _metadata_charge_policy =
          kDefaultCacheMetadataChargePolicy)
      : ShardedCacheOptions(_capacity, _num_shard_bits, _strict_capacity_limit,
                            std::move(_memory_allocator),
                            _metadata_charge_policy),
        estimated_entry_charge(_estimated_entry_charge) {}

  // Construct an instance of HyperClockCache using these options
  std::shared_ptr<Cache> MakeSharedCache() const;
};

// DEPRECATED - The old Clock Cache implementation had an unresolved bug and
// has been removed. The new HyperClockCache requires an additional
// configuration parameter that is not provided by this API. This function
// simply returns a new LRUCache for functional compatibility.
extern std::shared_ptr<Cache> NewClockCache(
    size_t capacity, int num_shard_bits = -1,
    bool strict_capacity_limit = false,
    CacheMetadataChargePolicy metadata_charge_policy =
        kDefaultCacheMetadataChargePolicy);

// Classifications of block cache entries.
//
// Developer notes: Adding a new enum to this class requires corresponding
// updates to `kCacheEntryRoleToCamelString` and
// `kCacheEntryRoleToHyphenString`. Do not add to this enum after `kMisc` since
// `kNumCacheEntryRoles` assumes `kMisc` comes last.
enum class CacheEntryRole {
  // Block-based table data block
  kDataBlock,
  // Block-based table filter block (full or partitioned)
  kFilterBlock,
  // Block-based table metadata block for partitioned filter
  kFilterMetaBlock,
  // OBSOLETE / DEPRECATED: old/removed block-based filter
  kDeprecatedFilterBlock,
  // Block-based table index block
  kIndexBlock,
  // Other kinds of block-based table block
  kOtherBlock,
  // WriteBufferManager's charge to account for its memtable usage
  kWriteBuffer,
  // Compression dictionary building buffer's charge to account for
  // its memory usage
  kCompressionDictionaryBuildingBuffer,
  // Filter's charge to account for
  // (new) bloom and ribbon filter construction's memory usage
  kFilterConstruction,
  // BlockBasedTableReader's charge to account for its memory usage
  kBlockBasedTableReader,
  // FileMetadata's charge to account for its memory usage
  kFileMetadata,
  // Blob value (when using the same cache as block cache and blob cache)
  kBlobValue,
  // Blob cache's charge to account for its memory usage (when using a
  // separate block cache and blob cache)
  kBlobCache,
  // Default bucket, for miscellaneous cache entries. Do not use for
  // entries that could potentially add up to large usage.
  kMisc,
};
constexpr uint32_t kNumCacheEntryRoles =
    static_cast<uint32_t>(CacheEntryRole::kMisc) + 1;

// Obtain a hyphen-separated, lowercase name of a `CacheEntryRole`.
const std::string& GetCacheEntryRoleName(CacheEntryRole);

// For use with `GetMapProperty()` for property
// `DB::Properties::kBlockCacheEntryStats`. On success, the map will
// be populated with all keys that can be obtained from these functions.
struct BlockCacheEntryStatsMapKeys {
  static const std::string& CacheId();
  static const std::string& CacheCapacityBytes();
  static const std::string& LastCollectionDurationSeconds();
  static const std::string& LastCollectionAgeSeconds();

  static std::string EntryCount(CacheEntryRole);
  static std::string UsedBytes(CacheEntryRole);
  static std::string UsedPercent(CacheEntryRole);
};

namespace cache {

// Depending on implementation, cache entries with higher priority levels
// could be less likely to get evicted than entries with lower priority
// levels. The "high" priority level applies to certain SST metablocks (e.g.
// index and filter blocks) if the option
// cache_index_and_filter_blocks_with_high_priority is set. The "low" priority
// level is used for other kinds of SST blocks (most importantly, data
// blocks), as well as the above metablocks in case
// cache_index_and_filter_blocks_with_high_priority is
// not set. The "bottom" priority level is for BlobDB's blob values.
enum class Priority { HIGH, LOW, BOTTOM };

// A set of callbacks to allow objects in the primary block cache to be
// be persisted in a secondary cache. The purpose of the secondary cache
// is to support other ways of caching the object, such as persistent or
// compressed data, that may require the object to be parsed and transformed
// in some way. Since the primary cache holds C++ objects and the secondary
// cache may only hold flat data that doesn't need relocation, these
// callbacks need to be provided by the user of the block
// cache to do the conversion.
// The CacheItemHelper is passed to Insert() and Lookup(). It has pointers
// to callback functions for size, saving and deletion of the
// object. The callbacks are defined in C-style in order to make them
// stateless and not add to the cache metadata size.
// Saving multiple std::function objects will take up 32 bytes per
// function, even if its not bound to an object and does no capture.
//
// All the callbacks are C-style function pointers in order to simplify
// lifecycle management. Objects in the cache can outlive the parent DB,
// so anything required for these operations should be contained in the
// object itself.
//
// The SizeCallback takes a void* pointer to the object and returns the size
// of the persistable data. It can be used by the secondary cache to allocate
// memory if needed.
//
// RocksDB callbacks are NOT exception-safe. A callback completing with an
// exception can lead to undefined behavior in RocksDB, including data loss,
// unreported corruption, deadlocks, and more.
using SizeCallback = size_t (*)(void* obj);

// The SaveToCallback takes a void* object pointer and saves the persistable
// data into a buffer. The secondary cache may decide to not store it in a
// contiguous buffer, in which case this callback will be called multiple
// times with increasing offset
using SaveToCallback = Status (*)(void* from_obj, size_t from_offset,
                                  size_t length, void* out);

// A function pointer type for custom destruction of an entry's
// value. The Cache is responsible for copying and reclaiming space
// for the key, but values are managed by the caller.
using DeleterFn = void (*)(const Slice& key, void* value);

// A struct with pointers to helper functions for spilling items from the
// cache into the secondary cache. May be extended in the future. An
// instance of this struct is expected to outlive the cache.
struct CacheItemHelper {
  SizeCallback size_cb;
  SaveToCallback saveto_cb;
  DeleterFn del_cb;

  CacheItemHelper() : size_cb(nullptr), saveto_cb(nullptr), del_cb(nullptr) {}
  CacheItemHelper(SizeCallback _size_cb, SaveToCallback _saveto_cb,
                  DeleterFn _del_cb)
      : size_cb(_size_cb), saveto_cb(_saveto_cb), del_cb(_del_cb) {}
  };

  // The CreateCallback is passed by the block cache user to Lookup(). It
  // takes in a buffer from the NVM cache and constructs an object using
  // it. The callback doesn't have ownership of the buffer and should
  // copy the contents into its own buffer.
  using CreateCallback = std::function<Status(const void* buf, size_t size,
                                              void** out_obj, size_t* charge)>;

  }  // namespace cache

}  // namespace ROCKSDB_NAMESPACE
