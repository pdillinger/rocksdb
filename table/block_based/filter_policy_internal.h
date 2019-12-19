// Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
// Copyright (c) 2012 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <atomic>

#include "rocksdb/filter_policy.h"
#include "rocksdb/table.h"

namespace rocksdb {

class Slice;

// Exposes any extra information needed for testing built-in
// FilterBitsBuilders
class BuiltinFilterBitsBuilder : public FilterBitsBuilder {
 public:
  // Calculate number of bytes needed for a new filter, including
  // metadata. Passing the result to CalculateNumEntry should
  // return >= the num_entry passed in.
  virtual uint32_t CalculateSpace(const int num_entry) = 0;
};

// EXPERIMENTAL/SUBJECT TO CHANGE: This setting hasn't been integrated into
// the options because we are expecting bigger changes to the options around
// SST filters, and it will be easier to integrate then. For now, it is
// externally controlled by an environment variable,
// ROCKSDB_EXPERIMENTAL_FSP=0, 1, or 2 (recommended), or by calling
// SetFilterSizePolicy on BloomFilterPolicy.
//
// Different ways of tweaking filter sizes for better performance.
// This currently only has an effect with format_version>=5.
enum class FilterSizePolicy : char {
  // Each filter should use the number of bytes that is most consistent
  // with the filter_policy's bits_per_key setting that is supported by
  // the implementation. This is the "old" behavior.
  //
  // This setting also optimizes the size of filters on disk (without
  // regard to filesystem internal fragmentation on the SST file) for a
  // given accuracy (false positive rate).
  kOptimizeRawSize = 0x00,

  // kOptimizeForMemory*: Chooses filter sizes that optimize runtime memory
  // footprint by minimizing internal fragmentation in the memory allocator
  // for the block cache, which is jemalloc by default.
  //
  // For example, if using 8 bits_per_key (for simplicity; also 8 bits per
  // byte), then 9*1024 keys in a filter would use 9KB with
  // kOptimizeRawSize. But the nearest allocation sizes used by Jemalloc
  // are 8KB and 10KB, so this size filter would leave 10% of the memory
  // allocated for it unused.
  //
  // The simplest way to reclaim this unused memory would simply be to
  // "round up" the filter size to just fit in the memory allocation size
  // that would be used for it anyway. This approach is not recommended
  // and not implemented because it is expected to produce too much
  // variance in behavior depending on filter sizes.

  // Optimizes filter sizes for memory footprint (0 - 20% savings) while
  // maintaining the same *average* size in SST files as kOptimizeRawSize,
  // by mixing rounding up and down to the nearest allocation sizes. Because
  // of non-linearity of false positive rates with respect to filter bits
  // per key, this increases FP rate by TODO
  kOptimizeForMemorySameAvgDiskSize = 0x01,

  // RECOMMENDED
  // Optimizes filter sizes for memory footprint (0 - TODO% savings) while
  // maintaining roughly the same *average* false positive rate in filters
  // (weighted by number of keys) as kOptimizeRawSize, by mixing rounding
  // up and down to the nearest allocation sizes. Because of non-linearity
  // of false positive rates with respect to filter bits per key, this
  // increases filter size in SST files by TODO
  kOptimizeForMemorySameAvgFpRate = 0x02,
};

// RocksDB built-in filter policy for Bloom or Bloom-like filters.
// This class is considered internal API and subject to change.
// See NewBloomFilterPolicy.
class BloomFilterPolicy : public FilterPolicy {
 public:
  // An internal marker for operating modes of BloomFilterPolicy, in terms
  // of selecting an implementation. This makes it easier for tests to track
  // or to walk over the built-in set of Bloom filter implementations. The
  // only variance in BloomFilterPolicy by mode/implementation is in
  // GetFilterBitsBuilder(), so an enum is practical here vs. subclasses.
  //
  // This enum is essentially the union of all the different kinds of return
  // value from GetFilterBitsBuilder, or "underlying implementation", and
  // higher-level modes that choose an underlying implementation based on
  // context information.
  enum Mode {
    // Legacy implementation of Bloom filter for full and partitioned filters.
    // Set to 0 in case of value confusion with bool use_block_based_builder
    // NOTE: TESTING ONLY as this mode does not use best compatible
    // implementation
    kLegacyBloom = 0,
    // Deprecated block-based Bloom filter implementation.
    // Set to 1 in case of value confusion with bool use_block_based_builder
    // NOTE: DEPRECATED but user exposed
    kDeprecatedBlock = 1,
    // A fast, cache-local Bloom filter implementation. See description in
    // FastLocalBloomImpl.
    // NOTE: TESTING ONLY as this mode does not check format_version
    kFastLocalBloom = 2,
    // Automatically choose from the above (except kDeprecatedBlock) based on
    // context at build time, including compatibility with format_version.
    // NOTE: This is currently the only recommended mode that is user exposed.
    kAuto = 100,
  };
  // All the different underlying implementations that a BloomFilterPolicy
  // might use, as a mode that says "always use this implementation."
  // Only appropriate for unit tests.
  static const std::vector<Mode> kAllFixedImpls;

  // All the different modes of BloomFilterPolicy that are exposed from
  // user APIs. Only appropriate for higher-level unit tests. Integration
  // tests should prefer using NewBloomFilterPolicy (user-exposed).
  static const std::vector<Mode> kAllUserModes;

  explicit BloomFilterPolicy(double bits_per_key, Mode mode);

  ~BloomFilterPolicy() override;

  const char* Name() const override;

  // Deprecated block-based filter only
  void CreateFilter(const Slice* keys, int n, std::string* dst) const override;

  // Deprecated block-based filter only
  bool KeyMayMatch(const Slice& key, const Slice& bloom_filter) const override;

  FilterBitsBuilder* GetFilterBitsBuilder() const override;

  // To use this function, call GetBuilderFromContext().
  //
  // Neither the context nor any objects therein should be saved beyond
  // the call to this function, unless it's shared_ptr.
  FilterBitsBuilder* GetBuilderWithContext(
      const FilterBuildingContext&) const override;

  // Returns a new FilterBitsBuilder from the filter_policy in
  // table_options of a context, or nullptr if not applicable.
  // (An internal convenience function to save boilerplate.)
  static FilterBitsBuilder* GetBuilderFromContext(const FilterBuildingContext&);

  // Read metadata to determine what kind of FilterBitsReader is needed
  // and return a new one. This must successfully process any filter data
  // generated by a built-in FilterBitsBuilder, regardless of the impl
  // chosen for this BloomFilterPolicy. Not compatible with CreateFilter.
  FilterBitsReader* GetFilterBitsReader(const Slice& contents) const override;

  // Essentially for testing only: configured millibits/key
  int GetMillibitsPerKey() const { return millibits_per_key_; }
  // Essentially for testing only: legacy whole bits/key
  int GetWholeBitsPerKey() const { return whole_bits_per_key_; }

  // EXPERIMENTAL
  void SetFilterSizePolicy(FilterSizePolicy fsp) { filter_size_policy_ = fsp; }

 private:
  // Newer filters support fractional bits per key. For predictable behavior
  // of 0.001-precision values across floating point implementations, we
  // round to thousandths of a bit (on average) per key.
  int millibits_per_key_;

  // Older filters round to whole number bits per key. (There *should* be no
  // compatibility issue with fractional bits per key, but preserving old
  // behavior with format_version < 5 just in case.)
  int whole_bits_per_key_;

  // Selected mode (a specific implementation or way of selecting an
  // implementation) for building new SST filters.
  Mode mode_;

  // The sizing policy in effect (see FilterSizePolicy)
  FilterSizePolicy filter_size_policy_;

  // State for implementing a FilterSizePolicy. Essentially, this tracks a
  // surplus or deficit of filter bytes stored to SST files, due to "rounding"
  // up or down to memory allocation sizes, vs. what would have been used
  // otherwise (kOptimizeRawSize).
  mutable std::atomic<int64_t> rounding_balance_bytes_;

  // For newer Bloom filter implementation(s)
  FilterBitsReader* GetBloomBitsReader(const Slice& contents) const;
};

}  // namespace rocksdb
