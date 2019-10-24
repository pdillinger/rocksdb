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

#include "rocksdb/filter_policy.h"

namespace rocksdb {

class Slice;

class FullFilterBitsBuilder : public FilterBitsBuilder {
 public:
  explicit FullFilterBitsBuilder(const int bits_per_key, const int num_probes);

  // No Copy allowed
  FullFilterBitsBuilder(const FullFilterBitsBuilder&) = delete;
  void operator=(const FullFilterBitsBuilder&) = delete;

  ~FullFilterBitsBuilder();

  virtual void AddKey(const Slice& key) override;

  // Create a filter that for hashes [0, n-1], the filter is allocated here
  // When creating filter, it is ensured that
  // total_bits = num_lines * CACHE_LINE_SIZE * 8
  // dst len is >= 5, 1 for num_probes, 4 for num_lines
  // Then total_bits = (len - 5) * 8, and cache_line_size could be calculated
  // +----------------------------------------------------------------+
  // |              filter data with length total_bits/8              |
  // +----------------------------------------------------------------+
  // |                                                                |
  // | ...                                                            |
  // |                                                                |
  // +----------------------------------------------------------------+
  // | ...                | num_probes : 1 byte | num_lines : 4 bytes |
  // +----------------------------------------------------------------+
  virtual Slice Finish(std::unique_ptr<const char[]>* buf) override;

  // Calculate num of entries fit into a space.
  virtual int CalculateNumEntry(const uint32_t space) override;

  // Calculate space for new filter. This is reverse of CalculateNumEntry.
  uint32_t CalculateSpace(const int num_entry, uint32_t* total_bits,
                          uint32_t* num_lines);

 private:
  friend class FullFilterBlockTest_DuplicateEntries_Test;
  int bits_per_key_;
  int num_probes_;
  std::vector<uint32_t> hash_entries_;

  // Get totalbits that optimized for cpu cache line
  uint32_t GetTotalBitsForLocality(uint32_t total_bits);

  // Reserve space for new filter
  char* ReserveSpace(const int num_entry, uint32_t* total_bits,
                     uint32_t* num_lines);

  // Assuming single threaded access to this function.
  void AddHash(uint32_t h, char* data, uint32_t num_lines, uint32_t total_bits);
};

// RocksDB built-in filter policy for Bloom or Bloom-like filters.
// This class is considered internal API and subject to change.
// See NewBloomFilterPolicy.
class BloomFilterPolicy : public FilterPolicy {
 public:
  // An internal marker for which Bloom filter implementation to use.
  // This makes it easier for tests to track or to walk over the built-in
  // set of Bloom filter policies. The only variance in BloomFilterPolicy
  // by implementation is in GetFilterBitsBuilder(), so an enum is practical
  // here vs. subclasses.
  enum Impl {
    // Implementation of Bloom filter for full and partitioned filters.
    // Set to 0 in case of value confusion with bool use_block_based_builder
    kFull = 0,
    // Deprecated block-based Bloom filter implementation.
    // Set to 1 in case of value confusion with bool use_block_based_builder
    kBlock = 1,
  };
  static const std::vector<Impl> kAllImpls;

  explicit BloomFilterPolicy(int bits_per_key, Impl impl);

  ~BloomFilterPolicy() override;

  const char* Name() const override;

  // Deprecated block-based filter only
  void CreateFilter(const Slice* keys, int n, std::string* dst) const override;

  // Deprecated block-based filter only
  bool KeyMayMatch(const Slice& key, const Slice& bloom_filter) const override;

  FilterBitsBuilder* GetFilterBitsBuilder() const override;

  // Read metadata to determine what kind of FilterBitsReader is needed
  // and return a new one. This must successfully process any filter data
  // generated by a built-in FilterBitsBuilder, regardless of the impl
  // chosen for this BloomFilterPolicy. Not compatible with CreateFilter.
  FilterBitsReader* GetFilterBitsReader(const Slice& contents) const override;

 private:
  int bits_per_key_;
  int num_probes_;
  // Selected implementation for building new SST filters
  Impl impl_;
};

}  // namespace rocksdb
