// Copyright (c) 2019-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include <rocksdb/slice.h>
#include <memory>

namespace rocksdb {

// Encapsulates a way of interpreting Bloom(-like) filter bits, assuming that
// each query provides a base pointer for the filter and its length, in
// addition to the query key. (In contrast to FilterBitsReader, this enables
// sharing a configuration among filter objects in the same SST file and while
// they might be swapped in and out of cache.)
class FilterBitsConfig {
 public:
  virtual ~FilterBitsConfig() {}

  virtual uint64_t Hash(const Slice& key) const = 0;

  virtual bool HashMayMatch(const Slice& filter, uint64_t hash) const = 0;

  virtual void PrepareHashMayMatch(const Slice& filter, uint64_t hash) const {
    (void)filter;
    (void)hash;
  }

  inline bool MayMatch(const Slice& filter, const Slice& key) const {
    uint64_t h = Hash(key);
    return HashMayMatch(filter, h);
  }

  virtual void MayMatch(const Slice& filter, int num_keys, Slice** keys,
                        bool* may_match) const {
    for (int i = 0; i < num_keys; ++i) {
      uint64_t h = Hash(*(keys[i]));
      may_match[i] = HashMayMatch(filter, h);
    }
  }

  virtual std::string ToConfigString() const = 0;

  static std::shared_ptr<const FilterBitsConfig> FromConfigString(
      const std::string& str);
};

}  // namespace rocksdb
