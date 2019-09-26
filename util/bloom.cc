//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2012 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "rocksdb/filter_policy.h"

#include "rocksdb/slice.h"
#include "table/block_based/block_based_filter_block.h"
#include "table/block_based/full_filter_block.h"
#include "table/full_filter_bits_builder.h"
#include "third-party/folly/folly/ConstexprMath.h"
#include "util/bloom_impl.h"
#include "util/coding.h"
#include "util/hash.h"

namespace rocksdb {

typedef LegacyLocalityBloomImpl</*ExtraRotates*/ false> LegacyFullFilterImpl;
class BlockBasedFilterBlockBuilder;
class FullFilterBlockBuilder;

FullFilterBitsBuilder::FullFilterBitsBuilder(const int bits_per_key,
                                             const int num_probes)
    : bits_per_key_(bits_per_key), num_probes_(num_probes) {
  assert(bits_per_key_);
}

  FullFilterBitsBuilder::~FullFilterBitsBuilder() {}

  void FullFilterBitsBuilder::AddKey(const Slice& key) {
    uint32_t hash = BloomHash(key);
    if (hash_entries_.size() == 0 || hash != hash_entries_.back()) {
      hash_entries_.push_back(hash);
    }
  }

  Slice FullFilterBitsBuilder::Finish(std::unique_ptr<const char[]>* buf) {
    uint32_t total_bits, num_lines;
    char* data = ReserveSpace(static_cast<int>(hash_entries_.size()),
                              &total_bits, &num_lines);
    assert(data);

    if (total_bits != 0 && num_lines != 0) {
      for (auto h : hash_entries_) {
        AddHash(h, data, num_lines, total_bits);
      }
    }
    data[total_bits/8] = static_cast<char>(num_probes_);
    EncodeFixed32(data + total_bits/8 + 1, static_cast<uint32_t>(num_lines));

    const char* const_data = data;
    buf->reset(const_data);
    hash_entries_.clear();

    return Slice(data, total_bits / 8 + 5);
  }

uint32_t FullFilterBitsBuilder::GetTotalBitsForLocality(uint32_t total_bits) {
  uint32_t num_lines =
      (total_bits + CACHE_LINE_SIZE * 8 - 1) / (CACHE_LINE_SIZE * 8);

  // Make num_lines an odd number to make sure more bits are involved
  // when determining which block.
  if (num_lines % 2 == 0) {
    num_lines++;
  }
  return num_lines * (CACHE_LINE_SIZE * 8);
}

uint32_t FullFilterBitsBuilder::CalculateSpace(const int num_entry,
                                               uint32_t* total_bits,
                                               uint32_t* num_lines) {
  assert(bits_per_key_);
  if (num_entry != 0) {
    uint32_t total_bits_tmp = static_cast<uint32_t>(num_entry * bits_per_key_);

    *total_bits = GetTotalBitsForLocality(total_bits_tmp);
    *num_lines = *total_bits / (CACHE_LINE_SIZE * 8);
    assert(*total_bits > 0 && *total_bits % 8 == 0);
  } else {
    // filter is empty, just leave space for metadata
    *total_bits = 0;
    *num_lines = 0;
  }

  // Reserve space for Filter
  uint32_t sz = *total_bits / 8;
  sz += 5;  // 4 bytes for num_lines, 1 byte for num_probes
  return sz;
}

char* FullFilterBitsBuilder::ReserveSpace(const int num_entry,
                                          uint32_t* total_bits,
                                          uint32_t* num_lines) {
  uint32_t sz = CalculateSpace(num_entry, total_bits, num_lines);
  char* data = new char[sz];
  memset(data, 0, sz);
  return data;
}

int FullFilterBitsBuilder::CalculateNumEntry(const uint32_t space) {
  assert(bits_per_key_);
  assert(space > 0);
  uint32_t dont_care1, dont_care2;
  int high = static_cast<int>(space * 8 / bits_per_key_ + 1);
  int low = 1;
  int n = high;
  for (; n >= low; n--) {
    uint32_t sz = CalculateSpace(n, &dont_care1, &dont_care2);
    if (sz <= space) {
      break;
    }
  }
  assert(n < high);  // High should be an overestimation
  return n;
}

inline void FullFilterBitsBuilder::AddHash(uint32_t h, char* data,
    uint32_t num_lines, uint32_t total_bits) {
#ifdef NDEBUG
  static_cast<void>(total_bits);
#endif
  assert(num_lines > 0 && total_bits > 0);

  LegacyFullFilterImpl::AddHash(h, num_lines, num_probes_, data,
                                folly::constexpr_log2(CACHE_LINE_SIZE));
}

namespace {

class MaxCacheFilterBitsBuilder : public FilterBitsBuilder {
 public:
  MaxCacheFilterBitsBuilder(const int bits_per_key)
      : bits_per_key_(bits_per_key) {
    assert(bits_per_key_);
  }

  // No Copy allowed
  MaxCacheFilterBitsBuilder(const MaxCacheFilterBitsBuilder&) = delete;
  void operator=(const MaxCacheFilterBitsBuilder&) = delete;

  ~MaxCacheFilterBitsBuilder() {}

  virtual void AddKey(const Slice& key) override {
    uint32_t hash = BloomHash(key);
    if (hash_entries_.size() == 0 || hash != hash_entries_.back()) {
      hash_entries_.push_back(hash);
    }
  }

  virtual Slice Finish(std::unique_ptr<const char[]>* buf) override {
    uint32_t num_cache_lines = 0;
    if (bits_per_key_ > 0 && hash_entries_.size() > 0) {
      num_cache_lines = static_cast<uint32_t>((uint64_t(1) * hash_entries_.size() * bits_per_key_ + 511 /*XXX + hash_entries_.size() / 2*/) / 512);
    }
    uint32_t len = num_cache_lines * 64;
    uint32_t len_with_metadata = len + 5;
    char* data = new char[len_with_metadata];
    memset(data, 0, len_with_metadata);
    assert(data);

    //fprintf(stderr, "-> Building with %d cache lines\n", num_cache_lines);
    if (len > 0) {
      // Too slow:
      //std::sort(hash_entries_.begin(), hash_entries_.end());

      // Partition into buckets with temporary allocation
      // (Fast enough for now; could save temp space with counting sort)
      std::unique_ptr<std::vector<uint32_t>[]> buckets(new std::vector<uint32_t>[num_cache_lines]);
      for (size_t i = 0; i < hash_entries_.size(); ++i) {
        uint32_t cache_line = fastrange32(num_cache_lines, hash_entries_[i]);
        //fprintf(stderr, "Preparing hash %x on cache line %d\n", hash_entries_[i], cache_line);
        buckets[cache_line].push_back(hash_entries_[i] * 0x9e3779b9);
      }
      for (uint32_t i = 0; i < num_cache_lines; ++i) {
        BuildCacheLine(data + (i << 6), buckets[i].begin(), buckets[i].end());
      }
      // Alternative
      // Partition into buckets with counting
      /*
      std::unique_ptr<uint32_t[]> index_by_bucket(new uint32[num_cache_lines+1]);
      for (uint32_t i = 0; i < num_cache_lines + 1; ++i) {
        index_by_bucket[i] = 0;
      }
      // First store counts in index_by_bucket
      for (size_t i = 0; i < hash_entries_.size(); ++i) {
        uint32_t cache_line = fastrange32(num_cache_lines, hash_entries_[i]);
        //fprintf(stderr, "Preparing hash %x on cache line %d\n", hash_entries_[i], cache_line);
        index_by_bucket[cache_line + 1]++;
      }
      // Change to indexes with summation
      for (uint32_t i = 2; i < num_cache_lines + 1; ++i) {
        index_by_bucket[i] += index_by_bucket[i - 1];
      }
      ...
      */
      // END Alternative
    }

    data[len] = static_cast<char>(-2);

    const char* const_data = data;
    buf->reset(const_data);
    hash_entries_.clear();

    return Slice(data, len_with_metadata);
  }

  int CalculateNumEntry(const uint32_t bytes) {
    return static_cast<int>(uint64_t(8) * bytes / bits_per_key_);
  }

 private:
  int bits_per_key_;
  int num_probes_;
  std::vector<uint32_t> hash_entries_;

  void BuildCacheLine(char *data_at_cache_line, std::vector<uint32_t>::iterator begin, std::vector<uint32_t>::iterator end) {
    if (end - begin >= 64) {
      BuildBloomCacheLine(data_at_cache_line, begin, end);
      return;
    }
    // Option 1 (too slow - modified for verification)
    //std::vector<uint32_t> copy(begin, end);
    /*
    std::sort(begin, end);
    int full_slots = 0;
    for (auto it = end - 1; it >= begin; --it) {
      uint32_t hash = *it;
      int cur = hash & 63;
      int delta = ((hash >> 5) | 1) & 63;
      int val = 1 + fastrange32(255, hash);
      int probes_rem = 64;
      //fprintf(stderr, "Adding(1) sct for %x at %d delta %d\n", val, cur, delta);
      do {
        if (data_at_cache_line[cur] == 0) {
          ++full_slots;
          if (full_slots >= 64) {
            memset(data_at_cache_line, 0, 64);
            BuildBloomCacheLine(data_at_cache_line, begin, end);
            return;
          }
          data_at_cache_line[cur] = val;
          //fprintf(stderr, "Stored(1) at %d\n", cur);
          break;
        } else if (data_at_cache_line[cur] == val) {
          //fprintf(stderr, "Collapsed(1) at %d\n", cur);
          break;
        }
        cur = (cur + delta) & 63;
        --probes_rem;
      } while (probes_rem > 0);
      if (probes_rem == 0) {
        assert(false);
        memset(data_at_cache_line, 0, 64);
        BuildBloomCacheLine(data_at_cache_line, begin, end);
        return;
      }
    }
    */

    // Option 2
    // Instead build full data in temporary table and then strip down.
    // Zero is a valid hash, so need external occupancy marker.
    std::array<uint32_t, 64> entries;
    uint64_t packed_full_slots = 0;

    for (auto it = begin; it != end; ++it) {
      uint32_t hash = *it;
      int cur = hash & 63;
      int delta = ((hash >> 5) | 1) & 63;
      int val = 1 + fastrange32(255, hash);
      int probes_rem = 64;
      //fprintf(stderr, "Adding(2) sct for %x at %d delta %d\n", val, cur, delta);
      do {
        if ((packed_full_slots & (uint64_t(1) << cur)) == 0) {
          packed_full_slots |= (uint64_t(1) << cur);
          if (packed_full_slots == 0xffffffffffffffffULL) {
            // Above should have excluded case of 64 or more values
            assert(false);
            BuildBloomCacheLine(data_at_cache_line, begin, end);
            return;
          }
          entries[cur] = hash;
          //fprintf(stderr, "Stored(2) at %d\n", cur);
          break;
        } else {
          int other_val = 1 + fastrange32(255, entries[cur]);
          if (other_val == val) {
            // NB: can't safely collapse here because both might need to be
            // bumped again. For now (simpler code), keep potentially
            // redundant entry. (Keep probing.) Seems to inflate FP rate by
            // factor of ~1.007.
          } else if (val > other_val) {
            std::swap(hash, entries[cur]);
            delta = ((hash >> 5) | 1) & 63;
            val = other_val;
            //fprintf(stderr, "Changed(2) to %x at %d delta %d\n", val, cur, delta);
          }
        }
        cur = (cur + delta) & 63;
        --probes_rem;
      } while (probes_rem > 0);
      if (probes_rem == 0) {
        assert(false);
        BuildBloomCacheLine(data_at_cache_line, begin, end);
        return;
      }
    }
    for (int i = 0; i < 64; ++i) {
      //fprintf(stderr, "Verif: val %x at %d\n", static_cast<uint8_t>(data_at_cache_line[i]), i);
      if ((packed_full_slots & (uint64_t(1) << i)) != 0) {
        data_at_cache_line[i] = static_cast<char>(1 + fastrange32(255, entries[i]));
      } else {
        assert(data_at_cache_line[i] == 0);
      }
    }
    //fprintf(stderr, "Built sct\n");
  }

  void BuildBloomCacheLine(char *data_at_cache_line, std::vector<uint32_t>::iterator begin, std::vector<uint32_t>::iterator end) {
    for (auto it = begin; it != end; ++it) {
      uint32_t hash = *it;
      int bit = hash & 255;
      data_at_cache_line[bit >> 3] |= (1 << (bit & 7));
      hash >>= 8;
      bit = hash & 255;
      data_at_cache_line[bit >> 3] |= (1 << (bit & 7));
      hash >>= 8;
      bit = hash & 255;
      data_at_cache_line[32 + (bit >> 3)] |= (1 << (bit & 7));
      hash >>= 8;
      bit = hash & 255;
      data_at_cache_line[32 + (bit >> 3)] |= (1 << (bit & 7));
    }
    for (int i = 0; i < 64; ++i) {
      if (data_at_cache_line[i] == 0) {
        // set one bit to make it non-zero
        data_at_cache_line[i] = 16;
      }
    }
    //fprintf(stderr, "Built bloom\n");
  }
};

class MaxCacheFilterBitsReader : public FilterBitsReader {
 public:
  MaxCacheFilterBitsReader(const char* data,
    uint32_t num_cache_lines)
    : data_(data), num_cache_lines_(num_cache_lines) {
    }

  // No Copy allowed
  MaxCacheFilterBitsReader(const MaxCacheFilterBitsReader&) = delete;
  void operator=(const MaxCacheFilterBitsReader&) = delete;

  ~MaxCacheFilterBitsReader() override {}

  bool MayMatch(const Slice& key) override {
    uint32_t hash = BloomHash(key);
    uint32_t line_offset = fastrange32(num_cache_lines_, hash) << 6;
    //fprintf(stderr, "-> Checking for %x on cache line %d\n", hash, line_offset >> 6);
    hash *= 0x9e3779b9;
    const uint8_t *data_at_cache_line = reinterpret_cast<const uint8_t *>(data_ + line_offset);
    // Any whole byte is zero -> marker for sct8 vs. bloom
    if (memchr(data_at_cache_line, 0, 64)) {
      // Use sct8
      int cur = hash & 63;
      int delta = ((hash >> 5) | 1) & 63;
      int val = 1 + fastrange32(255, hash);
      //fprintf(stderr, "Checking sct for %x at %d delta %d\n", val, cur, delta);
      int probes_rem = 63;
      do {
        if (data_at_cache_line[cur] < val) {
          return false;
        } else if (data_at_cache_line[cur] == val) {
          return true;
        }
        cur = (cur + delta) & 63;
        --probes_rem;
      } while (probes_rem > 0);
      return false;
    } else {
      //fprintf(stderr, "Checking bloom for %x\n", hash);
      // Use k=4 Bloom, two probes in lower 256 bits, two in upper
      int bit = hash & 255;
      if ((data_at_cache_line[bit >> 3] & (1 << (bit & 7))) == 0) {
        return false;
      }
      hash >>= 8;
      bit = hash & 255;
      if ((data_at_cache_line[bit >> 3] & (1 << (bit & 7))) == 0) {
        return false;
      }
      hash >>= 8;
      bit = hash & 255;
      if ((data_at_cache_line[32 + (bit >> 3)] & (1 << (bit & 7))) == 0) {
        return false;
      }
      hash >>= 8;
      bit = hash & 255;
      return (data_at_cache_line[32 + (bit >> 3)] & (1 << (bit & 7))) != 0;
    }
  }

  using FilterBitsReader::MayMatch; // inherit overload

/*
  virtual void MayMatch(int num_keys, Slice** keys, bool* may_match) override {
    uint32_t hashes[MultiGetContext::MAX_BATCH_SIZE];
    uint32_t byte_offsets[MultiGetContext::MAX_BATCH_SIZE];
    for (int i = 0; i < num_keys; ++i) {
      hashes[i] = BloomHash(*keys[i]);
      SemiLocalBloomImpl::PrepareHashMayMatch(hashes[i], len_bytes_, data_,
                                              &byte_offsets[i]);
    }
    for (int i = 0; i < num_keys; ++i) {
      may_match[i] = SemiLocalBloomImpl::HashMayMatchPrepared(hashes[i], len_bytes_, num_probes_,
                                                      data_, byte_offsets[i]);
    }
  }
*/
 private:
  const char* data_;
  const uint32_t num_cache_lines_;
};

class SemiLocalBloomBitsBuilder : public FilterBitsBuilder {
 public:
  SemiLocalBloomBitsBuilder(const int bits_per_key, const int num_probes)
      : bits_per_key_(bits_per_key), num_probes_(num_probes) {
    assert(bits_per_key_);
  }

  // No Copy allowed
  SemiLocalBloomBitsBuilder(const SemiLocalBloomBitsBuilder&) = delete;
  void operator=(const SemiLocalBloomBitsBuilder&) = delete;

  ~SemiLocalBloomBitsBuilder() {}

  virtual void AddKey(const Slice& key) override {
    uint32_t hash = BloomHash(key);
    if (hash_entries_.size() == 0 || hash != hash_entries_.back()) {
      hash_entries_.push_back(hash);
    }
  }

  virtual Slice Finish(std::unique_ptr<const char[]>* buf) override {
    uint32_t len = 0;
    if (bits_per_key_ > 0 && hash_entries_.size() > 0) {
      len = static_cast<uint32_t>((uint64_t(1) * hash_entries_.size() * bits_per_key_ + 7) / 8);
      if (len < 64) {
        // Minimum working size because of implementation details
        len = 64;
      }
    }
    uint32_t len_with_metadata = len + 5;
    char* data = new char[len_with_metadata];
    memset(data, 0, len_with_metadata);
    assert(data);

    if (len > 0) {
      for (auto h : hash_entries_) {
        SemiLocalBloomImpl::AddHash(h, len, num_probes_, data);
      }
    }

    data[len] = static_cast<char>(-1);
    data[len + 1] = static_cast<char>(1);
    data[len + 2] = static_cast<char>(num_probes_);

    const char* const_data = data;
    buf->reset(const_data);
    hash_entries_.clear();

    return Slice(data, len_with_metadata);
  }

  int CalculateNumEntry(const uint32_t bytes) {
    return static_cast<int>(uint64_t(8) * bytes / bits_per_key_);
  }

 private:
  int bits_per_key_;
  int num_probes_;
  std::vector<uint32_t> hash_entries_;
};

class SemiLocalFilterBitsReader : public FilterBitsReader {
 public:
  SemiLocalFilterBitsReader(const char* data,
    int num_probes,
    uint32_t len_bytes)
    : data_(data), num_probes_(num_probes), len_bytes_(len_bytes) {
    }

  // No Copy allowed
  SemiLocalFilterBitsReader(const SemiLocalFilterBitsReader&) = delete;
  void operator=(const SemiLocalFilterBitsReader&) = delete;

  ~SemiLocalFilterBitsReader() override {}

  bool MayMatch(const Slice& key) override {
    uint32_t hash = BloomHash(key);
    uint32_t byte_offset;
    SemiLocalBloomImpl::PrepareHashMayMatch(
        hash, len_bytes_, data_, /*out*/ &byte_offset);
    return SemiLocalBloomImpl::HashMayMatchPrepared(
        hash, len_bytes_, num_probes_, data_, byte_offset);
  }

  virtual void MayMatch(int num_keys, Slice** keys, bool* may_match) override {
    uint32_t hashes[MultiGetContext::MAX_BATCH_SIZE];
    uint32_t byte_offsets[MultiGetContext::MAX_BATCH_SIZE];
    for (int i = 0; i < num_keys; ++i) {
      hashes[i] = BloomHash(*keys[i]);
      SemiLocalBloomImpl::PrepareHashMayMatch(hashes[i], len_bytes_, data_,
                                              /*out*/ &byte_offsets[i]);
    }
    for (int i = 0; i < num_keys; ++i) {
      may_match[i] = SemiLocalBloomImpl::HashMayMatchPrepared(hashes[i], len_bytes_, num_probes_,
                                                      data_, byte_offsets[i]);
    }
  }

 private:
  const char* data_;
  const int num_probes_;
  const uint32_t len_bytes_;
};

class FastLocalBloomBitsBuilder : public FilterBitsBuilder {
 public:
  FastLocalBloomBitsBuilder(const int bits_per_key, const int num_probes)
      : bits_per_key_(bits_per_key), num_probes_(num_probes) {
    assert(bits_per_key_);
  }

  // No Copy allowed
  FastLocalBloomBitsBuilder(const FastLocalBloomBitsBuilder&) = delete;
  void operator=(const FastLocalBloomBitsBuilder&) = delete;

  ~FastLocalBloomBitsBuilder() {}

  virtual void AddKey(const Slice& key) override {
    uint32_t hash = BloomHash(key);
    if (hash_entries_.size() == 0 || hash != hash_entries_.back()) {
      hash_entries_.push_back(hash);
    }
  }

  virtual Slice Finish(std::unique_ptr<const char[]>* buf) override {
    uint32_t num_cache_lines = 0;
    if (bits_per_key_ > 0 && hash_entries_.size() > 0) {
      num_cache_lines = static_cast<uint32_t>((uint64_t(1) * hash_entries_.size() * bits_per_key_ + 511 /*XXX + hash_entries_.size() / 2*/) / 512);
    }
    uint32_t len = num_cache_lines * 64;
    uint32_t len_with_metadata = len + 5;
    char* data = new char[len_with_metadata];
    memset(data, 0, len_with_metadata);
    assert(data);

    if (len > 0) {
      for (auto h : hash_entries_) {
        FastLocalBloomImpl::AddHash(h, len, num_probes_, data);
      }
    }

    data[len] = static_cast<char>(-1);
    data[len + 1] = static_cast<char>(2);
    data[len + 2] = static_cast<char>(num_probes_);

    const char* const_data = data;
    buf->reset(const_data);
    hash_entries_.clear();

    return Slice(data, len_with_metadata);
  }

  int CalculateNumEntry(const uint32_t bytes) {
    return static_cast<int>(uint64_t(8) * bytes / bits_per_key_);
  }

 private:
  int bits_per_key_;
  int num_probes_;
  std::vector<uint32_t> hash_entries_;
};

class FastLocalFilterBitsReader : public FilterBitsReader {
 public:
  FastLocalFilterBitsReader(const char* data,
    int num_probes,
    uint32_t len_bytes)
    : data_(data), num_probes_(num_probes), len_bytes_(len_bytes) {
    }

  // No Copy allowed
  FastLocalFilterBitsReader(const FastLocalFilterBitsReader&) = delete;
  void operator=(const FastLocalFilterBitsReader&) = delete;

  ~FastLocalFilterBitsReader() override {}

  bool MayMatch(const Slice& key) override {
    uint32_t hash = BloomHash(key);
    uint32_t byte_offset;
    FastLocalBloomImpl::PrepareHashMayMatch(
        hash, len_bytes_, data_, /*out*/ &byte_offset);
    return FastLocalBloomImpl::HashMayMatchPrepared(
        hash, len_bytes_, num_probes_, data_, byte_offset);
  }

  virtual void MayMatch(int num_keys, Slice** keys, bool* may_match) override {
    uint32_t hashes[MultiGetContext::MAX_BATCH_SIZE];
    uint32_t byte_offsets[MultiGetContext::MAX_BATCH_SIZE];
    for (int i = 0; i < num_keys; ++i) {
      hashes[i] = BloomHash(*keys[i]);
      FastLocalBloomImpl::PrepareHashMayMatch(hashes[i], len_bytes_, data_,
                                              /*out*/ &byte_offsets[i]);
    }
    for (int i = 0; i < num_keys; ++i) {
      may_match[i] = FastLocalBloomImpl::HashMayMatchPrepared(hashes[i], len_bytes_, num_probes_,
                                                      data_, byte_offsets[i]);
    }
  }

 private:
  const char* data_;
  const int num_probes_;
  const uint32_t len_bytes_;
};


class AlwaysTrueFilter : public FilterBitsReader {
 public:
  bool MayMatch(const Slice&) override {
    return true;
  }
  using FilterBitsReader::MayMatch; // inherit overload
};

class AlwaysFalseFilter : public FilterBitsReader {
 public:
  bool MayMatch(const Slice&) override {
    return false;
  }
  using FilterBitsReader::MayMatch; // inherit overload
};

class FullFilterBitsReader : public FilterBitsReader {
 public:
  FullFilterBitsReader(const char* data,
    int num_probes,
    uint32_t num_lines,
    uint32_t log2_cache_line_size)
    : data_(data), num_probes_(num_probes), num_lines_(num_lines), log2_cache_line_size_(log2_cache_line_size) {
    }

  // No Copy allowed
  FullFilterBitsReader(const FullFilterBitsReader&) = delete;
  void operator=(const FullFilterBitsReader&) = delete;

  ~FullFilterBitsReader() override {}

  // "contents" contains the data built by a preceding call to
  // FilterBitsBuilder::Finish. MayMatch must return true if the key was
  // passed to FilterBitsBuilder::AddKey. This method may return true or false
  // if the key was not on the list, but it should aim to return false with a
  // high probability.
  bool MayMatch(const Slice& key) override {
    uint32_t hash = BloomHash(key);
    uint32_t byte_offset;
    LegacyFullFilterImpl::PrepareHashMayMatch(
        hash, num_lines_, data_, /*out*/ &byte_offset, log2_cache_line_size_);
    return LegacyFullFilterImpl::HashMayMatchPrepared(
        hash, num_probes_, data_ + byte_offset, log2_cache_line_size_);
  }

  virtual void MayMatch(int num_keys, Slice** keys, bool* may_match) override {
    uint32_t hashes[MultiGetContext::MAX_BATCH_SIZE];
    uint32_t byte_offsets[MultiGetContext::MAX_BATCH_SIZE];
    for (int i = 0; i < num_keys; ++i) {
      hashes[i] = BloomHash(*keys[i]);
      LegacyFullFilterImpl::PrepareHashMayMatch(hashes[i], num_lines_, data_,
                                                /*out*/ &byte_offsets[i],
                                                log2_cache_line_size_);
    }
    for (int i = 0; i < num_keys; ++i) {
      may_match[i] = LegacyFullFilterImpl::HashMayMatchPrepared(hashes[i], num_probes_,
                                                      data_ + byte_offsets[i],
                                                      log2_cache_line_size_);
    }
  }

 private:
  const char* data_;
  const int num_probes_;
  const uint32_t num_lines_;
  const uint32_t log2_cache_line_size_;
};

// An implementation of filter policy
class BloomFilterPolicy : public FilterPolicy {
 public:
  explicit BloomFilterPolicy(int bits_per_key, bool use_block_based_builder)
      : bits_per_key_(bits_per_key), hash_func_(BloomHash),
        use_block_based_builder_(use_block_based_builder) {
    initialize();
  }

  ~BloomFilterPolicy() override {}

  const char* Name() const override { return "rocksdb.BuiltinBloomFilter"; }

  void CreateFilter(const Slice* keys, int n, std::string* dst) const override {
    // Compute bloom filter size (in both bits and bytes)
    uint32_t bits = static_cast<uint32_t>(n * bits_per_key_);

    // For small n, we can see a very high false positive rate.  Fix it
    // by enforcing a minimum bloom filter length.
    if (bits < 64) bits = 64;

    uint32_t bytes = (bits + 7) / 8;
    bits = bytes * 8;

    const size_t init_size = dst->size();
    dst->resize(init_size + bytes, 0);
    dst->push_back(static_cast<char>(num_probes_));  // Remember # of probes
    char* array = &(*dst)[init_size];
    for (int i = 0; i < n; i++) {
      LegacyNoLocalityBloomImpl::AddHash(hash_func_(keys[i]), bits, num_probes_,
                                         array);
    }
  }

  bool KeyMayMatch(const Slice& key, const Slice& bloom_filter) const override {
    const size_t len = bloom_filter.size();
    if (len < 2 || len > 0xffffffffU) {
      return false;
    }

    const char* array = bloom_filter.data();
    const uint32_t bits = static_cast<uint32_t>(len - 1) * 8;

    // Use the encoded k so that we can read filters generated by
    // bloom filters created using different parameters.
    const int k = static_cast<uint8_t>(array[len - 1]);
    if (k > 30) {
      // Reserved for potentially new encodings for short bloom filters.
      // Consider it a match.
      return true;
    }
    // NB: using k not num_probes_
    return LegacyNoLocalityBloomImpl::HashMayMatch(hash_func_(key), bits, k,
                                                   array);
  }

  FilterBitsBuilder* GetFilterBitsBuilder() const override {
    if (use_block_based_builder_) {
      return nullptr;
    }
    static const char *custom_filter_impl = getenv("ROCKSDB_CUSTOM_FILTER_IMPL");
    if (custom_filter_impl != nullptr) {
      if (0 == strcmp(custom_filter_impl, "fastlocalbloom")) {
        return new FastLocalBloomBitsBuilder(bits_per_key_, num_probes_);
      }
      if (0 == strcmp(custom_filter_impl, "semilocalbloom")) {
        return new SemiLocalBloomBitsBuilder(bits_per_key_, num_probes_);
      }
      if (0 == strcmp(custom_filter_impl, "maxcache")) {
        return new MaxCacheFilterBitsBuilder(bits_per_key_);
      }
    }

    return new FullFilterBitsBuilder(bits_per_key_, num_probes_);
  }

  // Read metadata to determine what kind of FilterBitsReader is needed
  // and return a new one.
  FilterBitsReader* GetFilterBitsReader(const Slice& contents) const override {
    uint32_t len_with_meta = static_cast<uint32_t>(contents.size());
    if (len_with_meta <= 5) {
      // filter is empty or broken. Treat like zero keys added.
      return new AlwaysFalseFilter();
    }

    char raw_num_probes = contents.data()[len_with_meta - 5];
    // NB: *num_probes > 30 and < 128 probably have not been used, because of
    // BloomFilterPolicy::initialize, unless directly calling
    // FullFilterBitsBuilder as an API, but we are leaving those cases in
    // limbo with FullFilterBitsReader for now.

    if (raw_num_probes < 1) {
      // Note: < 0 (or unsigned > 127) indicate special new implementations
      // (or reserved for future use)
      if (raw_num_probes == -1) {
        // Marker for new Bloom implementations
        // Read more metadata
        char raw_locality_val = contents.data()[len_with_meta - 4];
        raw_num_probes = contents.data()[len_with_meta - 3];
        uint32_t len = len_with_meta - 5;
        if (raw_locality_val == 1 && raw_num_probes > 0 && raw_num_probes <= 16) {
          return new SemiLocalFilterBitsReader(contents.data(), raw_num_probes, len);
        }
        if (raw_locality_val == 2 && raw_num_probes > 0 && raw_num_probes <= 16) {
          return new FastLocalFilterBitsReader(contents.data(), raw_num_probes, len);
        }
      } else if (raw_num_probes == -2) {
        // Marker for MaxCache implementation
        uint32_t len = len_with_meta - 5;
        assert((len & 63) == 0);
        return new MaxCacheFilterBitsReader(contents.data(), len / 64);
      }
      // otherwise
      // Treat as zero probes (always FP) for now.
      return new AlwaysTrueFilter();
    }
    // else attempt decode for FullFilterBitsReader

    int num_probes = raw_num_probes;
    assert(num_probes >= 1);
    assert(num_probes <= 127);

    uint32_t len = len_with_meta - 5;
    assert(len > 0);

    uint32_t num_lines = DecodeFixed32(contents.data() + len_with_meta - 4);
    uint32_t log2_cache_line_size;

    if (num_lines * CACHE_LINE_SIZE == len) {
      // Common case
      log2_cache_line_size = folly::constexpr_log2(CACHE_LINE_SIZE);
    } else if (num_lines == 0 || len % num_lines != 0) {
      // Invalid (no solution to num_lines * x == len)
      // Treat as zero probes (always FP) for now.
      return new AlwaysTrueFilter();
    } else {
      // Determine the non-native cache line size (from another system)
      log2_cache_line_size = 0;
      while ((num_lines << log2_cache_line_size) < len) {
        ++log2_cache_line_size;
      }
      if ((num_lines << log2_cache_line_size) != len) {
        // Invalid (block size not a power of two)
        // Treat as zero probes (always FP) for now.
        return new AlwaysTrueFilter();
      }
    }
    // if not early return
    return new FullFilterBitsReader(contents.data(), num_probes, num_lines, log2_cache_line_size);
  }

  // If choose to use block based builder
  bool UseBlockBasedBuilder() { return use_block_based_builder_; }

 private:
  int bits_per_key_;
  int num_probes_;
  uint32_t (*hash_func_)(const Slice& key);

  const bool use_block_based_builder_;

  void initialize() {
    // We intentionally round down to reduce probing cost a little bit
    num_probes_ = static_cast<int>(bits_per_key_ * 0.69);  // 0.69 =~ ln(2)
    if (num_probes_ < 1) num_probes_ = 1;
    if (num_probes_ > 30) num_probes_ = 30;
  }
};

}  // namespace

const FilterPolicy* NewBloomFilterPolicy(int bits_per_key,
                                         bool use_block_based_builder) {
  return new BloomFilterPolicy(bits_per_key, use_block_based_builder);
}

}  // namespace rocksdb
