//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2012 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include <array>

#include "rocksdb/filter_policy.h"

#include "rocksdb/slice.h"
#include "table/block_based/block_based_filter_block.h"
#include "table/block_based/full_filter_block.h"
#include "table/block_based/filter_policy_internal.h"
#include "third-party/folly/folly/ConstexprMath.h"
#include "util/bloom_impl.h"
#include "util/coding.h"
#include "util/hash.h"

#ifdef HAVE_POPCNT
#include <x86intrin.h>
#endif
#if defined(HAVE_AVX2) || defined(HAVE_BMI2)
#include <immintrin.h>
#endif
// git diff 1d4b2632ee7a8b36383ef9b3b814d722c203022b..new-filters

namespace rocksdb {

namespace {

using LegacyFullFilterImpl = LegacyLocalityBloomImpl</*ExtraRotates*/ false>;

class LegacyBloomBitsBuilder : public BuiltinFilterBitsBuilder {
 public:
  explicit LegacyBloomBitsBuilder(const int bits_per_key, const int num_probes);

  // No Copy allowed
  LegacyBloomBitsBuilder(const LegacyBloomBitsBuilder&) = delete;
  void operator=(const LegacyBloomBitsBuilder&) = delete;

  ~LegacyBloomBitsBuilder() override;

  void AddKey(const Slice& key) override;

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
  Slice Finish(std::unique_ptr<const char[]>* buf) override;

  int CalculateNumEntry(const uint32_t bytes) override;

  uint32_t CalculateSpace(const int num_entry) override {
    uint32_t dont_care1;
    uint32_t dont_care2;
    return CalculateSpace(num_entry, &dont_care1, &dont_care2);
  }

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

  // Implementation-specific variant of public CalculateSpace
  uint32_t CalculateSpace(const int num_entry, uint32_t* total_bits,
                          uint32_t* num_lines);

  // Assuming single threaded access to this function.
  void AddHash(uint32_t h, char* data, uint32_t num_lines, uint32_t total_bits);
};

LegacyBloomBitsBuilder::LegacyBloomBitsBuilder(const int bits_per_key,
                                               const int num_probes)
    : bits_per_key_(bits_per_key), num_probes_(num_probes) {
  assert(bits_per_key_);
}

LegacyBloomBitsBuilder::~LegacyBloomBitsBuilder() {}

void LegacyBloomBitsBuilder::AddKey(const Slice& key) {
  uint32_t hash = BloomHash(key);
  if (hash_entries_.size() == 0 || hash != hash_entries_.back()) {
    hash_entries_.push_back(hash);
  }
}

Slice LegacyBloomBitsBuilder::Finish(std::unique_ptr<const char[]>* buf) {
  uint32_t total_bits, num_lines;
  char* data = ReserveSpace(static_cast<int>(hash_entries_.size()), &total_bits,
                            &num_lines);
  assert(data);

  if (total_bits != 0 && num_lines != 0) {
    for (auto h : hash_entries_) {
      AddHash(h, data, num_lines, total_bits);
    }
  }
  data[total_bits / 8] = static_cast<char>(num_probes_);
  EncodeFixed32(data + total_bits / 8 + 1, static_cast<uint32_t>(num_lines));

  const char* const_data = data;
  buf->reset(const_data);
  hash_entries_.clear();

  return Slice(data, total_bits / 8 + 5);
}

uint32_t LegacyBloomBitsBuilder::GetTotalBitsForLocality(uint32_t total_bits) {
  uint32_t num_lines =
      (total_bits + CACHE_LINE_SIZE * 8 - 1) / (CACHE_LINE_SIZE * 8);

  // Make num_lines an odd number to make sure more bits are involved
  // when determining which block.
  if (num_lines % 2 == 0) {
    num_lines++;
  }
  return num_lines * (CACHE_LINE_SIZE * 8);
}

uint32_t LegacyBloomBitsBuilder::CalculateSpace(const int num_entry,
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

char* LegacyBloomBitsBuilder::ReserveSpace(const int num_entry,
                                           uint32_t* total_bits,
                                           uint32_t* num_lines) {
  uint32_t sz = CalculateSpace(num_entry, total_bits, num_lines);
  char* data = new char[sz];
  memset(data, 0, sz);
  return data;
}

int LegacyBloomBitsBuilder::CalculateNumEntry(const uint32_t bytes) {
  assert(bits_per_key_);
  assert(bytes > 0);
  int high = static_cast<int>(bytes * 8 / bits_per_key_ + 1);
  int low = 1;
  int n = high;
  for (; n >= low; n--) {
    if (CalculateSpace(n) <= bytes) {
      break;
    }
  }
  assert(n < high);  // High should be an overestimation
  return n;
}

inline void LegacyBloomBitsBuilder::AddHash(uint32_t h, char* data,
                                            uint32_t num_lines,
                                            uint32_t total_bits) {
#ifdef NDEBUG
  static_cast<void>(total_bits);
#endif
  assert(num_lines > 0 && total_bits > 0);

  LegacyFullFilterImpl::AddHash(h, num_lines, num_probes_, data,
                                folly::constexpr_log2(CACHE_LINE_SIZE));
}

// See description in FastLocalBloomImpl
class FastLocalBloomBitsBuilder : public BuiltinFilterBitsBuilder {
 public:
  FastLocalBloomBitsBuilder(const int bits_per_key, const int num_probes)
      : bits_per_key_(bits_per_key), num_probes_(num_probes) {
    assert(bits_per_key_);
  }

  // No Copy allowed
  FastLocalBloomBitsBuilder(const FastLocalBloomBitsBuilder&) = delete;
  void operator=(const FastLocalBloomBitsBuilder&) = delete;

  ~FastLocalBloomBitsBuilder() override {}

  virtual void AddKey(const Slice& key) override {
    uint64_t hash = GetSliceHash64(key);
    if (hash_entries_.size() == 0 || hash != hash_entries_.back()) {
      hash_entries_.push_back(hash);
    }
  }

  virtual Slice Finish(std::unique_ptr<const char[]>* buf) override {
    uint32_t len_with_metadata =
        CalculateSpace(static_cast<uint32_t>(hash_entries_.size()));
    char* data = new char[len_with_metadata];
    memset(data, 0, len_with_metadata);

    assert(data);
    assert(len_with_metadata >= 5);

    uint32_t len = len_with_metadata - 5;
    if (len > 0) {
      AddAllEntries(data, len);
    }

    // -1 = Marker for newer Bloom implementations
    data[len] = static_cast<char>(-1);
    // 0 = Marker for this sub-implementation
    data[len + 1] = static_cast<char>(0);
    // num_probes (and 0 in upper bits for 64-byte block size)
    data[len + 2] = static_cast<char>(num_probes_);
    // rest of metadata stays zero

    const char* const_data = data;
    buf->reset(const_data);
    hash_entries_.clear();

    return Slice(data, len_with_metadata);
  }

  int CalculateNumEntry(const uint32_t bytes) override {
    uint32_t bytes_no_meta = bytes >= 5u ? bytes - 5u : 0;
    return static_cast<int>(uint64_t{8} * bytes_no_meta / bits_per_key_);
  }

  uint32_t CalculateSpace(const int num_entry) override {
    uint32_t num_cache_lines = 0;
    if (bits_per_key_ > 0 && num_entry > 0) {
      num_cache_lines = static_cast<uint32_t>(
          (int64_t{num_entry} * bits_per_key_ + 511) / 512);
    }
    return num_cache_lines * 64 + /*metadata*/ 5;
  }

 private:
  void AddAllEntries(char* data, uint32_t len) const {
    // Simple version without prefetching:
    //
    // for (auto h : hash_entries_) {
    //   FastLocalBloomImpl::AddHash(Lower32of64(h), Upper32of64(h), len,
    //                               num_probes_, data);
    // }

    const size_t num_entries = hash_entries_.size();
    constexpr size_t kBufferMask = 7;
    static_assert(((kBufferMask + 1) & kBufferMask) == 0,
                  "Must be power of 2 minus 1");

    std::array<uint32_t, kBufferMask + 1> hashes;
    std::array<uint32_t, kBufferMask + 1> byte_offsets;

    // Prime the buffer
    size_t i = 0;
    for (; i <= kBufferMask && i < num_entries; ++i) {
      uint64_t h = hash_entries_[i];
      FastLocalBloomImpl::PrepareHash(Lower32of64(h), len, data,
                                      /*out*/ &byte_offsets[i]);
      hashes[i] = Upper32of64(h);
    }

    // Process and buffer
    for (; i < num_entries; ++i) {
      uint32_t& hash_ref = hashes[i & kBufferMask];
      uint32_t& byte_offset_ref = byte_offsets[i & kBufferMask];
      // Process (add)
      FastLocalBloomImpl::AddHashPrepared(hash_ref, num_probes_,
                                          data + byte_offset_ref);
      // And buffer
      uint64_t h = hash_entries_[i];
      FastLocalBloomImpl::PrepareHash(Lower32of64(h), len, data,
                                      /*out*/ &byte_offset_ref);
      hash_ref = Upper32of64(h);
    }

    // Finish processing
    for (i = 0; i <= kBufferMask && i < num_entries; ++i) {
      FastLocalBloomImpl::AddHashPrepared(hashes[i], num_probes_,
                                          data + byte_offsets[i]);
    }
  }

  int bits_per_key_;
  int num_probes_;
  std::vector<uint64_t> hash_entries_;
};

// See description in FastLocalBloomImpl
class FastLocalBloomBitsReader : public FilterBitsReader {
 public:
  FastLocalBloomBitsReader(const char* data, int num_probes, uint32_t len_bytes)
      : data_(data), num_probes_(num_probes), len_bytes_(len_bytes) {}

  // No Copy allowed
  FastLocalBloomBitsReader(const FastLocalBloomBitsReader&) = delete;
  void operator=(const FastLocalBloomBitsReader&) = delete;

  ~FastLocalBloomBitsReader() override {}

  bool MayMatch(const Slice& key) override {
    uint64_t h = GetSliceHash64(key);
    uint32_t byte_offset;
    FastLocalBloomImpl::PrepareHash(Lower32of64(h), len_bytes_, data_,
                                    /*out*/ &byte_offset);
    return FastLocalBloomImpl::HashMayMatchPrepared(Upper32of64(h), num_probes_,
                                                    data_ + byte_offset);
  }

  virtual void MayMatch(int num_keys, Slice** keys, bool* may_match) override {
    std::array<uint32_t, MultiGetContext::MAX_BATCH_SIZE> hashes;
    std::array<uint32_t, MultiGetContext::MAX_BATCH_SIZE> byte_offsets;
    for (int i = 0; i < num_keys; ++i) {
      uint64_t h = GetSliceHash64(*keys[i]);
      FastLocalBloomImpl::PrepareHash(Lower32of64(h), len_bytes_, data_,
                                      /*out*/ &byte_offsets[i]);
      hashes[i] = Upper32of64(h);
    }
    for (int i = 0; i < num_keys; ++i) {
      may_match[i] = FastLocalBloomImpl::HashMayMatchPrepared(
          hashes[i], num_probes_, data_ + byte_offsets[i]);
    }
  }

 private:
  const char* data_;
  const int num_probes_;
  const uint32_t len_bytes_;
};

class LocalHybridBitsBuilder : public BuiltinFilterBitsBuilder {
 public:
  static constexpr uint32_t kCacheLineBits = 1024;
  static constexpr uint32_t kCacheLineBytes = kCacheLineBits / 8;

  LocalHybridBitsBuilder(const int bits_per_key)
      : bits_per_key_(bits_per_key) {
    assert(bits_per_key_);
  }

  // No Copy allowed
  LocalHybridBitsBuilder(const LocalHybridBitsBuilder&) = delete;
  void operator=(const LocalHybridBitsBuilder&) = delete;

  ~LocalHybridBitsBuilder() override {}

  virtual void AddKey(const Slice& key) override {
    uint64_t hash = GetSliceHash64(key);
    if (hash_entries_.size() == 0 || hash != hash_entries_.back()) {
      hash_entries_.push_back(hash);
    }
  }

  virtual Slice Finish(std::unique_ptr<const char[]>* buf) override {
    uint32_t len_with_metadata =
        CalculateSpace(static_cast<uint32_t>(hash_entries_.size()));
    char* data = new char[len_with_metadata];
    memset(data, 0, len_with_metadata);

    assert(data);
    assert(len_with_metadata >= 5);

    uint32_t len = len_with_metadata - 5;
    if (len > 0) {
      AddAllEntries(data, len);
    }

    // -2 = Marker for local hybrid filter implementation
    data[len] = static_cast<char>(-2);
    // 0 = Marker for this sub-implementation
    data[len + 1] = static_cast<char>(0);
    // rest of metadata stays zero (TODO: use for seed)

    const char* const_data = data;
    buf->reset(const_data);
    hash_entries_.clear();

    return Slice(data, len_with_metadata);
  }

  int CalculateNumEntry(const uint32_t bytes) override {
    uint32_t bytes_no_meta = bytes >= 5u ? bytes - 5u : 0;
    return static_cast<int>(uint64_t{8} * bytes_no_meta / bits_per_key_);
  }

  uint32_t CalculateSpace(const int num_entry) override {
    uint32_t num_cache_lines = 0;
    if (bits_per_key_ > 0 && num_entry > 0) {
      num_cache_lines = static_cast<uint32_t>(
          (static_cast<uint64_t>(int64_t{num_entry} * bits_per_key_) + kCacheLineBits - 1) / kCacheLineBits);
    }
    return num_cache_lines * kCacheLineBytes + /*metadata*/ 5;
  }

 private:
  void AddAllEntries(char* data, uint32_t len) const {
    // Too slow:
    //std::sort(hash_entries_.begin(), hash_entries_.end());

    const uint32_t num_cache_lines = len / kCacheLineBytes;

    // Partition into buckets with temporary allocation
    // (Fast enough for now; could save temp space with counting sort)
    std::unique_ptr<std::vector<uint32_t>[]> buckets(new std::vector<uint32_t>[num_cache_lines]);
    for (size_t i = 0; i < hash_entries_.size(); ++i) {
      uint32_t cache_line = fastrange32(Lower32of64(hash_entries_[i]), num_cache_lines);
      //fprintf(stderr, "Preparing hash %x on cache line %d\n", hash_entries_[i], cache_line);
      buckets[cache_line].push_back(Upper32of64(hash_entries_[i]));
    }
    double fp_rate_sum = 0;
    for (uint32_t i = 0; i < num_cache_lines; ++i) {
      BuildCacheLine(data + (i * kCacheLineBytes), buckets[i].begin(), buckets[i].end(), &fp_rate_sum);
    }
    fprintf(stderr, "Filter FP rate: %g\n", fp_rate_sum / num_cache_lines);
  }

  void BuildCacheLine(char *data_at_cache_line, std::vector<uint32_t>::iterator begin, std::vector<uint32_t>::iterator end, double *fp_rate_sum) const {
    const uint32_t count = static_cast<uint32_t>(end - begin);
    double fp_rate = std::pow(1.0 - exp(-4.0 * count / kCacheLineBits), 4.0); // bloom rate

    if (count * 6u > (kCacheLineBits - 8u)) {
      *fp_rate_sum += fp_rate;
      fprintf(stderr, "Cache line FP rate: %g (bloom %u)\n", fp_rate, count);
      BuildBloomCacheLine(data_at_cache_line, begin, end);
      return;
    }

    std::sort(begin, end);

    uint32_t range = static_cast<uint32_t>((uint64_t{std::max(count, (kCacheLineBits - 8u + 25u) / 26u)} << 25) * 0.75);
    std::vector<uint32_t> vals;
    uint32_t unary_bits = 0;

    uint32_t prev = 0;
    for (auto it = begin; it != end; ++it) {
      uint32_t val = fastrange32(*it, range);
      assert(val >= prev);
      vals.push_back(val);
      uint32_t diff = (val >> 20) - (prev >> 20);
      unary_bits += 1u + (diff >> 4);
      prev = val;
    }

    fprintf(stderr, "Rice count %u, unary_bits/count %g\n", count, (double)unary_bits / vals.size());

    uint32_t rem_bits = (kCacheLineBits - 8u) - ((unary_bits + 1u) & ~1u);
    uint32_t base_entry_bits = rem_bits / count;
    uint32_t extra_entry_bits;
    if (base_entry_bits >= 24u) {
      base_entry_bits = 24u;
      extra_entry_bits = 0;
      fp_rate = 0;
    } else {
      extra_entry_bits = rem_bits % count;
      fp_rate = 1.0 * extra_entry_bits / (range >> (24u - base_entry_bits - 1u));
    }
    fp_rate += 1.0 * (count - extra_entry_bits) / (range >> (24u - base_entry_bits));
    //*fp_rate_sum += fp_rate;
    fprintf(stderr, "Cache line FP rate: %g\n", fp_rate);
    /*
    // No more tries; fall back on Bloom
    *fp_rate_sum += fp_rate;
    fprintf(stderr, "Cache line FP rate: %g (bloom %u)\n", fp_rate, count);
    */

    // Simulate orderly filter
    rem_bits = (kCacheLineBits - 8u);
    base_entry_bits = rem_bits / count;
    if (base_entry_bits >= 24u) {
      base_entry_bits = 24u;
      extra_entry_bits = 0;
      fp_rate = 0;
    } else {
      extra_entry_bits = rem_bits % count;
      fp_rate = 4.0 * extra_entry_bits / (1 << (base_entry_bits + 1));
    }
    fp_rate += 4.0 * (count - extra_entry_bits) / (1 << base_entry_bits);
    fprintf(stderr, "Cache line FP rate (orderly): %g\n", fp_rate);
    *fp_rate_sum += fp_rate;


    BuildBloomCacheLine(data_at_cache_line, begin, end);
    // TODO/FIXME
    /*
    // A "before" bit index (append after)
    int unary_bit_ptr = 0;
    // An "after" bit index (append before)
    int entry_bit_ptr = 504;

    uint32_t prev = 0;
    for (uint32_t val : vals) {
      uint32_t diff = val - prev;
      unary_bit_ptr += (diff >> entry_bits);
      data_at_cache_line[unary_bit_ptr >> 3] |= char{1} << (unary_bit_ptr & 7);
      unary_bit_ptr++;


      prev = val;
    }
    assert (unary_bit_ptr <= entry_bit_ptr);
    */
  }

  void BuildBloomCacheLine(char *data_at_cache_line, std::vector<uint32_t>::iterator begin, std::vector<uint32_t>::iterator end) const {
    for (auto it = begin; it != end; ++it) {
      FastLocalBloomImpl::AddHashPrepared(*it, /*num_probes*/4, data_at_cache_line);
    }
    if ((data_at_cache_line[63] & char(0xc0)) == 0) {
      // set one bit to make one of the top two bits non-zero
      data_at_cache_line[63] |= char(0x80);
    }
    //fprintf(stderr, "Built bloom\n");
  }

  int bits_per_key_;
  std::vector<uint64_t> hash_entries_;
};

// See description in LocalHybridImpl
class LocalHybridBitsReader : public FilterBitsReader {
 public:
  LocalHybridBitsReader(const char* data, int num_probes, uint32_t len_bytes)
      : data_(data), num_probes_(num_probes), len_bytes_(len_bytes) {}

  // No Copy allowed
  LocalHybridBitsReader(const LocalHybridBitsReader&) = delete;
  void operator=(const LocalHybridBitsReader&) = delete;

  ~LocalHybridBitsReader() override {}

  bool MayMatch(const Slice& key) override {
    uint64_t h = GetSliceHash64(key);
    uint32_t byte_offset;
    FastLocalBloomImpl::PrepareHash(Lower32of64(h), len_bytes_, data_,
                                    /*out*/ &byte_offset);
    return FastLocalBloomImpl::HashMayMatchPrepared(Upper32of64(h), num_probes_,
                                                    data_ + byte_offset);
  }

  virtual void MayMatch(int num_keys, Slice** keys, bool* may_match) override {
    std::array<uint32_t, MultiGetContext::MAX_BATCH_SIZE> hashes;
    std::array<uint32_t, MultiGetContext::MAX_BATCH_SIZE> byte_offsets;
    for (int i = 0; i < num_keys; ++i) {
      uint64_t h = GetSliceHash64(*keys[i]);
      FastLocalBloomImpl::PrepareHash(Lower32of64(h), len_bytes_, data_,
                                      /*out*/ &byte_offsets[i]);
      hashes[i] = Upper32of64(h);
    }
    for (int i = 0; i < num_keys; ++i) {
      may_match[i] = FastLocalBloomImpl::HashMayMatchPrepared(
          hashes[i], num_probes_, data_ + byte_offsets[i]);
    }
  }

 private:
  const char* data_;
  const int num_probes_;
  const uint32_t len_bytes_;
};


class AlwaysTrueFilter : public FilterBitsReader {
 public:
  bool MayMatch(const Slice&) override { return true; }
  using FilterBitsReader::MayMatch;  // inherit overload
};

class AlwaysFalseFilter : public FilterBitsReader {
 public:
  bool MayMatch(const Slice&) override { return false; }
  using FilterBitsReader::MayMatch;  // inherit overload
};

class LegacyBloomBitsReader : public FilterBitsReader {
 public:
  LegacyBloomBitsReader(const char* data, int num_probes, uint32_t num_lines,
                        uint32_t log2_cache_line_size)
      : data_(data),
        num_probes_(num_probes),
        num_lines_(num_lines),
        log2_cache_line_size_(log2_cache_line_size) {}

  // No Copy allowed
  LegacyBloomBitsReader(const LegacyBloomBitsReader&) = delete;
  void operator=(const LegacyBloomBitsReader&) = delete;

  ~LegacyBloomBitsReader() override {}

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
    std::array<uint32_t, MultiGetContext::MAX_BATCH_SIZE> hashes;
    std::array<uint32_t, MultiGetContext::MAX_BATCH_SIZE> byte_offsets;
    for (int i = 0; i < num_keys; ++i) {
      hashes[i] = BloomHash(*keys[i]);
      LegacyFullFilterImpl::PrepareHashMayMatch(hashes[i], num_lines_, data_,
                                                /*out*/ &byte_offsets[i],
                                                log2_cache_line_size_);
    }
    for (int i = 0; i < num_keys; ++i) {
      may_match[i] = LegacyFullFilterImpl::HashMayMatchPrepared(
          hashes[i], num_probes_, data_ + byte_offsets[i],
          log2_cache_line_size_);
    }
  }

 private:
  const char* data_;
  const int num_probes_;
  const uint32_t num_lines_;
  const uint32_t log2_cache_line_size_;
};

}  // namespace

const std::vector<BloomFilterPolicy::Mode> BloomFilterPolicy::kAllFixedImpls = {
    kLegacyBloom,
    kDeprecatedBlock,
    kFastLocalBloom,
    kLocalHybrid,
};

const std::vector<BloomFilterPolicy::Mode> BloomFilterPolicy::kAllUserModes = {
    kDeprecatedBlock,
    kAuto,
};

BloomFilterPolicy::BloomFilterPolicy(int bits_per_key, Mode mode)
    : bits_per_key_(bits_per_key), mode_(mode) {
  // We intentionally round down to reduce probing cost a little bit
  num_probes_ = static_cast<int>(bits_per_key_ * 0.69);  // 0.69 =~ ln(2)
  if (num_probes_ < 1) num_probes_ = 1;
  if (num_probes_ > 30) num_probes_ = 30;
}

BloomFilterPolicy::~BloomFilterPolicy() {}

const char* BloomFilterPolicy::Name() const {
  return "rocksdb.BuiltinBloomFilter";
}

void BloomFilterPolicy::CreateFilter(const Slice* keys, int n,
                                     std::string* dst) const {
  // We should ideally only be using this deprecated interface for
  // appropriately constructed BloomFilterPolicy
  assert(mode_ == kDeprecatedBlock);

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
    LegacyNoLocalityBloomImpl::AddHash(BloomHash(keys[i]), bits, num_probes_,
                                       array);
  }
}

bool BloomFilterPolicy::KeyMayMatch(const Slice& key,
                                    const Slice& bloom_filter) const {
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
  return LegacyNoLocalityBloomImpl::HashMayMatch(BloomHash(key), bits, k,
                                                 array);
}

FilterBitsBuilder* BloomFilterPolicy::GetFilterBitsBuilder() const {
  // This code path should no longer be used, for the built-in
  // BloomFilterPolicy. Internal to RocksDB and outside BloomFilterPolicy,
  // only get a FilterBitsBuilder with FilterBuildingContext::GetBuilder(),
  // which will call BloomFilterPolicy::GetFilterBitsBuilderInternal.
  // RocksDB users have been warned (HISTORY.md) that they can no longer
  // call this on the built-in BloomFilterPolicy (unlikely).
  assert(false);
  return GetFilterBitsBuilderInternal(
      FilterBuildingContext(BlockBasedTableOptions()));
}

FilterBitsBuilder* BloomFilterPolicy::GetFilterBitsBuilderInternal(
    const FilterBuildingContext& context) const {
  Mode cur = mode_;
  // Unusual code construction so that we can have just
  // one exhaustive switch without (risky) recursion
  for (int i = 0; i < 2; ++i) {
    switch (cur) {
      case kAuto:
        if (context.table_options_.format_version < 5) {
          cur = kLegacyBloom;
        } else {
          cur = kLocalHybrid;
        }
        break;
      case kDeprecatedBlock:
        return nullptr;
      case kFastLocalBloom:
        return new FastLocalBloomBitsBuilder(bits_per_key_, num_probes_);
      case kLocalHybrid:
        return new LocalHybridBitsBuilder(bits_per_key_);
      case kLegacyBloom:
        return new LegacyBloomBitsBuilder(bits_per_key_, num_probes_);
    }
  }
  assert(false);
  return nullptr;  // something legal
}

// Read metadata to determine what kind of FilterBitsReader is needed
// and return a new one.
FilterBitsReader* BloomFilterPolicy::GetFilterBitsReader(
    const Slice& contents) const {
  uint32_t len_with_meta = static_cast<uint32_t>(contents.size());
  if (len_with_meta <= 5) {
    // filter is empty or broken. Treat like zero keys added.
    return new AlwaysFalseFilter();
  }

  char raw_num_probes = contents.data()[len_with_meta - 5];
  // NB: *num_probes > 30 and < 128 probably have not been used, because of
  // BloomFilterPolicy::initialize, unless directly calling
  // LegacyBloomBitsBuilder as an API, but we are leaving those cases in
  // limbo with LegacyBloomBitsReader for now.

  if (raw_num_probes < 1) {
    // Note: < 0 (or unsigned > 127) indicate special new implementations
    // (or reserved for future use)
    if (raw_num_probes == -1) {
      // Marker for newer Bloom implementations
      return GetBloomBitsReader(contents);
    }
    // otherwise
    // Treat as zero probes (always FP) for now.
    return new AlwaysTrueFilter();
  }
  // else attempt decode for LegacyBloomBitsReader

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
  return new LegacyBloomBitsReader(contents.data(), num_probes, num_lines,
                                   log2_cache_line_size);
}

// For newer Bloom filter implementations
FilterBitsReader* BloomFilterPolicy::GetBloomBitsReader(
    const Slice& contents) const {
  uint32_t len_with_meta = static_cast<uint32_t>(contents.size());
  uint32_t len = len_with_meta - 5;

  assert(len > 0);  // precondition

  // Read more metadata
  char sub_impl_val = contents.data()[len_with_meta - 4];
  // 0: FastLocalBloom
  // other: reserved

  char block_and_probes = contents.data()[len_with_meta - 3];
  int log2_block_bytes = ((block_and_probes >> 5) & 7) + 6;
  // 0 in top 3 bits -> 6 -> 64-byte (Intel cache line)
  // reserved:
  // 1 in top 3 bits -> 7 -> 128-byte
  // 2 in top 3 bits -> 8 -> 256-byte
  // ...

  int num_probes = (block_and_probes & 31);
  // num_probes in bottom 5 bits, except 0 and 31 reserved

  if (num_probes < 1 || num_probes > 30) {
    // Reserved / future safe
    return new AlwaysTrueFilter();
  }

  uint16_t rest = DecodeFixed16(contents.data() + len_with_meta - 2);
  if (rest != 0) {
    // Reserved, possibly for hash seed
    // Future safe
    return new AlwaysTrueFilter();
  }

  if (sub_impl_val == 0) {        // FastLocalBloom
    if (log2_block_bytes == 6) {  // Only block size supported for now
      return new FastLocalBloomBitsReader(contents.data(), num_probes, len);
    }
  }
  // otherwise
  // Reserved / future safe
  return new AlwaysTrueFilter();
}

const FilterPolicy* NewBloomFilterPolicy(int bits_per_key,
                                         bool use_block_based_builder) {
  BloomFilterPolicy::Mode m;
  if (use_block_based_builder) {
    m = BloomFilterPolicy::kDeprecatedBlock;
  } else {
    m = BloomFilterPolicy::kAuto;
  }
  assert(std::find(BloomFilterPolicy::kAllUserModes.begin(),
                   BloomFilterPolicy::kAllUserModes.end(),
                   m) != BloomFilterPolicy::kAllUserModes.end());
  return new BloomFilterPolicy(bits_per_key, m);
}

FilterPolicy::~FilterPolicy() { }

}  // namespace rocksdb
