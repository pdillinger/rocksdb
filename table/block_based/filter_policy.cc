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

    // See BloomFilterPolicy::GetBloomBitsReader re: metadata
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

inline void bit_table_set_bit(char *data, uint32_t bit_offset) {
  data[bit_offset >> 3] |= char{1} << (bit_offset & 7);
}

inline bool bit_table_get_bit(const char *data, uint32_t bit_offset) {
  return (data[bit_offset >> 3] >> (bit_offset & 7)) & 1;
}

inline void or_put_nibble(char *data, uint32_t nibble_offset, uint32_t val) {
  assert(val < 16);
  data[nibble_offset >> 1] |= val << ((nibble_offset & 1) * 4);
}

inline uint32_t get_nibble(const char *data, uint32_t nibble_offset) {
  return static_cast<uint32_t>(data[nibble_offset >> 1] >> ((nibble_offset & 1) * 4)) & uint32_t{15};
}

// Might access 5 bytes from starting address
inline uint32_t bit_table_get(const char *data, uint32_t bit_offset, uint32_t mask) {
  //fprintf(stderr, "--> Getting val at offset %d\n", bit_offset);
  data += bit_offset >> 3;
  bit_offset &= 7;
  const uint8_t *udata = reinterpret_cast<const uint8_t*>(data);
  //fprintf(stderr, "--> Bytes: %02x%02x%02x%02x%02x\n", udata[4], udata[3], udata[2], udata[1], udata[0]);
  uint64_t rv = static_cast<uint64_t>(udata[0]) + (static_cast<uint64_t>(udata[1]) << 8) + (static_cast<uint64_t>(udata[2]) << 16) + (static_cast<uint64_t>(udata[3]) << 24) + (static_cast<uint64_t>(udata[4]) << 32);
  //fprintf(stderr, "--> Raw:   %010lx\n", rv);
  return static_cast<uint32_t>(rv >> bit_offset) & mask;
}

// Might access 5 bytes from starting address, or extra aligned 32 bits
inline void bit_table_or_put(char *data, uint32_t bit_offset, uint32_t val) {
  //fprintf(stderr, "--> Putting val %x at offset %d\n", val, bit_offset);
#ifdef HAVE_SSE42 // XXX: stand-in for endianness
  uint32_t *data32 = reinterpret_cast<uint32_t*>(data);
  assert((bit_offset >> 5) < 15); // make sure we don't write beyond cache line
  data32 += bit_offset >> 5;
  bit_offset &= 31;
  data32[0] |= val << bit_offset;
  data32[1] |= val >> 1 >> (31 - bit_offset);
#else
  data += bit_offset >> 3;
  bit_offset &= 7;
  uint8_t *udata = reinterpret_cast<uint8_t*>(data);
  //fprintf(stderr, "--> Before: %02x%02x%02x%02x%02x\n", udata[4], udata[3], udata[2], udata[1], udata[0]);
  udata[0] |= static_cast<uint8_t>(val << bit_offset);
  udata[1] |= static_cast<uint8_t>(val >> (8 - bit_offset));
  udata[2] |= static_cast<uint8_t>(val >> (16 - bit_offset));
  udata[3] |= static_cast<uint8_t>(val >> (24 - bit_offset));
  udata[4] |= static_cast<uint8_t>(uint64_t(val) >> (32 - bit_offset));
  //fprintf(stderr, "--> After:  %02x%02x%02x%02x%02x\n", udata[4], udata[3], udata[2], udata[1], udata[0]);
#endif
}

inline uint32_t popcnt_char(char v) {
  uint32_t a = v & 0x55U;
  uint32_t b = a + ((v & 0xaaU) >> 1);
  a = b & 0x33;
  b = a + ((b & 0xcc) >> 2);
  a = b & 0xf;
  b = a + ((b & 0xf0) >> 4);
  return b;
}

inline uint32_t nbits_popcnt(const char *data, uint32_t nbits) {
  uint32_t rv = 0;
#ifdef HAVE_POPCNT
  while (nbits > 0) {
    uint64_t tmp;
    memcpy(&tmp, data, sizeof(tmp));
    if (nbits < 64) {
      tmp <<= (64 - nbits);
      rv += _popcnt64(tmp);
      nbits = 0;
      break;
    }
    rv += _popcnt64(tmp);
    nbits -= 64;
    data += sizeof(tmp);
  }
#else
  while (nbits >= 8) {
    rv += popcnt_char(*(data++));
    nbits -= 8;
  }
  if (nbits > 0) {
    rv += popcnt_char(static_cast<char>(static_cast<uint8_t>(*data) << (8 - nbits)));
  }
#endif
  return rv;
}

inline uint32_t general_tzcnt(const char *data, uint32_t from) {
  uint32_t rv = 0;
#if defined(HAVE_POPCNT) && defined(HAVE_BMI2)
  for (;;) {
    uint64_t v = *reinterpret_cast<const uint64_t*>(data + (from / 8));
    uint32_t bit_shift = from % 8;
    v >>= bit_shift;
    if (v != 0) {
      rv += _tzcnt_u64(v);
      return rv;
    }
    from += (64 - bit_shift);
    rv += (64 - bit_shift);
  }
#else
  while (bit_table_get_bit(data, from + rv) == false) {
    ++rv;
  }
  return rv;
#endif
}

// Returns 1-based index, or zero if n = 0.
inline uint32_t nth_set_bit_index1(const char *data, uint32_t from, uint32_t n) {
  if (n == 0) {
    return 0;
  }
#if defined(HAVE_POPCNT) && defined(HAVE_BMI2)
  assert(n < 64);
  const uint64_t *data64 = reinterpret_cast<const uint64_t*>(data + (from / 8));
  uint32_t rv = 1; // 1-based return
  uint64_t v = *data64 >> (from % 8);
  uint64_t p = _pdep_u64(uint64_t(1) << (n - 1), v);
  if (p != 0) {
    rv += _tzcnt_u64(p);
    return rv;
  }
  // else not in that uint64
  n -= _popcnt64(v);
  ++data64;
  rv += 64 - (from % 8);
  for (;;) {
    assert(n < 64);
    p = _pdep_u64(uint64_t(1) << (n - 1), *data64);
    if (p != 0) {
      rv += _tzcnt_u64(p);
      return rv;
    }
    // else not in that uint64
    n -= _popcnt64(*data64);
    ++data64;
    rv += 64;
  }
#else
  uint32_t to = from;
  do {
    n -= bit_table_get_bit(data, to);
    ++to;
  } while (n > 0);
  return to - from;
#endif
}

class LocalHybridBitsBuilder : public BuiltinFilterBitsBuilder {
 public:
  static constexpr uint32_t kCacheLineLog2Bytes = 6u;
  static constexpr uint32_t kCacheLineLog2Bits = kCacheLineLog2Bytes + 3u;
  static constexpr uint32_t kCacheLineBits = uint32_t{1} << kCacheLineLog2Bits;
  static constexpr uint32_t kCacheLineBytes = uint32_t{1} << kCacheLineLog2Bytes;
  static constexpr uint32_t kCacheLineMetaBits = 8u;
  static constexpr uint32_t kUnaryBitBlockSize = kCacheLineBits / 256;
  static constexpr uint32_t kMaxUnaryBits = (((kCacheLineBits - kCacheLineMetaBits) / 3) & ~uint32_t{kUnaryBitBlockSize - 1}) + 4;

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

  static uint32_t GetRange(uint32_t count) {
    // Range roughly 0.65 - 0.80 depending on count
    double f = 0.80 - (count * count * 2 / 100000.0);
    uint64_t corrected = std::max(count, (kCacheLineBits - kCacheLineMetaBits + 25u) / 26u);
    return static_cast<uint32_t>((corrected << 25) * f);
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
    //fprintf(stderr, "Building cache line %u\n", i);
      BuildCacheLine(data + (i * kCacheLineBytes), buckets[i].begin(), buckets[i].end(), &fp_rate_sum);
    }
  //fprintf(stderr, "Filter FP rate: %g\n", fp_rate_sum / num_cache_lines);
  }

  void BuildCacheLine(char *data_at_cache_line, std::vector<uint32_t>::iterator begin, std::vector<uint32_t>::iterator end, double *fp_rate_sum) const {
    const uint32_t count = static_cast<uint32_t>(end - begin);
    double fp_rate = std::pow(1.0 - exp(-4.0 * count / kCacheLineBits), 4.0); // bloom rate

    // TODO: pivot point closer to 5? But then we don't have min. 4 entry bits?
    if (count * 6u > (kCacheLineBits - kCacheLineMetaBits)) {
      *fp_rate_sum += fp_rate;
    //fprintf(stderr, "Cache line FP rate: %g (bloom1 %u)\n", fp_rate, count);
      BuildBloomCacheLine(data_at_cache_line, begin, end);
      return;
    }

    std::sort(begin, end);

    uint32_t range = GetRange(count);
    std::vector<uint32_t> rem_vals;
    uint32_t unary_bits = 0;
    uint32_t nibble_bwd_idx = (kCacheLineBits - kCacheLineMetaBits) / 4;

    uint32_t prev = 0;
    for (auto it = begin; it != end; ++it) {
      uint32_t val = fastrange32(*it, range);
    //fprintf(stderr, "Preparing hash %x (val %x)\n", *it, val);
      assert(val >= prev);
      uint32_t diff = (val >> 20) - (prev >> 20);
      unary_bits += (diff >> 4);
      bit_table_set_bit(data_at_cache_line, unary_bits);
      unary_bits++;
      nibble_bwd_idx--;
      if (unary_bits > kMaxUnaryBits || nibble_bwd_idx * 4 < unary_bits) {
        *fp_rate_sum += fp_rate;
      //fprintf(stderr, "Cache line FP rate: %g (bloom2 %u)\n", fp_rate, count);
        BuildBloomCacheLine(data_at_cache_line, begin, end);
        return;
      }
      or_put_nibble(data_at_cache_line, nibble_bwd_idx, diff & uint32_t{0xf});
      // Use (TBD part of) bottom 20 bits soon
      rem_vals.push_back(val & uint32_t{0xfffff});
      prev = val;
    }

    unary_bits = std::max(kMaxUnaryBits - 63 * kUnaryBitBlockSize, (unary_bits + kUnaryBitBlockSize - 1) & ~(kUnaryBitBlockSize - 1));

    uint32_t rem_bits = nibble_bwd_idx * 4 - unary_bits;

  //fprintf(stderr, "Rice count %u, unary_bits/count %g\n", count, (double)unary_bits / rem_vals.size());

    uint32_t base_rem_entry_bits = rem_bits / count;
    uint32_t entries_with_extra_bit;
    if (base_rem_entry_bits >= 20u) {
      base_rem_entry_bits = 20u;
      entries_with_extra_bit = 0;
      fp_rate = 0;
    } else {
      entries_with_extra_bit = rem_bits % count;
      fp_rate = 1.0 * entries_with_extra_bit / (range >> (20u - base_rem_entry_bits - 1u));
    }
    fp_rate += 1.0 * (count - entries_with_extra_bit) / (range >> (20u - base_rem_entry_bits));
    *fp_rate_sum += fp_rate;
  //fprintf(stderr, "Cache line FP rate: %g\n", fp_rate);

    uint32_t cur_bit_ptr = unary_bits;
    for (uint32_t rem_val : rem_vals) {
    //fprintf(stderr, "Putting %x at %u\n", rem_val >> (20 - base_rem_entry_bits), cur_bit_ptr);
      bit_table_or_put(data_at_cache_line, cur_bit_ptr, rem_val >> (20 - base_rem_entry_bits));
      cur_bit_ptr += base_rem_entry_bits;
    }
    if (base_rem_entry_bits == 20u) {
      assert(cur_bit_ptr + entries_with_extra_bit <= nibble_bwd_idx * 4);
    } else {
      assert(cur_bit_ptr + entries_with_extra_bit == nibble_bwd_idx * 4);
    }
    for (uint32_t i = 0; i < entries_with_extra_bit; ++i) {
      if ((rem_vals[i] >> (20 - base_rem_entry_bits - 1)) & uint32_t{1}) {
        bit_table_set_bit(data_at_cache_line, cur_bit_ptr + i);
      }
    }

    // Set metadata
    data_at_cache_line[kCacheLineBytes - 1] = static_cast<char>(unary_bits / kUnaryBitBlockSize - (kMaxUnaryBits / kUnaryBitBlockSize - 63));
  }

  void BuildBloomCacheLine(char *data_at_cache_line, std::vector<uint32_t>::iterator begin, std::vector<uint32_t>::iterator end) const {
    for (auto it = begin; it != end; ++it) {
      FastLocalBloomImpl::AddHashPrepared(*it, /*num_probes*/4, data_at_cache_line);
    }
    if ((data_at_cache_line[kCacheLineBytes - 1] & static_cast<char>(0xc0)) == 0) {
      // set one bit to make one of the top two bits non-zero
      data_at_cache_line[kCacheLineBytes - 1] |= static_cast<char>(0x80);
    }
    //fprintf(stderr, "Built bloom\n");
  }

  int bits_per_key_;
  std::vector<uint64_t> hash_entries_;
};

// See description in LocalHybridImpl
class LocalHybridBitsReader : public FilterBitsReader {
 public:
  LocalHybridBitsReader(const char* data, uint32_t len_bytes)
      : data_(data), num_cache_lines_(len_bytes / LocalHybridBitsBuilder::kCacheLineBytes) {}

  // No Copy allowed
  LocalHybridBitsReader(const LocalHybridBitsReader&) = delete;
  void operator=(const LocalHybridBitsReader&) = delete;

  ~LocalHybridBitsReader() override {}

  bool MayMatch(const Slice& key) override {
    const uint64_t h64 = GetSliceHash64(key);
    const uint32_t cache_line = fastrange32(Lower32of64(h64), num_cache_lines_);
    const char *data_at_cache_line = data_ + LocalHybridBitsBuilder::kCacheLineBytes * cache_line;
    return HashMayMatchPrepared(Upper32of64(h64), data_at_cache_line);
  }

  bool HashMayMatchPrepared(const uint32_t h, const char *data_at_cache_line) {
    const uint32_t meta = static_cast<uint8_t>(data_at_cache_line[LocalHybridBitsBuilder::kCacheLineBytes - 1]);
    if ((meta & 0xc0u) != 0) {
      return FastLocalBloomImpl::HashMayMatchPrepared(h, 4,
                                                      data_at_cache_line);
    }

    const uint32_t unary_bits = meta * LocalHybridBitsBuilder::kUnaryBitBlockSize + (LocalHybridBitsBuilder::kMaxUnaryBits - 63 * LocalHybridBitsBuilder::kUnaryBitBlockSize);
    const uint32_t count = nbits_popcnt(data_at_cache_line, unary_bits);

    const uint32_t val_to_find = fastrange32(h, LocalHybridBitsBuilder::GetRange(count));
  //fprintf(stderr, "Preparing hash %x (val %x) on cache line %u\n", h, val_to_find, cache_line);
    const uint32_t partial_val_to_find = val_to_find >> 20;

    const uint32_t rem_bits = LocalHybridBitsBuilder::kCacheLineBits - LocalHybridBitsBuilder::kCacheLineMetaBits - count * 4 - unary_bits;

  //fprintf(stderr, "Rice count %u, unary_bits/count %g\n", count, (double)unary_bits / count);

    uint32_t base_rem_entry_bits = rem_bits / count;
    uint32_t entries_with_extra_bit;
    if (base_rem_entry_bits >= 20u) {
      base_rem_entry_bits = 20u;
      entries_with_extra_bit = 0;
    } else {
      entries_with_extra_bit = rem_bits % count;
    }

    const uint32_t base_rem_entry_mask = (uint32_t{1} << base_rem_entry_bits) - 1u;
    const uint32_t rem_entry_to_find = (val_to_find & 0xfffffu) >> (20u - base_rem_entry_bits);

    /* A timing test hook */
    /*
    if ((entries_with_extra_bit+rem_entry_to_find+partial_val_to_find) & 1) {
      return true;
    }
    */

    uint32_t unary_cur = 0;
    const char *nibble_bwd_ptr = data_at_cache_line + (LocalHybridBitsBuilder::kCacheLineBits - LocalHybridBitsBuilder::kCacheLineMetaBits) / 8u;
    uint32_t cur_partial_val = 0;

    uint32_t i = 0;
    // Try skipping ahead by 8 nibbles at a time
    // Didn't seem to help: skipping by 16 and by 4 also
    while (i + 8 <= count && cur_partial_val + 30 < partial_val_to_find) {
      uint32_t nibbles_sum = 0;
#if defined(HAVE_POPCNT) && defined(HAVE_BMI2)
      {
        uint32_t nibbles = reinterpret_cast<const uint32_t*>(nibble_bwd_ptr)[-1];
        nibbles = (nibbles & uint32_t{0xf0f0f0f}) + ((nibbles & uint32_t{0xf0f0f0f0}) >> 4);
        nibbles = (nibbles & uint32_t{0xff00ff}) + ((nibbles & uint32_t{0xff00ff00}) >> 8);
        nibbles = (nibbles & uint32_t{0xffff}) + ((nibbles & uint32_t{0xffff0000}) >> 16);
        nibbles_sum += nibbles;
      }
#else
      for (uint32_t i = 1; i <= 8; i++) {
        nibbles_sum += get_nibble(data_at_cache_line, nibble_bwd_idx - i);
      }
#endif
      uint32_t nth1 = nth_set_bit_index1(data_at_cache_line, unary_cur, 8);
      uint32_t skip_partial_val = cur_partial_val + nibbles_sum + ((nth1 - 8) << 4);
      // NB: not easy to accept == case, because potentially earlier == cases
      if (skip_partial_val < partial_val_to_find) {
        cur_partial_val = skip_partial_val;
        i += 8;
        nibble_bwd_ptr -= 4;
        unary_cur += nth1;
      } else {
        break;
      }
    }

    /* A timing test hook */
    /*
    if ((entries_with_extra_bit+rem_entry_to_find+partial_val_to_find+unary_cur) & 1) {
      return true;
    }
    */

    uint32_t nibble_bwd_idx = (nibble_bwd_ptr - data_at_cache_line) * 2;

    for (; i < count; ++i) {
      uint32_t tz = general_tzcnt(data_at_cache_line, unary_cur);
      unary_cur += tz + 1;
      cur_partial_val += tz << 4;
      cur_partial_val += get_nibble(data_at_cache_line, --nibble_bwd_idx);
    //fprintf(stderr, "Vs. partial val %x\n", cur_partial_val);
      if (cur_partial_val > partial_val_to_find) {
        return false;
      } else if (cur_partial_val == partial_val_to_find) {
        if (base_rem_entry_bits > 0) {
          uint32_t rem_entry = bit_table_get(data_at_cache_line, unary_bits + i * base_rem_entry_bits, base_rem_entry_mask);
        //fprintf(stderr, "Looking for %x at %u, found %x\n", rem_entry_to_find, unary_bits + i * base_rem_entry_bits, rem_entry);
          if (rem_entry_to_find != rem_entry) {
            continue;
          }
        }
        if (i < entries_with_extra_bit) {
          bool extra = bit_table_get_bit(data_at_cache_line, unary_bits + count * base_rem_entry_bits + i);
          if (extra != (1u & (val_to_find >> (20u - base_rem_entry_bits - 1u)))) {
            continue;
          }
        }
        return true;
      }
    }
    return false;
  }

  virtual void MayMatch(int num_keys, Slice** keys, bool* may_match) override {
    std::array<uint32_t, MultiGetContext::MAX_BATCH_SIZE> hashes;
    std::array<uint32_t, MultiGetContext::MAX_BATCH_SIZE> byte_offsets;
    for (int i = 0; i < num_keys; ++i) {
      const uint64_t h64 = GetSliceHash64(*keys[i]);
      const uint32_t cache_line = fastrange32(Lower32of64(h64), num_cache_lines_);
      const uint32_t offset = LocalHybridBitsBuilder::kCacheLineBytes * cache_line;
      PREFETCH(data_ + offset, 0 /* rw */, 1 /* locality */);
      PREFETCH(data_ + offset + 63, 0 /* rw */, 1 /* locality */);
      byte_offsets[i] = offset;
      hashes[i] = Upper32of64(h64);
    }
    for (int i = 0; i < num_keys; ++i) {
      may_match[i] = HashMayMatchPrepared(hashes[i], data_ + byte_offsets[i]);
    }
  }
 private:
  const char* data_;
  const uint32_t num_cache_lines_;
};

using LegacyBloomImpl = LegacyLocalityBloomImpl</*ExtraRotates*/ false>;

class LegacyBloomBitsBuilder : public BuiltinFilterBitsBuilder {
 public:
  explicit LegacyBloomBitsBuilder(const int bits_per_key, const int num_probes);

  // No Copy allowed
  LegacyBloomBitsBuilder(const LegacyBloomBitsBuilder&) = delete;
  void operator=(const LegacyBloomBitsBuilder&) = delete;

  ~LegacyBloomBitsBuilder() override;

  void AddKey(const Slice& key) override;

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
  // See BloomFilterPolicy::GetFilterBitsReader for metadata
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

  LegacyBloomImpl::AddHash(h, num_lines, num_probes_, data,
                           folly::constexpr_log2(CACHE_LINE_SIZE));
}

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
    LegacyBloomImpl::PrepareHashMayMatch(
        hash, num_lines_, data_, /*out*/ &byte_offset, log2_cache_line_size_);
    return LegacyBloomImpl::HashMayMatchPrepared(
        hash, num_probes_, data_ + byte_offset, log2_cache_line_size_);
  }

  virtual void MayMatch(int num_keys, Slice** keys, bool* may_match) override {
    std::array<uint32_t, MultiGetContext::MAX_BATCH_SIZE> hashes;
    std::array<uint32_t, MultiGetContext::MAX_BATCH_SIZE> byte_offsets;
    for (int i = 0; i < num_keys; ++i) {
      hashes[i] = BloomHash(*keys[i]);
      LegacyBloomImpl::PrepareHashMayMatch(hashes[i], num_lines_, data_,
                                           /*out*/ &byte_offsets[i],
                                           log2_cache_line_size_);
    }
    for (int i = 0; i < num_keys; ++i) {
      may_match[i] = LegacyBloomImpl::HashMayMatchPrepared(
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

  // Legacy Bloom filter data:
  //             0 +-----------------------------------+
  //               | Raw Bloom filter data             |
  //               | ...                               |
  //           len +-----------------------------------+
  //               | byte for num_probes or            |
  //               |   marker for new implementations  |
  //         len+1 +-----------------------------------+
  //               | four bytes for number of cache    |
  //               |   lines                           |
  // len_with_meta +-----------------------------------+

  int8_t raw_num_probes =
      static_cast<int8_t>(contents.data()[len_with_meta - 5]);
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
    } else if (raw_num_probes == -2) {
      // Marker for local hybrid filter implementation
      return new LocalHybridBitsReader(contents.data(), len_with_meta - 5);
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

  // New Bloom filter data:
  //             0 +-----------------------------------+
  //               | Raw Bloom filter data             |
  //               | ...                               |
  //           len +-----------------------------------+
  //               | char{-1} byte -> new Bloom filter |
  //         len+1 +-----------------------------------+
  //               | byte for subimplementation        |
  //               |   0: FastLocalBloom               |
  //               |   other: reserved                 |
  //         len+2 +-----------------------------------+
  //               | byte for block_and_probes         |
  //               |   0 in top 3 bits -> 6 -> 64-byte |
  //               |   reserved:                       |
  //               |   1 in top 3 bits -> 7 -> 128-byte|
  //               |   2 in top 3 bits -> 8 -> 256-byte|
  //               |   ...                             |
  //               |   num_probes in bottom 5 bits,    |
  //               |     except 0 and 31 reserved      |
  //         len+3 +-----------------------------------+
  //               | two bytes reserved                |
  //               |   possibly for hash seed          |
  // len_with_meta +-----------------------------------+

  // Read more metadata (see above)
  char sub_impl_val = contents.data()[len_with_meta - 4];
  char block_and_probes = contents.data()[len_with_meta - 3];
  int log2_block_bytes = ((block_and_probes >> 5) & 7) + 6;

  int num_probes = (block_and_probes & 31);
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
