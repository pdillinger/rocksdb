//  Copyright (c) 2019-present, Facebook, Inc. All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Implementation details of newer Bloom and Bloom-like filters

#pragma once
#include <stddef.h>
#include <stdint.h>

#include "rocksdb/filter_bits_config.h"
#include "util/hash.h"
#include "util/xxhash.h"

#ifdef HAVE_AVX2
#include <immintrin.h>
#endif

namespace rocksdb {

// A partial implementation of FilterBitsConfig specialized for filter
// implementations (Bloom or otherwise) that access a predefined section of
// the filter bits (generally a CPU cache line) for each key query, based
// on that key's hash.
//
// This assumes that after using a 64-bit hash to select a cache line,
// the hash is trimmed down to 32 bits for adding within the cache line. This
// permits essentially the same scalable accuracy as a 64-bit hash, while
// if peak memory use during construction ever becomes a problem, that could
// be cut (back) to about 32 bits per key, if the number of keys is known ahead
// of time so that partial hashes could be grouped by cache line.
class CacheLocalFilterBitsConfig : public FilterBitsConfig {
 public:
  virtual void PreprocessHash(uint64_t hash, size_t filter_len,
                              size_t *byte_offset,
                              uint32_t *partial_hash) const = 0;

  virtual bool LocalMayMatch(const char *data_at_offset,
                             uint32_t partial_hash) const = 0;

  void PrepareHashMayMatch(const Slice &filter,
                           uint64_t hash) const override final {
    size_t byte_offset;
    uint32_t partial_hash;
    PreprocessHash(hash, filter.size(), &byte_offset, &partial_hash);
    PREFETCH(filter.data() + byte_offset, 0 /* rw */, 1 /* locality */);
    (void)partial_hash;
  }

  bool HashMayMatch(const Slice &filter, uint64_t hash) const override final {
    size_t byte_offset;
    uint32_t partial_hash;
    PreprocessHash(hash, filter.size(), &byte_offset, &partial_hash);
    return LocalMayMatch(filter.data() + byte_offset, partial_hash);
  }

  void MayMatch(const Slice &filter, int num_keys, Slice **keys,
                bool *may_match) const override final {
    uint32_t partial_hashes[MultiGetContext::MAX_BATCH_SIZE];
    size_t byte_offsets[MultiGetContext::MAX_BATCH_SIZE];
    for (int i = 0; i < num_keys; ++i) {
      uint64_t h = Hash(*(keys[i]));
      PreprocessHash(h, filter.size(), byte_offsets + i, partial_hashes + i);
      PREFETCH(filter.data() + byte_offsets[i], 0 /* rw */, 1 /* locality */);
    }
    for (int i = 0; i < num_keys; ++i) {
      may_match[i] =
          LocalMayMatch(filter.data() + byte_offsets[i], partial_hashes[i]);
    }
  }
};

// Expects Config to extend FilterBitsConfig
// and also implement ::BuildFromHashes and ::Log2BlockSize
template <class Config>
class GenericFilterBitsBuilder : public FilterBitsBuilder {
 public:
  GenericFilterBitsBuilder(int millibits_per_key,
                           std::shared_ptr<const Config> config)
      : millibits_per_key_(millibits_per_key), config_(config) {
    assert(config_);
    assert(millibits_per_key_ >= 1000);
  }
  ~GenericFilterBitsBuilder() {}

  // No Copy allowed
  GenericFilterBitsBuilder(const GenericFilterBitsBuilder &) = delete;
  void operator=(const GenericFilterBitsBuilder &) = delete;

  void AddKey(const Slice &key) override {
    uint64_t hash = config_->Hash(key);
    if (hashes_.size() == 0 || hash != hashes_.back()) {
      hashes_.push_back(hash);
    }
  }

  Slice Finish(std::unique_ptr<const char[]> *buf) override {
    size_t len = CalculateSpace(hashes_.size());
    char *data = new char[len];
    memset(data, 0, len);
    assert(data);

    config_->BuildFromHashes(data, len, hashes_);

    buf->reset(data);
    hashes_.clear();

    return Slice(data, len);
  }

  std::shared_ptr<const FilterBitsConfig> GetConfig() const override {
    return config_;
  }

  size_t CalculateSpace(const uint32_t keys) {
    size_t block_mask = (uint32_t{1} << config_->Log2BlockSize()) - 1;
    // Get number of bytes without blocking (rounding up)
    size_t sz = static_cast<size_t>(
        (uint64_t(keys) * millibits_per_key_ + 7999) / 8000);
    // Round up to next block boundary
    return (sz + block_mask) & ~block_mask;
  }

  int CalculateNumEntry(const uint32_t bytes) override {
    uint32_t block_mask = (uint32_t{1} << config_->Log2BlockSize()) - 1;
    // Round down to number of blocks in that limit
    uint32_t block_bytes = bytes & ~block_mask;
    // Round down to entries
    return static_cast<int>(uint64_t(block_bytes) * 8000 / millibits_per_key_);
  }

 private:
  int millibits_per_key_;
  std::shared_ptr<const Config> config_;
  std::vector<uint64_t> hashes_;
};

// TODO: full description
//
// For other cache line sizes, I suggest a somewhat different implementation
// (should there be sufficient demand) since it wouldn't have to be compatible
// with Intel SIMD.
// (1) Get more than one bit index from each re-mix,
// (2) Re-mix full 64 bit hash (to minimize re-mixing), and
// (3) Use rotation in addition to multiplication for remixing
// (like murmur hash). (Using multiplication only slightly hurts accuracy
// because lower bits never depend on original upper bits.)
//
class FastLocalBloomConfig : public CacheLocalFilterBitsConfig {
 public:
  FastLocalBloomConfig(int num_probes) : num_probes_(num_probes) {}

  uint64_t Hash(const Slice &key) const override final {
    return XXH64(key.data(), key.size(), 0 /* seed */);
  }

  void PreprocessHash(uint64_t hash, size_t filter_len, size_t *byte_offset,
                      uint32_t *partial_hash) const override final {
    // fastrange depends primarily on upper bits
    *byte_offset = fastrange64(filter_len >> 6, hash) << 6;
    // pass along lower bits for matching within cache line
    *partial_hash = static_cast<uint32_t>(hash);
  }

  virtual bool LocalMayMatch(const char *data_at_offset,
                             uint32_t partial_hash) const override final {
    uint32_t h = partial_hash;
    const uint8_t *data_at_cache_line =
        reinterpret_cast<const uint8_t *>(data_at_offset);
#ifdef HAVE_AVX2
    int rem_probes = num_probes_;
    for (;;) {
      // Eight copies of hash
      __m256i v = _mm256_set1_epi32(h);

      // Powers of 32-bit golden ratio, mod 2**32
      const __m256i multipliers =
          _mm256_setr_epi32(0x9e3779b9, 0xe35e67b1, 0x734297e9, 0x35fbe861,
                            0xdeb7c719, 0x448b211, 0x3459b749, 0xab25f4c1);

      v = _mm256_mullo_epi32(v, multipliers);

      __m256i x = _mm256_srli_epi32(v, 28);

      // Option 1 (Requires AVX512 - unverified)
      //__m256i lower = _mm256_loadu_si256(reinterpret_cast<const
      //__m256i*>(data_at_cache_line));
      //__m256i upper = _mm256_loadu_si256(reinterpret_cast<const
      //__m256i*>(data_at_cache_line) + 1); x = _mm256_permutex2var_epi32(lower,
      // x, upper);
      // END Option 1
      // Option 2
      // x = _mm256_i32gather_epi32((const int *)(data_at_cache_line), x,
      // /*bytes / i32*/4);
      // END Option 2
      // Option 3
      __m256i lower = _mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(data_at_cache_line));
      __m256i upper = _mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(data_at_cache_line) + 1);
      lower = _mm256_permutevar8x32_epi32(lower, x);
      upper = _mm256_permutevar8x32_epi32(upper, x);
      __m256i junk = _mm256_srai_epi32(v, 31);
      x = _mm256_blendv_epi8(lower, upper, junk);
      // END Option 3

      __m256i k_selector = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
      k_selector = _mm256_sub_epi32(k_selector, _mm256_set1_epi32(rem_probes));
      // Keep only high bit; negative after subtract -> use/select
      k_selector = _mm256_srli_epi32(k_selector, 31);

      v = _mm256_slli_epi32(v, 4);
      v = _mm256_srli_epi32(v, 27);
      v = _mm256_sllv_epi32(k_selector, v);
      if (rem_probes <= 8) {
        // Like ((~val) & mask) == 0)
        return _mm256_testc_si256(x, v);
      } else if (!_mm256_testc_si256(x, v)) {
        return false;
      } else {
        // Need another iteration
        h *= 0xab25f4c1;
        rem_probes -= 8;
      }
    }
#else
    for (int i = 0; i < num_probes; ++i) {
      h *= 0x9e3779b9UL;
      int bitpos = h >> (32 - 9);
      if ((data_at_cache_line[bitpos >> 3] & (uint8_t{1} << (bitpos & 7))) ==
          0) {
        return false;
      }
    }
    return true;
#endif
  }

  void BuildFromHashes(char *data, size_t filter_len,
                       std::vector<uint64_t> &hashes) const {
    for (auto hash : hashes) {
      size_t byte_offset;
      uint32_t partial_hash;
      PreprocessHash(hash, filter_len, &byte_offset, &partial_hash);
      LocalAddHash(data + byte_offset, partial_hash);
    }
  }

  void LocalAddHash(char *data_at_offset, uint32_t partial_hash) const {
    uint32_t h = partial_hash;
    uint8_t *data_at_cache_line = reinterpret_cast<uint8_t *>(data_at_offset);
    for (int i = 0; i < num_probes_; ++i) {
      h *= 0x9e3779b9UL;
      int bitpos = h >> (32 - 9);
      data_at_cache_line[bitpos >> 3] |= (uint8_t{1} << (bitpos & 7));
    }
  }

  inline int Log2BlockSize() const {
    // Built for 64 byte CPU cache line.
    return 6;
  }

  static const std::string ID;

  std::string ToConfigString() const override final {
    return ID + "(" + std::to_string(num_probes_) + ")";
  }

 private:
  int num_probes_;
};

typedef GenericFilterBitsBuilder<FastLocalBloomConfig> FastLocalBloomBuilder;

}  // namespace rocksdb
