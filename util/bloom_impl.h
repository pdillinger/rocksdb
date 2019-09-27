//  Copyright (c) 2019-present, Facebook, Inc. All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Implementation details of various Bloom filter implementations used in
// RocksDB. (DynamicBloom is in a separate file for now because it
// supports concurrent write.)

#pragma once
#include <stddef.h>
#include <stdint.h>

#include "rocksdb/slice.h"
#include "util/hash.h"

#ifdef HAVE_AVX2
#include <immintrin.h>
#endif

namespace rocksdb {

// A legacy Bloom filter implementation with no locality of probes (slow).
// It uses double hashing to generate a sequence of hash values.
// Asymptotic analysis is in [Kirsch,Mitzenmacher 2006], but known to have
// subtle accuracy flaws for practical sizes [Dillinger,Manolios 2004].
//
// DO NOT REUSE - faster and more predictably accurate implementations
// are available at
// https://github.com/pdillinger/wormhashing/blob/master/bloom_simulation_tests/foo.cc
// See e.g. RocksDB DynamicBloom.
//
class LegacyNoLocalityBloomImpl {
public:
  static inline void AddHash(uint32_t h, uint32_t total_bits,
                             int num_probes, char *data) {
    const uint32_t delta = (h >> 17) | (h << 15);  // Rotate right 17 bits
    for (int i = 0; i < num_probes; i++) {
      const uint32_t bitpos = h % total_bits;
      data[bitpos/8] |= (1 << (bitpos % 8));
      h += delta;
    }
  }

  static inline bool HashMayMatch(uint32_t h, uint32_t total_bits,
                                  int num_probes, const char *data) {
    const uint32_t delta = (h >> 17) | (h << 15);  // Rotate right 17 bits
    for (int i = 0; i < num_probes; i++) {
      const uint32_t bitpos = h % total_bits;
      if ((data[bitpos/8] & (1 << (bitpos % 8))) == 0) {
        return false;
      }
      h += delta;
    }
    return true;
  }
};


// A legacy Bloom filter implementation with probes local to a single
// cache line (fast). Because SST files might be transported between
// platforms, the cache line size is a parameter rather than hard coded.
// (But if specified as a constant parameter, an optimizing compiler
// should take advantage of that.)
//
// When ExtraRotates is false, this implementation is notably deficient in
// accuracy. Specifically, it uses double hashing with a 1/512 chance of the
// increment being zero (when cache line size is 512 bits). Thus, there's a
// 1/512 chance of probing only one index, which we'd expect to incur about
// a 1/2 * 1/512 or absolute 0.1% FP rate penalty. More detail at
// https://github.com/facebook/rocksdb/issues/4120
//
// DO NOT REUSE - faster and more predictably accurate implementations
// are available at
// https://github.com/pdillinger/wormhashing/blob/master/bloom_simulation_tests/foo.cc
// See e.g. RocksDB DynamicBloom.
//
template <bool ExtraRotates>
class LegacyLocalityBloomImpl {
private:
  static inline uint32_t GetLine(uint32_t h, uint32_t num_lines) {
    uint32_t offset_h = ExtraRotates ? (h >> 11) | (h << 21) : h;
    return offset_h % num_lines;
  }
public:
  static inline void AddHash(uint32_t h, uint32_t num_lines,
                             int num_probes, char *data,
                             int log2_cache_line_bytes) {
    const int log2_cache_line_bits = log2_cache_line_bytes + 3;

    char *data_at_offset =
        data + (GetLine(h, num_lines) << log2_cache_line_bytes);
    const uint32_t delta = (h >> 17) | (h << 15);
    for (int i = 0; i < num_probes; ++i) {
      // Mask to bit-within-cache-line address
      const uint32_t bitpos = h & ((1 << log2_cache_line_bits) - 1);
      data_at_offset[bitpos / 8] |= (1 << (bitpos % 8));
      if (ExtraRotates) {
        h = (h >> log2_cache_line_bits) | (h << (32 - log2_cache_line_bits));
      }
      h += delta;
    }
  }

  static inline void PrepareHashMayMatch(uint32_t h, uint32_t num_lines,
                                         const char *data,
                                         uint32_t /*out*/*byte_offset,
                                         int log2_cache_line_bytes) {
    uint32_t b = GetLine(h, num_lines) << log2_cache_line_bytes;
    PREFETCH(data + b, 0 /* rw */, 1 /* locality */);
    PREFETCH(data + b + ((1 << log2_cache_line_bytes) - 1),
             0 /* rw */, 1 /* locality */);
    *byte_offset = b;
  }

  static inline bool HashMayMatch(uint32_t h, uint32_t num_lines,
                                  int num_probes, const char *data,
                                  int log2_cache_line_bytes) {
    uint32_t b = GetLine(h, num_lines) << log2_cache_line_bytes;
    return HashMayMatchPrepared(h, num_probes,
                                data + b, log2_cache_line_bytes);
  }

  static inline bool HashMayMatchPrepared(uint32_t h, int num_probes,
                                          const char *data_at_offset,
                                          int log2_cache_line_bytes) {
    const int log2_cache_line_bits = log2_cache_line_bytes + 3;

    const uint32_t delta = (h >> 17) | (h << 15);
    for (int i = 0; i < num_probes; ++i) {
      // Mask to bit-within-cache-line address
      const uint32_t bitpos = h & ((1 << log2_cache_line_bits) - 1);
      if (((data_at_offset[bitpos / 8]) & (1 << (bitpos % 8))) == 0) {
        return false;
      }
      if (ExtraRotates) {
        h = (h >> log2_cache_line_bits) | (h << (32 - log2_cache_line_bits));
      }
      h += delta;
    }
    return true;
  }
};

class SemiLocalBloomImpl {
public:
  static inline void AddHash(uint32_t h32, uint32_t len_bytes,
                             int num_probes, char *data) {
    uint32_t a = fastrange32(len_bytes, h32);
    uint64_t h = h32;
    for (int i = 0;;) {
      h *= 0x9e3779b97f4a7c13ULL;
      for (int j = 0; j < 7; ++j) {
        data[a] |= (1 << (h & 7));
        ++i;
        if (i >= num_probes) {
          return;
        }
        a += ((h >> 3) & 63);
        if (a >= len_bytes) { a -= len_bytes; }
        h = (h >> 9) | (h << 55);
      }
    }
  }

  static inline void PrepareHashMayMatch(uint32_t h32, uint32_t len_bytes,
                                         const char *data,
                                         uint32_t /*out*/*byte_offset) {
    uint32_t a = fastrange32(len_bytes, h32);
    PREFETCH(data + a, 0 /* rw */, 1 /* locality */);
    *byte_offset = a;
  }

  static inline bool HashMayMatch(uint32_t h32, uint32_t len_bytes,
                                  int num_probes, const char *data) {
    uint32_t a = fastrange32(len_bytes, h32);
    return HashMayMatchPrepared(h32, len_bytes, num_probes,
                                data, a);
  }

  static inline bool HashMayMatchPrepared(uint32_t h32, uint32_t len_bytes,
                                          int num_probes, const char *data,
                                          uint32_t byte_offset) {
    uint32_t a = byte_offset;
    uint64_t h = h32;
    for (int i = 0;;) {
      h *= 0x9e3779b97f4a7c13ULL;
      for (int j = 0; j < 7; ++j) {
        if ((data[a] & (1 << (h & 7))) == 0) {
          return false;
        }
        ++i;
        if (i >= num_probes) {
          return true;
        }
        a += ((h >> 3) & 63);
        if (a >= len_bytes) { a -= len_bytes; }
        h = (h >> 9) | (h << 55);
      }
    }
  }
};

class FastLocalBloomImpl {
public:
  static inline void AddHash(uint32_t h, uint32_t len_bytes,
                             int num_probes, char *data) {
    uint32_t bytes_to_cache_line = fastrange32(len_bytes >> 6, h) << 6;
    char *data_at_cache_line = data + bytes_to_cache_line;
    PREFETCH(data_at_cache_line, 1 /* rw */, 1 /* locality */);
    AddHashPrepared(h, num_probes, data_at_cache_line);
  }

  static inline void AddHashPrepared(uint32_t h, int num_probes,
                                     char *data_at_cache_line) {
    for (int i = 0; i < num_probes; ++i) {
      h *= 0x9e3779b9UL;
      int bitpos = h >> (32 - 9);
      data_at_cache_line[bitpos >> 3] |= (uint8_t(1) << (bitpos & 7));
    }
  }

  static inline void PrepareHashMayMatch(uint32_t h, uint32_t len_bytes,
                                         const char *data,
                                         uint32_t /*out*/*byte_offset) {
    uint32_t bytes_to_cache_line = fastrange32(len_bytes >> 6, h) << 6;
    PREFETCH(data + bytes_to_cache_line, 0 /* rw */, 1 /* locality */);
    *byte_offset = bytes_to_cache_line;
  }

  static inline bool HashMayMatch(uint32_t h, uint32_t len_bytes,
                                  int num_probes, const char *data) {
    uint32_t bytes_to_cache_line = fastrange32(len_bytes >> 6, h) << 6;
    return HashMayMatchPrepared(h, num_probes, data + bytes_to_cache_line);
  }

  static inline bool HashMayMatchPrepared(uint32_t h, int num_probes,
                                          const char *data_at_cache_line) {
#ifdef HAVE_AVX2
    for (;;) {
      // Eight copies of hash
      __m256i v = _mm256_set1_epi32(h);

      // Powers of 32-bit golden ratio, mod 2**32
      const __m256i multipliers =
          _mm256_setr_epi32(0x9e3779b9,
                            0xe35e67b1,
                            0x734297e9,
                            0x35fbe861,
                            0xdeb7c719,
                            0x448b211,
                            0x3459b749,
                            0xab25f4c1);

      v = _mm256_mullo_epi32(v, multipliers);

      __m256i x = _mm256_srli_epi32(v, 28);

      // Option 1 (Requires AVX512 - unverified)
      //__m256i lower = reinterpret_cast<__m256i*>(table + a)[0];
      //__m256i upper = reinterpret_cast<__m256i*>(table + a)[1];
      //x = _mm256_permutex2var_epi32(lower, x, upper);
      // END Option 1
      // Option 2
      //x = _mm256_i32gather_epi32((const int *)(table + a), x, /*bytes / i32*/4);
      // END Option 2
      // Option 3
      // Potentially unaligned:
      const __m256i *mm_data =
          reinterpret_cast<const __m256i*>(data_at_cache_line);
      __m256i lower = _mm256_loadu_si256(mm_data);
      __m256i upper = _mm256_loadu_si256(mm_data + 1);
      lower = _mm256_permutevar8x32_epi32(lower, x);
      upper = _mm256_permutevar8x32_epi32(upper, x);
      __m256i junk = _mm256_srai_epi32(v, 31);
      x = _mm256_blendv_epi8(lower, upper, junk);
      // END Option 3

      __m256i k_selector = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
      k_selector = _mm256_sub_epi32(k_selector, _mm256_set1_epi32(num_probes));
      // Keep only high bit; negative after subtract -> use/select
      k_selector = _mm256_srli_epi32(k_selector, 31);

      v = _mm256_slli_epi32(v, 4);
      v = _mm256_srli_epi32(v, 27);
      v = _mm256_sllv_epi32(k_selector, v);
      if (num_probes <= 8) {
        // Like ((~val) & mask) == 0)
        return _mm256_testc_si256(x, v);
      } else if (!_mm256_testc_si256(x, v)) {
        return false;
      } else {
        // Need another iteration
        h *= 0xab25f4c1;
        num_probes -= 8;
      }
    }
#else
    for (int i = 0; i < num_probes; ++i) {
      h *= 0x9e3779b9UL;
      int bitpos = h >> (32 - 9);
      if ((data_at_cache_line[bitpos >> 3] & (char(1) << (bitpos & 7))) == 0) {
        return false;
      }
    }
    return true;
#endif
  }
};

}  // namespace rocksdb
