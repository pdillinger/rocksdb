// Copyright (c) 2011-present, Facebook, Inc. All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include <string>

#include "rocksdb/slice.h"

#include "port/port.h"
#include "util/hash.h"

#include <atomic>
#include <memory>

#ifdef HAVE_AVX2
#include <immintrin.h>
#endif

namespace rocksdb {

class Slice;
class Allocator;
class Logger;

// A Bloom filter intended only to be used in memory, never serialized in a way
// that could lead to schema incompatibility. Supports opt-in lock-free
// concurrent access.
//
// This implementation is also intended for applications generally preferring
// speed vs. maximum accuracy: roughly 0.9x BF op latency for 1.1x FP rate.
// For 1% FP rate, that means that the latency of a look-up triggered by an FP
// should be less than roughly 100x the cost of a Bloom filter op.
//
// For simplicity and performance, the current implementation requires
// num_probes to be a multiple of two and <= 12 (<= 8 when using AVX2).
//
class DynamicBloom {
 public:
  // allocator: pass allocator to bloom filter, hence trace the usage of memory
  // total_bits: fixed total bits for the bloom
  // num_probes: number of hash probes for a single key
  // hash_func:  customized hash function
  // huge_page_tlb_size:  if >0, try to allocate bloom bytes from huge page TLB
  //                      within this page size. Need to reserve huge pages for
  //                      it to be allocated, like:
  //                         sysctl -w vm.nr_hugepages=20
  //                     See linux doc Documentation/vm/hugetlbpage.txt
  explicit DynamicBloom(Allocator* allocator,
                        uint32_t total_bits,
                        uint32_t num_probes = 6,
                        size_t huge_page_tlb_size = 0,
                        Logger* logger = nullptr);

  ~DynamicBloom() {}

  // Assuming single threaded access to this function.
  void Add(const Slice& key);

  // Like Add, but may be called concurrent with other functions.
  void AddConcurrently(const Slice& key);

  // Assuming single threaded access to this function.
  void AddHash(uint32_t hash);

  // Like AddHash, but may be called concurrent with other functions.
  void AddHashConcurrently(uint32_t hash);

  // Multithreaded access to this function is OK
  bool MayContain(const Slice& key) const;

  // Multithreaded access to this function is OK
  bool MayContainHash(uint32_t hash) const;

  // Multithreaded access to this function is OK
  void Prefetch(uint32_t hash);

  static const char * const IMPL_NAME;

#ifdef HAVE_AVX2
  struct ShiftsAndSelectors {
    ShiftsAndSelectors(unsigned double_probes, unsigned offset);
    __m256i shifts_;
    __m256i selectors_;
  };
  static ShiftsAndSelectors matrix_[5][4];
  __m256i GetMask(uint64_t h, const ShiftsAndSelectors &ss) const;
#endif

 private:
  // Length of the structure, in 64-bit words. For this structure, "word"
  // will always refer to 64-bit words.
  uint32_t kLen;
  // We make the k probes in pairs, two for each 64-bit read/write. Thus,
  // this stores k/2, the number of words to double-probe.
  const uint32_t kNumDoubleProbes;
  // Raw filter data
  std::atomic<uint64_t>* data_;

  // or_func(ptr, mask) should effect *ptr |= mask with the appropriate
  // concurrency safety, working with bytes.
  template <typename OrFunc>
  void AddHashNoSimd(uint32_t hash, const OrFunc& or_func);
};

inline void DynamicBloom::Add(const Slice& key) { AddHash(BloomHash(key)); }

inline void DynamicBloom::AddConcurrently(const Slice& key) {
  AddHashConcurrently(BloomHash(key));
}

inline void DynamicBloom::AddHashConcurrently(uint32_t hash) {
  AddHashNoSimd(hash, [](std::atomic<uint64_t>* ptr, uint64_t mask) {
    // Happens-before between AddHash and MaybeContains is handled by
    // access to versions_->LastSequence(), so all we have to do here is
    // avoid races (so we don't give the compiler a license to mess up
    // our code) and not lose bits.  std::memory_order_relaxed is enough
    // for that.
    if ((mask & ptr->load(std::memory_order_relaxed)) != mask) {
      ptr->fetch_or(mask, std::memory_order_relaxed);
    }
  });
}

inline bool DynamicBloom::MayContain(const Slice& key) const {
  return (MayContainHash(BloomHash(key)));
}

#if defined(_MSC_VER)
#pragma warning(push)
// local variable is initialized but not referenced
#pragma warning(disable : 4189)
#endif
inline void DynamicBloom::Prefetch(uint32_t h32) {
  size_t a = fastrange32(kLen, h32);
  PREFETCH(data_ + a, 0, 3);
}
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

// Speed hacks in this implementation:
// * Uses fastrange instead of %
// * Minimum logic to determine first (and all) probed memory addresses.
//   (Uses constant bit-xor offsets from the starting probe address.)
// * (Major) Two probes per 64-bit memory fetch/write.
//   Code simplification / optimization: only allow even number of probes.
// * Very fast and effective (murmur-like) hash expansion/re-mixing. (At
// least on recent CPUs, integer multiplication is very cheap. Each 64-bit
// remix provides five pairs of bit addresses within a uint64_t.)
//   Code simplification / optimization: only allow up to 10 probes, from a
//   single 64-bit remix.
//
// The FP rate penalty for this implementation, vs. standard Bloom filter, is
// roughly 1.12x on top of the 1.15x penalty for a 512-bit cache-local Bloom.
// This implementation does not explicitly use the cache line size, but is
// effectively cache-local (up to 16 probes) because of the bit-xor offsetting.
//
// NB: could easily be upgraded to support a 64-bit hash and
// total_bits > 2^32 (512MB). (The latter is a bad idea without the former,
// because of false positives.)

inline bool DynamicBloom::MayContainHash(uint32_t h32) const {
  size_t a = fastrange32(kLen, h32);
  PREFETCH(data_ + a, 0, 3);
  // Expand/remix with 64-bit golden ratio
  uint64_t h = 0x9e3779b97f4a7c13ULL * h32;
#ifdef HAVE_AVX2
  // Translate to vector address
  const __m256i *ptr = reinterpret_cast<const __m256i*>(data_) + (a >> 2);
  // Like ((~*ptr) & mask) == 0)
  return _mm256_testc_si256(*ptr, GetMask(h, matrix_[kNumDoubleProbes][a & 3]));
#else
  for (unsigned i = 0;; ++i) {
    // Two bit probes per uint64_t probe
    uint64_t mask = ((uint64_t)1 << (h & 31))
                  | ((uint64_t)1 << 32 << ((h >> 32) & 31));
    uint64_t val = data_[a ^ i].load(std::memory_order_relaxed);
    if (i + 1 >= kNumDoubleProbes) {
      return (val & mask) == mask;
    } else if ((val & mask) != mask) {
      return false;
    }
    h = (h >> 5) | (h << 59);
  }
#endif
}

template <typename OrFunc>
inline void DynamicBloom::AddHashNoSimd(uint32_t h32, const OrFunc& or_func) {
  size_t a = fastrange32(kLen, h32);
  PREFETCH(data_ + a, 1, 3);
  // Expand/remix with 64-bit golden ratio
  uint64_t h = 0x9e3779b97f4a7c13ULL * h32;
  for (unsigned i = 0;; ++i) {
    // Two bit probes per uint64_t probe
    uint64_t mask = ((uint64_t)1 << (h & 31))
                  | ((uint64_t)1 << 32 << ((h >> 32) & 31));
    or_func(&data_[a ^ i], mask);
    if (i + 1 >= kNumDoubleProbes) {
      return;
    }
    h = (h >> 5) | (h << 59);
  }
}

inline void DynamicBloom::AddHash(uint32_t h32) {
#ifdef HAVE_AVX2
  size_t a = fastrange32(kLen, h32);
  PREFETCH(data_ + a, 1, 3);
  // Expand/remix with 64-bit golden ratio
  uint64_t h = 0x9e3779b97f4a7c13ULL * h32;
  // Translate to vector address
  __m256i *ptr = reinterpret_cast<__m256i*>(data_) + (a >> 2);
  // Like *ptr |= mask
  _mm256_store_si256(ptr, _mm256_or_si256(*ptr, GetMask(h, matrix_[kNumDoubleProbes][a & 3])));
#else
  AddHashNoSimd(hash, [](std::atomic<uint64_t>* ptr, uint64_t mask) {
    ptr->store(ptr->load(std::memory_order_relaxed) | mask,
               std::memory_order_relaxed);
  });
#endif
}

inline __m256i DynamicBloom::GetMask(uint64_t h, const ShiftsAndSelectors &ss) const {
  // Make four copies of h (to be split into four each hi and low 32-bits)
  __m256i hash_data = _mm256_set1_epi64x(h);

  const __m256i all_thirty_ones = _mm256_set1_epi32(31);

  hash_data = _mm256_srlv_epi32(hash_data, ss.shifts_);

  hash_data = _mm256_and_si256(hash_data, all_thirty_ones);

  // Generate mask by left-shifting the k selected 1s by those hash quantities
  return _mm256_sllv_epi32(ss.selectors_, hash_data);
}

}  // rocksdb
