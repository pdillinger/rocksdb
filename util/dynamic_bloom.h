// Copyright (c) 2011-present, Facebook, Inc. All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include <string>

#include "rocksdb/slice.h"

#include "memory/allocator.h"
#include "port/port.h"
#include "util/hash.h"

#include <atomic>
#include <memory>

namespace rocksdb {

class Slice;
class Allocator;
class Logger;


// A Bloom filter intended only to be used in memory, never serialized in a way
// that could lead to schema incompatibility. Supports opt-in lock-free
// concurrent access.
//
// Implementation details are compile-time swappable to make DynamicBloom
// type below.
//
template <class Impl>
class DynamicBloomTemplate : public Impl {
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
  explicit DynamicBloomTemplate(Allocator* allocator,
                                uint32_t total_bits,
                                uint32_t num_probes = 6,
                                size_t huge_page_tlb_size = 0,
                                Logger* logger = nullptr);

  // Assuming single threaded access to this function.
  void Add(const Slice& key);

  // Like Add, but may be called concurrent with other functions.
  void AddConcurrently(const Slice& key);

  // Multithreaded access to this function is OK
  bool MayContain(const Slice& key) const;

// Inherited public:
  // Assuming single threaded access to this function.
  //void AddHash(uint32_t hash);

  // Like AddHash, but may be called concurrent with other functions.
  //void AddHashConcurrently(uint32_t hash);

  // Multithreaded access to this function is OK
  //bool MayContainHash(uint32_t hash) const;

  // Multithreaded access to this function is OK
  //void Prefetch(uint32_t hash);

// Inherited protected:
  //void SetNumProbes(uint32_t num_probes);

  //uint32_t GetBlockBytes();

  //void SetData(char *data, size_t size);
};

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
// num_probes to be a multiple of two and <= 10.
//
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
//
class DynamicBloomImpl {
public:
  DynamicBloomImpl() : num_double_probes_(0), len_(0), data_(nullptr) {}
  ~DynamicBloomImpl() {}

  void Prefetch(uint32_t hash);

  bool MayContainHash(uint32_t hash) const;

  void AddHash(uint32_t hash);

  void AddHashConcurrently(uint32_t hash);

protected:
  void SetNumProbes(uint32_t num_probes);

  uint32_t GetBlockBytes();

  void SetData(char *data, size_t size);

 private:
  // We make the k probes in pairs, two for each 64-bit read/write. Thus,
  // this stores k/2, the number of words to double-probe.
  uint32_t num_double_probes_;

  // Length of the structure, in 64-bit words. For this structure, "word"
  // will always refer to 64-bit words.
  uint32_t len_;

  // Raw filter data
  std::atomic<uint64_t>* data_;

  // or_func(ptr, mask) should effect *ptr |= mask with the appropriate
  // concurrency safety.
  template <typename OrFunc>
  inline void AddHash(uint32_t hash, const OrFunc& or_func);
};


typedef DynamicBloomTemplate<DynamicBloomImpl> DynamicBloom;


// Implementation details that are templated or should be inlined:

template <class Impl>
DynamicBloomTemplate<Impl>::DynamicBloomTemplate(Allocator* allocator,
                                           uint32_t total_bits,
                                           uint32_t num_probes,
                                           size_t huge_page_tlb_size,
                                           Logger* logger) {
  Impl::SetNumProbes(num_probes);
  // Determine how much to round off + align by
  uint32_t block_bytes = Impl::GetBlockBytes();
  uint32_t block_bits = block_bytes * 8;
  uint32_t blocks = (total_bits + block_bits - 1) / block_bits;
  uint32_t sz = blocks * block_bytes;

  // Padding to correct for allocation not originally aligned on block_bytes
  // boundary
  sz += block_bytes - 1;
  assert(allocator);

  char* raw = allocator->AllocateAligned(sz, huge_page_tlb_size, logger);
  memset(raw, 0, sz);
  auto block_offset = reinterpret_cast<uintptr_t>(raw) % block_bytes;
  if (block_offset > 0) {
    // Align on block_bytes boundary
    raw += block_bytes - block_offset;
  }
  Impl::SetData(raw, blocks * block_bytes);
}

template <class Impl>
inline void DynamicBloomTemplate<Impl>::Add(const Slice& key) {
  Impl::AddHash(BloomHash(key));
}

template <class Impl>
inline void DynamicBloomTemplate<Impl>::AddConcurrently(const Slice& key) {
  Impl::AddHashConcurrently(BloomHash(key));
}

template <class Impl>
inline bool DynamicBloomTemplate<Impl>::MayContain(const Slice& key) const {
  return Impl::MayContainHash(BloomHash(key));
}


inline void DynamicBloomImpl::AddHash(uint32_t hash) {
  AddHash(hash, [](std::atomic<uint64_t>* ptr, uint64_t mask) {
    ptr->store(ptr->load(std::memory_order_relaxed) | mask,
               std::memory_order_relaxed);
  });
}

inline void DynamicBloomImpl::AddHashConcurrently(uint32_t hash) {
  AddHash(hash, [](std::atomic<uint64_t>* ptr, uint64_t mask) {
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

#if defined(_MSC_VER)
#pragma warning(push)
// local variable is initialized but not referenced
#pragma warning(disable : 4189)
#endif
inline void DynamicBloomImpl::Prefetch(uint32_t h32) {
  size_t a = fastrange32(len_, h32);
  PREFETCH(data_ + a, 0, 3);
}
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

inline bool DynamicBloomImpl::MayContainHash(uint32_t h32) const {
  size_t a = fastrange32(len_, h32);
  PREFETCH(data_ + a, 0, 3);
  // Expand/remix with 64-bit golden ratio
  uint64_t h = 0x9e3779b97f4a7c13ULL * h32;
  for (unsigned i = 0;; ++i) {
    // Two bit probes per uint64_t probe
    uint64_t mask = ((uint64_t)1 << (h & 63))
                  | ((uint64_t)1 << ((h >> 6) & 63));
    uint64_t val = data_[a ^ i].load(std::memory_order_relaxed);
    if (i + 1 >= num_double_probes_) {
      return (val & mask) == mask;
    } else if ((val & mask) != mask) {
      return false;
    }
    h = (h >> 12) | (h << 52);
  }
}

template <typename OrFunc>
inline void DynamicBloomImpl::AddHash(uint32_t h32, const OrFunc& or_func) {
  size_t a = fastrange32(len_, h32);
  PREFETCH(data_ + a, 0, 3);
  // Expand/remix with 64-bit golden ratio
  uint64_t h = 0x9e3779b97f4a7c13ULL * h32;
  for (unsigned i = 0;; ++i) {
    // Two bit probes per uint64_t probe
    uint64_t mask = ((uint64_t)1 << (h & 63))
                  | ((uint64_t)1 << ((h >> 6) & 63));
    or_func(&data_[a ^ i], mask);
    if (i + 1 >= num_double_probes_) {
      return;
    }
    h = (h >> 12) | (h << 52);
  }
}

}  // rocksdb
