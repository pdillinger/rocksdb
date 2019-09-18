// Copyright (c) 2011-present, Facebook, Inc. All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "dynamic_bloom.h"

#include <algorithm>

#include "memory/allocator.h"
#include "port/port.h"
#include "rocksdb/slice.h"
#include "util/hash.h"

namespace rocksdb {

namespace {

#ifndef HAVE_AVX2
uint32_t roundUpToPow2(uint32_t x) {
  uint32_t rv = 1;
  while (rv < x) {
    rv <<= 1;
  }
  return rv;
}
#endif
}

DynamicBloom::DynamicBloom(Allocator* allocator, uint32_t total_bits,
                           uint32_t num_probes,
                           size_t huge_page_tlb_size, Logger* logger)
    // Round down, except round up with 1
    : kNumDoubleProbes((num_probes + (num_probes == 1)) / 2) {
  assert(num_probes % 2 == 0); // limitation of current implementation(s)
#ifdef HAVE_AVX2
  assert(num_probes <= 8); // limitation of current SIMD implementation
#else
  assert(num_probes <= 12); // limitation of current non-SIMD implementation
#endif
  assert(kNumDoubleProbes > 0);

#ifdef HAVE_AVX2
  uint32_t block_bytes = sizeof(__m256i);
#else
  // Determine how much to round off + align by so that x ^ i (that's xor) is
  // a valid u64 index if x is a valid u64 index and 0 <= i < kNumDoubleProbes.
  uint32_t block_bytes = /*bytes/u64*/ 8 *
                         /*u64s*/ std::max(1U, roundUpToPow2(kNumDoubleProbes));
#endif
  uint32_t block_bits = block_bytes * 8;
  uint32_t blocks = (total_bits + block_bits - 1) / block_bits;
  uint32_t sz = blocks * block_bytes;
  kLen = sz / /*bytes/u64*/8;
  assert(kLen > 0);
#ifndef NDEBUG
  for (uint32_t i = 0; i < kNumDoubleProbes; ++i) {
    // Ensure probes starting at last word are in range
    assert(((kLen - 1) ^ i) < kLen);
  }
#endif

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
  static_assert(sizeof(std::atomic<uint64_t>) == sizeof(uint64_t),
                "Expecting zero-space-overhead atomic");
  data_ = reinterpret_cast<std::atomic<uint64_t>*>(raw);

#ifdef HAVE_AVX2
  shift_and_selector_matrix_ = _mm256_set1_epi32(0);
  uint64_t *matrix = reinterpret_cast<uint64_t*>(&shift_and_selector_matrix_);
  for (unsigned i = 0; i < /* offset upper bound */ 4; ++i) {
    for (unsigned j = 0; j < kNumDoubleProbes; ++j) {
      matrix[i ^ j] |= ((32 + j * 5) << (i * 8));
    }
  }
  for (unsigned i = 0; i < /* offset upper bound */ 4; ++i) {
    matrix[i] |= matrix[i] << 32;
  }
#endif
}

const char * const DynamicBloom::IMPL_NAME =
#ifdef HAVE_AVX2
    "Avx2";
#else
    "NoSimd";
#endif

}  // rocksdb
