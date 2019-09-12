// Copyright (c) 2011-present, Facebook, Inc. All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "dynamic_bloom.h"

namespace rocksdb {

void DynamicBloomImplNoSimd::SetNumProbes(uint32_t num_probes) {
  assert(num_probes % 2 == 0); // limitation of current implementation
  assert(num_probes <= 10); // limitation of current implementation
  num_double_probes_ = (num_probes + (num_probes == 1)) / 2;
  assert(num_double_probes_ > 0);
  assert(num_double_probes_ <= 5);
}

uint32_t DynamicBloomImplNoSimd::GetBlockBytes() const {
  uint32_t nextPowTwo = 1;
  while (nextPowTwo < num_double_probes_) { nextPowTwo <<= 1; }
  // How much to round off + align by so that x ^ i (that's xor) is a valid
  // u64 index if x is a valid u64 index and 0 <= i < num_double_probes_.
  return /*bytes/u64*/ 8 * /*u64s*/nextPowTwo;
}

void DynamicBloomImplNoSimd::SetData(char *data, size_t size) {
  static_assert(sizeof(std::atomic<uint64_t>) == sizeof(uint64_t),
                "Expecting zero-space-overhead atomic");
  data_ = reinterpret_cast<std::atomic<uint64_t>*>(data);
  // Ensure aligned size and pointer
  assert(size % GetBlockBytes() == 0);
  assert(reinterpret_cast<uintptr_t>(data) % GetBlockBytes() == 0);
  len_ = size / /*bytes/u64*/8;
}

const char * const DynamicBloomImplNoSimd::IMPL_NAME =
    "DynamicBloomImplNoSimd";

#ifdef HAVE_AVX2

void DynamicBloomImplAvx2::SetNumProbes(uint32_t k) {
  assert(k > 0); // limitation of this implementation
  assert(k <= 8); // limitation of this implementation

  k_selector_ = _mm256_setr_epi32(k >= 1, k >= 2, k >= 3, k >= 4,
                                  k >= 5, k >= 6, k >= 7, k >= 8);
}

uint32_t DynamicBloomImplAvx2::GetBlockBytes() const {
  return sizeof(__m256i);
}

void DynamicBloomImplAvx2::SetData(char *data, size_t size) {
  data_ = reinterpret_cast<__m256i*>(data);
  // Ensure aligned size and pointer
  assert(size % sizeof(__m256i) == 0);
  assert(reinterpret_cast<uintptr_t>(data) % sizeof(__m256i) == 0);
  len_ = size / sizeof(__m256i);
}

const char * const DynamicBloomImplAvx2::IMPL_NAME =
    "DynamicBloomImplAvx2";

#endif // HAVE_AVX2

}  // rocksdb
