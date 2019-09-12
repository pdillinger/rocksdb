// Copyright (c) 2011-present, Facebook, Inc. All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "dynamic_bloom.h"

namespace rocksdb {

void DynamicBloomImpl::SetNumProbes(uint32_t num_probes) {
  assert(num_probes % 2 == 0); // limitation of current implementation
  assert(num_probes <= 10); // limitation of current implementation
  num_double_probes_ = (num_probes + (num_probes == 1)) / 2;
  assert(num_double_probes_ > 0);
  assert(num_double_probes_ <= 5);
}

uint32_t DynamicBloomImpl::GetBlockBytes() {
  uint32_t nextPowTwo = 1;
  while (nextPowTwo < num_double_probes_) { nextPowTwo <<= 1; }
  // How much to round off + align by so that x ^ i (that's xor) is a valid
  // u64 index if x is a valid u64 index and 0 <= i < num_double_probes_.
  return /*bytes/u64*/ 8 * /*u64s*/nextPowTwo;
}

void DynamicBloomImpl::SetData(char *data, size_t size) {
  static_assert(sizeof(std::atomic<uint64_t>) == sizeof(uint64_t),
                "Expecting zero-space-overhead atomic");
  data_ = reinterpret_cast<std::atomic<uint64_t>*>(data);
  // Ensure aligned size and pointer
  assert(size % GetBlockBytes() == 0);
  assert(reinterpret_cast<uintptr_t>(data) % GetBlockBytes() == 0);
  len_ = size / /*bytes/u64*/8;
}

}  // rocksdb
