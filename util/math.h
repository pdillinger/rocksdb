//  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include <assert.h>
#include <stdint.h>
#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace rocksdb {

inline int FloorLog2(uint64_t v) {
  assert(v > 0);
#ifdef _MSC_VER
  unsigned long lz = 0;
  _BitScanReverse64(&lz, v);
  return 63 - static_cast<int>(lz);
#else
  return int{sizeof(unsigned long long)} * 8 - 1 - __builtin_clzll(v);
#endif
}

inline int FloorLog2(uint32_t v) {
  assert(v > 0);
#ifdef _MSC_VER
  unsigned long lz = 0;
  _BitScanReverse(&lz, v);
  return 31 - static_cast<int>(lz);
#else
  return int{sizeof(unsigned int)} * 8 - 1 - __builtin_clz(v);
#endif
}

}  // namespace rocksdb
