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

namespace ROCKSDB_NAMESPACE {

template <typename T>
inline int FloorLog2(T v) {
  static_assert(std::is_integral<T>::value, "non-integral type");
  assert(v > 0);
#ifdef _MSC_VER
  static_assert(sizeof(T) >= sizeof(uint32_t), "type too small");
  static_assert(sizeof(T) <= sizeof(uint64_t), "type too big");
  unsigned long lz = 0;
  if (sizeof(T) == sizeof(uint32_t)) {
    _BitScanReverse(&lz, static_cast<uint32_t>(v));
  } else {
    _BitScanReverse64(&lz, static_cast<uint64_t>(v));
  }
  return 63 - static_cast<int>(lz);
#else
  static_assert(sizeof(T) >= sizeof(unsigned int), "type too small");
  static_assert(sizeof(T) <= sizeof(unsigned long long), "type too big");
  if (sizeof(T) == sizeof(unsigned int)) {
    int lz = __builtin_clz(static_cast<unsigned int>(v));
    return int{sizeof(unsigned int)} * 8 - 1 - lz;
  } else {
    int lz = __builtin_clzll(static_cast<unsigned long long>(v));
    return int{sizeof(unsigned long long)} * 8 - 1 - lz;
  }
#endif
}

template <typename T>
inline int CountTrailingZeroBits(T v) {
  static_assert(std::is_integral<T>::value, "non-integral type");
  assert(v > 0);
#ifdef _MSC_VER
  static_assert(sizeof(T) <= sizeof(uint64_t), "type too big");
  unsigned long tz = 0;
  if (sizeof(T) <= sizeof(uint32_t)) {
    _BitScanForward(&tz, static_cast<uint32_t>(v));
  } else {
    _BitScanForward64(&tz, static_cast<uint64_t>(v));
  }
  return static_cast<int>(tz);
#else
  static_assert(sizeof(T) <= sizeof(unsigned long long), "type too big");
  if (sizeof(T) <= sizeof(unsigned int)) {
    return __builtin_ctz(static_cast<unsigned int>(v));
  } else if (sizeof(T) <= sizeof(unsigned long)) {
    return __builtin_ctzl(static_cast<unsigned long>(v));
  } else {
    return __builtin_ctzll(static_cast<unsigned long>(v));
  }
#endif
}

template <typename T>
inline int BitsSetToOne(T v) {
  static_assert(std::is_integral<T>::value, "non-integral type");
#ifdef _MSC_VER
  static_assert(sizeof(T) <= sizeof(uint64_t), "type too big");
  if (sizeof(T) > sizeof(uint32_t)) {
    return static_cast<int>(__popcnt64(static_cast<uint64_t>(v)));
  } else {
    return static_cast<int>(__popcnt(static_cast<uint32_t>(v)));
  }
#else
  static_assert(sizeof(T) <= sizeof(unsigned long long), "type too big");
  if (sizeof(T) > sizeof(unsigned long)) {
    return __builtin_popcountll(static_cast<unsigned long long>(v));
  } else if (sizeof(T) > sizeof(unsigned int)) {
    return __builtin_popcountl(static_cast<unsigned long>(v));
  } else {
    return __builtin_popcount(static_cast<unsigned int>(v));
  }
#endif
}

template <typename T>
inline int BitParity(T v) {
  static_assert(std::is_integral<T>::value, "non-integral type");
#ifdef _MSC_VER
  return BitsSetToOne(v) & 1;
#else
  static_assert(sizeof(T) <= sizeof(unsigned long long), "type too big");
  if (sizeof(T) > sizeof(unsigned long)) {
    return __builtin_parityll(static_cast<unsigned long long>(v));
  } else if (sizeof(T) > sizeof(unsigned int)) {
    return __builtin_parityl(static_cast<unsigned long>(v));
  } else {
    return __builtin_parity(static_cast<unsigned int>(v));
  }
#endif
}


}  // namespace ROCKSDB_NAMESPACE
