//  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include "util/math.h"

namespace ROCKSDB_NAMESPACE {

#undef HAVE_UINT128_EXTENSION

#ifdef HAVE_UINT128_EXTENSION
using Unsigned128 = __uint128_t;
#else
struct Unsigned128 {
  uint64_t lo;
  uint64_t hi;

  inline Unsigned128() {
    static_assert(sizeof(Unsigned128) == 2 * sizeof(uint64_t),
                  "unexpected overhead in representation");
    lo = 0;
    hi = 0;
  }

  inline Unsigned128(uint64_t lower) {
    lo = lower;
    hi = 0;
  }

  inline Unsigned128(uint64_t lower, uint64_t upper) {
    lo = lower;
    hi = upper;
  }
};

inline Unsigned128 operator<<(const Unsigned128& lhs, unsigned shift) {
  shift &= 127;
  Unsigned128 rv;
  if (shift >= 64) {
    rv.lo = 0;
    rv.hi = lhs.lo << (shift & 63);
  } else {
    uint64_t tmp = lhs.lo;
    rv.lo = tmp << shift;
    // Ensure shift==0 shifts away everything
    tmp = tmp >> 1 >> (63 - shift);
    rv.hi = tmp | (lhs.hi << shift);
  }
  return rv;
}

inline Unsigned128& operator<<=(Unsigned128& lhs, unsigned shift) {
  lhs = lhs << shift;
  return lhs;
}

inline Unsigned128 operator>>(const Unsigned128& lhs, unsigned shift) {
  shift &= 127;
  Unsigned128 rv;
  if (shift >= 64) {
    rv.hi = 0;
    rv.lo = lhs.hi >> (shift & 63);
  } else {
    uint64_t tmp = lhs.hi;
    rv.hi = tmp >> shift;
    // Ensure shift==0 shifts away everything
    tmp = tmp << 1 << (63 - shift);
    rv.lo = tmp | (lhs.lo >> shift);
  }
  return rv;
}

inline Unsigned128& operator>>=(Unsigned128& lhs, unsigned shift) {
  lhs = lhs >> shift;
  return lhs;
}

inline Unsigned128 operator&(const Unsigned128& lhs, const Unsigned128& rhs) {
  return Unsigned128(lhs.lo & rhs.lo, lhs.hi & rhs.hi);
}

inline Unsigned128& operator&=(Unsigned128& lhs, const Unsigned128& rhs) {
  lhs = lhs & rhs;
  return lhs;
}

inline Unsigned128 operator|(const Unsigned128& lhs, const Unsigned128& rhs) {
  return Unsigned128(lhs.lo | rhs.lo, lhs.hi | rhs.hi);
}

inline Unsigned128& operator|=(Unsigned128& lhs, const Unsigned128& rhs) {
  lhs = lhs | rhs;
  return lhs;
}

inline Unsigned128 operator^(const Unsigned128& lhs, const Unsigned128& rhs) {
  return Unsigned128(lhs.lo ^ rhs.lo, lhs.hi ^ rhs.hi);
}

inline Unsigned128& operator^=(Unsigned128& lhs, const Unsigned128& rhs) {
  lhs = lhs ^ rhs;
  return lhs;
}

inline Unsigned128 operator~(const Unsigned128& v) {
  return Unsigned128(~v.lo, ~v.hi);
}

inline bool operator==(const Unsigned128& lhs, const Unsigned128& rhs) {
  return lhs.lo == rhs.lo && lhs.hi == rhs.hi;
}
#endif

inline uint64_t Lower64Of128(Unsigned128 v) {
#ifdef HAVE_UINT128_EXTENSION
  return static_cast<uint64_t>(v);
#else
  return v.lo;
#endif
}

inline uint64_t Upper64Of128(Unsigned128 v) {
#ifdef HAVE_UINT128_EXTENSION
  return static_cast<uint64_t>(v >> 64);
#else
  return v.hi;
#endif
}

inline Unsigned128 Multiply64to128(uint64_t a, uint64_t b) {
#ifdef HAVE_UINT128_EXTENSION
  return Unsigned128(a) * Unsigned128(b);
#else
  // Full decomposition
  // NOTE: GCC seems to fully understand this code as 64-bit x 64-bit
  // -> 128-bit multiplication and optimize it appropriately.
  uint64_t tmp = uint64_t{b & 0xffffFFFF} * uint64_t{a & 0xffffFFFF};
  uint64_t lower = tmp & 0xffffFFFF;
  tmp >>= 32;
  tmp += uint64_t{b & 0xffffFFFF} * uint64_t{a >> 32};
  // Avoid overflow: first add lower 32 of tmp2, and later upper 32
  uint64_t tmp2 = uint64_t{b >> 32} * uint64_t{a & 0xffffFFFF};
  tmp += static_cast<uint32_t>(tmp2);
  lower |= tmp << 32;
  tmp >>= 32;
  tmp += (tmp2 >> 32);
  tmp += uint64_t{b >> 32} * uint64_t{a >> 32};
  return Unsigned128(lower, tmp);
#endif
}

template <>
inline int BitParity(Unsigned128 v) {
  return BitParity(Lower64Of128(v)) ^ BitParity(Upper64Of128(v));
}

}  // namespace ROCKSDB_NAMESPACE
