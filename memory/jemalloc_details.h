//  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include "util/math.h"

namespace rocksdb {

inline size_t RoundUpToJemallocSize(size_t len) {
  if (len <= 128) {
    if (len <= 8) {
      // TODO: also zero?
      return 8;
    } else {
      // Every 16
      return (len + 15) & ~15;
    }
  } else {
    // Powers of two times either 4, 5, 6, or 7.
    // How: Add one to 2nd bit below highest set bit, and keep only those
    // top three bits. Except we need to subtract 1 first to get back the
    // original top three bits when len is exactly an allocation size.
    const size_t len1 = len - 1;
    const int bit_index = FloorLog2(len1) - 2;
    const size_t add_bit = size_t{1} << bit_index;
    return (len1 + add_bit) & ~(add_bit - 1);
  }
}

inline size_t RoundDownToJemallocSize(size_t len) {
  if (len < 128) {
    if (len < 8) {
      return 0;
    } else if (len < 16) {
      return 8;
    } else {
      // Every 16
      return len & ~15;
    }
  } else {
    // Powers of two times either 4, 5, 6, or 7.
    // How: Keep only three bits starting at highest set bit.
    const int bit_index = FloorLog2(len) - 2;
    const size_t keep_from_bit = size_t{1} << bit_index;
    return len & ~(keep_from_bit - 1);
  }
}

}  // namespace rocksdb
