//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include <string.h>
#include "util/coding.h"
#include "util/hash.h"
#include "util/util.h"
#include "util/xxhash.h"

namespace rocksdb {

namespace {
inline uint32_t HashInline(const char* data, size_t n, uint32_t seed) {
  // MurmurHash1 - fast but mediocre quality
  // https://github.com/aappleby/smhasher/wiki/MurmurHash1
  //
  const uint32_t m = 0xc6a4a793;
  const uint32_t r = 24;
  const char* limit = data + n;
  uint32_t h = static_cast<uint32_t>(seed ^ (n * m));

  // Pick up four bytes at a time
  while (data + 4 <= limit) {
    uint32_t w = DecodeFixed32(data);
    data += 4;
    h += w;
    h *= m;
    h ^= (h >> 16);
  }

  // Pick up remaining bytes
  switch (limit - data) {
    // Note: The original hash implementation used data[i] << shift, which
    // promotes the char to int and then performs the shift. If the char is
    // negative, the shift is undefined behavior in C++. The hash algorithm is
    // part of the format definition, so we cannot change it; to obtain the same
    // behavior in a legal way we just cast to uint32_t, which will do
    // sign-extension. To guarantee compatibility with architectures where chars
    // are unsigned we first cast the char to int8_t.
    case 3:
      h += static_cast<uint32_t>(static_cast<int8_t>(data[2])) << 16;
      FALLTHROUGH_INTENDED;
    case 2:
      h += static_cast<uint32_t>(static_cast<int8_t>(data[1])) << 8;
      FALLTHROUGH_INTENDED;
    case 1:
      h += static_cast<uint32_t>(static_cast<int8_t>(data[0]));
      h *= m;
      h ^= (h >> r);
      break;
  }
  return h;
}
}

uint32_t Hash(const char* data, size_t n, uint32_t seed) {
  return HashInline(data, n, seed);
}

uint64_t Hash64(const char* key, size_t len, uint64_t seed) {
  // XXX For testing: above 32-bit hash
  //*
  (void)seed;
  uint32_t h = HashInline(key, len, seed);
  return (uint64_t(h) << 32) + h;
  //*/

  // Attempted 64-bit port of above
  /*
  const uint64_t m = 0xc6a4a7935bd1e995;
  const int r = 47;

  uint64_t h = seed ^ (len * m);

  const uint64_t * data = (const uint64_t *)key;
  const uint64_t * end = data + (len/8);

  while(data != end)
  {
      uint64_t k = *data++;

      h += k;
      h *= m;
      h ^= h >> r;
  }

  const unsigned char * data2 = (const unsigned char*)data;

  switch(len & 7)
  {
  case 7: h ^= ((uint64_t)data2[6]) << 48; FALLTHROUGH_INTENDED;
  case 6: h ^= ((uint64_t)data2[5]) << 40; FALLTHROUGH_INTENDED;
  case 5: h ^= ((uint64_t)data2[4]) << 32; FALLTHROUGH_INTENDED;
  case 4: h ^= ((uint64_t)data2[3]) << 24; FALLTHROUGH_INTENDED;
  case 3: h ^= ((uint64_t)data2[2]) << 16; FALLTHROUGH_INTENDED;
  case 2: h ^= ((uint64_t)data2[1]) << 8;  FALLTHROUGH_INTENDED;
  case 1: h ^= ((uint64_t)data2[0]);
      h *= m;
      h ^= h >> r;
  };

  return h;
  //*/

  // NB: this is currently an experimental version of XXH3 and we are stuck
  // with it if this code is pushed to master.
  //(void)seed; return XXH3_64bits(key, len);
  //return XXH3_64bits_withSeed(key, len, seed);
}

}  // namespace rocksdb
