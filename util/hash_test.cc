//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2012 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "util/hash.h"

#include <array>
#include <cstring>
#include <random>
#include <set>
#include <vector>

#include "env/generate_uuid.h"
#include "test_util/testharness.h"
#include "util/coding.h"
#include "util/math128.h"
#include "util/random.h"
#include "util/uuid.h"

#ifndef GFLAGS
bool FLAGS_extra = false;
#else
#include "util/gflags_compat.h"
using GFLAGS_NAMESPACE::ParseCommandLineFlags;
DEFINE_bool(extra, false, "Run extra tests");
#endif  // GFLAGS

namespace ROCKSDB_NAMESPACE {

// The hash algorithm is part of the file format, for example for the Bloom
// filters. Test that the hash values are stable for a set of random strings of
// varying lengths.
TEST(HashTest, Values) {
  constexpr uint32_t kSeed = 0xbc9f1d34;  // Same as BloomHash.

  EXPECT_EQ(Hash("", 0, kSeed), 3164544308u);
  EXPECT_EQ(Hash("\x08", 1, kSeed), 422599524u);
  EXPECT_EQ(Hash("\x17", 1, kSeed), 3168152998u);
  EXPECT_EQ(Hash("\x9a", 1, kSeed), 3195034349u);
  EXPECT_EQ(Hash("\x1c", 1, kSeed), 2651681383u);
  EXPECT_EQ(Hash("\x4d\x76", 2, kSeed), 2447836956u);
  EXPECT_EQ(Hash("\x52\xd5", 2, kSeed), 3854228105u);
  EXPECT_EQ(Hash("\x91\xf7", 2, kSeed), 31066776u);
  EXPECT_EQ(Hash("\xd6\x27", 2, kSeed), 1806091603u);
  EXPECT_EQ(Hash("\x30\x46\x0b", 3, kSeed), 3808221797u);
  EXPECT_EQ(Hash("\x56\xdc\xd6", 3, kSeed), 2157698265u);
  EXPECT_EQ(Hash("\xd4\x52\x33", 3, kSeed), 1721992661u);
  EXPECT_EQ(Hash("\x6a\xb5\xf4", 3, kSeed), 2469105222u);
  EXPECT_EQ(Hash("\x67\x53\x81\x1c", 4, kSeed), 118283265u);
  EXPECT_EQ(Hash("\x69\xb8\xc0\x88", 4, kSeed), 3416318611u);
  EXPECT_EQ(Hash("\x1e\x84\xaf\x2d", 4, kSeed), 3315003572u);
  EXPECT_EQ(Hash("\x46\xdc\x54\xbe", 4, kSeed), 447346355u);
  EXPECT_EQ(Hash("\xd0\x7a\x6e\xea\x56", 5, kSeed), 4255445370u);
  EXPECT_EQ(Hash("\x86\x83\xd5\xa4\xd8", 5, kSeed), 2390603402u);
  EXPECT_EQ(Hash("\xb7\x46\xbb\x77\xce", 5, kSeed), 2048907743u);
  EXPECT_EQ(Hash("\x6c\xa8\xbc\xe5\x99", 5, kSeed), 2177978500u);
  EXPECT_EQ(Hash("\x5c\x5e\xe1\xa0\x73\x81", 6, kSeed), 1036846008u);
  EXPECT_EQ(Hash("\x08\x5d\x73\x1c\xe5\x2e", 6, kSeed), 229980482u);
  EXPECT_EQ(Hash("\x42\xfb\xf2\x52\xb4\x10", 6, kSeed), 3655585422u);
  EXPECT_EQ(Hash("\x73\xe1\xff\x56\x9c\xce", 6, kSeed), 3502708029u);
  EXPECT_EQ(Hash("\x5c\xbe\x97\x75\x54\x9a\x52", 7, kSeed), 815120748u);
  EXPECT_EQ(Hash("\x16\x82\x39\x49\x88\x2b\x36", 7, kSeed), 3056033698u);
  EXPECT_EQ(Hash("\x59\x77\xf0\xa7\x24\xf4\x78", 7, kSeed), 587205227u);
  EXPECT_EQ(Hash("\xd3\xa5\x7c\x0e\xc0\x02\x07", 7, kSeed), 2030937252u);
  EXPECT_EQ(Hash("\x31\x1b\x98\x75\x96\x22\xd3\x9a", 8, kSeed), 469635402u);
  EXPECT_EQ(Hash("\x38\xd6\xf7\x28\x20\xb4\x8a\xe9", 8, kSeed), 3530274698u);
  EXPECT_EQ(Hash("\xbb\x18\x5d\xf4\x12\x03\xf7\x99", 8, kSeed), 1974545809u);
  EXPECT_EQ(Hash("\x80\xd4\x3b\x3b\xae\x22\xa2\x78", 8, kSeed), 3563570120u);
  EXPECT_EQ(Hash("\x1a\xb5\xd0\xfe\xab\xc3\x61\xb2\x99", 9, kSeed),
            2706087434u);
  EXPECT_EQ(Hash("\x8e\x4a\xc3\x18\x20\x2f\x06\xe6\x3c", 9, kSeed),
            1534654151u);
  EXPECT_EQ(Hash("\xb6\xc0\xdd\x05\x3f\xc4\x86\x4c\xef", 9, kSeed),
            2355554696u);
  EXPECT_EQ(Hash("\x9a\x5f\x78\x0d\xaf\x50\xe1\x1f\x55", 9, kSeed),
            1400800912u);
  EXPECT_EQ(Hash("\x22\x6f\x39\x1f\xf8\xdd\x4f\x52\x17\x94", 10, kSeed),
            3420325137u);
  EXPECT_EQ(Hash("\x32\x89\x2a\x75\x48\x3a\x4a\x02\x69\xdd", 10, kSeed),
            3427803584u);
  EXPECT_EQ(Hash("\x06\x92\x5c\xf4\x88\x0e\x7e\x68\x38\x3e", 10, kSeed),
            1152407945u);
  EXPECT_EQ(Hash("\xbd\x2c\x63\x38\xbf\xe9\x78\xb7\xbf\x15", 10, kSeed),
            3382479516u);
}

// The hash algorithm is part of the file format, for example for the Bloom
// filters.
TEST(HashTest, Hash64Misc) {
  constexpr uint32_t kSeed = 0;  // Same as GetSliceHash64

  for (char fill : {'\0', 'a', '1', '\xff'}) {
    const size_t max_size = 1000;
    const std::string str(max_size, fill);

    for (size_t size = 0; size <= max_size; ++size) {
      uint64_t here = Hash64(str.data(), size, kSeed);

      // Must be same as GetSliceHash64
      EXPECT_EQ(here, GetSliceHash64(Slice(str.data(), size)));

      // Upper and Lower must reconstruct hash
      EXPECT_EQ(here, (uint64_t{Upper32of64(here)} << 32) | Lower32of64(here));
      EXPECT_EQ(here, (uint64_t{Upper32of64(here)} << 32) + Lower32of64(here));
      EXPECT_EQ(here, (uint64_t{Upper32of64(here)} << 32) ^ Lower32of64(here));

      // Seed changes hash value (with high probability)
      for (uint64_t var_seed = 1; var_seed != 0; var_seed <<= 1) {
        EXPECT_NE(here, Hash64(str.data(), size, var_seed));
      }

      // Size changes hash value (with high probability)
      size_t max_smaller_by = std::min(size_t{30}, size);
      for (size_t smaller_by = 1; smaller_by <= max_smaller_by; ++smaller_by) {
        EXPECT_NE(here, Hash64(str.data(), size - smaller_by, kSeed));
      }
    }
  }
}

// Test that hash values are "non-trivial" for "trivial" inputs
TEST(HashTest, Hash64Trivial) {
  // Thorough test too slow for regression testing
  constexpr bool thorough = false;

  // For various seeds, make sure hash of empty string is not zero.
  constexpr uint64_t max_seed = thorough ? 0x1000000 : 0x10000;
  for (uint64_t seed = 0; seed < max_seed; ++seed) {
    uint64_t here = Hash64("", 0, seed);
    EXPECT_NE(Lower32of64(here), 0u);
    EXPECT_NE(Upper32of64(here), 0u);
  }

  // For standard seed, make sure hash of small strings are not zero
  constexpr uint32_t kSeed = 0;  // Same as GetSliceHash64
  char input[4];
  constexpr int max_len = thorough ? 3 : 2;
  for (int len = 1; len <= max_len; ++len) {
    for (uint32_t i = 0; (i >> (len * 8)) == 0; ++i) {
      EncodeFixed32(input, i);
      uint64_t here = Hash64(input, len, kSeed);
      EXPECT_NE(Lower32of64(here), 0u);
      EXPECT_NE(Upper32of64(here), 0u);
    }
  }
}

// Test that the hash values are stable for a set of random strings of
// varying small lengths.
TEST(HashTest, Hash64SmallValueSchema) {
  constexpr uint32_t kSeed = 0;  // Same as GetSliceHash64

  EXPECT_EQ(Hash64("", 0, kSeed), uint64_t{5999572062939766020u});
  EXPECT_EQ(Hash64("\x08", 1, kSeed), uint64_t{583283813901344696u});
  EXPECT_EQ(Hash64("\x17", 1, kSeed), uint64_t{16175549975585474943u});
  EXPECT_EQ(Hash64("\x9a", 1, kSeed), uint64_t{16322991629225003903u});
  EXPECT_EQ(Hash64("\x1c", 1, kSeed), uint64_t{13269285487706833447u});
  EXPECT_EQ(Hash64("\x4d\x76", 2, kSeed), uint64_t{6859542833406258115u});
  EXPECT_EQ(Hash64("\x52\xd5", 2, kSeed), uint64_t{4919611532550636959u});
  EXPECT_EQ(Hash64("\x91\xf7", 2, kSeed), uint64_t{14199427467559720719u});
  EXPECT_EQ(Hash64("\xd6\x27", 2, kSeed), uint64_t{12292689282614532691u});
  EXPECT_EQ(Hash64("\x30\x46\x0b", 3, kSeed), uint64_t{11404699285340020889u});
  EXPECT_EQ(Hash64("\x56\xdc\xd6", 3, kSeed), uint64_t{12404347133785524237u});
  EXPECT_EQ(Hash64("\xd4\x52\x33", 3, kSeed), uint64_t{15853805298481534034u});
  EXPECT_EQ(Hash64("\x6a\xb5\xf4", 3, kSeed), uint64_t{16863488758399383382u});
  EXPECT_EQ(Hash64("\x67\x53\x81\x1c", 4, kSeed),
            uint64_t{9010661983527562386u});
  EXPECT_EQ(Hash64("\x69\xb8\xc0\x88", 4, kSeed),
            uint64_t{6611781377647041447u});
  EXPECT_EQ(Hash64("\x1e\x84\xaf\x2d", 4, kSeed),
            uint64_t{15290969111616346501u});
  EXPECT_EQ(Hash64("\x46\xdc\x54\xbe", 4, kSeed),
            uint64_t{7063754590279313623u});
  EXPECT_EQ(Hash64("\xd0\x7a\x6e\xea\x56", 5, kSeed),
            uint64_t{6384167718754869899u});
  EXPECT_EQ(Hash64("\x86\x83\xd5\xa4\xd8", 5, kSeed),
            uint64_t{16874407254108011067u});
  EXPECT_EQ(Hash64("\xb7\x46\xbb\x77\xce", 5, kSeed),
            uint64_t{16809880630149135206u});
  EXPECT_EQ(Hash64("\x6c\xa8\xbc\xe5\x99", 5, kSeed),
            uint64_t{1249038833153141148u});
  EXPECT_EQ(Hash64("\x5c\x5e\xe1\xa0\x73\x81", 6, kSeed),
            uint64_t{17358142495308219330u});
  EXPECT_EQ(Hash64("\x08\x5d\x73\x1c\xe5\x2e", 6, kSeed),
            uint64_t{4237646583134806322u});
  EXPECT_EQ(Hash64("\x42\xfb\xf2\x52\xb4\x10", 6, kSeed),
            uint64_t{4373664924115234051u});
  EXPECT_EQ(Hash64("\x73\xe1\xff\x56\x9c\xce", 6, kSeed),
            uint64_t{12012981210634596029u});
  EXPECT_EQ(Hash64("\x5c\xbe\x97\x75\x54\x9a\x52", 7, kSeed),
            uint64_t{5716522398211028826u});
  EXPECT_EQ(Hash64("\x16\x82\x39\x49\x88\x2b\x36", 7, kSeed),
            uint64_t{15604531309862565013u});
  EXPECT_EQ(Hash64("\x59\x77\xf0\xa7\x24\xf4\x78", 7, kSeed),
            uint64_t{8601330687345614172u});
  EXPECT_EQ(Hash64("\xd3\xa5\x7c\x0e\xc0\x02\x07", 7, kSeed),
            uint64_t{8088079329364056942u});
  EXPECT_EQ(Hash64("\x31\x1b\x98\x75\x96\x22\xd3\x9a", 8, kSeed),
            uint64_t{9844314944338447628u});
  EXPECT_EQ(Hash64("\x38\xd6\xf7\x28\x20\xb4\x8a\xe9", 8, kSeed),
            uint64_t{10973293517982163143u});
  EXPECT_EQ(Hash64("\xbb\x18\x5d\xf4\x12\x03\xf7\x99", 8, kSeed),
            uint64_t{9986007080564743219u});
  EXPECT_EQ(Hash64("\x80\xd4\x3b\x3b\xae\x22\xa2\x78", 8, kSeed),
            uint64_t{1729303145008254458u});
  EXPECT_EQ(Hash64("\x1a\xb5\xd0\xfe\xab\xc3\x61\xb2\x99", 9, kSeed),
            uint64_t{13253403748084181481u});
  EXPECT_EQ(Hash64("\x8e\x4a\xc3\x18\x20\x2f\x06\xe6\x3c", 9, kSeed),
            uint64_t{7768754303876232188u});
  EXPECT_EQ(Hash64("\xb6\xc0\xdd\x05\x3f\xc4\x86\x4c\xef", 9, kSeed),
            uint64_t{12439346786701492u});
  EXPECT_EQ(Hash64("\x9a\x5f\x78\x0d\xaf\x50\xe1\x1f\x55", 9, kSeed),
            uint64_t{10841838338450144690u});
  EXPECT_EQ(Hash64("\x22\x6f\x39\x1f\xf8\xdd\x4f\x52\x17\x94", 10, kSeed),
            uint64_t{12883919702069153152u});
  EXPECT_EQ(Hash64("\x32\x89\x2a\x75\x48\x3a\x4a\x02\x69\xdd", 10, kSeed),
            uint64_t{12692903507676842188u});
  EXPECT_EQ(Hash64("\x06\x92\x5c\xf4\x88\x0e\x7e\x68\x38\x3e", 10, kSeed),
            uint64_t{6540985900674032620u});
  EXPECT_EQ(Hash64("\xbd\x2c\x63\x38\xbf\xe9\x78\xb7\xbf\x15", 10, kSeed),
            uint64_t{10551812464348219044u});
}

static std::string Hash64TestDescriptor(const char *repeat, size_t limit) {
  const char *mod61_encode =
      "abcdefghijklmnopqrstuvwxyz123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";

  std::string input;
  while (input.size() < limit) {
    input.append(repeat);
  }
  std::string rv;
  for (size_t i = 0; i < limit; ++i) {
    uint64_t h = GetSliceHash64(Slice(input.data(), i));
    rv.append(1, mod61_encode[static_cast<size_t>(h % 61)]);
  }
  return rv;
}

// XXH3p changes its algorithm for various sizes up through 250 bytes, so
// we need to check the stability of larger sizes also.
TEST(HashTest, Hash64LargeValueSchema) {
  // Each of these derives a "descriptor" from the hash values for all
  // lengths up to 430.
  // Note that "c" is common for the zero-length string.
  EXPECT_EQ(
      Hash64TestDescriptor("foo", 430),
      "cRhyWsY67B6klRA1udmOuiYuX7IthyGBKqbeosz2hzVglWCmQx8nEdnpkvPfYX56Up2OWOTV"
      "lTzfAoYwvtqKzjD8E9xttR2unelbXbIV67NUe6bOO23BxaSFRcA3njGu5cUWfgwOqNoTsszp"
      "uPvKRP6qaUR5VdoBkJUCFIefd7edlNK5mv6JYWaGdwxehg65hTkTmjZoPKxTZo4PLyzbL9U4"
      "xt12ITSfeP2MfBHuLI2z2pDlBb44UQKVMx27LEoAHsdLp3WfWfgH3sdRBRCHm33UxCM4QmE2"
      "xJ7gqSvNwTeH7v9GlC8zWbGroyD3UVNeShMLx29O7tH1biemLULwAHyIw8zdtLMDpEJ8m2ic"
      "l6Lb4fDuuFNAs1GCVUthjK8CV8SWI8Rsz5THSwn5CGhpqUwSZcFknjwWIl5rNCvDxXJqYr");
  // Note that "1EeRk" is common for "Rocks"
  EXPECT_EQ(
      Hash64TestDescriptor("Rocks", 430),
      "c1EeRkrzgOYWLA8PuhJrwTePJewoB44WdXYDfhbk3ZxTqqg25WlPExDl7IKIQLJvnA6gJxxn"
      "9TCSLkFGfJeXehaSS1GBqWSzfhEH4VXiXIUCuxJXxtKXcSC6FrNIQGTZbYDiUOLD6Y5inzrF"
      "9etwQhXUBanw55xAUdNMFQAm2GjJ6UDWp2mISLiMMkLjANWMKLaZMqaFLX37qB4MRO1ooVRv"
      "zSvaNRSCLxlggQCasQq8icWjzf3HjBlZtU6pd4rkaUxSzHqmo9oM5MghbU5Rtxg8wEfO7lVN"
      "5wdMONYecslQTwjZUpO1K3LDf3K3XK6sUXM6ShQQ3RHmMn2acB4YtTZ3QQcHYJSOHn2DuWpa"
      "Q8RqzX5lab92YmOLaCdOHq1BPsM7SIBzMdLgePNsJ1vvMALxAaoDUHPxoFLO2wx18IXnyX");
  EXPECT_EQ(
      Hash64TestDescriptor("RocksDB", 430),
      "c1EeRkukbkb28wLTahwD2sfUhZzaBEnF8SVrxnPVB6A7b8CaAl3UKsDZISF92GSq2wDCukOq"
      "Jgrsp7A3KZhDiLW8dFXp8UPqPxMCRlMdZeVeJ2dJxrmA6cyt99zkQFj7ELbut6jAeVqARFnw"
      "fnWVXOsaLrq7bDCbMcns2DKvTaaqTCLMYxI7nhtLpFN1jR755FRQFcOzrrDbh7QhypjdvlYw"
      "cdAMSZgp9JMHxbM23wPSuH6BOFgxejz35PScZfhDPvTOxIy1jc3MZsWrMC3P324zNolO7JdW"
      "CX2I5UDKjjaEJfxbgVgJIXxtQGlmj2xkO5sPpjULQV4X2HlY7FQleJ4QRaJIB4buhCA4vUTF"
      "eMFlxCIYUpTCsal2qsmnGOWa8WCcefrohMjDj1fjzSvSaQwlpyR1GZHF2uPOoQagiCpHpm");
}

TEST(FastRange32Test, Values) {
  using ROCKSDB_NAMESPACE::FastRange32;
  // Zero range
  EXPECT_EQ(FastRange32(0, 0), 0U);
  EXPECT_EQ(FastRange32(123, 0), 0U);
  EXPECT_EQ(FastRange32(0xffffffff, 0), 0U);

  // One range
  EXPECT_EQ(FastRange32(0, 1), 0U);
  EXPECT_EQ(FastRange32(123, 1), 0U);
  EXPECT_EQ(FastRange32(0xffffffff, 1), 0U);

  // Two range
  EXPECT_EQ(FastRange32(0, 2), 0U);
  EXPECT_EQ(FastRange32(123, 2), 0U);
  EXPECT_EQ(FastRange32(0x7fffffff, 2), 0U);
  EXPECT_EQ(FastRange32(0x80000000, 2), 1U);
  EXPECT_EQ(FastRange32(0xffffffff, 2), 1U);

  // Seven range
  EXPECT_EQ(FastRange32(0, 7), 0U);
  EXPECT_EQ(FastRange32(123, 7), 0U);
  EXPECT_EQ(FastRange32(613566756, 7), 0U);
  EXPECT_EQ(FastRange32(613566757, 7), 1U);
  EXPECT_EQ(FastRange32(1227133513, 7), 1U);
  EXPECT_EQ(FastRange32(1227133514, 7), 2U);
  // etc.
  EXPECT_EQ(FastRange32(0xffffffff, 7), 6U);

  // Big
  EXPECT_EQ(FastRange32(1, 0x80000000), 0U);
  EXPECT_EQ(FastRange32(2, 0x80000000), 1U);
  EXPECT_EQ(FastRange32(4, 0x7fffffff), 1U);
  EXPECT_EQ(FastRange32(4, 0x80000000), 2U);
  EXPECT_EQ(FastRange32(0xffffffff, 0x7fffffff), 0x7ffffffeU);
  EXPECT_EQ(FastRange32(0xffffffff, 0x80000000), 0x7fffffffU);
}

TEST(FastRange64Test, Values) {
  using ROCKSDB_NAMESPACE::FastRange64;
  // Zero range
  EXPECT_EQ(FastRange64(0, 0), 0U);
  EXPECT_EQ(FastRange64(123, 0), 0U);
  EXPECT_EQ(FastRange64(0xffffFFFF, 0), 0U);
  EXPECT_EQ(FastRange64(0xffffFFFFffffFFFF, 0), 0U);

  // One range
  EXPECT_EQ(FastRange64(0, 1), 0U);
  EXPECT_EQ(FastRange64(123, 1), 0U);
  EXPECT_EQ(FastRange64(0xffffFFFF, 1), 0U);
  EXPECT_EQ(FastRange64(0xffffFFFFffffFFFF, 1), 0U);

  // Two range
  EXPECT_EQ(FastRange64(0, 2), 0U);
  EXPECT_EQ(FastRange64(123, 2), 0U);
  EXPECT_EQ(FastRange64(0xffffFFFF, 2), 0U);
  EXPECT_EQ(FastRange64(0x7fffFFFFffffFFFF, 2), 0U);
  EXPECT_EQ(FastRange64(0x8000000000000000, 2), 1U);
  EXPECT_EQ(FastRange64(0xffffFFFFffffFFFF, 2), 1U);

  // Seven range
  EXPECT_EQ(FastRange64(0, 7), 0U);
  EXPECT_EQ(FastRange64(123, 7), 0U);
  EXPECT_EQ(FastRange64(0xffffFFFF, 7), 0U);
  EXPECT_EQ(FastRange64(2635249153387078802, 7), 0U);
  EXPECT_EQ(FastRange64(2635249153387078803, 7), 1U);
  EXPECT_EQ(FastRange64(5270498306774157604, 7), 1U);
  EXPECT_EQ(FastRange64(5270498306774157605, 7), 2U);
  EXPECT_EQ(FastRange64(0x7fffFFFFffffFFFF, 7), 3U);
  EXPECT_EQ(FastRange64(0x8000000000000000, 7), 3U);
  EXPECT_EQ(FastRange64(0xffffFFFFffffFFFF, 7), 6U);

  // Big but 32-bit range
  EXPECT_EQ(FastRange64(0x100000000, 0x80000000), 0U);
  EXPECT_EQ(FastRange64(0x200000000, 0x80000000), 1U);
  EXPECT_EQ(FastRange64(0x400000000, 0x7fffFFFF), 1U);
  EXPECT_EQ(FastRange64(0x400000000, 0x80000000), 2U);
  EXPECT_EQ(FastRange64(0xffffFFFFffffFFFF, 0x7fffFFFF), 0x7fffFFFEU);
  EXPECT_EQ(FastRange64(0xffffFFFFffffFFFF, 0x80000000), 0x7fffFFFFU);

  // Big, > 32-bit range
#if SIZE_MAX == UINT64_MAX
  EXPECT_EQ(FastRange64(0x7fffFFFFffffFFFF, 0x4200000002), 0x2100000000U);
  EXPECT_EQ(FastRange64(0x8000000000000000, 0x4200000002), 0x2100000001U);

  EXPECT_EQ(FastRange64(0x0000000000000000, 420000000002), 0U);
  EXPECT_EQ(FastRange64(0x7fffFFFFffffFFFF, 420000000002), 210000000000U);
  EXPECT_EQ(FastRange64(0x8000000000000000, 420000000002), 210000000001U);
  EXPECT_EQ(FastRange64(0xffffFFFFffffFFFF, 420000000002), 420000000001U);

  EXPECT_EQ(FastRange64(0xffffFFFFffffFFFF, 0xffffFFFFffffFFFF),
            0xffffFFFFffffFFFEU);
#endif
}

TEST(FastRangeGenericTest, Values) {
  // Generic (including big and small)
  // Note that FastRangeGeneric is also tested indirectly above via
  // FastRange32 and FastRange64.
  EXPECT_EQ(
      FastRangeGeneric(uint64_t{0x8000000000000000}, uint64_t{420000000002}),
      uint64_t{210000000001});
  EXPECT_EQ(FastRangeGeneric(uint64_t{0x8000000000000000}, uint16_t{12468}),
            uint16_t{6234});
  EXPECT_EQ(FastRangeGeneric(uint32_t{0x80000000}, uint16_t{12468}),
            uint16_t{6234});
  // Not recommended for typical use because for example this could fail on
  // some platforms and pass on others:
  //EXPECT_EQ(FastRangeGeneric(static_cast<unsigned long>(0x80000000),
  //                           uint16_t{12468}),
  //          uint16_t{6234});
}

// for inspection of disassembly
uint32_t MyFastRange32(uint32_t hash, uint32_t range) {
  return ROCKSDB_NAMESPACE::FastRange32(hash, range);
}

// for inspection of disassembly
size_t MyFastRange64(uint64_t hash, size_t range) {
  return ROCKSDB_NAMESPACE::FastRange64(hash, range);
}

// Tests for math.h / math128.h (not worth a separate test binary)
template <typename T>
static void test_BitOps() {
  // This complex code is to generalize to 128-bit values. Otherwise
  // we could just use = static_cast<T>(0x5555555555555555ULL);
  T everyOtherBit = 0;
  for (unsigned i = 0; i < sizeof(T); ++i) {
    everyOtherBit = (everyOtherBit << 8) | T{0x55};
  }

  for (int i = 0; i < int{8 * sizeof(T)}; ++i) {
    T v = T{1} << i;
    T vm1 = static_cast<T>(v - 1);

    // FloorLog2
    if (v > 0) {
      EXPECT_EQ(FloorLog2(v), i);
    }
    if (vm1 > 0) {
      EXPECT_EQ(FloorLog2(vm1), i - 1);
      EXPECT_EQ(FloorLog2(everyOtherBit & vm1), (i - 1) & ~1);
    }

    // CountTrailingZeroBits
    if (v != 0) {
      EXPECT_EQ(CountTrailingZeroBits(v), i);
    }
    if (vm1 != 0) {
      EXPECT_EQ(CountTrailingZeroBits(vm1), 0);
    }
    if (i < int{8 * sizeof(T)} - 1) {
      EXPECT_EQ(CountTrailingZeroBits(~vm1 & everyOtherBit), (i + 1) & ~1);
    }

    // BitsSetToOne
    EXPECT_EQ(BitsSetToOne(v), 1);
    EXPECT_EQ(BitsSetToOne(vm1), i);
    EXPECT_EQ(BitsSetToOne(vm1 & everyOtherBit), (i + 1) / 2);

    // BitParity
    EXPECT_EQ(BitParity(v), 1);
    EXPECT_EQ(BitParity(vm1), i & 1);
    EXPECT_EQ(BitParity(vm1 & everyOtherBit), ((i + 1) / 2) & 1);

    // EndianSwapValue
    T ev = T{1} << (((sizeof(T) - 1 - (i / 8)) * 8) + i % 8);
    EXPECT_EQ(EndianSwapValue(v), ev);

    // ReverseBits
    T rv = T{1} << (8 * sizeof(T) - 1 - i);
    EXPECT_EQ(ReverseBits(v), static_cast<T>(T{1} << (8 * sizeof(T) - 1 - i)));
    EXPECT_EQ(ReverseBits(vm1), static_cast<T>(rv * ~T{1}));
  }
}

TEST(MathTest, BitOps) {
  test_BitOps<uint32_t>();
  test_BitOps<uint64_t>();
  test_BitOps<uint16_t>();
  test_BitOps<uint8_t>();
  test_BitOps<unsigned char>();
  test_BitOps<unsigned short>();
  test_BitOps<unsigned int>();
  test_BitOps<unsigned long>();
  test_BitOps<unsigned long long>();
  test_BitOps<char>();
  test_BitOps<size_t>();
  test_BitOps<int32_t>();
  test_BitOps<int64_t>();
  test_BitOps<int16_t>();
  test_BitOps<int8_t>();
  test_BitOps<signed char>();
  test_BitOps<short>();
  test_BitOps<int>();
  test_BitOps<long>();
  test_BitOps<long long>();
  test_BitOps<ptrdiff_t>();
}

TEST(MathTest, BitOps128) { test_BitOps<Unsigned128>(); }

TEST(MathTest, Math128) {
  const Unsigned128 sixteenHexOnes = 0x1111111111111111U;
  const Unsigned128 thirtyHexOnes = (sixteenHexOnes << 56) | sixteenHexOnes;
  const Unsigned128 sixteenHexTwos = 0x2222222222222222U;
  const Unsigned128 thirtyHexTwos = (sixteenHexTwos << 56) | sixteenHexTwos;

  // v will slide from all hex ones to all hex twos
  Unsigned128 v = thirtyHexOnes;
  for (int i = 0; i <= 30; ++i) {
    // Test bitwise operations
    EXPECT_EQ(BitsSetToOne(v), 30);
    EXPECT_EQ(BitsSetToOne(~v), 128 - 30);
    EXPECT_EQ(BitsSetToOne(v & thirtyHexOnes), 30 - i);
    EXPECT_EQ(BitsSetToOne(v | thirtyHexOnes), 30 + i);
    EXPECT_EQ(BitsSetToOne(v ^ thirtyHexOnes), 2 * i);
    EXPECT_EQ(BitsSetToOne(v & thirtyHexTwos), i);
    EXPECT_EQ(BitsSetToOne(v | thirtyHexTwos), 60 - i);
    EXPECT_EQ(BitsSetToOne(v ^ thirtyHexTwos), 60 - 2 * i);

    // Test comparisons
    EXPECT_EQ(v == thirtyHexOnes, i == 0);
    EXPECT_EQ(v == thirtyHexTwos, i == 30);
    EXPECT_EQ(v > thirtyHexOnes, i > 0);
    EXPECT_EQ(v > thirtyHexTwos, false);
    EXPECT_EQ(v >= thirtyHexOnes, true);
    EXPECT_EQ(v >= thirtyHexTwos, i == 30);
    EXPECT_EQ(v < thirtyHexOnes, false);
    EXPECT_EQ(v < thirtyHexTwos, i < 30);
    EXPECT_EQ(v <= thirtyHexOnes, i == 0);
    EXPECT_EQ(v <= thirtyHexTwos, true);

    // Update v, clearing upper-most byte
    v = ((v << 12) >> 8) | 0x2;
  }

  for (int i = 0; i < 128; ++i) {
    // Test shifts
    Unsigned128 sl = thirtyHexOnes << i;
    Unsigned128 sr = thirtyHexOnes >> i;
    EXPECT_EQ(BitsSetToOne(sl), std::min(30, 32 - i / 4));
    EXPECT_EQ(BitsSetToOne(sr), std::max(0, 30 - (i + 3) / 4));
    EXPECT_EQ(BitsSetToOne(sl & sr), i % 2 ? 0 : std::max(0, 30 - i / 2));
  }

  // Test 64x64->128 multiply
  Unsigned128 product =
      Multiply64to128(0x1111111111111111U, 0x2222222222222222U);
  EXPECT_EQ(Lower64of128(product), 2295594818061633090U);
  EXPECT_EQ(Upper64of128(product), 163971058432973792U);

  // Test 128x128->256 multiply
  Unsigned128 big1 = (Unsigned128{0x8c7d1ebeae} << 64) + 0xea18f0547f9ec4cfU;
  Unsigned128 big2 = (Unsigned128{0x1846d550e37} << 64) + 0xb5063df43be64e08U;
  Unsigned128 p_upper;
  Unsigned128 p_lower;
  Multiply128to256(big1, big2, &p_upper, &p_lower);
  EXPECT_EQ(Upper64of128(p_upper), 54569U);
  EXPECT_EQ(Lower64of128(p_upper), 11712860883347098737U);
  EXPECT_EQ(Upper64of128(p_lower), 8294330302176911072U);
  EXPECT_EQ(Lower64of128(p_lower), 13792390213309773944U);

  // And 128x128->128 multiply (truncated)
  EXPECT_EQ(big1 * big2, p_lower);

  // And also in a fastrange sort of way. (2/3rd of 666666...)
  Unsigned128 two_thirds =
      (Unsigned128{0xaaaaaaaaaaaaaaaa} << 64) + 0xaaaaaaaaaaaaaaabU;
  Multiply128to256(two_thirds, 0x66666666666666U, &p_upper, &p_lower);
  uint64_t result = Lower64of128(p_upper);
  EXPECT_EQ(result, 0x44444444444444U);
}

TEST(MathTest, Coding128) {
  const char *in = "_1234567890123456";
  // Note: in + 1 is likely unaligned
  Unsigned128 decoded = DecodeFixed128(in + 1);
  EXPECT_EQ(Lower64of128(decoded), 0x3837363534333231U);
  EXPECT_EQ(Upper64of128(decoded), 0x3635343332313039U);
  char out[18];
  out[0] = '_';
  EncodeFixed128(out + 1, decoded);
  out[17] = '\0';
  EXPECT_EQ(std::string(in), std::string(out));
}

TEST(MathTest, CodingGeneric) {
  const char *in = "_1234567890123456";
  // Decode
  // Note: in + 1 is likely unaligned
  Unsigned128 decoded128 = DecodeFixedGeneric<Unsigned128>(in + 1);
  EXPECT_EQ(Lower64of128(decoded128), 0x3837363534333231U);
  EXPECT_EQ(Upper64of128(decoded128), 0x3635343332313039U);

  uint64_t decoded64 = DecodeFixedGeneric<uint64_t>(in + 1);
  EXPECT_EQ(decoded64, 0x3837363534333231U);

  uint32_t decoded32 = DecodeFixedGeneric<uint32_t>(in + 1);
  EXPECT_EQ(decoded32, 0x34333231U);

  uint16_t decoded16 = DecodeFixedGeneric<uint16_t>(in + 1);
  EXPECT_EQ(decoded16, 0x3231U);

  // Encode
  char out[18];
  out[0] = '_';
  memset(out + 1, '\0', 17);
  EncodeFixedGeneric(out + 1, decoded128);
  EXPECT_EQ(std::string(in), std::string(out));

  memset(out + 1, '\0', 9);
  EncodeFixedGeneric(out + 1, decoded64);
  EXPECT_EQ(std::string("_12345678"), std::string(out));

  memset(out + 1, '\0', 5);
  EncodeFixedGeneric(out + 1, decoded32);
  EXPECT_EQ(std::string("_1234"), std::string(out));

  memset(out + 1, '\0', 3);
  EncodeFixedGeneric(out + 1, decoded16);
  EXPECT_EQ(std::string("_12"), std::string(out));
}

// Tests for uuid.h (not worth a separate test binary)
// (See env_test for generate_uuid.h tests.)

// This is a proof-of-concept Monte Carlo test demonstrating, in short,
// that more randomness is not always better for uniqueness. For example,
// suppose that we are generating UUIDs, sometimes with knowledge and
// "ownership" of the recent UUID we generated and sometimes with
// no such knowledge or guarantees, so we are forced to choose a UUID at
// random.
//
// The question is what is the best approach for generating unique UUIDs
// after that first random UUID, when we know (with high probability) that no
// one else is also building from that same UUID? Is it better to use a
// structured approach (e.g. add 1 to previous UUID) or to ignore history and
// keep being random?
//
// This test shows that the structured approach is better. In fact, building
// on previously-generated UUIDs using xor is slightly better than addition.
//
// Note that all these approaches have the same AVERAGE (EXPECTED) number of
// collisions. Because we consider the "badness" of a single collision to be
// similar to the "badness" of many collisions, by clustering the rare cases
// of collisions, we reduce the overall probability of any collisions.
TEST(UuidTest, Approaches) {
  if (!FLAGS_extra) {
    ROCKSDB_GTEST_BYPASS("Demonstration/investigation purposes only");
    return;
  }

  ROCKSDB_NAMESPACE::Random64 r{std::random_device()()};

  constexpr uint64_t kMaxIter = 100000;
  constexpr int kBits = 20;
  constexpr size_t kSize = size_t{1} << kBits;
  constexpr size_t kMask = kSize - 1;

  enum Approach {
    kRandom,
    kAddition,
    kXor,
    kMultAddition,
    kMultXor,
  };

  std::vector<bool> v;
  for (auto approach : {kRandom, kAddition, kXor, kMultAddition, kMultXor}) {
    uint64_t threshold_total = 0;
    for (uint64_t iter = 0; iter < kMaxIter; ++iter) {
      v.assign(kSize, false);
      bool collision = false;
      do {
        size_t start = static_cast<size_t>(r.Next()) & kMask;
        size_t count;
        if (approach == kRandom) {
          count = 1;
        } else {
          count = 1000;
        }
        for (size_t i = 0; i < count && !collision; ++i) {
          size_t pos;
          size_t mult_offset = static_cast<size_t>(
              (i * uint64_t{0xf01f39c13d6a7d81}) >> (64 - kBits));
          if (approach == kAddition) {
            pos = (start + i) & kMask;
          } else if (approach == kMultAddition) {
            pos = (start + mult_offset) & kMask;
          } else if (approach == kMultXor) {
            pos = start ^ mult_offset;
          } else {
            pos = start ^ i;
          }
          if (v[pos]) {
            collision = true;
          } else {
            v[pos] = true;
            ++threshold_total;
          }
        }
      } while (!collision);
    }
    const char *name = approach == kAddition       ? "addition"
                       : approach == kXor          ? "xor"
                       : approach == kMultAddition ? "multaddition"
                       : approach == kMultXor      ? "multxor"
                                                   : "random";
    fprintf(stderr, "Average threshold with %s: %g / %u\n", name,
            1.0 * threshold_total / kMaxIter, (unsigned)kSize);
  }
  // Example output with non-random count=1000:
  // Average threshold with random: 1283.67 / 1048576
  // Average threshold with addition: 28653.2 / 1048576
  // Average threshold with xor: 39762.9 / 1048576
  // Average threshold with multaddition: 20369.2 / 1048576
  // Average threshold with multxor: 2810.48 / 1048576
  //
  // Winner is xor because there's lowest chance of any overlap, but addition
  // is close. Multiplicative hash + addition is within a factor of two, which
  // 1-2 bits of uniqueness (1 structured, e.g. counter, or 2 unstructured,
  // e.g. hash). Note that the random case matches roughly sqrt(n) we would
  // expect from the Birthday paradox.
  //
  // Example output with non-random count=50:
  // Average threshold with random: 1284.02 / 1048576
  // Average threshold with addition: 6460.97 / 1048576
  // Average threshold with xor: 8001.48 / 1048576
  // Average threshold with multaddition: 4665.97 / 1048576
  // Average threshold with multxor: 2057.52 / 1048576
}

TEST(UuidTest, RfcUuidTest) {
  // Default ctor
  RfcUuid u0;
  EXPECT_TRUE(u0.IsEmpty());
  EXPECT_EQ(u0.GetVariant(), 0);
  EXPECT_EQ(u0.GetVersion(), 0);
  EXPECT_EQ(u0.data, Unsigned128{0});

  // Parse valid, check ordering relation, etc.
  ASSERT_OK(RfcUuid::Parse("00000000-0000-0000-0000-000000000000", &u0));
  EXPECT_TRUE(u0.IsEmpty());
  EXPECT_EQ(u0.GetVariant(), 0);
  EXPECT_EQ(u0.GetVersion(), 0);
  EXPECT_EQ(u0.data, Unsigned128{0});
  EXPECT_EQ(u0.ToString(), "00000000-0000-0000-0000-000000000000");

  RfcUuid u1;
  ASSERT_OK(RfcUuid::Parse("00000000-0000-0000-0000-000000000001", &u1));
  EXPECT_FALSE(u1.IsEmpty());
  EXPECT_EQ(u1.GetVariant(), 0);
  EXPECT_EQ(u1.GetVersion(), 0);
  EXPECT_EQ(u1.data, Unsigned128{1});
  EXPECT_TRUE(u0 < u1);
  EXPECT_TRUE(u0.data < u1.data);  // implementation detail
  EXPECT_FALSE(u1 < u0);
  EXPECT_FALSE(u0 == u1);

  RfcUuid u2;
  ASSERT_OK(RfcUuid::Parse("00000000-0000-0000-0001-000000000000", &u2));
  EXPECT_EQ(u2.data, Unsigned128{1} << 48);
  EXPECT_TRUE(u1 < u2);
  EXPECT_TRUE(u1.data < u2.data);
  EXPECT_FALSE(u2 < u1);

  RfcUuid u3;
  ASSERT_OK(RfcUuid::Parse("00000000-0000-0001-0000-000000000000", &u3));
  EXPECT_EQ(u3.data, Unsigned128{1} << 64);
  EXPECT_TRUE(u2 < u3);
  EXPECT_TRUE(u2.data < u3.data);
  EXPECT_FALSE(u3 < u2);

  RfcUuid u4;
  ASSERT_OK(RfcUuid::Parse("00000000-0001-0000-0000-000000000000", &u4));
  EXPECT_EQ(u4.data, Unsigned128{1} << 80);
  EXPECT_TRUE(u3 < u4);
  EXPECT_TRUE(u3.data < u4.data);
  EXPECT_FALSE(u4 < u3);

  RfcUuid u5;
  ASSERT_OK(RfcUuid::Parse("00000001-0000-0000-0000-000000000000", &u5));
  EXPECT_EQ(u5.data, Unsigned128{1} << 96);
  EXPECT_TRUE(u4 < u5);
  EXPECT_TRUE(u4.data < u5.data);
  EXPECT_FALSE(u5 < u4);

  std::string s = "123e4567-e89b-12d3-a456-426614174000";
  RfcUuid ua;
  ASSERT_OK(RfcUuid::Parse(s, &ua));
  EXPECT_FALSE(ua.IsEmpty());
  EXPECT_EQ(ua.GetVariant(), 1);
  EXPECT_EQ(ua.GetVersion(), 1);
  EXPECT_EQ(ua.ToString(), s);

  // Upper case OK
  RfcUuid ua2;
  std::string s2 = s;
  s2[3] = 'E';
  s2[12] = 'B';
  ASSERT_OK(RfcUuid::Parse(s2, &ua2));
  EXPECT_TRUE(ua == ua2);
  EXPECT_EQ(ua.ToString(), s);

  // Test version
  for (int i = 0; i < 16; ++i) {
    ua.SetVersion(i);
    EXPECT_EQ(ua.GetVersion(), i);
    s[14] = "0123456789abcdef"[i];
    EXPECT_EQ(ua.ToString(), s);
  }

  // Test variant (weird because of variable number of bits)
  ua.SetVariant(0);
  EXPECT_EQ(ua.GetVariant(), 0);
  s[19] = '2';
  EXPECT_EQ(ua.ToString(), s);

  ua.SetVariant(1);
  EXPECT_EQ(ua.GetVariant(), 1);
  s[19] = 'a';
  EXPECT_EQ(ua.ToString(), s);

  ua.SetVariant(2);
  EXPECT_EQ(ua.GetVariant(), 2);
  s[19] = 'c';
  EXPECT_EQ(ua.ToString(), s);

  ua.SetVariant(1);
  EXPECT_EQ(ua.GetVariant(), 1);
  s[19] = '8';
  EXPECT_EQ(ua.ToString(), s);

  ua.SetVariant(3);
  EXPECT_EQ(ua.GetVariant(), 3);
  s[19] = 'e';
  EXPECT_EQ(ua.ToString(), s);

  // Pseudorandom, ensure coverage of the value space (assuming
  // version 4 variant 1) based on strings
  ROCKSDB_NAMESPACE::Random64 r{42U};
  std::array<std::set<char>, 36U> seen{};

  for (int i = 0; i < 1000; ++i) {
    // Generate random bits
    RawUuid ru = r.Next();
    ru = (ru << 64) | r.Next();

    RfcUuid ur = RfcUuid::FromRawLoseData(ru);
    std::array<char, 36U> buf;
    ur.PutString(buf.data());

    for (uint32_t j = 0; j < 36U; ++j) {
      seen[j].insert(buf[j]);
    }

    // Also check ToString
    s = ur.ToString();
    EXPECT_EQ(s, std::string(buf.data(), 36U));

    // Check that bottom 6 bits are completely ignored by FromRawLoseData.
    RawUuid ru2 = ru ^ (r.Next() & 0x3fU);
    RfcUuid ur2 = RfcUuid::FromRawLoseData(ru2);
    EXPECT_EQ(ur, ur2);
  }

  // Values covered for various positions
  for (uint32_t j = 0; j < 36U; ++j) {
    if (j == 8 || j == 13 || j == 18 || j == 23) {
      // Dash positions
      ASSERT_EQ(seen[j].size(), 1);
    } else if (j == 14) {
      // Version
      ASSERT_EQ(seen[j].size(), 1);
    } else if (j == 19) {
      // Variant + 2 bits data
      ASSERT_EQ(seen[j].size(), 4);
    } else {
      ASSERT_EQ(seen[j].size(), 16);
    }
  }
}

TEST(UuidTest, RocksMuidTest) {
  // Default ctor
  RocksMuid m0;
  EXPECT_EQ(m0.scaled_data, 0U);
  EXPECT_EQ(m0.ToString(), "00000000000000000000");

  // Valid
  ASSERT_OK(RocksMuid::Parse("00000000000000000000", &m0));
  EXPECT_EQ(m0.scaled_data, 0U);
  EXPECT_EQ(m0.ToString(), "00000000000000000000");

  RocksMuid m1;
  ASSERT_OK(RocksMuid::Parse("00000000000000000001", &m1));
  EXPECT_EQ(m1.scaled_data, Unsigned128{25455958});  // implementation detail
  EXPECT_EQ(m1.ToString(), "00000000000000000001");

  EXPECT_FALSE(m0 == m1);
  EXPECT_TRUE(m0 < m1);
  EXPECT_FALSE(m1 < m0);
  EXPECT_FALSE(m0 < m0);

  // Upper and lower case OK
  RocksMuid ma;
  ASSERT_OK(RocksMuid::Parse("abcdeABCDEvwxyzVWXYZ", &ma));
  RocksMuid mb;
  ASSERT_OK(RocksMuid::Parse("ABCDEabcdeVWXYZvwxyz", &mb));
  EXPECT_EQ(ma, mb);
  EXPECT_TRUE(m0 < ma);
  EXPECT_TRUE(m0.scaled_data < ma.scaled_data);

  // Pseudorandom, ensure coverage of the value space based on strings
  ROCKSDB_NAMESPACE::Random64 r{42U};
  std::array<std::set<char>, 20U> seen{};

  for (int i = 0; i < 1000; ++i) {
    // Generate random bits
    RawUuid u = r.Next();
    u = (u << 64) | r.Next();

    RocksMuid mr = RocksMuid::FromRawLoseData(u);
    std::array<char, 20U> buf;
    mr.PutString(buf.data());

    for (uint32_t j = 0; j < 20U; ++j) {
      seen[j].insert(buf[j]);
    }

    // Also check ToString
    std::string s = mr.ToString();
    EXPECT_EQ(s, std::string(buf.data(), 20U));

    // Check that bottom 24 bits play little role in FromRawLoseData. Such a
    // range can hit at most two output values.
    RawUuid u2 = u + (r.Next() & 0xffffffU);
    RawUuid u3 = u + (r.Next() & 0xffffffU);
    RocksMuid mr2 = RocksMuid::FromRawLoseData(u2);
    RocksMuid mr3 = RocksMuid::FromRawLoseData(u3);
    EXPECT_TRUE(mr == mr2 || mr2 == mr3 || mr == mr3);
  }

  // All values covered in each position
  for (uint32_t j = 0; j < 20U; ++j) {
    ASSERT_EQ(seen[j].size(), 36);
  }

  // Invalid, too short or long
  EXPECT_TRUE(RocksMuid::Parse("0", &ma).IsCorruption());
  EXPECT_TRUE(RocksMuid::Parse("0000011111222220000", &ma).IsCorruption());
  EXPECT_TRUE(RocksMuid::Parse("000001111122222000000", &ma).IsCorruption());
  // Bad character
  EXPECT_TRUE(RocksMuid::Parse("000001111122222-00000", &ma).IsCorruption());
  EXPECT_TRUE(RocksMuid::Parse("000001111122222-0000", &ma).IsCorruption());
  EXPECT_TRUE(RocksMuid::Parse("0000011111222220000-", &ma).IsCorruption());
  EXPECT_TRUE(RocksMuid::Parse("-0000011111222220000", &ma).IsCorruption());
  EXPECT_TRUE(RocksMuid::Parse("-00000111112222200000", &ma).IsCorruption());
}

namespace {

RawUuid GetCacheKey(const RocksMuid& base_session, uint64_t session_counter, uint64_t file_number, uint64_t file_offset) {
  RawUuid u = base_session.scaled_data;
  u = XorInMiddleOf3Counter(u, session_counter);
  u = XorInBottomOf3Counter(u, file_number);
  u = XorInTopOf3Counter(u, file_offset);
  return u;
}

struct ReverseEngineerCacheKey {
  // Input:
  int session_counter_max_bits;
  int file_number_max_bits;
  int file_offset_max_bits;

  // Computed on Prepare():
  RawUuid base_session_provide_mask;

  // Computed on Execute():
  RawUuid base_session;
  uint64_t session_counter;
  uint64_t file_number;
  uint64_t file_offset;

  void Prepare() {
    // To recover fields added, we need to be provided the bits that overlap
    // with them.
    RawUuid provide = 0;

    // File offset uses top counter
    // TODO: support reverse engineering overflow
    assert(file_offset_max_bits <= 44);
    provide |= ((Unsigned128{1} << file_offset_max_bits) - 1) << 84;

    // Session counter uses middle counter
    // TODO: support reverse engineering overflow
    assert(session_counter_max_bits <= 40);
    provide |= ((Unsigned128{1} << session_counter_max_bits) - 1) << 44;

    // File number uses bottom counter
    // TODO: support reverse engineering overflow
    assert(file_number_max_bits <= 44);
    provide |= (Unsigned128{1} << file_number_max_bits) - 1;

    /*
    TODO:
    // Except RocksMuid only has roughly 104 bits entropy, essentially in
    // upper bits, so we can always infer the bottom 24 bits.
    provide &= (Unsigned128{0} - 1) << 24;
    */

    base_session_provide_mask = provide;
  }

  void Execute(RawUuid cache_key, RawUuid base_session_provided_bits) {
    RawUuid base_session_infer_mask = ~base_session_provide_mask;

    assert((base_session_provided_bits & base_session_infer_mask) == 0);

    RawUuid tmp = cache_key;
    tmp ^= base_session_provided_bits;

    file_offset =
        ((uint64_t{1} << file_offset_max_bits) - 1) & Lower64of128(tmp >> 84);

    session_counter = ((uint64_t{1} << session_counter_max_bits) - 1) &
                      Lower64of128(tmp >> 44);

    file_number =
        ((uint64_t{1} << file_number_max_bits) - 1) & Lower64of128(tmp);

    tmp = GetCacheKey(RocksMuid(), session_counter, file_number, file_offset);

    base_session = tmp ^ cache_key;
  }
};

}  // namespace

TEST(UuidTest, PrototypeCacheKeys) {
  ReverseEngineerCacheKey r;
  r.file_offset_max_bits = 33;
  r.session_counter_max_bits = 34;
  r.file_number_max_bits = 35;
  r.Prepare();

  RocksMuid base_session = GenerateMuid();
  Random64 rnd{Upper64of128(base_session.scaled_data)};
  uint64_t file_offset =
      rnd.Next() & ((uint64_t{1} << r.file_offset_max_bits) - 1);
  uint64_t session_counter =
      rnd.Next() & ((uint64_t{1} << r.session_counter_max_bits) - 1);
  uint64_t file_number =
      rnd.Next() & ((uint64_t{1} << r.file_number_max_bits) - 1);

  RawUuid cache_key =
      GetCacheKey(base_session, session_counter, file_number, file_offset);

  ASSERT_EQ(BitsSetToOne(r.base_session_provide_mask),
            r.file_offset_max_bits + r.session_counter_max_bits +
                r.file_number_max_bits);
  r.Execute(cache_key, base_session.scaled_data & r.base_session_provide_mask);

  ASSERT_EQ(base_session.scaled_data, r.base_session);
  ASSERT_EQ(file_offset, r.file_offset);
  ASSERT_EQ(session_counter, r.session_counter);
  ASSERT_EQ(file_number, r.file_number);
}

TEST(UuidTest, AltCacheKeys) {
  if (!FLAGS_extra) {
    ROCKSDB_GTEST_BYPASS("Demonstration/investigation purposes only");
    return;
  }

  // constexpr uint64_t kPrime64_A = 0xe6143049bc367e43;
  ROCKSDB_NAMESPACE::Random64 r{(unsigned)getpid()};
  // constexpr uint64_t kPrime64_B = ~kPrime64_A + 1;

  for (;;) {
    // Trying 5017b98a1afc9a77, bd6cabd1a6fc377b (through 6, all three)
    // Trying 5017b6d8a311cdd3, bd6ca47d7851ca6f (through 7, all three)
    // Trying 5017b7356eeab2d5, bd6ca483db871c93 (through 8, all three)

    // Trying 0x743e68b4fd45a8b3, 0xcd01262457eccf07 (through 13, +1both only)
    // Trying 0x743e68a6f2adcaad, 0xcd01260710abfd79 (through 14)

    /*
  Trying 0x751661862ec6fe55, 0x40c555e660a9a045
  bits: 15 +1both: 0
  Trying 0xd8a79bdb867e1f5b, 0xd0ebc28596340057
  bits: 15 +1both: 0
  Trying 0xa0898ccaf49385a9, 0x8602c470d97d60cf
  bits: 15 +1both: 0
  Trying 0x7d0024c10a1deac9, 0x4def1b4e0405da21
  bits: 15 +1both: 0
  Trying 0x79f3ddc0516a5977, 0x2f6f520b223eccc9
  bits: 15 +1both: 0
  Trying 0xf582d7ad09e998ed, 0x9daf788867c13f09
  bits: 15 +1both: 0
  */

    /*
    Found 0x3d0e38c0003322e9, 0x75ae5cdff2030a3d
    bits: 12 +1both: 0 +2ao: 0 +2bo: 0
    Found 0xbc8e6deacbbb03ab, 0x31e8a2c7285ffedb
    bits: 12 +1both: 0 +2ao: 0 +2bo: 0
    Found 0x6be3888d41b849d9, 0xe179e1bc0e6d04fb
    bits: 12 +1both: 0 +2ao: 0 +2bo: 0
    Found 0x6be3888d41b849d9, 0xe179e1bc0e6d04fb
    bits: 13 +1both: 0 +2ao: 0 +2bo: 0
    Found 0x736759a8b0fde6ff, 0xa1763a3f6726d87d
    bits: 12 +1both: 0 +2ao: 0 +2bo: 0
    Found 0x736759a8b0fde6ff, 0xa1763a3f6726d87d
    bits: 13 +1both: 0 +2ao: 0 +2bo: 0
    */

    /*
    Found 0xf01f39c13d6a7d81, 0x9683ac7883fc67b7
    bits: 15 +1both: 0 +2ao: 0 +2bo: 0
    WINNER!!!!!!!!!!!!! 0xf01f39c13d6a7d81, 0x9683ac7883fc67b7
    */

    uint64_t factorA = r.Next() | 1;
    uint64_t factorB = r.Next() | 1;

    std::vector<bool> seen_plus1both;
    // std::vector<bool> seen_plus2a;
    // std::vector<bool> seen_plus2b;
    std::vector<bool> seen_plus2ao;
    std::vector<bool> seen_plus2bo;

    for (int bits = 1; bits <= 17; ++bits) {
      int nbits = 64 - bits;
      uint64_t max = uint64_t{1} << (2 * bits);
      seen_plus1both.assign(max << 2, false);
      // seen_plus2a.assign(max << 2, false);
      // seen_plus2b.assign(max << 2, false);
      seen_plus2ao.assign(max << 2, false);
      seen_plus2bo.assign(max << 2, false);

      int plus1both_collision = 0;
      // int plus2a_collision = 0;
      // int plus2b_collision = 0;
      int plus2ao_collision = 0;
      int plus2bo_collision = 0;

      for (uint64_t i = 0; i < max; ++i) {
        uint64_t a = i * factorA;
        uint64_t b = i * factorB;

        uint64_t v = ((a >> (nbits - 1)) << (bits + 1)) | (b >> (nbits - 1));
        plus1both_collision += seen_plus1both[v];
        seen_plus1both[v] = true;

        /*
        v = ((a >> (nbits - 2)) << bits) | (b >> nbits);
        plus2a_collision += seen_plus2a[v];
        seen_plus2a[v] = true;

        v = ((a >> nbits) << (bits + 2)) | (b >> (nbits - 2));
        plus2b_collision += seen_plus2b[v];
        seen_plus2b[v] = true;
        */

        v = a >> (nbits * 2 - 2);
        plus2ao_collision += seen_plus2ao[v];
        seen_plus2ao[v] = true;

        v = b >> (nbits * 2 - 2);
        plus2bo_collision += seen_plus2bo[v];
        seen_plus2bo[v] = true;
      }

      // fprintf(stderr, "bits: %d +1both: %d +1ao: %d +1bo: %d +2ao: %d +2bo:
      // %d\n", bits, plus1both_collision, plus1ao_collision, plus1bo_collision,
      // plus2ao_collision, plus2bo_collision);
      if (plus1both_collision || plus2ao_collision || plus2bo_collision) break;
      // if (bits > 8 && (plus2a_collision || plus2b_collision)) break;
      //*
      if (bits >= 13) {
        fprintf(stderr, "Found 0x%lx, 0x%lx\n", factorA, factorB);
        fprintf(stderr, "bits: %d +1both: %d +2ao: %d +2bo: %d\n", bits,
                plus1both_collision, plus2ao_collision, plus2bo_collision);
      }
      if (bits >= 15) {
        fprintf(stderr, "WINNER!!!!!!!!!!!!! 0x%lx, 0x%lx\n", factorA, factorB);
      }
      //*/
    }
  }
}

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  fprintf(stderr, "NPHash64 id: %x\n",
          static_cast<int>(ROCKSDB_NAMESPACE::GetSliceNPHash64("RocksDB")));
  ::testing::InitGoogleTest(&argc, argv);
#ifdef GFLAGS
  ParseCommandLineFlags(&argc, &argv, true);
#endif  // GFLAGS

  return RUN_ALL_TESTS();
}
