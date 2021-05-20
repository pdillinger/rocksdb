//  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include "rocksdb/status.h"
#include "util/math128.h"

namespace ROCKSDB_NAMESPACE {

// A holder for an identifier of up to 128 bits.
using RawUuid = Unsigned128;

// TODO: add Hash128 to generate RawUuid from arbitrary strings/data

// An RFC 4122 Universally Unique IDentifier. Variant 1 version 4 is the
// most common, which offers 122 bits of entropy. The text form is 32
// hexadecimal digits and four hyphens, for a total of 36 characters.
// https://en.wikipedia.org/wiki/Universally_unique_identifier
//
// This is considered suitable for identifying medium-high frequency events.
// For example, if 1 million computers each generate 10 random RfcUuids every
// second for a year, the chance of one being repeated in that time is 1 in
// millions. See GenerateRfcUuid().
//
// Implementation detail: RocksDB treats the uuid as a 128-bit scalar with
// native byte order in memory.
struct RfcUuid {
  RawUuid data = 0;

  // Parses from the 36-character form with hexadecimal digits and four
  // hyphens. Returns Corruption if malformed.
  static Status Parse(const Slice &input, RfcUuid *out);

  // Assuming the input is random with 128 bits of entropy, this converts to
  // an RfcUuid by setting variant 1 version 4, losing 6 bits of entropy.
  //
  // Implementation detail: some bits are shifted right to make room for
  // encoding variant and version. Only the lowest 6 bits are dropped from the
  // input, to ensure preserving the guarantees of GenerateRawUuid().
  static RfcUuid FromRawLoseData(const RawUuid &raw);

  // Generates the 36-character text form
  std::string ToString() const;
  // Generates the 36-character text form directly into a buffer,
  // returning a Slice into the buffer for the 36 characters written.
  Slice PutString(char *buf_of_36) const;

  // Should only be true when unset
  inline bool IsEmpty() { return data == 0; }

  // From RFC 4122, except reserved/unspecified variants are unchecked and
  // might be returned. Variant is >= 0 and <= 4
  int GetVariant() const;
  // Sets variant while leaving other data bits in place. Note that higher
  // variants overwrite more data bits, so the variant should generally only
  // be set once.
  void SetVariant(int variant);

  // From RFC 4122, Version is >= 0 and <= 15
  int GetVersion() const;
  // Sets version while leaving other data bits in place.
  void SetVersion(int version);
};

inline bool operator<(const RfcUuid &lhs, const RfcUuid &rhs) {
  return lhs.data < rhs.data;
}

inline bool operator==(const RfcUuid &lhs, const RfcUuid &rhs) {
  return lhs.data == rhs.data;
}

// A "mostly unique ID" encodable into 20 characters in base 36 ([0-9][A-Z])
// This provides a compact, portable text representation with 103.4 bits of
// entropy. This was originally developed for DB Session IDs (see
// DB::GetDbSessionId(). Any value including "all zeros" could be a valid
// generated RocksMuid. See GenerateMuid().
//
// This is considered suitable for identifying medium-frequency events or
// narrow scope events. For example, if 1 million computers each generate a
// random RocksMuid every minute for a year, or one computer generates a
// million every second for a week, the chance of a RocksMuid being repeated
// in that time is 1 in millions. And this does not account for guaranteed
// uniqueness within process lifetime, which reduces the likelihood of any
// collisions.
//
// In text form, the digits are in standard written order: lowest-order last.
struct RocksMuid {
  // Implementation detail: for better integration with some other Uuid
  // manipulation, the range of 36**20 values are spread evenly (scaled)
  // over the 2**128 raw value range, essentially occupying the "upper" bits
  // of this data field. With rounding, the scaled raw interval between
  // muid values is usually 25455957, sometimes 25455958. This means, for
  // example, that we can simply add an offset of up to 25 million without
  // data loss.
  RawUuid scaled_data = 0;

  // Parses from the 20-digit base-36 text form. Returns Corruption if
  // malformed.
  static Status Parse(const Slice &input, RocksMuid *out);

  // This converts a RawUuid to a RocksMuid by dropping some data.
  // Assuming the input is random with 128 bits of entropy, about 24.6 bits
  // of entropy are lost in the conversion. This conversion is idempotent
  // (no-op) when applied to an existing RocksMuid::scaled_data.
  // Note: This is not appropriate for RfcUuid::data because of the position
  // of missing entropy for version and variant.
  static RocksMuid FromRawLoseData(const RawUuid &raw);

  // For generating a human readable string, base-36 with 20 digits
  std::string ToString() const;
  // For generating a human readable string (base-36 with 20 digits)
  // into an existing buffer. No trailing nul is written or expected.
  // Returns a Slice for the buffer and 20 written digits.
  Slice PutString(char *buf_of_20) const;

 private:
  static Unsigned128 ReducedToScaled(const Unsigned128 &reduced);
  static void ScaledToPieces(const Unsigned128 &scaled, uint64_t *upper_piece,
                             uint64_t *lower_piece);
  static Unsigned128 PiecesToReduced(uint64_t upper_piece,
                                     uint64_t lower_piece);
  static Unsigned128 ScaledToReduced(const Unsigned128 &scaled);

  // 36 to the 10th power
  static constexpr uint64_t kPieceMod = 3656158440062976;
};

inline bool operator<(const RocksMuid &lhs, const RocksMuid &rhs) {
  return lhs.scaled_data < rhs.scaled_data;
}

inline bool operator==(const RocksMuid &lhs, const RocksMuid &rhs) {
  return lhs.scaled_data == rhs.scaled_data;
}

// Auxiliary function for mixing a counter into a Uuid with a balance of
// nice uniqueness-preserving properties and hash-like mixing properties.
// The uniqueness-preserving properties
// * For a fixed `counter`, output is 1:1 with input `u`.
// * Each output bit depends on at least the bottom 32 `counter` bits.
//   (There is no attempt to mix up `u` in the output, just preserve its
//   entropy.)
// For a fixed input `u`:
// * Either 64-bit half of output is sufficient to uniquely identify the
//   input counter.
// * For a counter of up to n bits, n <= 30, the input counter can be
//   uniquely identified using either
//   * n + 2 upper-most bits of *either* 64-bit half of output
//   * ceil(n / 2) + 1 upper-most bits of *both* 64-bit halfs of output.
//
// Implementation note:
RawUuid MixAndAddCounterToUuid(const RawUuid &u, uint64_t counter);

// For mixing in up to three "counters" into a RawUuid. They are generally
// expected to only vary in the lowest 40 bits (counter up to 1 trillion),
// but higher bits can spill over into the upper part of an adjacent value.
// Specific guarantees:
// * For the same `u`, can preserve without collision the following
//   combinations:
//   * "Balanced" combination
//     * Lowest 44 bits of top_counter
//     * Lowest 40 bits of middle_counter
//     * Lowest 44 bits of bottom_counter
//   * "Top overflow" combination
//     * Lowest 44 + n bits of top_counter
//     * Lowest 40 bits of middle_counter
//     * Lowest 44 - n - 1 bits of bottom_counter
//   * "Middle overflow" combination
//     * Lowest 44 - n - 1 bits of top_counter
//     * Lowest 40 + n bits of middle_counter
//     * Lowest 44 bits of bottom_counter
//   * "Bottom overflow" combination
//     * Lowest 44 bits of top_counter
//     * Lowest 40 - n - 1 bits of middle_counter
//     * Lowest 44 + n bits of bottom_counter
// * For the same `u`, no naive bit correlations between counters can lead
//   to collision. At least ~20 other counter bits need to change to cause
//   a collision when one bit is flipped. This provides for reasonably good
//   probabilistic uniqueness even when pushing this approach to the extreme.
//   * For such an extreme example, suppose only the bottom 40 bits of
//     middle_counter and the top 20 bits of bottom_counter vary. Because of
//     funneling those top 20 "overflow" bits in with the middle 40 bit
//     counter, we still need about 2^20 ~= 1 million samples to expect a
//     collision (Birthday problem). This is considered an extreme worst case
//     because counters are supposed to generally have more entropy in lower
//     bits than in higher bits, so variance in the extra 44 bits of
//     bottom_counter should make collisions generally impossible even in
//     overflow cases.
RawUuid XorInTopOf3Counter(const RawUuid &u, uint64_t top_counter);
RawUuid XorInMiddleOf3Counter(const RawUuid &u, uint64_t middle_counter);
RawUuid XorInBottomOf3Counter(const RawUuid &u, uint64_t bottom_counter);

// If XorInMiddleOf3Counter is already used and you only need to mix in one
// more counter, this will (almost) perfectly complement
// XorInMiddleOf3Counter for no possibility of collision for the same `u`
// (up to 127 bits in counters).
RawUuid XorInNonMiddleCounter(const RawUuid &u, uint64_t middle_counter);

}  // namespace ROCKSDB_NAMESPACE

namespace std {
template <>
struct hash<ROCKSDB_NAMESPACE::RfcUuid> {
  std::size_t operator()(ROCKSDB_NAMESPACE::RfcUuid const &u) const noexcept {
    return std::hash<ROCKSDB_NAMESPACE::Unsigned128>()(u.data);
  }
};

template <>
struct hash<ROCKSDB_NAMESPACE::RocksMuid> {
  std::size_t operator()(ROCKSDB_NAMESPACE::RocksMuid const &u) const noexcept {
    return std::hash<ROCKSDB_NAMESPACE::Unsigned128>()(u.scaled_data);
  }
};
}  // namespace std
