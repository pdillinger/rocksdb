//  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include <array>
#include <cassert>
#include <cmath>

#include "port/lang.h"  // for FALLTHROUGH_INTENDED
#include "rocksdb/rocksdb_namespace.h"

namespace ROCKSDB_NAMESPACE {

namespace ribbon {

// RIBBON PHSF & RIBBON Filter (Rapid Incremental Boolean Banding ON-the-fly)
//
// ribbon_config.h: APIs for relating numbers of slots with numbers of
// additions for tolerable construction failure probabilities. This is
// separate from ribbon_impl.h because it might not be needed for
// some applications.

// Represents a chosen chance of successful Ribbon construction for a single
// seed. Allowing higher chance of failed construction can reduce space
// overhead but takes extra time in construction.
enum ConstructionFailureChance {
  kOneIn2,
  kOneIn20,
  // When using kHomogeneous==true, construction failure chance should
  // not generally exceed target FP rate, so it unlikely useful to
  // allow a higher "failure" chance. In some cases, even more overhead
  // is appropriate. (TODO)
  kOneIn1000,
};

// Based on data from FindOccupancyForSuccessRate in ribbon_test

// For sufficiently large number of slots m, the number of entries that
// can be added with failure chance approximately no worse than kCfc is
// m / f where overhead factor f = kBaseFactor + kFactorPerPow2 * log2(m)

namespace detail {

template <uint64_t kCoeffBits>
struct BandingConfigHelperData1 {
  static const double kFactorPerPow2;
};

template <ConstructionFailureChance kCfc, uint64_t kCoeffBits>
struct BandingConfigHelperData2 : public BandingConfigHelperData1<kCoeffBits> {
  static const double kBaseFactor;
};

template <ConstructionFailureChance kCfc, uint64_t kCoeffBits, bool kUseSmash>
struct BandingConfigHelperData3
    : public detail::BandingConfigHelperData2<kCfc, kCoeffBits> {
  static const std::array<double, 18> kKnownByPow2;
};

template <ConstructionFailureChance kCfc, uint64_t kCoeffBits, bool kUseSmash,
          bool kHomogeneous, bool kIsSupported>
struct BandingConfigHelper1MaybeSupported {
 public:
  static uint32_t GetNumToAdd(uint32_t num_slots) {
    // Unsupported
    assert(num_slots == 0);
    return 0;
  }

  static uint32_t GetNumSlots(uint32_t num_to_add) {
    // Unsupported
    assert(num_to_add == 0);
    return 0;
  }
};

template <ConstructionFailureChance kCfc, uint64_t kCoeffBits, bool kUseSmash,
          bool kHomogeneous>
struct BandingConfigHelper1MaybeSupported<
    kCfc, kCoeffBits, kUseSmash, kHomogeneous, true /* kIsSupported */> {
 public:
  // See BandingConfigHelper1
  static uint32_t GetNumToAdd(uint32_t num_slots) {
    if (num_slots == 0) {
      return 0;
    }
    uint32_t num_to_add;
    using Data = detail::BandingConfigHelperData3<kCfc, kCoeffBits, kUseSmash>;
    double log2_num_slots = std::log(num_slots) * 1.4426950409;
    size_t floor_log2 = static_cast<size_t>(log2_num_slots);
    if (floor_log2 + 1 < Data::kKnownByPow2.size()) {
      double ceil_portion = log2_num_slots - floor_log2;
      // Must be a supported number of slots
      assert(Data::kKnownByPow2[floor_log2] > 0.0);
      // Weighted average of two nearest known data points
      num_to_add = ceil_portion * Data::kKnownByPow2[floor_log2 + 1] +
                   (1.0 - ceil_portion) * Data::kKnownByPow2[floor_log2];
    } else {
      // Use formula for large values
      double factor = Data::kBaseFactor + log2_num_slots * Data::kFactorPerPow2;
      assert(factor >= 1.0);
      num_to_add = static_cast<uint32_t>(num_slots / factor);
    }
    if (kHomogeneous) {
      // Even when standard filter construction would succeed, we might
      // have loaded things up too much for Homogeneous filter. (Complete
      // explanation not known but observed empirically.) This seems to
      // correct for that, mostly affecting small filter configurations.
      if (num_to_add >= 8) {
        num_to_add -= 8;
      } else {
        assert(false);
      }
    }
    return num_to_add;
  }

  // See BandingConfigHelper1
  static uint32_t GetNumSlots(uint32_t num_to_add) {
    if (num_to_add == 0) {
      return 0;
    }
    if (kHomogeneous) {
      // Reverse of above in GetNumToAdd
      num_to_add += 8;
    }
    using Data = detail::BandingConfigHelperData3<kCfc, kCoeffBits, kUseSmash>;
    double log2_num_to_add = std::log(num_to_add) * 1.4426950409;
    size_t approx_log2_slots = static_cast<size_t>(log2_num_to_add + 0.5);
    double lower_num_to_add;
    double upper_num_to_add;
    double lower_num_slots = uint32_t{1} << approx_log2_slots;
    if (approx_log2_slots + 1 < Data::kKnownByPow2.size()) {
      if (num_to_add < Data::kKnownByPow2[approx_log2_slots]) {
        --approx_log2_slots;
        lower_num_slots /= 2.0;
      }
      // approx_log2_slots is now floor(log2(slots))
      lower_num_to_add = Data::kKnownByPow2[approx_log2_slots];
      if (lower_num_to_add == 0 /* unsupported */) {
        // Return minimum non-zero slots in standard implementation
        return kUseSmash ? kCoeffBits : 2 * kCoeffBits;
      }
      upper_num_to_add = Data::kKnownByPow2[approx_log2_slots + 1];
    } else {
      if (num_to_add <
          lower_num_slots /
              (Data::kBaseFactor + approx_log2_slots * Data::kFactorPerPow2)) {
        --approx_log2_slots;
        lower_num_slots /= 2.0;
      }
      // approx_log2_slots is now floor(log2(slots))
      lower_num_to_add =
          lower_num_slots /
          (Data::kBaseFactor + approx_log2_slots * Data::kFactorPerPow2);
      upper_num_to_add =
          lower_num_slots * 2.0 /
          (Data::kBaseFactor + (approx_log2_slots + 1) * Data::kFactorPerPow2);
    }
    assert(num_to_add >= lower_num_to_add);
    assert(num_to_add < upper_num_to_add);

    // FIXME? when lower_num_to_add == 0
    double upper_portion =
        (num_to_add - lower_num_to_add) / (upper_num_to_add - lower_num_to_add);

    // Interpolation, round up
    return static_cast<uint32_t>(upper_portion * lower_num_slots +
                                 lower_num_slots + 0.999999);
  }
};

}  // namespace detail

template <ConstructionFailureChance kCfc, uint64_t kCoeffBits, bool kUseSmash,
          bool kHomogeneous>
struct BandingConfigHelper1
    : public detail::BandingConfigHelper1MaybeSupported<
          kCfc, kCoeffBits, kUseSmash, kHomogeneous,
          /* kIsSupported */ kCoeffBits == 64 || kCoeffBits == 128> {
 public:
  // Returns a number of entries that can be added to a given number of
  // slots, with roughly kCfc chance of construction failure per seed,
  // or better. Does NOT do rounding for InterleavedSoln; call
  // RoundUpNumSlots for that.
  //
  // inherited:
  // static uint32_t GetNumToAdd(uint32_t num_slots);

  // Returns a number of slots for a given number of entries to add
  // that should have roughly kCfc chance of construction failure per
  // seed, or better. Does NOT do rounding for InterleavedSoln; call
  // RoundUpNumSlots for that.
  //
  // num_to_add should not exceed roughly 2/3rds of the maximum value
  // of the uint32_t type to avoid overflow.
  //
  // inherited:
  // static uint32_t GetNumSlots(uint32_t num_to_add);
};

// Configured using TypesAndSettings as in ribbon_impl.h
template <ConstructionFailureChance kCfc, class TypesAndSettings>
struct BandingConfigHelper1TS
    : public BandingConfigHelper1<
          kCfc,
          /* kCoeffBits */ sizeof(typename TypesAndSettings::CoeffRow) * 8U,
          TypesAndSettings::kUseSmash, TypesAndSettings::kHomogeneous> {};

// Failure chance can be a runtime rather than compile time value.
template <class TypesAndSettings>
struct BandingConfigHelper {
 public:
  static constexpr ConstructionFailureChance kDefaultFailureChance =
      TypesAndSettings::kHomogeneous ? kOneIn1000 : kOneIn20;

  static uint32_t GetNumToAdd(
      uint32_t num_slots,
      ConstructionFailureChance max_failure = kDefaultFailureChance) {
    switch (max_failure) {
      default:
        assert(false);
        FALLTHROUGH_INTENDED;
      case kOneIn20: {
        using H1 = BandingConfigHelper1TS<kOneIn20, TypesAndSettings>;
        return H1::GetNumToAdd(num_slots);
      }
      case kOneIn2: {
        using H1 = BandingConfigHelper1TS<kOneIn2, TypesAndSettings>;
        return H1::GetNumToAdd(num_slots);
      }
      case kOneIn1000: {
        using H1 = BandingConfigHelper1TS<kOneIn1000, TypesAndSettings>;
        return H1::GetNumToAdd(num_slots);
      }
    }
  }

  static uint32_t GetNumSlots(
      uint32_t num_to_add,
      ConstructionFailureChance max_failure = kDefaultFailureChance) {
    switch (max_failure) {
      default:
        assert(false);
        FALLTHROUGH_INTENDED;
      case kOneIn20: {
        using H1 = BandingConfigHelper1TS<kOneIn20, TypesAndSettings>;
        return H1::GetNumSlots(num_to_add);
      }
      case kOneIn2: {
        using H1 = BandingConfigHelper1TS<kOneIn2, TypesAndSettings>;
        return H1::GetNumSlots(num_to_add);
      }
      case kOneIn1000: {
        using H1 = BandingConfigHelper1TS<kOneIn1000, TypesAndSettings>;
        return H1::GetNumSlots(num_to_add);
      }
    }
  }
};

}  // namespace ribbon

}  // namespace ROCKSDB_NAMESPACE
