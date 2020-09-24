//  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include "util/sgauss_alg.h"

namespace ROCKSDB_NAMESPACE {

namespace SGauss {

// concept TypesAndSettings {
//   typename CoeffRow;
//   typename ResultRow;
//   typename Index;
//   typename Hash;
//   typename Key;
//   typename Seed;
//   static constexpr bool kIsFilter;
//   static constexpr bool kFirstCoeffAlwaysOne;
//   static constexpr bool kUsePrefetch;
//   static Hash HashFn(const Key &, Seed);
// };

// A bit of a hack to automatically construct the type for
// BuilderInput based on a constexpr bool.
template<class TypesAndSettings, bool IsFilter>
class BuilderInputSelector : public TypesAndSettings {
public:
  // For general PHSF, not filter
  using BuilderInput = std::pair<Key, ResultRow>;

  inline ResultRow GetResultRowMask() const {
    // all bits set
    return ResultRow{0} - ResultRow{1};
  }
};

template<class TypesAndSettings>
class BuilderInputSelector<TypesAndSettings, true /*IsFilter*/> : public TypesAndSettings {
public:
  // For Filter
  using BuilderInput = Key;

  inline ResultRow GetResultRowMask() const {
    return rr_mask_;
  }
protected:
  ResultRow rr_mask_ = ResultRow{0} - ResultRow{1};
};

template<class TypesAndSettings>
class StandardHasher : public BuilderInputSelector<TypesAndSettings, TypesAndSettings::kIsFilter> {
public:
  using QueryInput = Key;

  inline Hash GetHash(const Key& key) const {
    return HashFn(key, seed_);
  };
  inline Hash GetHash(const std::pair<Key, ResultRow>& bi) const {
    return GetHash(bi.first);
  };
  inline Index GetStart(Hash h) const {
    // TODO
  }
  inline CoeffRow GetCoeffRow(Hash h) const {
    // TODO
  }
  inline ResultRow GetResultRowFromHash(Hash h) const {
    if (kFilter) {
      // TODO
      // TODO: mask?
    } else {
      return 0;
    }
  }
  inline ResultRow GetResultRowFromInput(const Key&) const {
    return 0;
  }
  inline ResultRow GetResultRowFromInput(const std::pair<Key, ResultRow>& bi) const {
    return bi.second;
  }

  bool NextSeed(Seed max_seed) {
    if (seed_ >= max_seed) {
      return false;
    } else {
      ++seed_;
      return true;
    }
  }
  Seed GetSeed() const {
    return seed_;
  }
protected:
  Seed seed_ = 0;
};





}  // namespace SGauss

}  // namespace ROCKSDB_NAMESPACE
