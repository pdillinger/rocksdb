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
//   static constexpr bool kUseSmash;
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
  constexpr auto kCoeffBits = static_cast<Index>(sizeof(CoeffRow) * 8U);

  inline Hash GetHash(const Key& key) const {
    return HashFn(key, seed_);
  };
  // For when BuilderInput == pair<Key, ResultRow> (kIsFilter == false)
  inline Hash GetHash(const std::pair<Key, ResultRow>& bi) const {
    return GetHash(bi.first);
  };
  inline Index GetStart(Hash h, Index num_starts) const {
    // This is "critical path" code because it's required before memory
    // lookup.
    //
    // FastRange gives us a fast and effective mapping from h to the
    // approriate range. This depends most, sometimes exclusively, on
    // upper bits of h.
    if (kUseSmash) {
      // TODO: explain more
      // These seem to work well, and happen to work out to fastrange over
      // number of slots (vs. number of starts).
      constexpr auto kFrontSmash = kCoeffBits / 2;
      constexpr auto kBackSmash = kCoeffBits / 2 - 1;
      Index start = FastRangeGeneric(h, num_starts + kFrontSmash + kBackSmash);
      start = std::max(start, kFrontSmash);
      start -= kFrontSmash;
      start = std::min(start, num_starts - 1);
      return start;
    } else {
      // For query speed, we allow small number of initial and final
      // entries to be under-utilized. Recommended when typical
      // num_starts >= 10k.
      return FastRangeGeneric(h, num_starts);
    }
  }
  inline CoeffRow GetCoeffRow(Hash h) const {
    // This is a reasonably cheap but empirically effective remix/expansion
    // of the hash data to fill CoeffRow.
    // This is not so much "critical path" code because it can be done in
    // parallel (instruction level) with memory lookup.
    uint128_t a = Multiply64to128(h, 0x9e3779b97f4a7c13U); // FIXME: better values
    uint128_t b = Multiply64to128(h, 0xa4398ab94d038781U);
    return static_cast<CoeffRow>(b ^ (a << 64) ^ (a >> 64)) | CoeffRow{kFirstCoeffAlwaysOne ? 1 : 0};
  }
  inline ResultRow GetResultRowFromHash(Hash h) const {
    if (kIsFilter) {
      // In contrast to GetStart, here we draw primarily from lower bits,
      // but not literally, which seemed to cause FP rate hit in some cases.
      // This is not so much "critical path" code because it can be done in
      // parallel (instruction level) with memory lookup.
      auto rr = static_cast<ResultRow>(h ^ (h >> 13) ^ (h >> 26));
      return rr & GetResultRowMask();
    } else {
      // Must be zero
      return 0;
    }
  }
  // For when BuilderInput == Key (kIsFilter == true)
  inline ResultRow GetResultRowFromInput(const Key&) const {
    // Must be zero
    return 0;
  }
  // For when BuilderInput == pair<Key, ResultRow> (kIsFilter == false)
  inline ResultRow GetResultRowFromInput(const std::pair<Key, ResultRow>& bi) const {
    // Simple extraction
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

template<class TypesAndSettings>
class StandardSolver : public StandardHasher<TypesAndSettings> {
public:
  StandardSolver(Index num_slots = 0, Index backtrack_size = 0) {
    Reset(num_slots, Index backtrack_size);
  }
  void Reset(Index num_slots, Index backtrack_size = 0) {
    assert(num_slots >= kCoeffBits);
    if (num_slots > num_slots_allocated_) {
      coeff_rows.reset(new CoeffRow[num_slots]());
      // Note: don't strictly have to zero-init result_rows,
      // except possible information leakage ;)
      result_rows.reset(new ResultRow[num_slots]());
      num_slots_allocated_ = num_slots;
    } else {
      for (Index i = 0; i < num_slots; ++i) {
        coeff_rows_[i] = 0;
        result_rows_[i] = 0;
      }
    }
    num_starts_ = num_slots - kCoeffBits + 1;
    if (backtrack_size > backtrack_size_) {
      backtrack_.reset(new Index[backtrack_size]);
      backtrack_size_ = backtrack_size;
    }
  }

  // from concept SolverStorage
  inline bool UsePrefetch() const {
    return kUsePrefetch;
  }
  inline void Prefetch(Index i) const {
    // TODO
  }
  inline CoeffRow* CoeffRowPtr(Index i) {
    return coeff_rows_[i];
  }
  inline ResultRow* ResultRowPtr(Index i) {
    return result_rows_[i];
  }
  inline Index GetNumStarts() const {
    return num_starts_;
  }

  // from concept BacktrackStorage, for when backtracking is used
  inline bool UseBacktrack() const {
    return true;
  }
  inline void BacktrackPut(Index i, Index to_save) {
    backtrack_[i] = to_save;
  }
  inline Index BacktrackGet(Index i) const {
    return backtrack_[i];
  }

  // Some useful API
  // TODO: detail
  template<typename InputIterator>
  bool SolveMore(InputIterator begin, InputIterator end) {
    return Solve(this, *this, begin, end);
  }

  // TODO: detail
  template<typename InputIterator>
  bool BacktrackableSolveMore(InputIterator begin, InputIterator end) {
    return BacktrackableSolve(this, this, *this, begin, end);
  }

protected:
  // TODO: explore combining in a struct
  std::unique_ptr<CoeffRow[]> coeff_rows_;
  std::unique_ptr<ResultRow[]> result_rows_;
  Index num_starts_ = 0;
  Index num_slots_allocated_ = 0;
  std::unique_ptr<Index[]> backtrack_;
  Index backtrack_size_ = 0;
}




}  // namespace SGauss

}  // namespace ROCKSDB_NAMESPACE
