//  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include "util/sgauss_alg.h"

namespace ROCKSDB_NAMESPACE {

namespace sgauss {

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
template<typename Key, typename ResultRow, bool IsFilter>
struct BuilderInputSelector {
  // For general PHSF, not filter
  using T = std::pair<Key, ResultRow>;
};

template<typename Key, typename ResultRow>
struct BuilderInputSelector<Key, ResultRow, true /*IsFilter*/> {
  // For Filter
  using T = Key;
};

// To avoid writing 'typename' everwhere we use types like 'Index'
#define IMPORT_TYPES_AND_SETTINGS(TypesAndSettings) \
  using CoeffRow = typename TypesAndSettings::CoeffRow; \
  using ResultRow = typename TypesAndSettings::ResultRow; \
  using Index = typename TypesAndSettings::Index; \
  using Hash = typename TypesAndSettings::Hash; \
  using Key = typename TypesAndSettings::Key; \
  using Seed = typename TypesAndSettings::Seed; \
\
  /* Some more additions */ \
  using QueryInput = Key; \
  using BuilderInput = typename BuilderInputSelector<Key, ResultRow, TypesAndSettings::kIsFilter>::T; \
  static constexpr auto kCoeffBits = static_cast<Index>(sizeof(CoeffRow) * 8U); \
\
  /* Export to algorithm */ \
  static constexpr bool kFirstCoeffAlwaysOne = TypesAndSettings::kFirstCoeffAlwaysOne; \


template<class TypesAndSettings>
class StandardHasher {
public:
  IMPORT_TYPES_AND_SETTINGS(TypesAndSettings);

  inline Hash GetHash(const Key& key) const {
    return TypesAndSettings::HashFn(key, seed_);
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
    if (TypesAndSettings::kUseSmash) {
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
    Unsigned128 a = Multiply64to128(h, 0x9e3779b97f4a7c13U); // FIXME: better values
    Unsigned128 b = Multiply64to128(h, 0xa4398ab94d038781U);
    auto cr = static_cast<CoeffRow>(b ^ (a << 64) ^ (a >> 64));
    if (kFirstCoeffAlwaysOne) {
      cr |= 1;
    } else {
      // Still have to ensure non-zero
      cr |= static_cast<unsigned>(cr == 0);
    }
    return cr;
  }
  inline ResultRow GetResultRowMask() const {
    // TODO
    // For now, all bits set
    return ResultRow{0} - ResultRow{1};
  }
  inline ResultRow GetResultRowFromHash(Hash h) const {
    if (TypesAndSettings::kIsFilter) {
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
  void ResetSeed() {
    seed_ = 0;
  }
protected:
  Seed seed_ = 0;
};

template<class TypesAndSettings>
class StandardSolver : public StandardHasher<TypesAndSettings> {
public:
  IMPORT_TYPES_AND_SETTINGS(TypesAndSettings);

  StandardSolver(Index num_slots = 0, Index backtrack_size = 0) {
    Reset(num_slots, backtrack_size);
  }
  void Reset(Index num_slots, Index backtrack_size = 0) {
    assert(num_slots >= kCoeffBits);
    if (num_slots > num_slots_allocated_) {
      coeff_rows_.reset(new CoeffRow[num_slots]());
      // Note: don't strictly have to zero-init result_rows,
      // except possible information leakage ;)
      result_rows_.reset(new ResultRow[num_slots]());
      num_slots_allocated_ = num_slots;
    } else {
      for (Index i = 0; i < num_slots; ++i) {
        coeff_rows_[i] = 0;
        // Note: don't strictly have to zero-init result_rows
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
    return TypesAndSettings::kUsePrefetch;
  }
  inline void Prefetch(Index i) const {
    // TODO
  }
  inline CoeffRow* CoeffRowPtr(Index i) {
    return &coeff_rows_[i];
  }
  inline ResultRow* ResultRowPtr(Index i) {
    return &result_rows_[i];
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

  // Some useful API, still somewhat low level
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

  // High-level API
  // TODO: detail
  template<typename InputIterator>
  bool ResetAndFindSeedToSolve(Index num_slots, InputIterator begin, InputIterator end, Seed max_seed) {
    StandardHasher<TypesAndSettings>::ResetSeed();
    do {
      Reset(num_slots);
      bool success = SolveMore(begin, end);
      if (success) {
        return true;
      }
    } while (StandardHasher<TypesAndSettings>::NextSeed(max_seed));
    // no seed through max_seed worked
    return false;
  }
protected:
  // TODO: explore combining in a struct
  std::unique_ptr<CoeffRow[]> coeff_rows_;
  std::unique_ptr<ResultRow[]> result_rows_;
  Index num_starts_ = 0;
  Index num_slots_allocated_ = 0;
  std::unique_ptr<Index[]> backtrack_;
  Index backtrack_size_ = 0;
};

// Implements concept SimpleSolutionStorage
template<class TypesAndSettings>
class InMemSimpleSolution {
public:
  IMPORT_TYPES_AND_SETTINGS(TypesAndSettings);

  void PrepareForNumStarts(Index num_starts) {
    const Index num_slots = num_starts + kCoeffBits - 1;
    assert(num_slots >= kCoeffBits);
    if (num_slots > num_slots_allocated_) {
      // Do not need to init the memory
      solution_rows_.reset(new ResultRow[num_slots]);
      num_slots_allocated_ = num_slots;
    }
    num_starts_ = num_starts;
  }
  Index GetNumStarts() const {
    return num_starts_;
  }
  ResultRow Load(Index slot_num) const {
    return solution_rows_[slot_num];
  }
  void Store(Index slot_num, ResultRow solution_row) {
    solution_rows_[slot_num] = solution_row;
  }
  template<typename SolverStorage>
  void BackSubstFrom(const SolverStorage &ss) {
    SimpleBackSubst(this, ss);
  }
protected:
  Index num_starts_ = 0;
  Index num_slots_allocated_ = 0;
  std::unique_ptr<ResultRow[]> solution_rows_;
};



}  // namespace SGauss

}  // namespace ROCKSDB_NAMESPACE
