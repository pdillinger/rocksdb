//  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include "port/port.h"  // for PREFETCH
#include "util/ribbon_alg.h"

namespace ROCKSDB_NAMESPACE {

namespace ribbon {

// RIBBON PHSF & RIBBON Filter (Rapid Incremental Boolean Banding ON-the-fly)
//
// ribbon_impl.h: templated (parameterized) standard implementations
//
// Ribbon is a Perfect Hash Static Function construction useful as a compact
// static Bloom filter alternative. See ribbon_alg.h for core algorithms
// and core design details.
//
// TODO: more details on trade-offs and practical issues.

// Ribbon implementations in this file take these parameters, which must be
// provided in a class/struct type with members expressed in this concept:

// concept TypesAndSettings {
//   // See RibbonTypes and *Hasher in ribbon_alg.h, except here we have
//   // the added constraint that Hash be equivalent to either uint32_t or
//   // uint64_t.
//   typename Hash;
//   typename CoeffRow;
//   typename ResultRow;
//   typename Index;
//   typename Key;
//   static constexpr bool kFirstCoeffAlwaysOne;
//
//   // An unsigned integer type for identifying a hash seed, typically
//   // uint32_t or uint64_t.
//   typename Seed;
//
//   // When true, the PHSF implements a static filter, expecting just
//   // keys as inputs for construction. When false, implements a general
//   // PHSF and expects std::pair<Key, ResultRow> as inputs for
//   // construction.
//   static constexpr bool kIsFilter;
//
//   // When true, adds a tiny bit more hashing logic on queries and
//   // construction to improve utilization at the beginning and end of
//   // the structure.  Recommended when CoeffRow is only 64 bits (or
//   // less), so typical num_starts < 10k.
//   static constexpr bool kUseSmash;
//
//   // A seedable stock hash function on Keys. All bits of Hash must
//   // be reasonably high quality. XXH functions recommended, but
//   // Murmur, City, Farm, etc. also work.
//   //
//   // If sequential seeds are not sufficiently independent for your
//   // stock hash function, consider multiplying by a large odd constant.
//   // If seed 0 is still undesirable, consider adding 1 before the
//   // multiplication.
//   static Hash HashFn(const Key &, Seed);
// };

// A bit of a hack to automatically construct the type for
// AddInput based on a constexpr bool.
template <typename Key, typename ResultRow, bool IsFilter>
struct AddInputSelector {
  // For general PHSF, not filter
  using T = std::pair<Key, ResultRow>;
};

template <typename Key, typename ResultRow>
struct AddInputSelector<Key, ResultRow, true /*IsFilter*/> {
  // For Filter
  using T = Key;
};

// To avoid writing 'typename' everwhere that we use types like 'Index'
#define IMPORT_RIBBON_TYPES_AND_SETTINGS(TypesAndSettings)                   \
  using CoeffRow = typename TypesAndSettings::CoeffRow;                      \
  using ResultRow = typename TypesAndSettings::ResultRow;                    \
  using Index = typename TypesAndSettings::Index;                            \
  using Hash = typename TypesAndSettings::Hash;                              \
  using Key = typename TypesAndSettings::Key;                                \
  using Seed = typename TypesAndSettings::Seed;                              \
                                                                             \
  /* Some more additions */                                                  \
  using QueryInput = Key;                                                    \
  using AddInput = typename ROCKSDB_NAMESPACE::ribbon::AddInputSelector<     \
      Key, ResultRow, TypesAndSettings::kIsFilter>::T;                       \
  static constexpr auto kCoeffBits =                                         \
      static_cast<Index>(sizeof(CoeffRow) * 8U);                             \
                                                                             \
  /* Export to algorithm */                                                  \
  static constexpr bool kFirstCoeffAlwaysOne =                               \
      TypesAndSettings::kFirstCoeffAlwaysOne;                                \
                                                                             \
  static_assert(sizeof(CoeffRow) + sizeof(ResultRow) + sizeof(Index) +       \
                        sizeof(Hash) + sizeof(Key) + sizeof(Seed) +          \
                        sizeof(QueryInput) + sizeof(AddInput) + kCoeffBits + \
                        kFirstCoeffAlwaysOne >                               \
                    0,                                                       \
                "avoid unused warnings, semicolon expected after macro call")

// StandardHasher: A standard implementation of concepts RibbonTypes,
// PhsfQueryHasher, FilterQueryHasher, and BandingHasher from ribbon_alg.h.
//
// This implementation should be suitable for most all practical purposes
// as it "behaves" across a wide range of settings, with little room left
// for improvement. The key functionality in this hasher is generating
// CoeffRows, starts, and (for filters) ResultRows, which could be ~150
// bits of data or more, from a modest hash of 64 or even just 32 bits, with
// enough uniformity and bitwise independence to be close to "the best you
// can do" with available hash information in terms of FP rate and
// compactness. (64 bits recommended and sufficient for PHSF practical
// purposes.)
template <class TypesAndSettings>
class StandardHasher {
 public:
  IMPORT_RIBBON_TYPES_AND_SETTINGS(TypesAndSettings);

  StandardHasher(Seed seed = 0) : seed_(seed) {}

  inline Hash GetHash(const Key& key) const {
    return TypesAndSettings::HashFn(key, seed_);
  };
  // For when AddInput == pair<Key, ResultRow> (kIsFilter == false)
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
    //
    if (TypesAndSettings::kUseSmash) {
      // Extra logic to "smash" entries at beginning and end, for
      // better utilization. For example, without smash and with
      // kFirstCoeffAlwaysOne, there's about a 30% chance that the
      // first slot in the banding will be unused, and worse without
      // kFirstCoeffAlwaysOne. The ending slots are even less utilized
      // without smash.
      //
      // But since this only affects roughly kCoeffBits of the slots,
      // it's usually small enough to be ignorable (less computation in
      // this function) when number of slots is roughly 10k or larger.
      //
      // TODO: re-check these degress of smash, esp with kFirstCoeffAlwaysOne
      //
      constexpr auto kFrontSmash = kCoeffBits / 2 - 1;
      constexpr auto kBackSmash = kCoeffBits / 2;
      Index start = FastRangeGeneric(h, num_starts + kFrontSmash + kBackSmash);
      start = std::max(start, kFrontSmash);
      start -= kFrontSmash;
      start = std::min(start, num_starts - 1);
      return start;
    } else {
      // For query speed, we allow small number of initial and final
      // entries to be under-utilized.
      // NOTE: This call statically enforces that Hash is equivalent to
      // either uint32_t or uint64_t.
      return FastRangeGeneric(h, num_starts);
    }
  }
  inline CoeffRow GetCoeffRow(Hash h) const {
    // This is a reasonably cheap but empirically effective remix/expansion
    // of the hash data to fill CoeffRow. (Large primes)
    // This is not so much "critical path" code because it can be done in
    // parallel (instruction level) with memory lookup.
    Unsigned128 a = Multiply64to128(h, 0x85EBCA77C2B2AE63U);
    Unsigned128 b = Multiply64to128(h, 0x27D4EB2F165667C5U);
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
    // TODO: will be used with InterleavedSolutionStorage
    // For now, all bits set (note: might be a small type so might need to
    // narrow after promotion)
    return static_cast<ResultRow>(~ResultRow{0});
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
  // For when AddInput == Key (kIsFilter == true)
  inline ResultRow GetResultRowFromInput(const Key&) const {
    // Must be zero
    return 0;
  }
  // For when AddInput == pair<Key, ResultRow> (kIsFilter == false)
  inline ResultRow GetResultRowFromInput(
      const std::pair<Key, ResultRow>& bi) const {
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
  Seed GetSeed() const { return seed_; }
  void ResetSeed(Seed seed = 0) { seed_ = seed; }

 protected:
  Seed seed_;
};

// StandardRehasher (and StandardRehasherAdapter): A variant of
// StandardHasher that uses the same type for keys as for hashes.
// This is primarily intended for building a Ribbon filter/PHSF
// from existing hashes without going back to original inputs in order
// to apply a different seed. This hasher seeds a 1-to-1 mixing
// transformation to apply a seed to an existing hash (or hash-sized key).
//
// concept RehasherTypesAndSettings: like TypesAndSettings but
// does not require Key or HashFn.
template <class RehasherTypesAndSettings>
class StandardRehasherAdapter : public RehasherTypesAndSettings {
 public:
  using Hash = typename RehasherTypesAndSettings::Hash;
  using Key = Hash;
  using Seed = typename RehasherTypesAndSettings::Seed;

  static Hash HashFn(const Hash& input, Seed seed) {
    static_assert(sizeof(Hash) <= 8, "Hash too big");
    if (sizeof(Hash) > 4) {
      // XXH3_avalanche / XXH3p_avalanche (64-bit), modified for seed
      uint64_t h = input;
      h ^= h >> 37;
      h ^= seed * uint64_t{0xC2B2AE3D27D4EB4F};
      h *= uint64_t{0x165667B19E3779F9};
      h ^= h >> 32;
      return static_cast<Hash>(h);
    } else {
      // XXH32_avalanche (32-bit)
      uint32_t h32 = input;
      h32 ^= h32 >> 15;
      h32 ^= seed * uint32_t{0x27D4EB4F};
      h32 *= uint32_t{0x85EBCA77};
      h32 ^= h32 >> 13;
      h32 *= uint32_t{0xC2B2AE3D};
      h32 ^= h32 >> 16;
      return static_cast<Hash>(h32);
    }
  }
};

// See comment on StandardRehasherAdapter
template <class RehasherTypesAndSettings>
using StandardRehasher =
    StandardHasher<StandardRehasherAdapter<RehasherTypesAndSettings>>;

template <class TypesAndSettings>
class StandardBanding : public StandardHasher<TypesAndSettings> {
 public:
  IMPORT_RIBBON_TYPES_AND_SETTINGS(TypesAndSettings);

  StandardBanding(Index num_slots = 0, Index backtrack_size = 0) {
    if (num_slots > 0) {
      Reset(num_slots, backtrack_size);
    } else {
      EnsureBacktrackSize(backtrack_size);
    }
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
    EnsureBacktrackSize(backtrack_size);
  }
  void EnsureBacktrackSize(Index backtrack_size) {
    if (backtrack_size > backtrack_size_) {
      backtrack_.reset(new Index[backtrack_size]);
      backtrack_size_ = backtrack_size;
    }
  }

  // ********************************************************************
  // From concept SolverStorage

  inline bool UsePrefetch() const {
    // A rough guestimate of when prefetching during construction pays off.
    // TODO: verify/validate
    return num_starts_ > 1500;
  }
  inline void Prefetch(Index i) const {
    PREFETCH(&coeff_rows_[i], 1 /* rw */, 1 /* locality */);
    PREFETCH(&result_rows_[i], 1 /* rw */, 1 /* locality */);
  }
  inline CoeffRow* CoeffRowPtr(Index i) { return &coeff_rows_[i]; }
  inline ResultRow* ResultRowPtr(Index i) { return &result_rows_[i]; }
  inline Index GetNumStarts() const { return num_starts_; }

  // from concept BacktrackStorage, for when backtracking is used
  inline bool UseBacktrack() const { return true; }
  inline void BacktrackPut(Index i, Index to_save) { backtrack_[i] = to_save; }
  inline Index BacktrackGet(Index i) const { return backtrack_[i]; }

  // ********************************************************************
  // Some useful API, still somewhat low level. Here an input is
  // a Key for filters, or std::pair<Key, ResultRow> for general PHSF.

  // Adds a range of inputs to the banding, returning true if successful.
  // False means none or some may have been successfully added, so it's
  // best to Reset this banding before any further use.
  //
  // Adding can fail even before all the "slots" are completely "full".
  //
  template <typename InputIterator>
  bool AddRange(InputIterator begin, InputIterator end) {
    return BandingAddRange(this, *this, begin, end);
  }

  // Adds a range of inputs to the banding, returning true if successful,
  // or if unsuccessful, rolls back to state before this call and returns
  // false. Caller guarantees that the number of inputs in this batch
  // does not exceed `backtrack_size` provided to Reset.
  //
  // Adding can fail even before all the "slots" are completely "full".
  //
  template <typename InputIterator>
  bool AddRangeOrRollBack(InputIterator begin, InputIterator end) {
    return BandingAddRange(this, this, *this, begin, end);
  }

  // Adds a single input to the banding, returning true if successful.
  // If unsuccessful, returns false and banding state is unchanged.
  //
  // Adding can fail even before all the "slots" are completely "full".
  //
  bool Add(const AddInput& input) { return AddRange(&input, &input + 1); }

  // Return the number of "occupied" rows (with non-zero coefficients stored).
  Index GetOccupiedCount() const {
    Index count = 0;
    const Index num_slots = num_starts_ + kCoeffBits - 1;
    for (Index i = 0; i < num_slots; ++i) {
      if (coeff_rows_[i] != 0) {
        ++count;
      }
    }
    return count;
  }

  // ********************************************************************
  // High-level API

  // Iteratively (a) resets the structure for `num_slots`, (b) attempts
  // to add the range of inputs, and (c) if unsuccessful, chooses next
  // hash seed, until either successful or unsuccessful with max_seed
  // (minimum one seed attempted). Returns true if successful. In that
  // case, use GetSeed() to get the successful seed.
  template <typename InputIterator>
  bool ResetAndFindSeedToSolve(Index num_slots, InputIterator begin,
                               InputIterator end, Seed max_seed) {
    StandardHasher<TypesAndSettings>::ResetSeed();
    do {
      Reset(num_slots);
      bool success = AddRange(begin, end);
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

// Implements concept SimpleSolutionStorage, mostly for demonstration
// purposes. This is "in memory" only because it does not handle byte
// ordering issues for serialization.
template <class TypesAndSettings>
class InMemSimpleSolution {
 public:
  IMPORT_RIBBON_TYPES_AND_SETTINGS(TypesAndSettings);

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

  Index GetNumStarts() const { return num_starts_; }

  ResultRow Load(Index slot_num) const { return solution_rows_[slot_num]; }

  void Store(Index slot_num, ResultRow solution_row) {
    solution_rows_[slot_num] = solution_row;
  }

  // ********************************************************************
  // High-level API

  template <typename SolverStorage>
  void BackSubstFrom(const SolverStorage& ss) {
    SimpleBackSubst(this, ss);
  }

  template <typename PhsfQueryHasher>
  ResultRow PhsfQuery(const Key& input, const PhsfQueryHasher& hasher) {
    assert(!TypesAndSettings::kIsFilter);
    return SimplePhsfQuery(input, hasher, *this);
  }

  template <typename FilterQueryHasher>
  bool FilterQuery(const Key& input, const FilterQueryHasher& hasher) {
    assert(TypesAndSettings::kIsFilter);
    return SimpleFilterQuery(input, hasher, *this);
  }

 protected:
  Index num_starts_ = 0;
  Index num_slots_allocated_ = 0;
  std::unique_ptr<ResultRow[]> solution_rows_;
};

}  // namespace ribbon

}  // namespace ROCKSDB_NAMESPACE

// For convenience working with templates
#define IMPORT_RIBBON_IMPL_TYPES(TypesAndSettings)                            \
  using Hasher = ROCKSDB_NAMESPACE::ribbon::StandardHasher<TypesAndSettings>; \
  using Banding =                                                             \
      ROCKSDB_NAMESPACE::ribbon::StandardBanding<TypesAndSettings>;           \
  using SimpleSoln =                                                          \
      ROCKSDB_NAMESPACE::ribbon::InMemSimpleSolution<TypesAndSettings>;       \
  static_assert(sizeof(Hasher) + sizeof(Banding) + sizeof(SimpleSoln) > 0,    \
                "avoid unused warnings, semicolon expected after macro call")
