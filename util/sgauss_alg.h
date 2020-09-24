//  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include "util/math128.h"

namespace ROCKSDB_NAMESPACE {

namespace SGauss {

// concept SGaussTypes {
//   typename CoeffRow;
//   typename ResultRow;
//   typename Index;
// };

// concept PhsfQueryHasher extends SGaussTypes {
//   // For a filter, Input will be just a key. For a general PHSF, it must
//   // include key and result it maps to (e.g. in a pair).
//   typename QueryInput;
//   // Type for hashed summary of "key" part of Input. uint64_t recommended.
//   typename Hash;
//
//   Hash GetHash(const QueryInput &) const;
//   Index GetStart(Hash) const;
//   CoeffRow GetCoeffRow(Hash) const;
// };

// concept FilterQueryHasher extends PhsfQueryHasher {
//   // For building or querying a filter, this returns the expected
//   // result row associated with a hashed input. For general PHSF,
//   // this must return 0.
//   //
//   // Although not strictly required, there's a slightly better chance of
//   // solver success if result row is masked down here to only the bits
//   // actually needed.
//   ResultRow GetResultRowFromHash(Hash) const;
// }

// concept BuilderHasher extends FilterQueryHasher {
//   // For a filter, this will generally be the same as QueryInput (a key).
//   // For a general PHSF, it must either
//   // (a) include a key and a result it maps to (e.g. in a pair), or
//   // (b) GetResultRowFromInput looks up the result somewhere rather than
//   // extracting it.
//   typename BuilderInput;
//
//   // We don't need to directly extract QueryInput from BuilderInput, but
//   // this is simple if BuilderInput == QueryInput.
//   Hash GetHash(const BuilderInput &) const;
//
//   // For building a non-filter PHSF, this extracts or looks up the result
//   // row to associate with an input. For filter PHSF, this must return 0.
//   ResultRow GetResultRowFromInput(const BuilderInput &) const;
//
//   // Whether the solver can assume the lowest bit of GetCoeffRow is
//   // always 1. When true, it should improve solver efficiency slightly.
//   static bool kFirstCoeffAlwaysOne;
// }

// concept SolverStorage extends SGaussTypes {
//   bool UsePrefetch();
//   void Prefetch(Index i);
//   CoeffRow* CoeffRowPtr(Index i);
//   ResultRow* ResultRowPtr(Index i);
// };

// concept BacktrackStorage extends SGaussTypes {
//   bool UseBacktrack();
//   void BacktrackPut(Index i, Index to_save);
//   Index BacktrackGet(Index i);
// }

template<typename SolverStorage, typename BacktrackStorage>
bool SolverAdd(SolverStorage *ss, SolverStorage::Hash h,
               SolverStorage::Index start, SolverStorage::ResultRow rr, SolverStorage::CoeffRow cr
               BacktrackStorage *bts, SolverStorage::Index backtrack_pos) {
  using SS = SolverStorage;
  SS:Index i = start;

  if (!SS::kFirstCoeffAlwaysOne) {
    int tz = CountTrailingZeroBits(cr);
    i += static_cast<SS::Index>(tz);
    cr >>= tz;
  } else {
    assert((cr & 1) == 1);
  }

  for (;;) {
    SS::CoeffRow other = *(ss->CoeffRowPtr(i));
    if (other == 0) {
      *(ss->CoeffRowPtr(i)) = cr;
      *(ss->ResultRowPtr(i)) = rr;
      bts->BacktrackPut(backtrack_pos, i);
      return true;
    }
    assert((other & 1) == 1);
    cr ^= other;
    rr ^= *(ss->ResultRowPtr(i));
    if (cr == 0) {
      // Inconsistency or (less likely) redundancy
      break;
    }
    int tz = CountTrailingZeroBits(cr);
    i += static_cast<SS::Index>(tz);
    cr >>= tz;
  }
    // Failed, unless result row == 0 because e.g. a duplicate input or a
    // stock hash collision, with same result row. (For filter, stock hash
    // collision implies same result row.) Or we could have a full equation
    // equal to sum of other equations, which is very possible with
    // small range of values for result row.
  return rr == 0;
}

// Here "Input" is short for BuilderInput.
template<typename SolverStorage, typename BacktrackStorage, typename BuilderHasher, typename InputIterator>
bool BacktrackableSolve(SolverStorage *ss, BacktrackStorage *bts, const BuilderHasher &bh, InputIterator begin, InputIterator end) {
  using SS = SolverStorage;

  if (begin == end) {
    // trivial
    return true;
  }

  InputIterator cur = begin;
  Index backtrack_pos = 0;
  if (!ss->UsePrefetch()) {
    // Simple version, no prefetch
    for (;;) {
      SS::Hash h = bh.GetHash(*cur);
      SS::Index start = bh.GetStart(h);
      SS::ResultRow rr = bh.GetResultRowFromInput(*cur) | bh.GetResultRowFromHash(h);
      SS::CoeffRow cr = bh.GetCoeffRow(h);

      if (!SolverAdd(ss, h, start, rr, cr, bts, backtrack_pos)) {
        break;
      }
      if ((++cur) == end) {
        return true;
      }
      ++backtrack_pos;
    }
  } else {
    // Pipelined w/prefetch
    // Prime the pipeline
    SS::Hash h = bh.GetHash(*cur);
    SS::Index start = bh.GetStart(h);
    SS::ResultRow rr = bh.GetResultRowFromInput(*cur);
    ss->Prefetch(start);

    // Pipeline
    for (;;) {
      rr |= bh.GetResultRowFromHash(h);
      SS::CoeffRow cr = bh.GetCoeffRow(h);
      if ((++cur) == end) {
        if (!SolverAdd(ss, h, start, rr, cr, bts, backtrack_pos)) {
          break;
        }
        return true;
      }
      SS::Hash next_h = bh.GetHash(*cur);
      SS::Index next_start = bh.GetStart(h);
      SS::ResultRow next_rr = bh.GetResultRowFromInput(*cur);
      ss->Prefetch(next_start);
      if (!SolverAdd(ss, h, start, rr, cr, bts, backtrack_pos)) {
        break;
      }
      ++backtrack_pos;
      h = next_h;
      start = next_start;
      rr = next_rr;
    }
  }
  // failed; backtrack (if implemented)
  if (bts->UseBacktrack()) {
    while (backtrack_pos > 0) {
      --backtrack_pos;
      Index i = bts->BacktrackGet(backtrack_pos);
      *(ss->CoeffRowPtr(i)) = 0;
      // Not required: *(ss->ResultRowPtr(i)) = 0;
    }
  }
  return false;
}

// Here "Input" is short for BuilderInput.
template<typename SolverStorage, typename InputIterator>
bool Solve(SolverStorage *ss, InputIterator begin, InputIterator end) {
  struct NoopBacktrackStorage {
    bool UseBacktrack() { return false; }
    void BacktrackPut(SolverStorage::Index to_save, SolverStorage::Index i) {}
    Index BacktrackGet(SolverStorage::Index i) {
      assert(false);
      return 0;
    }
  } nbts;
  return BacktrackableSolve(ss, &nbts, begin, end);
}

/*
//   bool NextSeed();
//   void ResetFor(Index num_slots);
// TODO
template<typename SolverStorage>
bool SolveSGauss(SolverStorage *st, const std::deque<SolverStorage::Input> &inputs, SolverStorage::Index num_slots) {
  using SS = SolverStorage;
  if (inputs.empty()) {
    // trivial case
    st->ResetFor(num_slots);
    return true;
  }
  do {
    st->ResetFor(num_slots);

  } while (st->NextSeed());
  // no more seeds
  return false;
}
*/

template<typename SolverStorage>
void BackSubstStep(SolverStorage::CoeffRow *state, SolverStorage::Index result_bits, const SolverStorage *ss, SolverStorage::Index base_slot, SolverStorage::Index slot_count) {
  using SS = SolverStorage;
  for (SS::Index i = base_slot + slot_count; i > base_slot;) {
    --i;
    SS::CoeffRow cr = *ss->CoeffRowPtr(i);
    SS:ResultRow rr = *ss->ResultRowPtr(i);
    for (SS::Index j = 0; j < result_bits; ++j) {
      SS::CoeffRow tmp = state[j] << 1;
      tmp |= SS::CoeffRow{BitParity(tmp & cr) ^ ((rr >> j) & 1)};
      state[j] = tmp;
    }
  }
}

// concept ByColumnSolutionStorage extends SGaussTypes {
//   typename Unit;
//   Index GetNumColumns() const;
//   // Assuming little endian across blocks
//   Unit Load(Index block_num, Index column) const;
//   void Store(Index block_num, Index column, Unit data);
// };

template<typename SolutionStorageHelper, typename SolverStorage>
void ByColumnBackSubstRange(ByColumnSolutionStorage *bcss, SolverStorage::CoeffRow *state, const SolverStorage *ss, SolverStorage::Index start_block, SolverStorage::Index block_count) {
  using SS = SolverStorage;

  constexpr auto kUnitBits = static_cast<SS::Index>(sizeof(ByColumnSolutionStorage::Unit) * 8U);

  const SS::Index result_bits = bcss->GetNumColumns();

  for (SS::Index i = start_block + block_count; i > start_block;) {
    --i;
    BackSubstStep(state, result_bits, ss, i * kUnitBits, kUnitBits);
    for (SS::Index j = 0; j < result_bits; ++j) {
      // Extract lower bits (or all) of corresponding state
      auto v = static_cast<SSH::Unit>((*state)[j]);
      // And store in solution structure
      bcss->Store(i, j, v);
    }
  }
}

template<typename ByColumnSolutionStorage, typename PhsfQueryHasher>
bool ByColumnGeneralizedQuery(const PhsfQueryHasher &hasher, PhsfQueryHasher::Hash &hash, const ByColumnSolutionStorage &bcss, ByColumnSolutionStorage::Result *result, bool match) {
  using BCSS = ByColumnSolutionStorage;

  // always compile-time constants
  constexpr auto kUnitBits = static_cast<BCSS::Index>(sizeof(BCSS::Unit) * 8U);
  constexpr auto kCoeffBits = static_cast<BCSS::Index>(sizeof(BCSS::CoeffRow) * 8U);

  // sometimes compile-time constant
  const BCSS::Index num_columns = bcss.GetNumColumns();

  // always dynamic
  const BCSS::Index start_slot = hasher.GetStart(hash);
  const BCSS::CoeffRow cr = hasher.GetCoeffRow(hash);
  const BCSS::Index start_bit = start_slot % kUnitBits;
  const BCSS::Index start_block = start_slot / kUnitBits;
  const BCSS::Index end_block = (start_slot + kCoeffBits - 1) / kUnitBits;

  for (BCSS::Index column = 0; column < num_columns; ++column) {
    // First block is the only one that we left shift cr for.
    // This approach makes most sense when kCoeffBits >= kUnitBits.
    // TODO? Good impl for kCoeffBits < kUnitBits
    BCSS::Unit val = bcss.Load(start_block, column) & static_cast<BCSS::Unit>(cr << start_bit);
    for (BCSS::Index i = 1; start_block + i <= end_block; ++i) {
      // The rest we right shift cr for
      val ^= bcss.Load(start_block + i, column) & static_cast<BCSS::Unit>(cr >> (i * kUnitBits - start_bit));
    }
    auto bit = BCSS::ResultRow{BitParity(val)};
    if (match) {
      // filter behavior
      if (((*result >> column) & 1U) != bit) {
        return false;
      }
    } else {
      // PHSF behavior
      *result |= bit << column;
    }
  }
  return true;
}

template<typename ByColumnSolutionStorage, typename PhsfQueryHasher>
ByColumnSolutionStorage::Result ByColumnPhsfQuery(const PhsfQueryHasher::QueryInput &input, const PhsfQueryHasher &hasher, const ByColumnSolutionStorage &bcss) {
  PhsfQueryHasher::Hash hash = hasher.GetHash(input);
  ByColumnSolutionStorage::Result result = 0;
  ByColumnGeneralizedQuery(hasher, hash, bcss, &result, false /*match*/);
  return result;
}

template<typename ByColumnSolutionStorage, typename FilterQueryHasher>
bool ByColumnFilterQuery(const FilterQueryHasher::QueryInput &input, const FilterQueryHasher &hasher, const ByColumnSolutionStorage &bcss) {
  FilterQueryHasher::Hash hash = hasher.GetHash(input);
  ByColumnSolutionStorage::Result result = hasher.GetResultRowFromHash(hash);
  return ByColumnGeneralizedQuery(hasher, hash, bcss, &result, true /*match*/);
}

}  // namespace SGauss

}  // namespace ROCKSDB_NAMESPACE
