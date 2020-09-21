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

// concept Hasher extends SGaussTypes {
//   // For a filter, Input will be just a key. For a general PHSF, it must
//   // include key and result it maps to (e.g. in a pair).
//   typename Input;
//   // Type for hashed summary of "key" part of Input. uint64_t recommended.
//   typename Hash;
//
//   Hash GetHash(const Input &);
//   Index GetStart(Hash);
//   CoeffRow GetCoeffRow(Hash);
//
//   // Whether the solver can assume the lowest bit of GetCoeffRow is
//   // always 1. When true, it should improve solver efficiency slightly.
//   static bool kFirstCoeffAlwaysOne;
//
//   // Either the Input (PHSF) or the Hash (filter) may be used to get
//   // the result row. Only one of these needs to return the value, and
//   // the other should return constant 0.
//   //
//   // Although not strictly required, there's a slightly better chance of
//   // solver success if result row is masked down here to only the bits
//   // actually needed.
//   ResultRow GetResultRowFromInput(const Input &);
//   ResultRow GetResultRowFromHash(const Hash &);
// };

// concept SolverStorage extends Hasher {
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
               SolverStorage::Index start, SolverStorage::ResultRow prelim_rr,
               BacktrackStorage *bts, SolverStorage::Index backtrack_pos) {
  using SS = SolverStorage;
  SS::ResultRow rr = prelim_rr | ss->GetResultRowFromHash(h);
  SS::CoeffRow cr = ss->GetCoeffRow(h);
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

template<typename SolverStorage, typename BacktrackStorage, typename InputIterator>
bool BacktrackableSolve(SolverStorage *ss, BacktrackStorage *bts, InputIterator begin, InputIterator end) {
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
      SS::Hash h = ss->GetHash(*cur);
      SS::Index start = ss->GetStart(h);
      SS::ResultRow rr = ss->GetResultRowFromInput(*cur);
      if (!SolverAdd(ss, h, start, rr, bts, backtrack_pos)) {
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
    SS::Hash h = ss->GetHash(*cur);
    SS::Index start = ss->GetStart(h);
    SS::ResultRow rr = ss->GetResultRowFromInput(*cur);
    ss->Prefetch(start);

    // Pipeline
    for (;;) {
      if ((++cur) == end) {
        if (!SolverAdd(ss, h, start, rr, bts, backtrack_pos)) {
          break;
        }
        return true;
      }
      SS::Hash next_h = ss->GetHash(*cur);
      SS::Index next_start = ss->GetStart(h);
      SS::ResultRow next_rr = ss->GetResultRowFromInput(*cur);
      if (!SolverAdd(ss, h, start, rr, bts, backtrack_pos)) {
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

template<typename ByColumnSolutionStorage, typename Hasher>
Hasher::ResultRow ByColumnPhsfQuery(const Hasher::Input &input, ByColumnSolutionStorage *bcss) {

}




/*
  constexpr size_t kRatio = sizeof(SSH::Unit) / sizeof(SSH::PtrUnit)
  static_assert(sizeof(SSH::Unit) == sizeof(SSH::PtrUnit) * kRatio, "Must evenly divide");

      SSH::Store(ptr + (i * result_bits + j) * kRatio, v);
*/


}  // namespace SGauss

}  // namespace ROCKSDB_NAMESPACE
