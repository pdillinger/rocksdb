//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#ifndef GFLAGS
#include <cstdio>
int main() {
  fprintf(stderr, "Please install gflags to run rocksdb tools\n");
  return 1;
}
#else

#include <cinttypes>
#include <iostream>
#include <random>
#include <vector>

#include "port/port.h"
#include "port/stack_trace.h"
#include "rocksdb/filter_policy.h"
#include "util/gflags_compat.h"
#include "util/hash.h"
#include "util/stop_watch.h"

using GFLAGS_NAMESPACE::ParseCommandLineFlags;
using GFLAGS_NAMESPACE::RegisterFlagValidator;
using GFLAGS_NAMESPACE::SetUsageMessage;

DEFINE_int64(seed, 0, "Seed for random number generators");

DEFINE_uint32(working_mem_size_mb, 200,
              "MB of memory to get up to among all filters");

DEFINE_uint32(average_keys_per_filter, 10000,
              "Average number of keys per filter");

DEFINE_uint32(key_size, 16, "Number of bytes each key should be");

DEFINE_uint32(batch_size, 8, "Number of keys to group in each batch");

DEFINE_uint32(bits_per_key, 10, "Bits per key setting for filters");

DEFINE_uint32(m_queries, 200, "Millions of queries for each test mode");

DEFINE_uint32(new_reader_every, 0,
              "Create new reader ~1/n queries (0 = never re-create)");

DEFINE_bool(quick, false, "Run more limited set of tests, fewer queries");

void _always_assert_fail(int line, const char *file, const char *expr) {
  fprintf(stderr, "%s: %d: Assertion %s failed\n", file, line, expr);
  abort();
}

#define always_assert(cond) \
  ((cond) ? (void)0 : ::_always_assert_fail(__LINE__, __FILE__, #cond))

using rocksdb::fastrange32;
using rocksdb::Slice;

struct KeyMaker {
  KeyMaker(size_t size)
      : data_(new char[size]),
        slice_(data_.get(), size),
        vals_(reinterpret_cast<uint32_t *>(data_.get())) {
    assert(size >= 8);
    memset(data_.get(), 0, size);
  }
  std::unique_ptr<char> data_;
  Slice slice_;
  uint32_t *vals_;

  Slice &Get(uint32_t filter_num, uint32_t val_num) {
    vals_[0] = filter_num + val_num;
    vals_[1] = val_num;
    return slice_;
  }
};

void PrintWarnings() {
#if defined(__GNUC__) && !defined(__OPTIMIZE__)
  fprintf(stdout,
          "WARNING: Optimization is disabled: benchmarks unnecessarily slow\n");
#endif
#ifndef NDEBUG
  fprintf(stdout,
          "WARNING: Assertions are enabled; benchmarks unnecessarily slow\n");
#endif
}

struct FilterInfo {
  uint32_t filter_id_ = 0;
  std::unique_ptr<const char[]> owner_;
  Slice filter_;
  uint32_t keys_added_ = 0;
  std::unique_ptr<rocksdb::FilterBitsReader> reader_;
  uint64_t outside_queries_ = 0;
  uint64_t false_positives_ = 0;
};

enum TestMode {
  TM_SingleFilter,
  TM_BatchPrepared,
  TM_BatchUnprepared,
  TM_FiftyOneFilter,
  TM_EightyTwentyFilter,
  TM_RandomFilter,
};

static const std::vector<TestMode> allTestModes = {
    TM_SingleFilter,   TM_BatchPrepared,      TM_BatchUnprepared,
    TM_FiftyOneFilter, TM_EightyTwentyFilter, TM_RandomFilter,
};

static const std::vector<TestMode> quickTestModes = {
    TM_SingleFilter,
    TM_RandomFilter,
};

const char *TestModeToString(TestMode tm) {
  switch (tm) {
    case TM_SingleFilter:
      return "Single filter";
    case TM_BatchPrepared:
      return "Batched, prepared";
    case TM_BatchUnprepared:
      return "Batched, unprepared";
    case TM_FiftyOneFilter:
      return "Skewed 50% in 1%";
    case TM_EightyTwentyFilter:
      return "Skewed 80% in 20%";
    case TM_RandomFilter:
      return "Random filter";
  }
  return "Bad TestMode";
}

struct FilterBench {
  std::vector<KeyMaker> kms;
  std::unique_ptr<const rocksdb::FilterPolicy> policy;
  std::vector<FilterInfo> infos;
  std::mt19937 random;

  FilterBench()
      : policy(rocksdb::NewBloomFilterPolicy(FLAGS_bits_per_key)),
        random(FLAGS_seed) {
    for (uint32_t i = 0; i < FLAGS_batch_size; ++i) {
      kms.emplace_back(FLAGS_key_size < 8 ? 8 : FLAGS_key_size);
    }
  }

  void Go();

  void RandomQueryTest(bool inside, bool dry_run, TestMode mode);
};

void FilterBench::Go() {
  std::unique_ptr<rocksdb::FilterBitsBuilder> builder(
      policy->GetFilterBitsBuilder());

  uint32_t varianceMask = 1;
  while (varianceMask * varianceMask * 4 < FLAGS_average_keys_per_filter) {
    varianceMask = varianceMask * 2 + 1;
  }

  const std::vector<TestMode> &testModes =
      FLAGS_quick ? quickTestModes : allTestModes;
  if (FLAGS_quick) {
    FLAGS_m_queries /= 10;
  }

  std::cout << "Building..." << std::endl;

  size_t totalMemoryUsed = 0;
  size_t totalKeysAdded = 0;

  while (totalMemoryUsed < 1024 * 1024 * FLAGS_working_mem_size_mb) {
    uint32_t filterId = random();
    uint32_t keysToAdd = FLAGS_average_keys_per_filter +
                         (random() & varianceMask) - (varianceMask / 2);
    for (uint32_t i = 0; i < keysToAdd; ++i) {
      builder->AddKey(kms[0].Get(filterId, i));
    }
    infos.emplace_back();
    FilterInfo &info = infos.back();
    info.filter_id_ = filterId;
    info.filter_ = builder->Finish(&info.owner_);
    info.keys_added_ = keysToAdd;
    info.reader_.reset(policy->GetFilterBitsReader(info.filter_));
    totalMemoryUsed += info.filter_.size();
    totalKeysAdded += keysToAdd;
  }

  std::cout << "Number of filters: " << infos.size() << std::endl;
  std::cout << "Total memory (MB): " << totalMemoryUsed / 1024.0 / 1024.0
            << std::endl;

  double bpk = totalMemoryUsed * 8.0 / totalKeysAdded;
  std::cout << "Bits/key actual: " << bpk << std::endl;
  if (!FLAGS_quick) {
    double tolerableRate = std::pow(2.0, -(bpk - 1.0) / 1.5);
    std::cout << "Best possible FP rate %: " << 100.0 * std::pow(2.0, -bpk)
              << std::endl;
    std::cout << "Tolerable FP rate %: " << 100.0 * tolerableRate << std::endl;

    std::cout << "----------------------------" << std::endl;
    std::cout << "Verifying..." << std::endl;

    uint32_t outside_q_per_f = 1000000 / infos.size();
    uint64_t fps = 0;
    for (uint32_t i = 0; i < infos.size(); ++i) {
      FilterInfo &info = infos[i];
      for (uint32_t j = 0; j < info.keys_added_; ++j) {
        always_assert(info.reader_->MayMatch(kms[0].Get(info.filter_id_, j)));
      }
      for (uint32_t j = 0; j < outside_q_per_f; ++j) {
        fps +=
            info.reader_->MayMatch(kms[0].Get(info.filter_id_, j | 0x80000000));
      }
    }
    std::cout << " No FNs :)" << std::endl;
    double prelimRate = (double)fps / outside_q_per_f / infos.size();
    std::cout << " Prelim FP rate %: " << (100.0 * prelimRate) << std::endl;

    always_assert(prelimRate < tolerableRate);
  }

  std::cout << "----------------------------" << std::endl;
  std::cout << "Inside queries..." << std::endl;
  random.seed(FLAGS_seed + 1);
  RandomQueryTest(/*inside*/ true, /*dry_run*/ true, TM_RandomFilter);
  for (TestMode tm : testModes) {
    random.seed(FLAGS_seed + 1);
    RandomQueryTest(/*inside*/ true, /*dry_run*/ false, tm);
  }

  std::cout << "----------------------------" << std::endl;
  std::cout << "Outside queries..." << std::endl;
  random.seed(FLAGS_seed + 2);
  RandomQueryTest(/*inside*/ false, /*dry_run*/ true, TM_RandomFilter);
  for (TestMode tm : testModes) {
    random.seed(FLAGS_seed + 2);
    RandomQueryTest(/*inside*/ false, /*dry_run*/ false, tm);
  }

  std::cout << "----------------------------" << std::endl;
  std::cout << "Done." << std::endl;
}

void FilterBench::RandomQueryTest(bool inside, bool dry_run, TestMode mode) {
  for (auto &info : infos) {
    info.outside_queries_ = 0;
    info.false_positives_ = 0;
  }

  uint32_t dryRunHash = 0;
  uint64_t maxQueries = static_cast<uint64_t>(FLAGS_m_queries) * 1000000;
  size_t numPrimaryFilters = infos.size();
  uint32_t primaryFilterThreshold = 0xffffffff;
  if (mode == TM_SingleFilter) {
    // 100% of queries to 1 filter
    numPrimaryFilters = 1;
  } else if (mode == TM_FiftyOneFilter) {
    // 50% of queries
    primaryFilterThreshold /= 2;
    // to 1% of filters
    numPrimaryFilters = (numPrimaryFilters + 99) / 100;
  } else if (mode == TM_EightyTwentyFilter) {
    // 80% of queries
    primaryFilterThreshold = primaryFilterThreshold / 5 * 4;
    // to 20% of filters
    numPrimaryFilters = (numPrimaryFilters + 4) / 5;
  }
  size_t batchSize = 1;
  std::unique_ptr<Slice *[]> batchSlices;
  std::unique_ptr<bool[]> batchResults;
  if (mode == TM_BatchPrepared || mode == TM_BatchUnprepared) {
    batchSize = kms.size();
    batchSlices.reset(new Slice *[batchSize]);
    batchResults.reset(new bool[batchSize]);
    for (size_t i = 0; i < batchSize; ++i) {
      batchSlices[i] = &kms[i].slice_;
      batchResults[i] = false;
    }
  }

  rocksdb::StopWatchNano timer(rocksdb::Env::Default(), true);

  for (uint64_t q = 0; q < maxQueries; q += batchSize) {
    uint32_t filterIndex;
    if (random() <= primaryFilterThreshold) {
      filterIndex = fastrange32(numPrimaryFilters, random());
    } else {
      // secondary
      filterIndex = numPrimaryFilters +
                    fastrange32(infos.size() - numPrimaryFilters, random());
    }
    FilterInfo &info = infos[filterIndex];
    if (FLAGS_new_reader_every > 0 &&
        fastrange32(FLAGS_new_reader_every, random()) < batchSize) {
      info.reader_.reset(policy->GetFilterBitsReader(info.filter_));
    }
    for (size_t i = 0; i < batchSize; ++i) {
      if (inside) {
        kms[i].Get(info.filter_id_, fastrange32(info.keys_added_, random()));
      } else {
        kms[i].Get(info.filter_id_, random() | 0x80000000);
        info.outside_queries_++;
      }
    }
    if (mode == TM_BatchPrepared && !dry_run) {
      for (size_t i = 0; i < batchSize; ++i) {
        batchResults[i] = false;
      }
      info.reader_->MayMatch(batchSize, batchSlices.get(), batchResults.get());
      for (size_t i = 0; i < batchSize; ++i) {
        if (inside) {
          always_assert(batchResults[i]);
        } else {
          info.false_positives_ += batchResults[i];
        }
      }
    } else {
      for (size_t i = 0; i < batchSize; ++i) {
        if (dry_run) {
          dryRunHash ^= rocksdb::BloomHash(kms[i].slice_);
        } else if (inside) {
          always_assert(info.reader_->MayMatch(kms[i].slice_));
        } else {
          info.false_positives_ += info.reader_->MayMatch(kms[i].slice_);
        }
      }
    }
  }

  uint64_t elapsedNanos = timer.ElapsedNanos();
  double ns = (double)elapsedNanos / maxQueries;

  if (dry_run) {
    // Printing part of hash prevents dry run components from being optimized
    // away by compiler
    std::cout << "  Dry run (" << std::hex << (dryRunHash & 0xfff) << std::dec
              << ") ";
  } else {
    std::cout << "  " << TestModeToString(mode) << " ";
  }
  std::cout << "ns/op: " << ns << std::endl;

  if (!inside && !dry_run && mode == TM_RandomFilter) {
    uint64_t q = 0;
    uint64_t fp = 0;
    double worst_fp_rate = 0.0;
    double best_fp_rate = 1.0;
    for (auto &info : infos) {
      q += info.outside_queries_;
      fp += info.false_positives_;
      if (info.outside_queries_ > 0) {
        double fp_rate = (double)info.false_positives_ / info.outside_queries_;
        worst_fp_rate = std::max(worst_fp_rate, fp_rate);
        best_fp_rate = std::min(best_fp_rate, fp_rate);
      }
    }
    std::cout << "    Average FP rate %: " << 100.0 * fp / q << std::endl;
    if (!FLAGS_quick) {
      std::cout << "    Worst   FP rate %: " << 100.0 * worst_fp_rate
                << std::endl;
      std::cout << "    Best    FP rate %: " << 100.0 * best_fp_rate
                << std::endl;
      std::cout << "    Best possible bits/key: "
                << -std::log((double)fp / q) / std::log(2.0) << std::endl;
    }
  }
}

int main(int argc, char **argv) {
  rocksdb::port::InstallStackTraceHandler();
  SetUsageMessage(std::string("\nUSAGE:\n") + std::string(argv[0]) +
                  " [OPTIONS]...");
  ParseCommandLineFlags(&argc, &argv, true);

  PrintWarnings();

  FilterBench b;
  b.Go();

  return 0;
}

#endif  // GFLAGS
