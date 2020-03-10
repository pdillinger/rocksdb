//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2012 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include <array>
#include <deque>

#include "rocksdb/filter_policy.h"

#include "rocksdb/slice.h"
#include "table/block_based/block_based_filter_block.h"
#include "table/block_based/full_filter_block.h"
#include "table/block_based/filter_policy_internal.h"
#include "third-party/folly/folly/ConstexprMath.h"
#include "util/bloom_impl.h"
#include "util/coding.h"
#include "util/hash.h"
#include "util/math.h"
#include "util/util.h"

namespace ROCKSDB_NAMESPACE {

namespace {

// See description in FastLocalBloomImpl
class FastLocalBloomBitsBuilder : public BuiltinFilterBitsBuilder {
 public:
  explicit FastLocalBloomBitsBuilder(const int millibits_per_key)
      : millibits_per_key_(millibits_per_key),
        num_probes_(FastLocalBloomImpl::ChooseNumProbes(millibits_per_key_)) {
    assert(millibits_per_key >= 1000);
  }

  // No Copy allowed
  FastLocalBloomBitsBuilder(const FastLocalBloomBitsBuilder&) = delete;
  void operator=(const FastLocalBloomBitsBuilder&) = delete;

  ~FastLocalBloomBitsBuilder() override {}

  virtual void AddKey(const Slice& key) override {
    uint64_t hash = GetSliceHash64(key);
    if (hash_entries_.empty() || hash != hash_entries_.back()) {
      hash_entries_.push_back(hash);
    }
  }

  virtual Slice Finish(std::unique_ptr<const char[]>* buf) override {
    uint32_t len_with_metadata =
        CalculateSpace(static_cast<uint32_t>(hash_entries_.size()));
    char* data = new char[len_with_metadata];
    memset(data, 0, len_with_metadata);

    assert(data);
    assert(len_with_metadata >= 5);

    uint32_t len = len_with_metadata - 5;
    if (len > 0) {
      AddAllEntries(data, len);
    }

    // See BloomFilterPolicy::GetBloomBitsReader re: metadata
    // -1 = Marker for newer Bloom implementations
    data[len] = static_cast<char>(-1);
    // 0 = Marker for this sub-implementation
    data[len + 1] = static_cast<char>(0);
    // num_probes (and 0 in upper bits for 64-byte block size)
    data[len + 2] = static_cast<char>(num_probes_);
    // rest of metadata stays zero

    const char* const_data = data;
    buf->reset(const_data);
    assert(hash_entries_.empty());

    return Slice(data, len_with_metadata);
  }

  int CalculateNumEntry(const uint32_t bytes) override {
    uint32_t bytes_no_meta = bytes >= 5u ? bytes - 5u : 0;
    return static_cast<int>(uint64_t{8000} * bytes_no_meta /
                            millibits_per_key_);
  }

  uint32_t CalculateSpace(const int num_entry) override {
    uint32_t num_cache_lines = 0;
    if (millibits_per_key_ > 0 && num_entry > 0) {
      num_cache_lines = static_cast<uint32_t>(
          (int64_t{num_entry} * millibits_per_key_ + 511999) / 512000);
    }
    return num_cache_lines * 64 + /*metadata*/ 5;
  }

  double EstimatedFpRate(size_t keys, size_t bytes) override {
    return FastLocalBloomImpl::EstimatedFpRate(keys, bytes - /*metadata*/ 5,
                                               num_probes_, /*hash bits*/ 64);
  }

 private:
  void AddAllEntries(char* data, uint32_t len) {
    // Simple version without prefetching:
    //
    // for (auto h : hash_entries_) {
    //   FastLocalBloomImpl::AddHash(Lower32of64(h), Upper32of64(h), len,
    //                               num_probes_, data);
    // }

    const size_t num_entries = hash_entries_.size();
    constexpr size_t kBufferMask = 7;
    static_assert(((kBufferMask + 1) & kBufferMask) == 0,
                  "Must be power of 2 minus 1");

    std::array<uint32_t, kBufferMask + 1> hashes;
    std::array<uint32_t, kBufferMask + 1> byte_offsets;

    // Prime the buffer
    size_t i = 0;
    for (; i <= kBufferMask && i < num_entries; ++i) {
      uint64_t h = hash_entries_.front();
      hash_entries_.pop_front();
      FastLocalBloomImpl::PrepareHash(Lower32of64(h), len, data,
                                      /*out*/ &byte_offsets[i]);
      hashes[i] = Upper32of64(h);
    }

    // Process and buffer
    for (; i < num_entries; ++i) {
      uint32_t& hash_ref = hashes[i & kBufferMask];
      uint32_t& byte_offset_ref = byte_offsets[i & kBufferMask];
      // Process (add)
      FastLocalBloomImpl::AddHashPrepared(hash_ref, num_probes_,
                                          data + byte_offset_ref);
      // And buffer
      uint64_t h = hash_entries_.front();
      hash_entries_.pop_front();
      FastLocalBloomImpl::PrepareHash(Lower32of64(h), len, data,
                                      /*out*/ &byte_offset_ref);
      hash_ref = Upper32of64(h);
    }

    // Finish processing
    for (i = 0; i <= kBufferMask && i < num_entries; ++i) {
      FastLocalBloomImpl::AddHashPrepared(hashes[i], num_probes_,
                                          data + byte_offsets[i]);
    }
  }

  int millibits_per_key_;
  int num_probes_;
  // A deque avoids unnecessary copying of already-saved values
  // and has near-minimal peak memory use.
  std::deque<uint64_t> hash_entries_;
};

// See description in FastLocalBloomImpl
class FastLocalBloomBitsReader : public FilterBitsReader {
 public:
  FastLocalBloomBitsReader(const char* data, int num_probes, uint32_t len_bytes)
      : data_(data), num_probes_(num_probes), len_bytes_(len_bytes) {}

  // No Copy allowed
  FastLocalBloomBitsReader(const FastLocalBloomBitsReader&) = delete;
  void operator=(const FastLocalBloomBitsReader&) = delete;

  ~FastLocalBloomBitsReader() override {}

  bool MayMatch(const Slice& key) override {
    uint64_t h = GetSliceHash64(key);
    uint32_t byte_offset;
    FastLocalBloomImpl::PrepareHash(Lower32of64(h), len_bytes_, data_,
                                    /*out*/ &byte_offset);
    return FastLocalBloomImpl::HashMayMatchPrepared(Upper32of64(h), num_probes_,
                                                    data_ + byte_offset);
  }

  virtual void MayMatch(int num_keys, Slice** keys, bool* may_match) override {
    std::array<uint32_t, MultiGetContext::MAX_BATCH_SIZE> hashes;
    std::array<uint32_t, MultiGetContext::MAX_BATCH_SIZE> byte_offsets;
    for (int i = 0; i < num_keys; ++i) {
      uint64_t h = GetSliceHash64(*keys[i]);
      FastLocalBloomImpl::PrepareHash(Lower32of64(h), len_bytes_, data_,
                                      /*out*/ &byte_offsets[i]);
      hashes[i] = Upper32of64(h);
    }
    for (int i = 0; i < num_keys; ++i) {
      may_match[i] = FastLocalBloomImpl::HashMayMatchPrepared(
          hashes[i], num_probes_, data_ + byte_offsets[i]);
    }
  }

 private:
  const char* data_;
  const int num_probes_;
  const uint32_t len_bytes_;
};

struct GaussData {
  // Contents of this row in the coefficient matrix, starting at `start`
  // (rest implicitly 0)
  // Originally based on hash, but updated in gaussian elimination.
  uint64_t coeff_row = 0;
  // Full contents of (corresponding) this row in the match matrix.
  // Originally based on hash, but updated in gaussian elimination.
  uint32_t match_row = 0;
  // The index of the first non-zero column in coeff for this row.
  // Equivalently, the starting index of output rows to be used in
  // query. Based on hash.
  uint16_t start = 0;
  // A row in output, or a column in coeff.
  // Computed during gaussian elimination.
  uint16_t pivot = 0;

  // Using 20/20 was found to be most compact in some circumstances, but now
  // 32/31 (fastest) seems most compact also.
  static constexpr uint32_t front_smash = 32;
  static constexpr uint32_t back_smash = 31;

  static inline uint16_t HashToStart(uint64_t h, uint32_t num_output_rows) {
    // NB: addrs == num_output_rows with 32/31 setting
    const uint32_t addrs = num_output_rows - 63 + front_smash + back_smash;
    uint32_t start = fastrange32(Upper32of64(h), addrs);
    uint32_t x = front_smash; // FIXME for DEBUG_LEVEL=2
    start = std::max(start, x);
    start -= front_smash;
    start = std::min(start, num_output_rows - 64);
    assert(start < num_output_rows - 63);
    assert(start < 0x10000U);
    return static_cast<uint16_t>(start);
  }

  static inline uint64_t HashToCoeffRow(uint64_t h) {
    uint64_t row = (h + (h >> 32)) * 0x9e3779b97f4a7c13;
    row |= uint64_t{1} << 63;
    return row;
  }

  static inline uint32_t HashToMatchRow(uint64_t h) {
    // NB: just h seems to cause some association affecting FP rate
    return Lower32of64(h ^ (h >> 13) ^ (h >> 26));
  }

  inline void Reset(uint64_t h, uint32_t num_output_rows, uint32_t match_row_mask) {
    start = HashToStart(h, num_output_rows);
    coeff_row = HashToCoeffRow(h);
    match_row = HashToMatchRow(h) & match_row_mask;
    pivot = 0;
  }

  // FIXME: better way?
  static inline uint64_t TweakPreHash(uint64_t h) {
    if ((h & uint64_t{0x8000000000000380}) == uint64_t{0x0000000000000380}) {
      return h + uint64_t{0x8000000000000000};
    } else {
      return h;
    }
  }

  static inline uint32_t PreHashToSection(uint64_t h) {
    uint32_t v = Lower32of64(h) & 1023;
    if (v < 300) {
        return v / 3;
    } else if (v < 428) {
        return v - 200;
    } else if (v < 512) {
        return (v + 256) / 3;
//    } else if (v < 532) {
//        return (v + 1516) / 8;
    } else {
        return 0;
    }
  }

  static inline uint64_t SeedPreHash(uint64_t h, uint32_t seed) {
    uint32_t rot = (seed * 39) & 63;
    return ((h << rot) | (h >> (64 - rot))) * 0x9e3779b97f4a7c13;
  }

  static inline bool DotCoeffRowWithOutputColumn(uint32_t start, uint64_t coeff_row, const uint64_t *output_data, uint32_t match_bits, uint32_t selected_match_bit) {
    uint32_t start_word = (start / 64) * match_bits + selected_match_bit;
    uint32_t start_bit = start % 64;
    // TODO: endianness
    uint64_t output_column = output_data[start_word] >> start_bit;
    if (start_bit > 0) {
      output_column |= output_data[start_word + match_bits] << (64 - start_bit);
    }
    return BitParity(output_column & coeff_row) != 0;
  }

  static inline void StoreOutputVal(bool val, uint64_t *output_data, uint32_t match_bits, uint32_t start, uint32_t selected_match_bit) {
    uint32_t selected_word = (start / 64) * match_bits + selected_match_bit;
    uint64_t bit_selector = uint64_t{1} << (start % 64);

    //printf("Storing %u @ %u.%u\n", (unsigned)val, (unsigned)(size_t)(output_data + selected_word), start % 64);

    // TODO: endianness
    if (val) {
      output_data[selected_word] |= bit_selector;
    } else {
      output_data[selected_word] &= ~bit_selector;
    }
  }

  static bool CompareByStart(const GaussData &a, const GaussData &b) {
    return a.start < b.start;
  }
};

static inline ptrdiff_t size_t_diff(size_t a, size_t b) {
  return static_cast<ptrdiff_t>(a) - static_cast<ptrdiff_t>(b);
}

struct SimpleGaussFilter {
  static constexpr uint32_t metadata_size = 5;

  char* data = nullptr;
  size_t bytes = metadata_size; // incl metadata
  size_t total_blocks = 0;
  size_t first_block_upper = 0;
  uint32_t log2_shards = 0;
  uint32_t avg_shard_slots = 0;
  uint32_t lower_match_bits = 0;
  uint32_t first_shard_extra_blocks = 0;

  inline size_t GetShardBeginBlock(uint32_t shard) const {
    return static_cast<size_t>((uint64_t{shard} * avg_shard_slots + first_shard_extra_blocks) / 64);
  }

  inline char* GetBlockDataStart(size_t block) const {
    assert(data);
    char *rv = data + block * 8 * lower_match_bits;
    rv += std::max(ptrdiff_t{0}, size_t_diff(block, first_block_upper)) * 8;
    return rv;
  }

  inline char* GetShardMetaDataStart() const {
    return GetBlockDataStart(GetShardBeginBlock(uint32_t{1} << log2_shards));
  }

  // Reads:
  //   bytes
  //   log2_shards
  //   avg_shard_slots
  // Reads+writes:
  //   first_shard_extra_blocks
  //   lower_match_bits
  //   total_blocks
  //   first_block_upper
  void ElasticShrinkForFirstShard(uint32_t new_first_shard_extra_blocks) {
    uint32_t old_first_shard_extra_blocks = first_shard_extra_blocks;
    uint32_t old_lower_match_bits = lower_match_bits;
    size_t old_total_blocks = total_blocks;
    //size_t old_first_block_upper = first_block_upper;

    this->first_shard_extra_blocks = new_first_shard_extra_blocks;
    CalculateMatchBitSettings();

    assert(first_shard_extra_blocks > old_first_shard_extra_blocks);
    assert(lower_match_bits <= old_lower_match_bits);
    assert(total_blocks > old_total_blocks);

    fprintf(stderr, "Elastic shrink for first shard not yet implemented. Aborting.\n");
    abort();
  }

  // Reads:
  //   bytes
  //   log2_shards
  //   avg_shard_slots
  //   first_shard_extra_blocks (default 0 OK)
  // Writes:
  //   lower_match_bits
  //   total_blocks
  //   first_block_upper
  void CalculateMatchBitSettings() {
    uint32_t shards = uint32_t{1} << log2_shards;
    uint64_t orig_total_slots = uint64_t{shards} * avg_shard_slots;
    // Must be multiple of 64 already
    assert((orig_total_slots & 63) == 0);
    this->total_blocks = static_cast<size_t>(orig_total_slots / 64) + first_shard_extra_blocks;

    // One byte per shard of sharding data
    size_t bytes_for_blocks = bytes - metadata_size - /*sharding data*/shards;
    // Slots only work in 64-bit word chunks
    size_t words_for_blocks = bytes_for_blocks / 8;
    this->lower_match_bits =
        static_cast<uint32_t>(std::min(uint64_t{31}, words_for_blocks / total_blocks));
    uint32_t upper_match_bits = lower_match_bits + 1;
    size_t words_for_upper_extra = words_for_blocks - lower_match_bits * total_blocks;
    this->first_block_upper = total_blocks - words_for_upper_extra;
    assert(words_for_blocks == (first_block_upper * lower_match_bits) +
                               ((total_blocks - first_block_upper) * upper_match_bits));
    //printf("%u %u %u %u %u %u %u\n", (unsigned) bytes, (unsigned)total_blocks, (unsigned) first_block_upper, log2_shards, avg_shard_slots, lower_match_bits, first_shard_extra_blocks);
  }

  // Reads no fields
  // Writes:
  //   log2_shards
  //   avg_shard_slots
  void CalculatePreferredSharding(size_t keys) {
    // FIXME: 1.004
    size_t total_slots = static_cast<size_t>(1.008 * keys + 0.5);
    // Make it a multiple of 64
    total_slots = (total_slots + 63) & ~size_t{63};
    // Find power of two number of shards that gets average slots per shard
    // closest to ~1000
    this->log2_shards = 0;
    while (log2_shards + 1 < 32 && (total_slots >> log2_shards) > 1414U) {
        ++log2_shards;
    }
    uint32_t shards = uint32_t{1} << log2_shards;
    this->avg_shard_slots = (total_slots + shards - 1) / shards;
  }

  // Reads no fields
  // Writes:
  //   log2_shards
  //   avg_shard_slots
  //   lower_match_bits
  //   total_blocks
  //   first_block_upper
  void CalculateSpace(size_t keys, int millibits_per_key) {
    CalculatePreferredSharding(keys);
    uint32_t shards = uint32_t{1} << log2_shards;
    size_t ideal_bytes = static_cast<size_t>((int64_t{millibits_per_key} * keys + 7999) / 8000);
    size_t rounded_bytes_for_blocks = ((ideal_bytes - shards - metadata_size + 7) & ~size_t{7});
    this->bytes = rounded_bytes_for_blocks + shards + metadata_size;
    CalculateMatchBitSettings();
  }

  // Reads:
  //   bytes
  // Writes:
  //   data
  //   owned_data
  void AllocateSpace(std::unique_ptr<const char[]>* buf) {
    assert(bytes >= metadata_size);
    buf->reset(data = new char[bytes]());
  }

  static bool TrySolve(std::vector<GaussData> &gauss, uint32_t num_coeff_rows, uint32_t num_output_rows, const std::vector<uint64_t> &shard_hashes, const std::vector<uint64_t> &bumped_hashes, uint32_t seed, uint32_t last_section, uint32_t match_row_mask) {
    assert(gauss.size() >= num_coeff_rows);
    // Build gauss data with seeded hashes
    uint32_t next_gauss_idx = 0;
    for (uint64_t pre_h : shard_hashes) {
      uint32_t section = GaussData::PreHashToSection(pre_h);
      if (section <= last_section) {
        uint64_t h = GaussData::SeedPreHash(pre_h, seed);
        GaussData &d = gauss[next_gauss_idx];
        d.Reset(h, num_output_rows, match_row_mask);
        assert(d.start < num_output_rows);
        ++next_gauss_idx;
      }
    }
    // ... except bumped hashes don't get the seed
    for (uint64_t h : bumped_hashes) {
      GaussData &d = gauss[next_gauss_idx];
      d.Reset(h, num_output_rows, match_row_mask);
      assert(d.start < num_output_rows);
      ++next_gauss_idx;
    }
    assert(next_gauss_idx == num_coeff_rows);
    // Sort based on start positions
    std::sort(gauss.begin(), gauss.begin() + num_coeff_rows, GaussData::CompareByStart);
    // Now "simple" gaussian elimination
    for (uint32_t i = 0; i < num_coeff_rows; ++i) {
      GaussData &di = gauss[i];
      if (di.coeff_row == 0) {
        // Might be a total duplicate or lucky coincidence in generating
        // desired output. For plain (over-approximate) sets, that's just
        // fine. Pivot of 0 is a safe fake here, because if a real row uses
        // pivot 0, it has to be the first row and will be back-propagated
        // last.
        if (false && di.match_row == 0) { // FIXME: doesn't work; handle duplicates?
          assert(di.pivot == 0);
        } else {
          // Attempt failed
          return false;
        }
      }
      int tz = CountTrailingZeroBits(di.coeff_row);
      di.pivot = di.start + tz;
      for (uint32_t j = i + 1; j < num_coeff_rows; ++j) {
        GaussData &dj = gauss[j];
        assert(dj.start >= di.start);
        if (di.pivot < dj.start) {
          break;
        }
        if ((dj.coeff_row >> (di.pivot - dj.start)) & 1) {
          dj.coeff_row ^= (di.coeff_row >> (dj.start - di.start));
          dj.match_row ^= di.match_row;
        }
      }
    }
    // Success
    return true;
  }

  static void BackPropAndStoreSingle(GaussData &d, uint64_t *word_data, uint32_t match_bits) {
    const uint32_t start = d.start;
    const uint64_t coeff_row = d.coeff_row;
    const uint32_t match_row = d.match_row;
    const uint32_t pivot = d.pivot;
    //printf("BPSS %u %u %u\n", (unsigned)(size_t)word_data, start, match_bits);

    for (uint32_t j = 0; j < match_bits; ++j) {
      bool val = GaussData::DotCoeffRowWithOutputColumn(start, coeff_row, word_data, match_bits, j);
      val ^= ((match_row >> j) & 1);
      GaussData::StoreOutputVal(val, word_data, match_bits, pivot, j);
    }
  }

  void BackPropAndStore(size_t starting_block, std::vector<GaussData> &gauss, uint32_t num_coeff_rows, uint32_t num_output_rows) {
    size_t ending_block = starting_block + num_output_rows / 64;
    uint32_t match_bits = lower_match_bits + (starting_block >= first_block_upper);
    uint32_t i = num_coeff_rows;
    uint64_t *word_data = reinterpret_cast<uint64_t *>(GetBlockDataStart(starting_block));
    if (first_block_upper > starting_block && first_block_upper < ending_block) {
      // Special handling (mixed match_bits in shard)
      uint32_t num_lower_blocks = first_block_upper - starting_block;
      uint32_t first_start_upper = num_lower_blocks * 64;
      // Where the starting pointer would be if the whole thing used
      // upper match bits and ended at the same pointer (but started
      // before actual start).
      uint64_t *special_word_data = word_data - num_lower_blocks;
      for (; i > 0 && gauss[i - 1].start >= first_start_upper; --i) {
        BackPropAndStoreSingle(gauss[i - 1], special_word_data, match_bits + 1);
      }
      // There is one intermediate block where we might need to use
      // upper match bits or lower match bits depending on position
      // of the pivot (where we are writing data to)
      for (; i > 0 && gauss[i - 1].start >= first_start_upper - 64; --i) {
        if (gauss[i - 1].pivot >= first_start_upper) {
          BackPropAndStoreSingle(gauss[i - 1], special_word_data, match_bits + 1);
        } else {
          BackPropAndStoreSingle(gauss[i - 1], word_data, match_bits);
        }
      }
      // fall through for lower portion
    }
    // Easy handling (consistent match_bits in shard, or lower portion)
    for (; i > 0; --i) {
      BackPropAndStoreSingle(gauss[i - 1], word_data, match_bits);
    }
  }

  void BuildShards(std::deque<uint64_t> *hashes) {
    assert(log2_shards < 32);
    uint32_t shards = uint32_t{1} << log2_shards;

    std::unique_ptr<std::vector<uint64_t>[]> shard_hashes{new std::vector<uint64_t>[shards]};
    std::unique_ptr<std::vector<uint64_t>[]> bumped_hashes{new std::vector<uint64_t>[shards]};
    std::unique_ptr<std::array<uint32_t, 256>[]> shard_section_counts{new std::array<uint32_t, 256>[shards]()};

    // TODO: pre-allocate vector capacity?

    uint32_t leave_shard_shift_minus1 = 64U - log2_shards - 1;
    while (!hashes->empty()) {
      uint64_t h = GaussData::TweakPreHash(hashes->front());
      hashes->pop_front();
      uint32_t shard = h >> 1 >> leave_shard_shift_minus1;
      uint32_t section = GaussData::PreHashToSection(h);
      assert(section < 256U);
      shard_hashes[shard].push_back(h);
      shard_section_counts[shard][section]++;
    }

    char *shard_metadata = GetShardMetaDataStart();

    std::vector<GaussData> gauss;
    gauss.resize(avg_shard_slots + 63);

    uint32_t match_row_mask = (uint32_t{1} << 1 << lower_match_bits) - 1;

    size_t shard_begin_block = total_blocks;
    size_t shard_end_block;
    for (uint32_t shard = shards - 1; shard > 0; --shard) {
      shard_end_block = shard_begin_block;
      shard_begin_block = GetShardBeginBlock(shard);

      uint32_t shard_slots = (shard_end_block - shard_begin_block) * 64;
      uint32_t last_section = 0;
      uint32_t count = shard_section_counts[shard][0] + bumped_hashes[shard].size();
      for (; last_section < 255U; ++last_section) {
        uint32_t next_count = count + shard_section_counts[shard][last_section + 1];
        if (next_count > shard_slots) {
          break;
        }
        count = next_count;
      }

      for (; last_section > 0; --last_section) {
        if (TrySolve(gauss, count, shard_slots, shard_hashes[shard], bumped_hashes[shard], /*seed*/last_section, last_section, match_row_mask)) {
          BackPropAndStore(shard_begin_block, gauss, count, shard_slots);
          break;
        }
        count -= shard_section_counts[shard][last_section];
      }
      shard_metadata[shard] = static_cast<char>(last_section);
      // Handle bumping
      // TODO: combine with solving passes
      if (last_section < 255) {
        uint32_t bumped_shard_or = (uint32_t{1} << FloorLog2(shard));
        uint32_t bumped_shard_mask = bumped_shard_or - 1;
        bumped_shard_or /= 2;
        bumped_shard_mask /= 2;
        for (uint64_t pre_h : shard_hashes[shard]) {
          uint32_t section = GaussData::PreHashToSection(pre_h);
          if (section > last_section) {
            // TODO: reconsider/refactor
            uint32_t bumped_shard = (Upper32of64(pre_h) & bumped_shard_mask) | bumped_shard_or;
            assert(bumped_shard < shard);
            uint64_t h = pre_h * 0x9e3779b97f4a7c13 * 0x9e3779b97f4a7c13;
            bumped_hashes[bumped_shard].push_back(h);
          }
        }
      }
      if (last_section == 0) {
        fprintf(stderr, "Bloom fallback not yet implemented. Aborting.\n");
        abort();
      }
    }
    // special treatment for first shard
    uint32_t original_shard_end_block = shard_begin_block;
    shard_end_block = original_shard_end_block;
    shard_begin_block = 0;
    uint32_t shard_slots = (shard_end_block - shard_begin_block) * 64;

    uint32_t count = bumped_hashes[0].size();
    for (uint32_t i = 0; i < 256U; ++i) {
      count += shard_section_counts[0][i];
    }
    while (count > shard_slots) {
      ++shard_end_block;
      shard_slots = (shard_end_block - shard_begin_block) * 64;
      gauss.resize(shard_slots);
    }
    uint32_t seed = 255;
    for (; shard_slots <= 0x10000U && seed > 0; --seed) {
      if (seed < 240 && shard_slots < 0x10000U) {
        ++shard_end_block;
        shard_slots = (shard_end_block - shard_begin_block) * 64;
        gauss.resize(shard_slots);
      }
      if (TrySolve(gauss, count, shard_slots, shard_hashes[0], bumped_hashes[0], seed, /*all*/ 255, match_row_mask)) {
        break;
      }
    }
    if (shard_slots > 0x10000U || seed == 0) {
      fprintf(stderr, "Bloom fallback not yet implemented (last shard). Aborting.\n");
      abort();
    }
    shard_metadata[0] = static_cast<char>(seed);
    if (shard_end_block != original_shard_end_block) {
      assert(shard_end_block > original_shard_end_block);
      // Needed to add extra blocks to last shard, so need to
      // elastic shrink earlier data to make room.
      ElasticShrinkForFirstShard(first_shard_extra_blocks + (shard_end_block - original_shard_end_block));
      assert(shard_end_block == GetShardBeginBlock(1));
    }
    BackPropAndStore(shard_begin_block, gauss, count, shard_slots);
  }

  void StoreFilterMetadata() {
    assert(data);
    assert(bytes >= metadata_size);

    char *metadata = data + bytes - metadata_size;

    // See BloomFilterPolicy::GetSimpleGaussBitsReader re: metadata
    // -2 = Marker for Simple Gauss
    metadata[0] = static_cast<char>(-2);

    // Log2 of number of shards
    assert(log2_shards < 64);
    metadata[1] = static_cast<char>(log2_shards);

    // Extra space going into last shard (x 64 x match_bits bits)
    assert(first_shard_extra_blocks < 256);
    metadata[2] = static_cast<char>(first_shard_extra_blocks);

    // Average number of slots per shard, in 16 bits
    assert(avg_shard_slots < 0x10000U);
    EncodeFixed16(metadata + 3, static_cast<uint16_t>(avg_shard_slots));
  }

  void InitializeFromMetadata() {
    assert(data);
    assert(bytes >= metadata_size);

    char *metadata = data + bytes - metadata_size;
    assert(metadata[0] == static_cast<char>(-2));

    this->log2_shards = static_cast<uint8_t>(metadata[1]);
    assert(log2_shards < 64);

    this->first_shard_extra_blocks = static_cast<uint8_t>(metadata[2]);

    this->avg_shard_slots = DecodeFixed16(metadata + 3);

    CalculateMatchBitSettings();
  }

  // Reads:
  //   lower_match_bits
  //   total_blocks
  //   first_block_upper
  double GetFpRate() const {
    if (total_blocks == 0) {
      return 1.0;
    }
    double upper_fp_rate = std::pow(0.5, lower_match_bits + 1);
    double lower_fp_rate = std::pow(0.5, lower_match_bits);
    double upper_portion = static_cast<double>(total_blocks - first_block_upper) /
                           static_cast<double>(total_blocks);
    double lower_portion = static_cast<double>(first_block_upper) /
                           static_cast<double>(total_blocks);
    return upper_portion * upper_fp_rate + lower_portion * lower_fp_rate;
  }
};

class SimpleGaussBitsBuilder : public BuiltinFilterBitsBuilder {
 public:
  explicit SimpleGaussBitsBuilder(const int millibits_per_key)
      : millibits_per_key_(millibits_per_key) {
    assert(millibits_per_key >= 1000);
  }

  // No Copy allowed
  SimpleGaussBitsBuilder(const SimpleGaussBitsBuilder&) = delete;
  void operator=(const SimpleGaussBitsBuilder&) = delete;

  ~SimpleGaussBitsBuilder() override {}

  virtual void AddKey(const Slice& key) override {
    uint64_t hash = GetSliceHash64(key);
    if (hash_entries_.empty() || hash != hash_entries_.back()) {
      hash_entries_.push_back(hash);
    }
  }

  virtual Slice Finish(std::unique_ptr<const char[]>* buf) override {
    SimpleGaussFilter f;

    f.CalculateSpace(hash_entries_.size(), millibits_per_key_);
    f.AllocateSpace(buf);
    f.BuildShards(&hash_entries_);
    f.StoreFilterMetadata();

    return Slice(f.data, f.bytes);
  }

  int CalculateNumEntry(const uint32_t bytes) override {
    // FIXME?
    uint32_t bytes_no_meta = bytes >= 5u ? bytes - 5u : 0;
    return static_cast<int>(uint64_t{8000} * bytes_no_meta /
                            millibits_per_key_);
  }

  uint32_t CalculateSpace(const int num_entry) override {
    SimpleGaussFilter f;
    assert(num_entry >= 0);
    f.CalculateSpace(static_cast<size_t>(num_entry), millibits_per_key_);
    return static_cast<uint32_t>(f.bytes);
  }

  double EstimatedFpRate(size_t keys, size_t bytes) override {
    SimpleGaussFilter f;
    f.CalculatePreferredSharding(keys);
    // TODO: set for more conservative estimate?
    assert(f.first_shard_extra_blocks == 0);
    f.bytes = bytes;
    f.CalculateMatchBitSettings();
    return f.GetFpRate();
  }

 private:
/*
  uint32_t GetPreferredNumOutputRows(uint32_t num_coeff_rows) {
    // Seems to be roughly 80% chance encoding success with this formula
    //double overhead = std::min(1.1, 1.0 + std::max(1.5, std::log2(num_coeff_rows) - 8) / 100.0);
    // More compact
    double overhead = std::min(1.1, 1.0 + std::max(0.8, std::log2(num_coeff_rows) - 9) / 100.0);
    return static_cast<uint32_t>(num_coeff_rows * overhead + 63) / 64 * 64;
  }
*/

  int millibits_per_key_;
  // A deque avoids unnecessary copying of already-saved values
  // and has near-minimal peak memory use.
  std::deque<uint64_t> hash_entries_;
};

class SimpleGaussBitsReader : public FilterBitsReader {
 public:
  SimpleGaussBitsReader(const char* data, size_t bytes) {
    // FIXME?
    f.data = const_cast<char *>(data);
    f.bytes = bytes;
    f.InitializeFromMetadata();
  }

  // No Copy allowed
  SimpleGaussBitsReader(const SimpleGaussBitsReader&) = delete;
  void operator=(const SimpleGaussBitsReader&) = delete;

  ~SimpleGaussBitsReader() override {}

  bool MayMatch(const Slice& key) override {
    const uint64_t pre_h = GaussData::TweakPreHash(GetSliceHash64(key));
    const uint32_t leave_shard_shift_minus1 = 64U - f.log2_shards - 1;
    uint32_t shard = pre_h >> 1 >> leave_shard_shift_minus1;
    const uint32_t section = GaussData::PreHashToSection(pre_h);

    const uint8_t shard_meta = static_cast<uint8_t>(f.GetShardMetaDataStart()[shard]);

    uint64_t h;
    if (shard > 0 && section > shard_meta) {
      // Bumped
      uint32_t bumped_shard_or = (uint32_t{1} << FloorLog2(shard));
      uint32_t bumped_shard_mask = bumped_shard_or - 1;
      bumped_shard_or /= 2;
      bumped_shard_mask /= 2;
      shard = (Upper32of64(pre_h) & bumped_shard_mask) | bumped_shard_or;
      h = pre_h * 0x9e3779b97f4a7c13 * 0x9e3779b97f4a7c13;
    } else {
      // Not bumped
      h = GaussData::SeedPreHash(pre_h, shard_meta);
    }

    const size_t begin_block = f.GetShardBeginBlock(shard);
    const size_t end_block = f.GetShardBeginBlock(shard + 1);
    const uint32_t num_output_rows = static_cast<uint32_t>(end_block - begin_block) * 64;

    const uint32_t start = GaussData::HashToStart(h, num_output_rows);
    const size_t start_block = begin_block + start / 64;
    const uint32_t start_bit = start % 64;
    const uint32_t match_bits = f.lower_match_bits + (start_block >= f.first_block_upper);
    const uint64_t *word_data = reinterpret_cast<const uint64_t *>(f.GetBlockDataStart(start_block));
    PREFETCH(word_data, 0 /* rw */, 1 /* locality */);
    const uint32_t maybe_offset = (start_bit != 0) * match_bits;
    PREFETCH(word_data + maybe_offset + match_bits - 1, 0 /* rw */, 1 /* locality */);

    const uint64_t coeff_row = GaussData::HashToCoeffRow(h);
    const uint32_t match_row = GaussData::HashToMatchRow(h);
    uint32_t versus_row = 0;
    uint64_t column;

    // TODO: endianness
    switch (match_bits) {
      default:
      case 10:
        column = (word_data[9] >> start_bit) | (word_data[9 + maybe_offset] << (64 - start_bit));
        versus_row |= static_cast<uint32_t>(BitParity(column & coeff_row)) << 9;
        FALLTHROUGH_INTENDED;
      case 9:
        column = (word_data[8] >> start_bit) | (word_data[8 + maybe_offset] << (64 - start_bit));
        versus_row |= static_cast<uint32_t>(BitParity(column & coeff_row)) << 8;
        FALLTHROUGH_INTENDED;
      case 8:
        column = (word_data[7] >> start_bit) | (word_data[7 + maybe_offset] << (64 - start_bit));
        versus_row |= static_cast<uint32_t>(BitParity(column & coeff_row)) << 7;
        FALLTHROUGH_INTENDED;
      case 7:
        column = (word_data[6] >> start_bit) | (word_data[6 + maybe_offset] << (64 - start_bit));
        versus_row |= static_cast<uint32_t>(BitParity(column & coeff_row)) << 6;
        FALLTHROUGH_INTENDED;
      case 6:
        column = (word_data[5] >> start_bit) | (word_data[5 + maybe_offset] << (64 - start_bit));
        versus_row |= static_cast<uint32_t>(BitParity(column & coeff_row)) << 5;
        FALLTHROUGH_INTENDED;
      case 5:
        column = (word_data[4] >> start_bit) | (word_data[4 + maybe_offset] << (64 - start_bit));
        versus_row |= static_cast<uint32_t>(BitParity(column & coeff_row)) << 4;
        FALLTHROUGH_INTENDED;
      case 4:
        column = (word_data[3] >> start_bit) | (word_data[3 + maybe_offset] << (64 - start_bit));
        versus_row |= static_cast<uint32_t>(BitParity(column & coeff_row)) << 3;
        FALLTHROUGH_INTENDED;
      case 3:
        column = (word_data[2] >> start_bit) | (word_data[2 + maybe_offset] << (64 - start_bit));
        versus_row |= static_cast<uint32_t>(BitParity(column & coeff_row)) << 2;
        FALLTHROUGH_INTENDED;
      case 2:
        column = (word_data[1] >> start_bit) | (word_data[1 + maybe_offset] << (64 - start_bit));
        versus_row |= static_cast<uint32_t>(BitParity(column & coeff_row)) << 1;
        FALLTHROUGH_INTENDED;
      case 1:
        column = (word_data[0] >> start_bit) | (word_data[0 + maybe_offset] << (64 - start_bit));
        versus_row |= static_cast<uint32_t>(BitParity(column & coeff_row)) << 0;
        FALLTHROUGH_INTENDED;
      case 0:
        FALLTHROUGH_INTENDED;
    }
    //printf("QM: %u\n", match_bits);
    return versus_row == (match_row & ((uint32_t{1} << match_bits) - 1));

    // Old, not as efficient?
    /*
    for (uint32_t i = 0; i < match_bits_; ++i) {
      bool v = GaussData::DotCoeffRowWithOutputColumn(start, coeff_row, word_data, match_bits, i);
      if (v != (match_row & 1)) {
        return false;
      }
      match_row >>= 1;
    }
    return true;
    */
  }

  virtual void MayMatch(int num_keys, Slice** keys, bool* may_match) override {
    (void)num_keys;
    (void)keys;
    (void)may_match;
/*
    std::array<uint32_t, MultiGetContext::MAX_BATCH_SIZE> hashes;
    std::array<uint32_t, MultiGetContext::MAX_BATCH_SIZE> byte_offsets;
    for (int i = 0; i < num_keys; ++i) {
      uint64_t h = GetSliceHash64(*keys[i]);
      SimpleGaussImpl::PrepareHash(Lower32of64(h), len_bytes_, data_,
                                      &byte_offsets[i]);
      hashes[i] = Upper32of64(h);
    }
    for (int i = 0; i < num_keys; ++i) {
      may_match[i] = SimpleGaussImpl::HashMayMatchPrepared(
          hashes[i], num_probes_, data_ + byte_offsets[i]);
    }
*/
  }

 private:
  SimpleGaussFilter f;
};

using LegacyBloomImpl = LegacyLocalityBloomImpl</*ExtraRotates*/ false>;

class LegacyBloomBitsBuilder : public BuiltinFilterBitsBuilder {
 public:
  explicit LegacyBloomBitsBuilder(const int bits_per_key, Logger* info_log);

  // No Copy allowed
  LegacyBloomBitsBuilder(const LegacyBloomBitsBuilder&) = delete;
  void operator=(const LegacyBloomBitsBuilder&) = delete;

  ~LegacyBloomBitsBuilder() override;

  void AddKey(const Slice& key) override;

  Slice Finish(std::unique_ptr<const char[]>* buf) override;

  int CalculateNumEntry(const uint32_t bytes) override;

  uint32_t CalculateSpace(const int num_entry) override {
    uint32_t dont_care1;
    uint32_t dont_care2;
    return CalculateSpace(num_entry, &dont_care1, &dont_care2);
  }

  double EstimatedFpRate(size_t keys, size_t bytes) override {
    return LegacyBloomImpl::EstimatedFpRate(keys, bytes - /*metadata*/ 5,
                                            num_probes_);
  }

 private:
  int bits_per_key_;
  int num_probes_;
  std::vector<uint32_t> hash_entries_;
  Logger* info_log_;

  // Get totalbits that optimized for cpu cache line
  uint32_t GetTotalBitsForLocality(uint32_t total_bits);

  // Reserve space for new filter
  char* ReserveSpace(const int num_entry, uint32_t* total_bits,
                     uint32_t* num_lines);

  // Implementation-specific variant of public CalculateSpace
  uint32_t CalculateSpace(const int num_entry, uint32_t* total_bits,
                          uint32_t* num_lines);

  // Assuming single threaded access to this function.
  void AddHash(uint32_t h, char* data, uint32_t num_lines, uint32_t total_bits);
};

LegacyBloomBitsBuilder::LegacyBloomBitsBuilder(const int bits_per_key,
                                               Logger* info_log)
    : bits_per_key_(bits_per_key),
      num_probes_(LegacyNoLocalityBloomImpl::ChooseNumProbes(bits_per_key_)),
      info_log_(info_log) {
  assert(bits_per_key_);
}

LegacyBloomBitsBuilder::~LegacyBloomBitsBuilder() {}

void LegacyBloomBitsBuilder::AddKey(const Slice& key) {
  uint32_t hash = BloomHash(key);
  if (hash_entries_.size() == 0 || hash != hash_entries_.back()) {
    hash_entries_.push_back(hash);
  }
}

Slice LegacyBloomBitsBuilder::Finish(std::unique_ptr<const char[]>* buf) {
  uint32_t total_bits, num_lines;
  size_t num_entries = hash_entries_.size();
  char* data =
      ReserveSpace(static_cast<int>(num_entries), &total_bits, &num_lines);
  assert(data);

  if (total_bits != 0 && num_lines != 0) {
    for (auto h : hash_entries_) {
      AddHash(h, data, num_lines, total_bits);
    }

    // Check for excessive entries for 32-bit hash function
    if (num_entries >= /* minimum of 3 million */ 3000000U) {
      // More specifically, we can detect that the 32-bit hash function
      // is causing significant increase in FP rate by comparing current
      // estimated FP rate to what we would get with a normal number of
      // keys at same memory ratio.
      double est_fp_rate = LegacyBloomImpl::EstimatedFpRate(
          num_entries, total_bits / 8, num_probes_);
      double vs_fp_rate = LegacyBloomImpl::EstimatedFpRate(
          1U << 16, (1U << 16) * bits_per_key_ / 8, num_probes_);

      if (est_fp_rate >= 1.50 * vs_fp_rate) {
        // For more details, see
        // https://github.com/facebook/rocksdb/wiki/RocksDB-Bloom-Filter
        ROCKS_LOG_WARN(
            info_log_,
            "Using legacy SST/BBT Bloom filter with excessive key count "
            "(%.1fM @ %dbpk), causing estimated %.1fx higher filter FP rate. "
            "Consider using new Bloom with format_version>=5, smaller SST "
            "file size, or partitioned filters.",
            num_entries / 1000000.0, bits_per_key_, est_fp_rate / vs_fp_rate);
      }
    }
  }
  // See BloomFilterPolicy::GetFilterBitsReader for metadata
  data[total_bits / 8] = static_cast<char>(num_probes_);
  EncodeFixed32(data + total_bits / 8 + 1, static_cast<uint32_t>(num_lines));

  const char* const_data = data;
  buf->reset(const_data);
  hash_entries_.clear();

  return Slice(data, total_bits / 8 + 5);
}

uint32_t LegacyBloomBitsBuilder::GetTotalBitsForLocality(uint32_t total_bits) {
  uint32_t num_lines =
      (total_bits + CACHE_LINE_SIZE * 8 - 1) / (CACHE_LINE_SIZE * 8);

  // Make num_lines an odd number to make sure more bits are involved
  // when determining which block.
  if (num_lines % 2 == 0) {
    num_lines++;
  }
  return num_lines * (CACHE_LINE_SIZE * 8);
}

uint32_t LegacyBloomBitsBuilder::CalculateSpace(const int num_entry,
                                                uint32_t* total_bits,
                                                uint32_t* num_lines) {
  assert(bits_per_key_);
  if (num_entry != 0) {
    uint32_t total_bits_tmp = static_cast<uint32_t>(num_entry * bits_per_key_);

    *total_bits = GetTotalBitsForLocality(total_bits_tmp);
    *num_lines = *total_bits / (CACHE_LINE_SIZE * 8);
    assert(*total_bits > 0 && *total_bits % 8 == 0);
  } else {
    // filter is empty, just leave space for metadata
    *total_bits = 0;
    *num_lines = 0;
  }

  // Reserve space for Filter
  uint32_t sz = *total_bits / 8;
  sz += 5;  // 4 bytes for num_lines, 1 byte for num_probes
  return sz;
}

char* LegacyBloomBitsBuilder::ReserveSpace(const int num_entry,
                                           uint32_t* total_bits,
                                           uint32_t* num_lines) {
  uint32_t sz = CalculateSpace(num_entry, total_bits, num_lines);
  char* data = new char[sz];
  memset(data, 0, sz);
  return data;
}

int LegacyBloomBitsBuilder::CalculateNumEntry(const uint32_t bytes) {
  assert(bits_per_key_);
  assert(bytes > 0);
  int high = static_cast<int>(bytes * 8 / bits_per_key_ + 1);
  int low = 1;
  int n = high;
  for (; n >= low; n--) {
    if (CalculateSpace(n) <= bytes) {
      break;
    }
  }
  assert(n < high);  // High should be an overestimation
  return n;
}

inline void LegacyBloomBitsBuilder::AddHash(uint32_t h, char* data,
                                            uint32_t num_lines,
                                            uint32_t total_bits) {
#ifdef NDEBUG
  static_cast<void>(total_bits);
#endif
  assert(num_lines > 0 && total_bits > 0);

  LegacyBloomImpl::AddHash(h, num_lines, num_probes_, data,
                           folly::constexpr_log2(CACHE_LINE_SIZE));
}

class LegacyBloomBitsReader : public FilterBitsReader {
 public:
  LegacyBloomBitsReader(const char* data, int num_probes, uint32_t num_lines,
                        uint32_t log2_cache_line_size)
      : data_(data),
        num_probes_(num_probes),
        num_lines_(num_lines),
        log2_cache_line_size_(log2_cache_line_size) {}

  // No Copy allowed
  LegacyBloomBitsReader(const LegacyBloomBitsReader&) = delete;
  void operator=(const LegacyBloomBitsReader&) = delete;

  ~LegacyBloomBitsReader() override {}

  // "contents" contains the data built by a preceding call to
  // FilterBitsBuilder::Finish. MayMatch must return true if the key was
  // passed to FilterBitsBuilder::AddKey. This method may return true or false
  // if the key was not on the list, but it should aim to return false with a
  // high probability.
  bool MayMatch(const Slice& key) override {
    uint32_t hash = BloomHash(key);
    uint32_t byte_offset;
    LegacyBloomImpl::PrepareHashMayMatch(
        hash, num_lines_, data_, /*out*/ &byte_offset, log2_cache_line_size_);
    return LegacyBloomImpl::HashMayMatchPrepared(
        hash, num_probes_, data_ + byte_offset, log2_cache_line_size_);
  }

  virtual void MayMatch(int num_keys, Slice** keys, bool* may_match) override {
    std::array<uint32_t, MultiGetContext::MAX_BATCH_SIZE> hashes;
    std::array<uint32_t, MultiGetContext::MAX_BATCH_SIZE> byte_offsets;
    for (int i = 0; i < num_keys; ++i) {
      hashes[i] = BloomHash(*keys[i]);
      LegacyBloomImpl::PrepareHashMayMatch(hashes[i], num_lines_, data_,
                                           /*out*/ &byte_offsets[i],
                                           log2_cache_line_size_);
    }
    for (int i = 0; i < num_keys; ++i) {
      may_match[i] = LegacyBloomImpl::HashMayMatchPrepared(
          hashes[i], num_probes_, data_ + byte_offsets[i],
          log2_cache_line_size_);
    }
  }

 private:
  const char* data_;
  const int num_probes_;
  const uint32_t num_lines_;
  const uint32_t log2_cache_line_size_;
};

class AlwaysTrueFilter : public FilterBitsReader {
 public:
  bool MayMatch(const Slice&) override { return true; }
  using FilterBitsReader::MayMatch;  // inherit overload
};

class AlwaysFalseFilter : public FilterBitsReader {
 public:
  bool MayMatch(const Slice&) override { return false; }
  using FilterBitsReader::MayMatch;  // inherit overload
};

}  // namespace

const std::vector<BloomFilterPolicy::Mode> BloomFilterPolicy::kAllFixedImpls = {
    kLegacyBloom,
    kDeprecatedBlock,
    kFastLocalBloom,
    kSimpleGauss,
};

const std::vector<BloomFilterPolicy::Mode> BloomFilterPolicy::kAllUserModes = {
    kDeprecatedBlock,
    kAuto,
};

BloomFilterPolicy::BloomFilterPolicy(double bits_per_key, Mode mode)
    : mode_(mode), warned_(false) {
  // Sanitize bits_per_key
  if (bits_per_key < 1.0) {
    bits_per_key = 1.0;
  } else if (!(bits_per_key < 100.0)) {  // including NaN
    bits_per_key = 100.0;
  }

  // Includes a nudge toward rounding up, to ensure on all platforms
  // that doubles specified with three decimal digits after the decimal
  // point are interpreted accurately.
  millibits_per_key_ = static_cast<int>(bits_per_key * 1000.0 + 0.500001);

  // For better or worse, this is a rounding up of a nudged rounding up,
  // e.g. 7.4999999999999 will round up to 8, but that provides more
  // predictability against small arithmetic errors in floating point.
  whole_bits_per_key_ = (millibits_per_key_ + 500) / 1000;
}

BloomFilterPolicy::~BloomFilterPolicy() {}

const char* BloomFilterPolicy::Name() const {
  return "rocksdb.BuiltinBloomFilter";
}

void BloomFilterPolicy::CreateFilter(const Slice* keys, int n,
                                     std::string* dst) const {
  // We should ideally only be using this deprecated interface for
  // appropriately constructed BloomFilterPolicy
  assert(mode_ == kDeprecatedBlock);

  // Compute bloom filter size (in both bits and bytes)
  uint32_t bits = static_cast<uint32_t>(n * whole_bits_per_key_);

  // For small n, we can see a very high false positive rate.  Fix it
  // by enforcing a minimum bloom filter length.
  if (bits < 64) bits = 64;

  uint32_t bytes = (bits + 7) / 8;
  bits = bytes * 8;

  int num_probes =
      LegacyNoLocalityBloomImpl::ChooseNumProbes(whole_bits_per_key_);

  const size_t init_size = dst->size();
  dst->resize(init_size + bytes, 0);
  dst->push_back(static_cast<char>(num_probes));  // Remember # of probes
  char* array = &(*dst)[init_size];
  for (int i = 0; i < n; i++) {
    LegacyNoLocalityBloomImpl::AddHash(BloomHash(keys[i]), bits, num_probes,
                                       array);
  }
}

bool BloomFilterPolicy::KeyMayMatch(const Slice& key,
                                    const Slice& bloom_filter) const {
  const size_t len = bloom_filter.size();
  if (len < 2 || len > 0xffffffffU) {
    return false;
  }

  const char* array = bloom_filter.data();
  const uint32_t bits = static_cast<uint32_t>(len - 1) * 8;

  // Use the encoded k so that we can read filters generated by
  // bloom filters created using different parameters.
  const int k = static_cast<uint8_t>(array[len - 1]);
  if (k > 30) {
    // Reserved for potentially new encodings for short bloom filters.
    // Consider it a match.
    return true;
  }
  // NB: using stored k not num_probes for whole_bits_per_key_
  return LegacyNoLocalityBloomImpl::HashMayMatch(BloomHash(key), bits, k,
                                                 array);
}

FilterBitsBuilder* BloomFilterPolicy::GetFilterBitsBuilder() const {
  // This code path should no longer be used, for the built-in
  // BloomFilterPolicy. Internal to RocksDB and outside
  // BloomFilterPolicy, only get a FilterBitsBuilder with
  // BloomFilterPolicy::GetBuilderFromContext(), which will call
  // BloomFilterPolicy::GetBuilderWithContext(). RocksDB users have
  // been warned (HISTORY.md) that they can no longer call this on
  // the built-in BloomFilterPolicy (unlikely).
  assert(false);
  return GetBuilderWithContext(FilterBuildingContext(BlockBasedTableOptions()));
}

FilterBitsBuilder* BloomFilterPolicy::GetBuilderWithContext(
    const FilterBuildingContext& context) const {
  Mode cur = mode_;
  // Unusual code construction so that we can have just
  // one exhaustive switch without (risky) recursion
  for (int i = 0; i < 2; ++i) {
    switch (cur) {
      case kAuto:
        if (context.table_options.format_version < 5) {
          cur = kLegacyBloom;
        } else {
          cur = kFastLocalBloom;
        }
        break;
      case kDeprecatedBlock:
        return nullptr;
      case kSimpleGauss:
        return new SimpleGaussBitsBuilder(millibits_per_key_);
      case kFastLocalBloom:
        return new FastLocalBloomBitsBuilder(millibits_per_key_);
      case kLegacyBloom:
        if (whole_bits_per_key_ >= 14 && context.info_log &&
            !warned_.load(std::memory_order_relaxed)) {
          warned_ = true;
          const char* adjective;
          if (whole_bits_per_key_ >= 20) {
            adjective = "Dramatic";
          } else {
            adjective = "Significant";
          }
          // For more details, see
          // https://github.com/facebook/rocksdb/wiki/RocksDB-Bloom-Filter
          ROCKS_LOG_WARN(
              context.info_log,
              "Using legacy Bloom filter with high (%d) bits/key. "
              "%s filter space and/or accuracy improvement is available "
              "with format_version>=5.",
              whole_bits_per_key_, adjective);
        }
        return new LegacyBloomBitsBuilder(whole_bits_per_key_,
                                          context.info_log);
    }
  }
  assert(false);
  return nullptr;  // something legal
}

FilterBitsBuilder* BloomFilterPolicy::GetBuilderFromContext(
    const FilterBuildingContext& context) {
  if (context.table_options.filter_policy) {
    return context.table_options.filter_policy->GetBuilderWithContext(context);
  } else {
    return nullptr;
  }
}

// Read metadata to determine what kind of FilterBitsReader is needed
// and return a new one.
FilterBitsReader* BloomFilterPolicy::GetFilterBitsReader(
    const Slice& contents) const {
  uint32_t len_with_meta = static_cast<uint32_t>(contents.size());
  if (len_with_meta <= 5) {
    // filter is empty or broken. Treat like zero keys added.
    return new AlwaysFalseFilter();
  }

  // Legacy Bloom filter data:
  //             0 +-----------------------------------+
  //               | Raw Bloom filter data             |
  //               | ...                               |
  //           len +-----------------------------------+
  //               | byte for num_probes or            |
  //               |   marker for new implementations  |
  //         len+1 +-----------------------------------+
  //               | four bytes for number of cache    |
  //               |   lines                           |
  // len_with_meta +-----------------------------------+

  int8_t raw_num_probes =
      static_cast<int8_t>(contents.data()[len_with_meta - 5]);
  // NB: *num_probes > 30 and < 128 probably have not been used, because of
  // BloomFilterPolicy::initialize, unless directly calling
  // LegacyBloomBitsBuilder as an API, but we are leaving those cases in
  // limbo with LegacyBloomBitsReader for now.

  if (raw_num_probes < 1) {
    // Note: < 0 (or unsigned > 127) indicate special new implementations
    // (or reserved for future use)
    if (raw_num_probes == -1) {
      // Marker for newer Bloom implementations
      return GetBloomBitsReader(contents);
    }
    if (raw_num_probes == -2) {
      // Marker for simple gauss filter
      return GetSimpleGaussBitsReader(contents);
    }
    // otherwise
    // Treat as zero probes (always FP) for now.
    return new AlwaysTrueFilter();
  }
  // else attempt decode for LegacyBloomBitsReader

  int num_probes = raw_num_probes;
  assert(num_probes >= 1);
  assert(num_probes <= 127);

  uint32_t len = len_with_meta - 5;
  assert(len > 0);

  uint32_t num_lines = DecodeFixed32(contents.data() + len_with_meta - 4);
  uint32_t log2_cache_line_size;

  if (num_lines * CACHE_LINE_SIZE == len) {
    // Common case
    log2_cache_line_size = folly::constexpr_log2(CACHE_LINE_SIZE);
  } else if (num_lines == 0 || len % num_lines != 0) {
    // Invalid (no solution to num_lines * x == len)
    // Treat as zero probes (always FP) for now.
    return new AlwaysTrueFilter();
  } else {
    // Determine the non-native cache line size (from another system)
    log2_cache_line_size = 0;
    while ((num_lines << log2_cache_line_size) < len) {
      ++log2_cache_line_size;
    }
    if ((num_lines << log2_cache_line_size) != len) {
      // Invalid (block size not a power of two)
      // Treat as zero probes (always FP) for now.
      return new AlwaysTrueFilter();
    }
  }
  // if not early return
  return new LegacyBloomBitsReader(contents.data(), num_probes, num_lines,
                                   log2_cache_line_size);
}

// For newer Bloom filter implementations
FilterBitsReader* BloomFilterPolicy::GetBloomBitsReader(
    const Slice& contents) const {
  uint32_t len_with_meta = static_cast<uint32_t>(contents.size());
  uint32_t len = len_with_meta - 5;

  assert(len > 0);  // precondition

  // New Bloom filter data:
  //             0 +-----------------------------------+
  //               | Raw Bloom filter data             |
  //               | ...                               |
  //           len +-----------------------------------+
  //               | char{-1} byte -> new Bloom filter |
  //         len+1 +-----------------------------------+
  //               | byte for subimplementation        |
  //               |   0: FastLocalBloom               |
  //               |   other: reserved                 |
  //         len+2 +-----------------------------------+
  //               | byte for block_and_probes         |
  //               |   0 in top 3 bits -> 6 -> 64-byte |
  //               |   reserved:                       |
  //               |   1 in top 3 bits -> 7 -> 128-byte|
  //               |   2 in top 3 bits -> 8 -> 256-byte|
  //               |   ...                             |
  //               |   num_probes in bottom 5 bits,    |
  //               |     except 0 and 31 reserved      |
  //         len+3 +-----------------------------------+
  //               | two bytes reserved                |
  //               |   possibly for hash seed          |
  // len_with_meta +-----------------------------------+

  // Read more metadata (see above)
  char sub_impl_val = contents.data()[len_with_meta - 4];
  char block_and_probes = contents.data()[len_with_meta - 3];
  int log2_block_bytes = ((block_and_probes >> 5) & 7) + 6;

  int num_probes = (block_and_probes & 31);
  if (num_probes < 1 || num_probes > 30) {
    // Reserved / future safe
    return new AlwaysTrueFilter();
  }

  uint16_t rest = DecodeFixed16(contents.data() + len_with_meta - 2);
  if (rest != 0) {
    // Reserved, possibly for hash seed
    // Future safe
    return new AlwaysTrueFilter();
  }

  if (sub_impl_val == 0) {        // FastLocalBloom
    if (log2_block_bytes == 6) {  // Only block size supported for now
      return new FastLocalBloomBitsReader(contents.data(), num_probes, len);
    }
  }
  // otherwise
  // Reserved / future safe
  return new AlwaysTrueFilter();
}

FilterBitsReader* BloomFilterPolicy::GetSimpleGaussBitsReader(
    const Slice& contents) const {
  // FIXME: more checks?
  return new SimpleGaussBitsReader(contents.data(), contents.size());
}

const FilterPolicy* NewBloomFilterPolicy(double bits_per_key,
                                         bool use_block_based_builder) {
  BloomFilterPolicy::Mode m;
  if (use_block_based_builder) {
    m = BloomFilterPolicy::kDeprecatedBlock;
  } else {
    m = BloomFilterPolicy::kAuto;
  }
  assert(std::find(BloomFilterPolicy::kAllUserModes.begin(),
                   BloomFilterPolicy::kAllUserModes.end(),
                   m) != BloomFilterPolicy::kAllUserModes.end());
  return new BloomFilterPolicy(bits_per_key, m);
}

FilterBuildingContext::FilterBuildingContext(
    const BlockBasedTableOptions& _table_options)
    : table_options(_table_options) {}

FilterPolicy::~FilterPolicy() { }

}  // namespace ROCKSDB_NAMESPACE
