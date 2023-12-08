//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "rocksdb/experimental.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "db/db_impl/db_impl.h"
#include "db/version_util.h"
#include "logging/logging.h"
#include "util/atomic.h"

namespace ROCKSDB_NAMESPACE {
namespace experimental {


Status SuggestCompactRange(DB* db, ColumnFamilyHandle* column_family,
                           const Slice* begin, const Slice* end) {
  if (db == nullptr) {
    return Status::InvalidArgument("DB is empty");
  }

  return db->SuggestCompactRange(column_family, begin, end);
}

Status PromoteL0(DB* db, ColumnFamilyHandle* column_family, int target_level) {
  if (db == nullptr) {
    return Status::InvalidArgument("Didn't recognize DB object");
  }
  return db->PromoteL0(column_family, target_level);
}


Status SuggestCompactRange(DB* db, const Slice* begin, const Slice* end) {
  return SuggestCompactRange(db, db->DefaultColumnFamily(), begin, end);
}

Status UpdateManifestForFilesState(
    const DBOptions& db_opts, const std::string& db_name,
    const std::vector<ColumnFamilyDescriptor>& column_families,
    const UpdateManifestForFilesStateOptions& opts) {
  // TODO: plumb Env::IOActivity, Env::IOPriority
  const ReadOptions read_options;
  const WriteOptions write_options;
  OfflineManifestWriter w(db_opts, db_name);
  Status s = w.Recover(column_families);

  size_t files_updated = 0;
  size_t cfs_updated = 0;
  auto fs = db_opts.env->GetFileSystem();

  for (auto cfd : *w.Versions().GetColumnFamilySet()) {
    if (!s.ok()) {
      break;
    }
    assert(cfd);

    if (cfd->IsDropped() || !cfd->initialized()) {
      continue;
    }

    const auto* current = cfd->current();
    assert(current);

    const auto* vstorage = current->storage_info();
    assert(vstorage);

    VersionEdit edit;
    edit.SetColumnFamily(cfd->GetID());

    /* SST files */
    for (int level = 0; level < cfd->NumberLevels(); level++) {
      if (!s.ok()) {
        break;
      }
      const auto& level_files = vstorage->LevelFiles(level);

      for (const auto& lf : level_files) {
        assert(lf);

        uint64_t number = lf->fd.GetNumber();
        std::string fname =
            TableFileName(w.IOptions().db_paths, number, lf->fd.GetPathId());

        std::unique_ptr<FSSequentialFile> f;
        FileOptions fopts;
        // Use kUnknown to signal the FileSystem to search all tiers for the
        // file.
        fopts.temperature = Temperature::kUnknown;

        IOStatus file_ios =
            fs->NewSequentialFile(fname, fopts, &f, /*dbg*/ nullptr);
        if (file_ios.ok()) {
          if (opts.update_temperatures) {
            Temperature temp = f->GetTemperature();
            if (temp != Temperature::kUnknown && temp != lf->temperature) {
              // Current state inconsistent with manifest
              ++files_updated;
              edit.DeleteFile(level, number);
              edit.AddFile(
                  level, number, lf->fd.GetPathId(), lf->fd.GetFileSize(),
                  lf->smallest, lf->largest, lf->fd.smallest_seqno,
                  lf->fd.largest_seqno, lf->marked_for_compaction, temp,
                  lf->oldest_blob_file_number, lf->oldest_ancester_time,
                  lf->file_creation_time, lf->epoch_number, lf->file_checksum,
                  lf->file_checksum_func_name, lf->unique_id,
                  lf->compensated_range_deletion_size, lf->tail_size,
                  lf->user_defined_timestamps_persisted);
            }
          }
        } else {
          s = file_ios;
          break;
        }
      }
    }

    if (s.ok() && edit.NumEntries() > 0) {
      std::unique_ptr<FSDirectory> db_dir;
      s = fs->NewDirectory(db_name, IOOptions(), &db_dir, nullptr);
      if (s.ok()) {
        s = w.LogAndApply(read_options, write_options, cfd, &edit,
                          db_dir.get());
      }
      if (s.ok()) {
        ++cfs_updated;
      }
    }
  }

  if (cfs_updated > 0) {
    ROCKS_LOG_INFO(db_opts.info_log,
                   "UpdateManifestForFilesState: updated %zu files in %zu CFs",
                   files_updated, cfs_updated);
  } else if (s.ok()) {
    ROCKS_LOG_INFO(db_opts.info_log,
                   "UpdateManifestForFilesState: no updates needed");
  }
  if (!s.ok()) {
    ROCKS_LOG_ERROR(db_opts.info_log, "UpdateManifestForFilesState failed: %s",
                    s.ToString().c_str());
  }

  return s;
}

// EXPERIMENTAL new filtering features

namespace {
void GetFilterInput(FilterInput select, const Slice& key,
                    const KeySegmentsExtractor::Result& extracted,
                    Slice* out_input, Slice* out_leadup) {
  struct FilterInputGetter {
    explicit FilterInputGetter(const Slice& _key,
                               const KeySegmentsExtractor::Result& _extracted)
        : key(_key), extracted(_extracted) {}
    const Slice& key;
    const KeySegmentsExtractor::Result& extracted;

    Slice operator()(SelectKeySegment select) {
      size_t count = extracted.segment_ends.size();
      if (count <= select.segment_index) {
        return Slice();
      }
      assert(count > 0);
      size_t start = select.segment_index > 0
                         ? extracted.segment_ends[select.segment_index - 1]
                         : 0;
      size_t end =
          extracted
              .segment_ends[std::min(size_t{select.segment_index}, count - 1)];
      return Slice(key.data() + start, end - start);
    }

    Slice operator()(SelectKeySegmentRange select) {
      assert(select.from_segment_index <= select.to_segment_index);
      size_t count = extracted.segment_ends.size();
      if (count <= select.from_segment_index) {
        return Slice();
      }
      assert(count > 0);
      size_t start = select.from_segment_index > 0
                         ? extracted.segment_ends[select.from_segment_index - 1]
                         : 0;
      size_t end = extracted.segment_ends[std::min(
          size_t{select.to_segment_index}, count - 1)];
      return Slice(key.data() + start, end - start);
    }

    Slice operator()(SelectWholeKey select) { return key; }

    Slice operator()(SelectLegacyKeyPrefix select) {
      // TODO
      assert(false);
      return Slice();
    }

    Slice operator()(SelectUserTimestamp select) {
      // TODO
      assert(false);
      return Slice();
    }

    Slice operator()(SelectColumnName select) {
      // TODO
      assert(false);
      return Slice();
    }

    Slice operator()(SelectValue select) {
      // TODO
      assert(false);
      return Slice();
    }
  };

  Slice input = std::visit(FilterInputGetter(key, extracted), select);
  *out_input = input;
  if (input.empty() || input.data() < key.data() ||
      input.data() > key.data() + key.size()) {
    *out_leadup = key;
  } else {
    *out_leadup = Slice(key.data(), input.data() - key.data());
  }
}

const char* DeserializeFilterInput(const char* p, const char* limit,
                                   FilterInput* out) {
  if (p >= limit) {
    return nullptr;
  }
  uint8_t b = static_cast<uint8_t>(*p++);
  if (b & 0x80) {
    // Reserved for future use to read more bytes
    return nullptr;
  }

  switch (b >> 4) {
    case 0:
      // Various cases that don't have an argument
      switch (b) {
        case 0:
          *out = SelectWholeKey{};
          return p;
        case 1:
          *out = SelectLegacyKeyPrefix{};
          return p;
        case 2:
          *out = SelectUserTimestamp{};
          return p;
        case 3:
          *out = SelectColumnName{};
          return p;
        case 4:
          *out = SelectValue{};
          return p;
        default:
          // Reserved for future use
          return nullptr;
      }
    case 1:
      // First 16 cases of SelectKeySegment
      *out = SelectKeySegment{BitwiseAnd(b, 0xf)};
      return p;
    case 2:
      // First 16 cases of SelectKeySegmentRange
      // that are not a single key segment
      // 0: 0-1
      // 1: 0-2
      // 2: 1-2
      // 3: 0-3
      // 4: 1-3
      // 5: 2-3
      // 6: 0-4
      // 7: 1-4
      // 8: 2-4
      // 9: 3-4
      // 10: 0-5
      // 11: 1-5
      // 12: 2-5
      // 13: 3-5
      // 14: 4-5
      // 15: 0-6
      if (b < 6) {
        if (b >= 3) {
          *out = SelectKeySegmentRange{static_cast<uint8_t>(b - 3), 3};
        } else if (b >= 1) {
          *out = SelectKeySegmentRange{static_cast<uint8_t>(b - 1), 2};
        } else {
          *out = SelectKeySegmentRange{0, 1};
        }
      } else if (b < 10) {
        *out = SelectKeySegmentRange{static_cast<uint8_t>(b - 6), 4};
      } else if (b < 15) {
        *out = SelectKeySegmentRange{static_cast<uint8_t>(b - 10), 5};
      } else {
        *out = SelectKeySegmentRange{0, 6};
      }
      return p;
    default:
      // Reserved for future use
      return nullptr;
  }
}

void SerializeFilterInput(std::string* out, const FilterInput& select) {
  struct FilterInputSerializer {
    std::string* out;
    void operator()(SelectWholeKey) { out->push_back(0); }
    void operator()(SelectLegacyKeyPrefix) { out->push_back(1); }
    void operator()(SelectUserTimestamp) { out->push_back(2); }
    void operator()(SelectColumnName) { out->push_back(3); }
    void operator()(SelectValue) { out->push_back(4); }
    void operator()(SelectKeySegment select) {
      // TODO: expand supported cases
      assert(select.segment_index < 16);
      out->push_back(static_cast<char>((1 << 4) | select.segment_index));
    }
    void operator()(SelectKeySegmentRange select) {
      auto from = select.from_segment_index;
      auto to = select.to_segment_index;
      // TODO: expand supported cases
      assert(from < 6);
      assert(to < 6 || (to == 6 && from == 0));
      assert(from < to);
      int start = (to - 1) * to / 2;
      assert(start + from < 16);
      out->push_back(static_cast<char>((2 << 4) | (start + from)));
    }
  };
  std::visit(FilterInputSerializer{out}, select);
}

size_t GetFilterInputSerializedLength(const FilterInput& select) {
  // TODO: expand supported cases
  return 1;
}

uint64_t CategorySetToUint(const KeySegmentsExtractor::KeyCategorySet& s) {
  static_assert(sizeof(KeySegmentsExtractor::KeyCategorySet) ==
                sizeof(uint64_t));
  return *reinterpret_cast<const uint64_t*>(&s);
}

KeySegmentsExtractor::KeyCategorySet UintToCategorySet(uint64_t s) {
  static_assert(sizeof(KeySegmentsExtractor::KeyCategorySet) ==
                sizeof(uint64_t));
  return *reinterpret_cast<const KeySegmentsExtractor::KeyCategorySet*>(&s);
}

enum BuiltinSstQueryFilters : char {
  // Wraps a set of filters such that they use a particular
  // KeySegmentsExtractor and version, and a set of categories covering
  // all keys seen.
  kExtrAndVerAndCatFilterWrapper = 0x1,

  // Wraps a set of filters to limit their scope to a particular set of
  // categories. (Unlike kExtrAndVerAndCatFilterWrapper,
  // keys in other categories may have been seen so are not filtered here.)
  kCategoryScopeFilterWrapper = 0x2,

  // A filter representing the bytewise min and max values of a numbered
  // segment or composite (range of segments). The empty value is tracked
  // and filtered independently because it might be a special case that is
  // not representative of the minimum in a spread of values.
  kBytewiseMinMaxFilter = 0x10,
};

class SstQueryFilterBuilder {
 public:
  virtual ~SstQueryFilterBuilder() {}
  virtual void Add(const Slice& key,
                   const KeySegmentsExtractor::Result& extracted,
                   const Slice* prev_key,
                   const KeySegmentsExtractor::Result* prev_extracted) = 0;
  virtual Status GetStatus() const = 0;
  virtual size_t GetEncodedLength() const = 0;
  virtual void Finish(std::string& append_to) = 0;
};

class SstQueryFilterConfigImpl : public SstQueryFilterConfig {
 public:
  explicit SstQueryFilterConfigImpl(
      const FilterInput& input,
      const KeySegmentsExtractor::KeyCategorySet& categories)
      : input_(input), categories_(categories) {}

  virtual ~SstQueryFilterConfigImpl() {}

  virtual std::unique_ptr<SstQueryFilterBuilder> NewBuilder(
      bool sanity_checks) const = 0;

 protected:
  FilterInput input_;
  KeySegmentsExtractor::KeyCategorySet categories_;
};

class CategoryScopeFilterWrapperBuilder : public SstQueryFilterBuilder {
 public:
  explicit CategoryScopeFilterWrapperBuilder(
      KeySegmentsExtractor::KeyCategorySet categories,
      std::unique_ptr<SstQueryFilterBuilder> wrapped)
      : categories_(categories), wrapped_(std::move(wrapped)) {}

  void Add(const Slice& key, const KeySegmentsExtractor::Result& extracted,
           const Slice* prev_key,
           const KeySegmentsExtractor::Result* prev_extracted) override {
    if (!categories_.Contains(extracted.category)) {
      // Category not in scope of the contituent filters
      return;
    }
    wrapped_->Add(key, extracted, prev_key, prev_extracted);
  }

  Status GetStatus() const override { return wrapped_->GetStatus(); }

  size_t GetEncodedLength() const override {
    size_t wrapped_length = wrapped_->GetEncodedLength();
    if (wrapped_length == 0) {
      // Use empty filter
      // FIXME: needs unit test
      return 0;
    } else {
      // For now in the code, wraps only 1 filter, but schema supports multiple
      return 1 + VarintLength(CategorySetToUint(categories_)) + 1 +
             wrapped_length;
    }
  }

  void Finish(std::string& append_to) override {
    size_t encoded_length = GetEncodedLength();
    if (encoded_length == 0) {
      // Nothing to do
      return;
    }
    size_t old_append_to_size = append_to.size();
    append_to.reserve(old_append_to_size + encoded_length);
    append_to.push_back(kCategoryScopeFilterWrapper);

    PutVarint64(&append_to, CategorySetToUint(categories_));

    // Wrapping just 1 filter for now
    PutVarint64(&append_to, 1);
    wrapped_->Finish(append_to);
  }

 private:
  KeySegmentsExtractor::KeyCategorySet categories_;
  std::unique_ptr<SstQueryFilterBuilder> wrapped_;
};

class BytewiseMinMaxSstQueryFilterConfig : public SstQueryFilterConfigImpl {
 public:
  using SstQueryFilterConfigImpl::SstQueryFilterConfigImpl;

  std::unique_ptr<SstQueryFilterBuilder> NewBuilder(
      bool sanity_checks) const override {
    auto b = std::make_unique<MyBuilder>(*this, sanity_checks);
    if (categories_ != KeySegmentsExtractor::KeyCategorySet::All()) {
      return std::make_unique<CategoryScopeFilterWrapperBuilder>(categories_,
                                                                 std::move(b));
    } else {
      return b;
    }
  }

  static bool RangeMayMatch(
      const Slice& filter, const Slice& lower_bound_incl,
      const KeySegmentsExtractor::Result& lower_bound_extracted,
      const Slice& upper_bound_excl,
      const KeySegmentsExtractor::Result& upper_bound_extracted) {
    assert(!filter.empty() && filter[0] == kBytewiseMinMaxFilter);
    if (filter.size() <= 4) {
      // Missing some data
      return true;
    }
    bool empty_included = (filter[1] & kEmptySeenFlag) != 0;
    const char* p = filter.data() + 2;
    const char* limit = filter.data() + filter.size();

    FilterInput in;
    p = DeserializeFilterInput(p, limit, &in);
    if (p == nullptr) {
      // Corrupt or unsupported
      return true;
    }

    uint32_t smallest_size;
    p = GetVarint32Ptr(p, limit, &smallest_size);
    if (p == nullptr || static_cast<size_t>(limit - p) <= smallest_size) {
      // Corrupt
      return true;
    }
    Slice smallest = Slice(p, smallest_size);
    p += smallest_size;

    size_t largest_size = static_cast<size_t>(limit - p);
    Slice largest = Slice(p, largest_size);

    Slice lower_bound_input, lower_bound_leadup;
    Slice upper_bound_input, upper_bound_leadup;
    GetFilterInput(in, lower_bound_incl, lower_bound_extracted,
                   &lower_bound_input, &lower_bound_leadup);
    GetFilterInput(in, upper_bound_excl, upper_bound_extracted,
                   &upper_bound_input, &upper_bound_leadup);

    if (lower_bound_leadup.compare(upper_bound_leadup) != 0) {
      // Unable to filter range when bounds have different lead-up to key
      // segment
      return true;
    }

    if (empty_included && lower_bound_input.empty()) {
      // May match on 0-length segment
      return true;
    }
    // TODO: potentially fix upper bound to actually be exclusive

    // May match if both the upper bound and lower bound indicate there could
    // be overlap
    return upper_bound_input.compare(smallest) >= 0 &&
           lower_bound_input.compare(largest) <= 0;
  }

 protected:
  struct MyBuilder : public SstQueryFilterBuilder {
    MyBuilder(const BytewiseMinMaxSstQueryFilterConfig& _parent,
              bool _sanity_checks)
        : parent(_parent), sanity_checks(_sanity_checks) {}

    void Add(const Slice& key, const KeySegmentsExtractor::Result& extracted,
             const Slice* prev_key,
             const KeySegmentsExtractor::Result* prev_extracted) override {
      Slice input, leadup;
      GetFilterInput(parent.input_, key, extracted, &input, &leadup);

      if (sanity_checks && prev_key && prev_extracted) {
        // Opportunistic checking of segment ordering invariant
        Slice prev_input, prev_leadup;
        GetFilterInput(parent.input_, *prev_key, *prev_extracted, &prev_input,
                       &prev_leadup);

        int compare = prev_leadup.compare(leadup);
        if (compare > 0) {
          status = Status::Corruption(
              "Ordering invariant violated from 0x" +
              prev_key->ToString(/*hex=*/true) + " with prefix 0x" +
              prev_leadup.ToString(/*hex=*/true) + " to 0x" +
              key.ToString(/*hex=*/true) + " with prefix 0x" +
              leadup.ToString(/*hex=*/true));
          return;
        } else if (compare == 0) {
          // On the same prefix leading up to the segment, the segments must
          // not be out of order.
          compare = prev_input.compare(input);
          if (compare > 0) {
            status = Status::Corruption(
                "Ordering invariant violated from 0x" +
                prev_key->ToString(/*hex=*/true) + " with segment 0x" +
                prev_input.ToString(/*hex=*/true) + " to 0x" +
                key.ToString(/*hex=*/true) + " with segment 0x" +
                input.ToString(/*hex=*/true));
            return;
          }
        }
      }

      // Now actually update state for the filter inputs
      // TODO: shorten largest and smallest if appropriate
      if (input.empty()) {
        empty_seen = true;
      } else if (largest.empty()) {
        // Step for first non-empty input
        smallest = largest = input.ToString();
      } else if (input.compare(largest) > 0) {
        largest = input.ToString();
      } else if (input.compare(smallest) < 0) {
        smallest = input.ToString();
      }
    }

    Status GetStatus() const override { return status; }

    size_t GetEncodedLength() const override {
      if (largest.empty()) {
        // Not an interesting filter -> 0 to indicate no filter
        // FIXME: needs unit test
        return 0;
      }
      return 2 + GetFilterInputSerializedLength(parent.input_) +
             VarintLength(smallest.size()) + smallest.size() + largest.size();
    }

    void Finish(std::string& append_to) override {
      assert(status.ok());
      size_t encoded_length = GetEncodedLength();
      if (encoded_length == 0) {
        // Nothing to do
        return;
      }
      size_t old_append_to_size = append_to.size();
      append_to.reserve(old_append_to_size + encoded_length);
      append_to.push_back(kBytewiseMinMaxFilter);

      append_to.push_back(empty_seen ? kEmptySeenFlag : 0);

      SerializeFilterInput(&append_to, parent.input_);

      PutVarint32(&append_to, static_cast<uint32_t>(smallest.size()));
      append_to.append(smallest);
      // The end of `largest` is given by the end of the filter
      append_to.append(largest);
      assert(append_to.size() == old_append_to_size + encoded_length);
    }

    const BytewiseMinMaxSstQueryFilterConfig& parent;
    const bool sanity_checks;
    // Smallest and largest segment seen, excluding the empty segment which
    // is tracked separately
    std::string smallest;
    std::string largest;
    bool empty_seen = false;

    // Only for sanity checks
    Status status;
  };

 private:
  static constexpr char kEmptySeenFlag = 0x1;
};

const SstQueryFilterConfigs kEmptySstQueryFilterConfigs{};

class SstQueryFilterConfigsManagerImpl : public SstQueryFilterConfigsManager {
 public:
  using ConfigVersionMap = std::map<FilteringVersion, SstQueryFilterConfigs>;

  Status Populate(const Data& data) {
    if (data.empty()) {
      return Status::OK();
    }
    // Populate only once
    assert(min_ver_ == 0 && max_ver_ == 0);
    min_ver_ = max_ver_ = data.begin()->first;

    FilteringVersion prev_ver = 0;
    for (const auto& ver_info : data) {
      if (ver_info.first == 0) {
        return Status::InvalidArgument(
            "Filtering version 0 is reserved for empty configuration and may "
            "not be overridden");
      }
      if (ver_info.first <= prev_ver) {
        return Status::InvalidArgument(
            "Filtering versions must increase without repeating: " +
            std::to_string(prev_ver) + " -> " + std::to_string(ver_info.first));
      }
      min_ver_ = std::min(min_ver_, ver_info.first);
      max_ver_ = std::max(max_ver_, ver_info.first);
      UnorderedSet<std::string> names_seen_this_ver;
      for (const auto& config : ver_info.second) {
        if (!names_seen_this_ver.insert(config.first).second) {
          return Status::InvalidArgument(
              "Duplicate name in filtering version " +
              std::to_string(ver_info.first) + ": " + config.first);
        }
        auto& ver_map = name_map_[config.first];
        ver_map[ver_info.first] = config.second;
        if (config.second.extractor) {
          extractor_map_[config.second.extractor->Name()] =
              config.second.extractor;
        }
      }
      prev_ver = ver_info.first;
    }
    return Status::OK();
  }

  struct MyCollector : public TablePropertiesCollector {
    // Keeps a reference to `configs` which should be kept alive by
    // SstQueryFilterConfigsManagerImpl, which should be kept alive by
    // any factories
    // TODO: sanity_checks option
    explicit MyCollector(const SstQueryFilterConfigs& configs,
                         const SstQueryFilterConfigsManagerImpl& _parent)
        : parent(_parent),
          extractor(configs.extractor.get()),
          extractor_version(configs.extractor_version),
          sanity_checks(true) {
      for (const auto& c : configs.filters) {
        builders.push_back(
            static_cast<SstQueryFilterConfigImpl&>(*c).NewBuilder(
                sanity_checks));
      }
    }

    Status AddUserKey(const Slice& key, const Slice& /*value*/,
                      EntryType /*type*/, SequenceNumber /*seq*/,
                      uint64_t /*file_size*/) override {
      KeySegmentsExtractor::Result extracted;
      if (extractor) {
        Status s = extractor->Extract(key, KeySegmentsExtractor::kFullUserKey,
                                      extractor_version, &extracted);
        if (!s.ok()) {
          return s;
        }
        bool new_category = categories_seen.Add(extracted.category);
        if (sanity_checks) {
          // Opportunistic checking of category ordering invariant
          if (!first_key) {
            if (prev_extracted.category != extracted.category &&
                !new_category) {
              return Status::Corruption(
                  "Category ordering invariant violated from key 0x" +
                  Slice(prev_key).ToString(/*hex=*/true) + " to 0x" +
                  key.ToString(/*hex=*/true));
            }
          }
        }
      }
      for (const auto& b : builders) {
        if (first_key) {
          b->Add(key, extracted, nullptr, nullptr);
        } else {
          Slice prev_key_slice = Slice(prev_key);
          b->Add(key, extracted, &prev_key_slice, &prev_extracted);
        }
      }
      prev_key.assign(key.data(), key.size());
      prev_extracted = extracted;
      first_key = false;
      return Status::OK();
    }
    Status Finish(UserCollectedProperties* properties) override {
      assert(properties != nullptr);

      size_t total_size = 1;
      // TODO: use autovector
      std::vector<std::pair<SstQueryFilterBuilder&, size_t>> filters_to_finish;
      // Need to determine number of filters before serializing them. Might
      // as well determine full length also.
      for (const auto& b : builders) {
        Status s = b->GetStatus();
        if (s.ok()) {
          size_t len = b->GetEncodedLength();
          if (len > 0) {
            total_size += VarintLength(len) + len;
            filters_to_finish.emplace_back(*b, len);
          }
        } else {
          // FIXME: no way to report partial failure without getting
          // remaining filters thrown out
        }
      }
      total_size += VarintLength(filters_to_finish.size());
      if (filters_to_finish.empty()) {
        // No filters to add
        return Status::OK();
      }
      // Length of the last filter is omitted
      total_size -= VarintLength(filters_to_finish.back().second);

      // Need to determine size of
      // kExtrAndVerAndCatFilterWrapper if used
      size_t name_len = 0;
      if (extractor) {
        name_len = strlen(extractor->Name());
        // identifier byte
        total_size += 1;
        // fields of the wrapper
        total_size += VarintLength(name_len) + name_len +
                      VarintLength(extractor_version) +
                      VarintLength(CategorySetToUint(categories_seen));
        // outer layer will have just 1 filter in its count (added here)
        // and this filter wrapper will have filters_to_finish.size()
        // (added above).
        total_size += 1;
      }

      std::string filters;
      filters.reserve(total_size);

      filters.push_back(kSchemaVersion);

      if (extractor) {
        // Wrap everything in a kExtrAndVerAndCatFilterWrapper
        // TODO in future: put whole key filters outside of this wrapper.
        // Also TODO in future: order the filters starting with broadest
        // applicability.

        // Just one top-level filter (wrapper). Because it's last, we don't
        // need to encode its length.
        PutVarint64(&filters, 1);
        // The filter(s) wrapper itself
        filters.push_back(kExtrAndVerAndCatFilterWrapper);
        PutVarint64(&filters, name_len);
        filters.append(extractor->Name(), name_len);
        PutVarint64(&filters, extractor_version);
        PutVarint64(&filters, CategorySetToUint(categories_seen));
      }

      PutVarint64(&filters, filters_to_finish.size());

      for (const auto& e : filters_to_finish) {
        // Encode filter length, except last filter
        if (&e != &filters_to_finish.back()) {
          PutVarint64(&filters, e.second);
        }
        // Encode filter
        e.first.Finish(filters);
      }
      if (filters.size() != total_size) {
        assert(false);
        return Status::Corruption(
            "Internal inconsistency building SST query filters");
      }

      (*properties)[kTablePropertyName] = std::move(filters);
      return Status::OK();
    }
    UserCollectedProperties GetReadableProperties() const override {
      // TODO?
      return {};
    }
    const char* Name() const override {
      // placeholder
      return "SstQueryFilterConfigsImpl::MyCollector";
    }

    const SstQueryFilterConfigsManagerImpl& parent;
    const KeySegmentsExtractor* const extractor;
    const uint32_t extractor_version;
    const bool sanity_checks;
    std::vector<std::shared_ptr<SstQueryFilterBuilder>> builders;
    bool first_key = true;
    std::string prev_key;
    KeySegmentsExtractor::Result prev_extracted;
    KeySegmentsExtractor::KeyCategorySet categories_seen;
  };

  struct RangeQueryFilterReader {
    Slice lower_bound_incl;
    Slice upper_bound_excl;
    const KeySegmentsExtractor* extractor;
    const UnorderedMap<std::string,
                       std::shared_ptr<const KeySegmentsExtractor>>&
        extractor_map;

    struct State {
      KeySegmentsExtractor::Result lb_extracted;
      KeySegmentsExtractor::Result ub_extracted;
    };

    bool MayMatch_CategoryScopeFilterWrapper(Slice wrapper,
                                             State& state) const {
      assert(!wrapper.empty() && wrapper[0] == kCategoryScopeFilterWrapper);

      // Regardless of the filter values (which we assume is not all
      // categories; that should skip the wrapper), we need upper bound and
      // lower bound to be in the same category to do any range filtering.
      // (There could be another category in range between the bounds.)
      if (state.lb_extracted.category != state.ub_extracted.category) {
        // Can't filter between categories
        return true;
      }

      const char* p = wrapper.data() + 1;
      const char* limit = wrapper.data() + wrapper.size();

      uint64_t cats_raw;
      p = GetVarint64Ptr(p, limit, &cats_raw);
      if (p == nullptr) {
        // Missing categories
        return true;
      }
      KeySegmentsExtractor::KeyCategorySet categories =
          UintToCategorySet(cats_raw);

      // Check category against those in scope
      if (!categories.Contains(state.lb_extracted.category)) {
        // Can't filter this category
        return true;
      }

      // Process the wrapped filters
      return MayMatch(Slice(p, limit - p), &state);
    }

    bool MayMatch_ExtrAndVerAndCatFilterWrapper(Slice wrapper) const {
      assert(!wrapper.empty() && wrapper[0] == kExtrAndVerAndCatFilterWrapper);
      if (wrapper.size() <= 4) {
        // Missing some data
        return true;
      }
      const char* p = wrapper.data() + 1;
      const char* limit = wrapper.data() + wrapper.size();
      uint64_t name_len;
      p = GetVarint64Ptr(p, limit, &name_len);
      if (p == nullptr || name_len == 0 ||
          static_cast<size_t>(limit - p) < name_len) {
        // Missing some data
        return true;
      }
      Slice name(p, name_len);
      p += name_len;
      const KeySegmentsExtractor* ex = nullptr;
      if (extractor && name == Slice(extractor->Name())) {
        ex = extractor;
      } else {
        auto it = extractor_map.find(name.ToString());
        if (it != extractor_map.end()) {
          ex = it->second.get();
        } else {
          // Extractor mismatch / not found
          // TODO future: try to get the extractor from the ObjectRegistry
          return true;
        }
      }

      // TODO future: cache extraction with default version
      uint32_t version;
      p = GetVarint32Ptr(p, limit, &version);
      if (p == nullptr) {
        // Missing some data
        return true;
      }

      // Ready to run extractor
      assert(ex);
      State state;
      Status s = ex->Extract(lower_bound_incl,
                             KeySegmentsExtractor::kInclusiveLowerBound,
                             version, &state.lb_extracted);
      if (!s.ok()) {
        // TODO? Report problem
        // No filtering
        return true;
      }
      s = ex->Extract(upper_bound_excl,
                      KeySegmentsExtractor::kExclusiveUpperBound, version,
                      &state.ub_extracted);
      if (!s.ok()) {
        // TODO? Report problem
        // No filtering
        return true;
      }

      uint64_t cats_raw;
      p = GetVarint64Ptr(p, limit, &cats_raw);
      if (p == nullptr) {
        // Missing categories
        return true;
      }
      KeySegmentsExtractor::KeyCategorySet categories =
          UintToCategorySet(cats_raw);

      // Can only filter out based on category if upper and lower bound have
      // the same category. (Each category is contiguous by key order, but we
      // don't know the order between categories.)
      if (state.lb_extracted.category == state.ub_extracted.category &&
          !categories.Contains(state.lb_extracted.category)) {
        // Filtered out
        return false;
      }

      // Process the wrapped filters
      return MayMatch(Slice(p, limit - p), &state);
    }

    bool MayMatch(Slice filters, State* state = nullptr) const {
      const char* p = filters.data();
      const char* limit = p + filters.size();
      uint64_t filter_count;
      p = GetVarint64Ptr(p, limit, &filter_count);
      if (p == nullptr || filter_count == 0) {
        // TODO? Report problem
        // No filtering
        return true;
      }

      for (size_t i = 0; i < filter_count; ++i) {
        uint64_t filter_len;
        if (i + 1 == filter_count) {
          // Last filter
          filter_len = static_cast<uint64_t>(limit - p);
        } else {
          p = GetVarint64Ptr(p, limit, &filter_len);
          if (p == nullptr || filter_len == 0 ||
              static_cast<size_t>(limit - p) < filter_len) {
            // TODO? Report problem
            // No filtering
            return true;
          }
        }
        Slice filter = Slice(p, filter_len);
        p += filter_len;
        bool may_match = true;
        char type = filter[0];
        switch (type) {
          case kExtrAndVerAndCatFilterWrapper:
            may_match = MayMatch_ExtrAndVerAndCatFilterWrapper(filter);
            break;
          case kCategoryScopeFilterWrapper:
            if (state == nullptr) {
              // TODO? Report problem
              // No filtering
              return true;
            }
            may_match = MayMatch_CategoryScopeFilterWrapper(filter, *state);
            break;
          case kBytewiseMinMaxFilter:
            if (state == nullptr) {
              // TODO? Report problem
              // No filtering
              return true;
            }
            may_match = BytewiseMinMaxSstQueryFilterConfig::RangeMayMatch(
                filter, lower_bound_incl, state->lb_extracted, upper_bound_excl,
                state->ub_extracted);
            break;
          default:
            // TODO? Report problem
            {}
            // Unknown filter type
        }
        if (!may_match) {
          // Successfully filtered
          return false;
        }
      }

      // Wasn't filtered
      return true;
    }
  };

  struct MyFactory : public TblPropCollFactory {
    explicit MyFactory(
        std::shared_ptr<const SstQueryFilterConfigsManagerImpl> _parent,
        const std::string& _configs_name)
        : parent(std::move(_parent)),
          ver_map(parent->GetVerMap(_configs_name)),
          configs_name(std::move(_configs_name)) {}

    TablePropertiesCollector* CreateTablePropertiesCollector(
        TablePropertiesCollectorFactory::Context /*context*/) override {
      return new MyCollector(GetConfigs(), *parent);
    }
    const char* Name() const override {
      // placeholder
      return "SstQueryFilterConfigsManagerImpl::MyFactory";
    }

    Status SetFilteringVersion(FilteringVersion ver) override {
      // FIXME: check version
      version.StoreRelaxed(ver);
      return Status::OK();
    }
    FilteringVersion GetFilteringVersion() const override {
      return version.LoadRelaxed();
    }
    const std::string& GetConfigsName() const override { return configs_name; }
    const SstQueryFilterConfigs& GetConfigs() const override {
      FilteringVersion ver = version.LoadRelaxed();
      if (ver == 0) {
        // Special case
        return kEmptySstQueryFilterConfigs;
      }
      auto it = ver_map.lower_bound(ver);
      if (it == ver_map.end()) {
        return kEmptySstQueryFilterConfigs;
      }
      return it->second;
    }

    // The buffers pointed to by the Slices must live as long as any read
    // operations using this table filter function.
    std::function<bool(const TableProperties&)> GetTableFilterForRangeQuery(
        Slice lower_bound_incl, Slice upper_bound_excl) const override {
      // TODO: cache extractor results between SST files, assuming most will
      // use the same version
      return
          [rqf = RangeQueryFilterReader{
               lower_bound_incl, upper_bound_excl, GetConfigs().extractor.get(),
               parent->extractor_map_}](const TableProperties& props) -> bool {
            auto it = props.user_collected_properties.find(kTablePropertyName);
            if (it == props.user_collected_properties.end()) {
              // No filtering
              return true;
            }
            auto& filters = it->second;
            // Parse the serialized filters string
            if (filters.size() < 2 || filters[0] != kSchemaVersion) {
              // TODO? Report problem
              // No filtering
              return true;
            }
            return rqf.MayMatch(Slice(filters.data() + 1, filters.size() - 1));
          };
    }

    const std::shared_ptr<const SstQueryFilterConfigsManagerImpl> parent;
    const ConfigVersionMap& ver_map;
    const std::string configs_name;
    RelaxedAtomic<FilteringVersion> version;
  };

  Status MakeSharedTblPropCollFactory(std::shared_ptr<TblPropCollFactory>* out,
                                      const std::string& configs_name,
                                      FilteringVersion ver) const override {
    auto obj = std::make_shared<MyFactory>(*this, configs_name);
    Status s = obj->SetFilteringVersion(ver);
    if (s.ok()) {
      *out = std::move(obj);
    }
    return s;
  }

  const ConfigVersionMap& GetVerMap(const std::string& configs_name) const {
    static const ConfigVersionMap kEmptyMap;
    auto it = name_map_.find(configs_name);
    if (it == name_map_.end()) {
      return kEmptyMap;
    }
    return it->second;
  }

 private:
  static const std::string kTablePropertyName;
  static constexpr char kSchemaVersion = 1;

 private:
  UnorderedMap<std::string, ConfigVersionMap> name_map_;
  UnorderedMap<std::string, std::shared_ptr<const KeySegmentsExtractor>>
      extractor_map_;
  FilteringVersion min_ver_ = 0;
  FilteringVersion max_ver_ = 0;
};

// SstQueryFilterConfigs
const std::string SstQueryFilterConfigsManagerImpl::kTablePropertyName =
    "rocksdb.sqfc";
}  // namespace

std::shared_ptr<SstQueryFilterConfig> MakeSharedBytewiseMinMaxSQFC(
    FilterInput input, KeySegmentsExtractor::KeyCategorySet categories) {
  return std::make_shared<BytewiseMinMaxSstQueryFilterConfig>(input,
                                                              categories);
}

Status SstQueryFilterConfigsManager::MakeShared(
    std::shared_ptr<SstQueryFilterConfigsManager>* out, const Data& data) {
  auto obj = std::make_shared<SstQueryFilterConfigsManagerImpl>();
  Status s = obj->Populate(data);
  if (s.ok()) {
    *out = std::move(obj);
  }
  return s;
}

}  // namespace experimental
}  // namespace ROCKSDB_NAMESPACE
