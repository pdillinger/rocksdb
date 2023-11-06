//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "rocksdb/experimental.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "db/db_impl/db_impl.h"
#include "db/version_util.h"
#include "logging/logging.h"

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
Slice GetSegmentsFromKey(size_t from_idx, size_t to_idx, const Slice& key,
                         const KeySegmentsExtractor::Result& extracted) {
  assert(from_idx <= to_idx);
  size_t count = extracted.segment_ends.size();
  if (count <= from_idx) {
    return Slice();
  }
  assert(count > 0);
  size_t start = from_idx > 0 ? extracted.segment_ends[from_idx - 1] : 0;
  size_t end = extracted.segment_ends[std::min(to_idx, count - 1)];
  return Slice(key.data() + start, end - start);
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

class SstQueryFilterConfigImpl {
 public:
  virtual ~SstQueryFilterConfigImpl() {}

  virtual std::unique_ptr<SstQueryFilterBuilder> NewBuilder(
      bool sanity_checks) const = 0;
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
  explicit BytewiseMinMaxSstQueryFilterConfig(
      uint32_t segment_index_from, uint32_t segment_index_to,
      KeySegmentsExtractor::KeyCategorySet categories)
      : segment_index_from_(segment_index_from),
        segment_index_to_(segment_index_to),
        categories_(categories) {}

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
    uint32_t segment_index_from = static_cast<uint32_t>(filter[2]);
    uint32_t segment_index_to = static_cast<uint32_t>(filter[3]);
    const char* p = filter.data() + 4;
    const char* limit = filter.data() + filter.size();

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

    if (segment_index_from > 0) {
      Slice lower_bound_prefix = GetSegmentsFromKey(
          0, segment_index_from - 1, lower_bound_incl, lower_bound_extracted);
      Slice upper_bound_prefix = GetSegmentsFromKey(
          0, segment_index_from - 1, upper_bound_excl, upper_bound_extracted);
      if (lower_bound_prefix.compare(upper_bound_prefix) != 0) {
        // Unable to filter when bounds cross prefix leading up to segment
        return true;
      }
    }
    Slice lower_bound_segment =
        GetSegmentsFromKey(segment_index_from, segment_index_to,
                           lower_bound_incl, lower_bound_extracted);
    if (empty_included && lower_bound_segment.empty()) {
      // May match on 0-length segment
      return true;
    }
    Slice upper_bound_segment =
        GetSegmentsFromKey(segment_index_from, segment_index_to,
                           upper_bound_excl, upper_bound_extracted);

    // TODO: potentially fix upper bound to actually be exclusive

    // May match if both the upper bound and lower bound indicate there could
    // be overlap
    return upper_bound_segment.compare(smallest) >= 0 &&
           lower_bound_segment.compare(largest) <= 0;
  }

 protected:
  struct MyBuilder : public SstQueryFilterBuilder {
    MyBuilder(const BytewiseMinMaxSstQueryFilterConfig& _parent,
              bool _sanity_checks)
        : parent(_parent), sanity_checks(_sanity_checks) {}

    void Add(const Slice& key, const KeySegmentsExtractor::Result& extracted,
             const Slice* prev_key,
             const KeySegmentsExtractor::Result* prev_extracted) override {
      Slice segment = GetSegmentsFromKey(
          parent.segment_index_from_, parent.segment_index_to_, key, extracted);

      if (sanity_checks && prev_key && prev_extracted) {
        // Opportunistic checking of segment ordering invariant
        int compare = 0;
        if (parent.segment_index_from_ > 0) {
          Slice prev_prefix = GetSegmentsFromKey(
              0, parent.segment_index_from_ - 1, *prev_key, *prev_extracted);
          Slice prefix = GetSegmentsFromKey(0, parent.segment_index_from_ - 1,
                                            key, extracted);
          compare = prev_prefix.compare(prefix);
          if (compare > 0) {
            status = Status::Corruption(
                "Ordering invariant violated from 0x" +
                prev_key->ToString(/*hex=*/true) + " with prefix 0x" +
                prev_prefix.ToString(/*hex=*/true) + " to 0x" +
                key.ToString(/*hex=*/true) + " with prefix 0x" +
                prefix.ToString(/*hex=*/true));
            return;
          }
        }
        if (compare == 0) {
          // On the same prefix leading up to the segment, the segments must
          // not be out of order.
          Slice prev_segment = GetSegmentsFromKey(parent.segment_index_from_,
                                                  parent.segment_index_to_,
                                                  *prev_key, *prev_extracted);
          compare = prev_segment.compare(segment);
          if (compare > 0) {
            status = Status::Corruption(
                "Ordering invariant violated from 0x" +
                prev_key->ToString(/*hex=*/true) + " with segment 0x" +
                prev_segment.ToString(/*hex=*/true) + " to 0x" +
                key.ToString(/*hex=*/true) + " with segment 0x" +
                segment.ToString(/*hex=*/true));
            return;
          }
        }
      }

      // Now actually update state for the key segments
      // TODO: shorten largest and smallest if appropriate
      if (segment.empty()) {
        empty_seen = true;
      } else if (largest.empty()) {
        // First step for non-empty segment
        smallest = largest = segment.ToString();
      } else if (segment.compare(largest) > 0) {
        largest = segment.ToString();
      } else if (segment.compare(smallest) < 0) {
        smallest = segment.ToString();
      }
    }

    Status GetStatus() const override { return status; }

    size_t GetEncodedLength() const override {
      if (largest.empty()) {
        // Not an interesting filter -> 0 to indicate no filter
        // FIXME: needs unit test
        return 0;
      }
      return 4 + VarintLength(smallest.size()) + smallest.size() +
             largest.size();
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

      // FIXME: check bounds
      append_to.push_back(static_cast<char>(parent.segment_index_from_));
      append_to.push_back(static_cast<char>(parent.segment_index_to_));

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
  uint32_t segment_index_from_;
  uint32_t segment_index_to_;
  KeySegmentsExtractor::KeyCategorySet categories_;

  static constexpr char kEmptySeenFlag = 0x1;
};

class SstQueryFilterConfigsImpl
    : public SstQueryFilterConfigs,
      public std::enable_shared_from_this<SstQueryFilterConfigsImpl> {
 public:
  Self& SetExtractorAndVersion(std::shared_ptr<KeySegmentsExtractor> extractor,
                               uint32_t version) override {
    extractor_ = std::move(extractor);
    version_ = version;
    return *this;
  }

  Self& SetSanityChecks(bool enabled) override {
    sanity_checks_ = enabled;
    return *this;
  }

  Self& AddMinMax(uint32_t from_segment_index, uint32_t to_segment_index,
                  KeySegmentsExtractor::KeyCategorySet categories) override {
    configs_.push_back(std::make_shared<BytewiseMinMaxSstQueryFilterConfig>(
        from_segment_index, to_segment_index, categories));
    return *this;
  }
  Self& AddApproximateSet(
      uint32_t from_segment_index, uint32_t to_segment_index,
      KeySegmentsExtractor::KeyCategorySet categories) override {
    // TODO
    (void)from_segment_index;
    (void)to_segment_index;
    (void)categories;
    return *this;
  }

  struct MyCollector : public TablePropertiesCollector {
    explicit MyCollector(
        std::shared_ptr<const SstQueryFilterConfigsImpl> _parent)
        : parent(std::move(_parent)) {
      for (const auto& c : parent->configs_) {
        builders.push_back(c->NewBuilder(parent->sanity_checks_));
      }
    }

    Status AddUserKey(const Slice& key, const Slice& /*value*/,
                      EntryType /*type*/, SequenceNumber /*seq*/,
                      uint64_t /*file_size*/) override {
      KeySegmentsExtractor::Result extracted;
      if (parent->extractor_) {
        Status s =
            parent->extractor_->Extract(key, KeySegmentsExtractor::kFullUserKey,
                                        parent->version_, &extracted);
        if (!s.ok()) {
          return s;
        }
        bool new_category = categories_seen.Add(extracted.category);
        if (parent->sanity_checks_) {
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
      if (parent->extractor_) {
        name_len = strlen(parent->extractor_->Name());
        // identifier byte
        total_size += 1;
        // fields of the wrapper
        total_size += VarintLength(name_len) + name_len +
                      VarintLength(parent->version_) +
                      VarintLength(CategorySetToUint(categories_seen));
        // outer layer will have just 1 filter in its count (added here)
        // and this filter wrapper will have filters_to_finish.size()
        // (added above).
        total_size += 1;
      }

      std::string filters;
      filters.reserve(total_size);

      filters.push_back(kSchemaVersion);

      if (parent->extractor_) {
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
        filters.append(parent->extractor_->Name(), name_len);
        PutVarint64(&filters, parent->version_);
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

      (*properties)[SstQueryFilterConfigsImpl::kTablePropertyName] =
          std::move(filters);
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

    std::shared_ptr<const SstQueryFilterConfigsImpl> parent;
    std::vector<std::shared_ptr<SstQueryFilterBuilder>> builders;
    bool first_key = true;
    std::string prev_key;
    KeySegmentsExtractor::Result prev_extracted;
    KeySegmentsExtractor::KeyCategorySet categories_seen;
  };

  struct RangeQueryFilterReader {
    Slice lower_bound_incl;
    Slice upper_bound_excl;
    std::shared_ptr<KeySegmentsExtractor> extractor;

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
      if (!extractor || name != Slice(extractor->Name())) {
        // Extractor mismatch
        // TODO future: try to get the extractor from the ObjectRegistry
        return true;
      }
      // TODO future: cache extraction with default version
      uint32_t version;
      p = GetVarint32Ptr(p, limit, &version);
      if (p == nullptr) {
        // Missing some data
        return true;
      }

      // Ready to run extractor
      assert(extractor);
      State state;
      Status s = extractor->Extract(lower_bound_incl,
                                    KeySegmentsExtractor::kInclusiveLowerBound,
                                    version, &state.lb_extracted);
      if (!s.ok()) {
        // TODO? Report problem
        // No filtering
        return true;
      }
      s = extractor->Extract(upper_bound_excl,
                             KeySegmentsExtractor::kExclusiveUpperBound,
                             version, &state.ub_extracted);
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

  struct MyFactory : public TablePropertiesCollectorFactory {
    explicit MyFactory(std::shared_ptr<const SstQueryFilterConfigsImpl> _parent)
        : parent(std::move(_parent)) {}
    TablePropertiesCollector* CreateTablePropertiesCollector(
        TablePropertiesCollectorFactory::Context /*context*/) override {
      return new MyCollector(parent);
    }
    const char* Name() const override {
      // placeholder
      return "SstQueryFilterConfigsImpl::MyFactory";
    }
    std::shared_ptr<const SstQueryFilterConfigsImpl> parent;
  };

  std::shared_ptr<TablePropertiesCollectorFactory> GetTblPropCollFactory()
      const override {
    return std::make_shared<MyFactory>(this->shared_from_this());
  }
  std::function<bool(const TableProperties&)> GetTableFilterForRangeQuery(
      Slice lower_bound_incl, Slice upper_bound_excl) const override {
    // TODO: cache extractor results between SST files, assuming most will
    // use the same version
    return [rqf = RangeQueryFilterReader{lower_bound_incl, upper_bound_excl,
                                         extractor_}](
               const TableProperties& props) -> bool {
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

 private:
  static const std::string kTablePropertyName;
  static constexpr char kSchemaVersion = 1;

 private:
  std::shared_ptr<KeySegmentsExtractor> extractor_;
  uint32_t version_ = 0;
  std::vector<std::shared_ptr<SstQueryFilterConfigImpl>> configs_;
  bool sanity_checks_ = false;
};

// SstQueryFilterConfigs
const std::string SstQueryFilterConfigsImpl::kTablePropertyName =
    "rocksdb.sqfc";

}  // namespace

std::shared_ptr<SstQueryFilterConfigs> SstQueryFilterConfigs::MakeShared() {
  return std::make_shared<SstQueryFilterConfigsImpl>();
}

}  // namespace experimental
}  // namespace ROCKSDB_NAMESPACE
