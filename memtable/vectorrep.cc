//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
#include <algorithm>
#include <cstring>
#include <memory>
#include <set>
#include <type_traits>
#include <unordered_set>
#include <utility>

#include "db/memtable.h"
#include "memory/arena.h"
#include "memtable/stl_wrappers.h"
#include "port/port.h"
#include "rocksdb/memtablerep.h"
#include "rocksdb/utilities/options_type.h"
#include "util/mutexlock.h"
#include "util/random.h"

namespace ROCKSDB_NAMESPACE {
namespace {

class VectorRep : public MemTableRep {
 public:
  VectorRep(const KeyComparator& compare, Allocator* allocator, size_t count);

  // Insert key into the collection. (The caller will pack key and value into a
  // single buffer and pass that in as the parameter to Insert)
  // REQUIRES: nothing that compares equal to key is currently in the
  // collection.
  void Insert(KeyHandle handle) override;

  void InsertConcurrently(KeyHandle handle) override;

  // Returns true iff an entry that compares equal to key is in the collection.
  bool Contains(const char* key) const override;

  void MarkReadOnly() override;

  size_t ApproximateMemoryUsage() override;

  void Get(const LookupKey& k, void* callback_args,
           bool (*callback_func)(void* arg, const char* entry)) override;

  void BatchPostProcess() override;

  ~VectorRep() override = default;

  class Iterator : public MemTableRep::Iterator {
    class VectorRep* vrep_;
    std::shared_ptr<std::vector<const char*>> bucket_;
    std::vector<const char*>::const_iterator mutable cit_;
    const KeyComparator& compare_;
    std::string tmp_;  // For passing to EncodeKey
    bool mutable sorted_;
    void DoSort() const;

   public:
    explicit Iterator(class VectorRep* vrep,
                      std::shared_ptr<std::vector<const char*>> bucket,
                      const KeyComparator& compare);

    // Initialize an iterator over the specified collection.
    // The returned iterator is not valid.
    // explicit Iterator(const MemTableRep* collection);
    ~Iterator() override = default;

    // Returns true iff the iterator is positioned at a valid node.
    bool Valid() const override;

    // Returns the key at the current position.
    // REQUIRES: Valid()
    const char* key() const override;

    // Advances to the next position.
    // REQUIRES: Valid()
    void Next() override;

    // Advances to the previous position.
    // REQUIRES: Valid()
    void Prev() override;

    // Advance to the first entry with a key >= target
    void Seek(const Slice& user_key, const char* memtable_key) override;

    // Seek and do some memory validation
    Status SeekAndValidate(const Slice& internal_key, const char* memtable_key,
                           bool allow_data_in_errors,
                           bool detect_key_out_of_order,
                           const std::function<Status(const char*, bool)>&
                               key_validation_callback) override;

    // Advance to the first entry with a key <= target
    void SeekForPrev(const Slice& user_key, const char* memtable_key) override;

    // Position at the first entry in collection.
    // Final state of iterator is Valid() iff collection is not empty.
    void SeekToFirst() override;

    // Position at the last entry in collection.
    // Final state of iterator is Valid() iff collection is not empty.
    void SeekToLast() override;
  };

  // Return an iterator over the keys in this representation.
  MemTableRep::Iterator* GetIterator(Arena* arena) override;

 private:
  friend class Iterator;
  ALIGN_AS(CACHE_LINE_SIZE) RelaxedAtomic<size_t> bucket_size_;
  using Bucket = std::vector<const char*>;
  std::shared_ptr<Bucket> bucket_;
  mutable port::RWMutex rwlock_;
  bool immutable_;
  bool sorted_;
  const KeyComparator& compare_;
  // Thread-local vector to buffer concurrent writes.
  using TlBucket = std::vector<const char*>;
  ThreadLocalPtr tl_writes_;

  static void DeleteTlBucket(void* ptr) {
    auto* v = static_cast<TlBucket*>(ptr);
    delete v;
  }
};

class VectorGCRep : public MemTableRep {
 public:
  using Bucket = std::vector<const char*>;
  struct Segment;
  struct EntryHeader {
    Segment* segment = nullptr;
    size_t len = 0;
  };

  struct Segment {
    explicit Segment(size_t count = 0) : bucket(new Bucket()) {
      bucket->reserve(count);
      entries.reserve(count);
    }

    char* Allocate(size_t len);

    std::shared_ptr<Bucket> bucket;
    std::vector<std::unique_ptr<char[]>> entries;
    size_t entry_bytes = 0;
    bool accepts_inserts = true;
  };

  struct Snapshot {
    std::vector<std::shared_ptr<Segment>> segments;
    std::shared_ptr<Bucket> bucket;
    bool sorted = false;

    Snapshot() : bucket(new Bucket()) {}
  };

  VectorGCRep(const KeyComparator& compare, Allocator* allocator, size_t count);

  KeyHandle Allocate(const size_t len, char** buf) override;
  void Insert(KeyHandle handle) override;
  void InsertConcurrently(KeyHandle handle) override;
  bool Contains(const char* key) const override;
  void MarkReadOnly() override;
  size_t ApproximateMemoryUsage() override;
  void Get(const LookupKey& k, void* callback_args,
           bool (*callback_func)(void* arg, const char* entry)) override;
  void BatchPostProcess() override;
  void UniqueRandomSample(const uint64_t num_entries,
                          const uint64_t target_sample_size,
                          std::unordered_set<const char*>* entries) override;
  bool SupportsGarbageCollection() const override { return true; }
  std::unique_ptr<GarbageCollectionContext> StartGarbageCollection() override;

  ~VectorGCRep() override = default;

  class Iterator : public MemTableRep::Iterator {
   public:
    explicit Iterator(std::shared_ptr<Snapshot> snapshot,
                      const KeyComparator& compare);

    ~Iterator() override = default;

    bool Valid() const override;
    const char* key() const override;
    void Next() override;
    void Prev() override;
    void Seek(const Slice& user_key, const char* memtable_key) override;
    void SeekForPrev(const Slice& user_key, const char* memtable_key) override;
    void SeekToFirst() override;
    void SeekToLast() override;

   private:
    void DoSort() const;

    std::shared_ptr<Snapshot> snapshot_;
    std::vector<const char*>::const_iterator mutable cit_;
    const KeyComparator& compare_;
    std::string tmp_;
    bool mutable sorted_;
  };

  MemTableRep::Iterator* GetIterator(Arena* arena) override;

 private:
  friend class Iterator;
  class GCContext;

  std::shared_ptr<Snapshot> MakeSnapshot() const;
  const char* ResolveHandleForInsert(KeyHandle handle);
  void RecomputeStatsLocked();
  void RebuildImmutableBucketLocked();
  static EntryHeader* GetEntryHeader(const char* entry);
  static size_t GetEntryLength(const char* entry);
  static GarbageCollectionStats GetStats(const Bucket& bucket);

  ALIGN_AS(CACHE_LINE_SIZE) RelaxedAtomic<size_t> bucket_size_;
  RelaxedAtomic<size_t> entry_bytes_;
  std::vector<std::shared_ptr<Segment>> segments_;
  std::shared_ptr<Segment> active_;
  std::shared_ptr<Bucket> immutable_bucket_;
  mutable port::RWMutex rwlock_;
  bool immutable_;
  bool gc_in_progress_;
  size_t count_;
  const KeyComparator& compare_;

  using TlBucket = std::vector<const char*>;
  ThreadLocalPtr tl_writes_;

  static void DeleteTlBucket(void* ptr) {
    auto* v = static_cast<TlBucket*>(ptr);
    delete v;
  }
};

void VectorRep::Insert(KeyHandle handle) {
  auto* key = static_cast<char*>(handle);
  {
    WriteLock l(&rwlock_);
    assert(!immutable_);
    bucket_->push_back(key);
  }
  bucket_size_.FetchAddRelaxed(1);
}

void VectorRep::InsertConcurrently(KeyHandle handle) {
  auto* v = static_cast<TlBucket*>(tl_writes_.Get());
  if (!v) {
    v = new TlBucket();
    tl_writes_.Reset(v);
  }
  v->push_back(static_cast<char*>(handle));
}

// Returns true iff an entry that compares equal to key is in the collection.
bool VectorRep::Contains(const char* key) const {
  ReadLock l(&rwlock_);
  return std::find(bucket_->begin(), bucket_->end(), key) != bucket_->end();
}

void VectorRep::MarkReadOnly() {
  WriteLock l(&rwlock_);
  immutable_ = true;
}

size_t VectorRep::ApproximateMemoryUsage() {
  return bucket_size_.LoadRelaxed() *
         sizeof(std::remove_reference<decltype(*bucket_)>::type::value_type);
}

void VectorRep::BatchPostProcess() {
  auto* v = static_cast<TlBucket*>(tl_writes_.Get());
  if (v) {
    {
      WriteLock l(&rwlock_);
      assert(!immutable_);
      for (auto& key : *v) {
        bucket_->push_back(key);
      }
    }
    bucket_size_.FetchAddRelaxed(v->size());
    delete v;
    tl_writes_.Reset(nullptr);
  }
}

VectorRep::VectorRep(const KeyComparator& compare, Allocator* allocator,
                     size_t count)
    : MemTableRep(allocator),
      bucket_size_(0),
      bucket_(new Bucket()),
      immutable_(false),
      sorted_(false),
      compare_(compare),
      tl_writes_(DeleteTlBucket) {
  bucket_.get()->reserve(count);
}

VectorRep::Iterator::Iterator(class VectorRep* vrep,
                              std::shared_ptr<std::vector<const char*>> bucket,
                              const KeyComparator& compare)
    : vrep_(vrep),
      bucket_(bucket),
      cit_(bucket_->end()),
      compare_(compare),
      sorted_(false) {}

void VectorRep::Iterator::DoSort() const {
  // vrep is non-null means that we are working on an immutable memtable
  if (!sorted_ && vrep_ != nullptr) {
    WriteLock l(&vrep_->rwlock_);
    if (!vrep_->sorted_) {
      std::sort(bucket_->begin(), bucket_->end(),
                stl_wrappers::Compare(compare_));
      cit_ = bucket_->begin();
      vrep_->sorted_ = true;
    }
    sorted_ = true;
  }
  if (!sorted_) {
    std::sort(bucket_->begin(), bucket_->end(),
              stl_wrappers::Compare(compare_));
    cit_ = bucket_->begin();
    sorted_ = true;
  }
  assert(sorted_);
  assert(vrep_ == nullptr || vrep_->sorted_);
}

// Returns true iff the iterator is positioned at a valid node.
bool VectorRep::Iterator::Valid() const {
  DoSort();
  return cit_ != bucket_->end();
}

// Returns the key at the current position.
// REQUIRES: Valid()
const char* VectorRep::Iterator::key() const {
  assert(sorted_);
  return *cit_;
}

// Advances to the next position.
// REQUIRES: Valid()
void VectorRep::Iterator::Next() {
  assert(sorted_);
  if (cit_ == bucket_->end()) {
    return;
  }
  ++cit_;
}

// Advances to the previous position.
// REQUIRES: Valid()
void VectorRep::Iterator::Prev() {
  assert(sorted_);
  if (cit_ == bucket_->begin()) {
    // If you try to go back from the first element, the iterator should be
    // invalidated. So we set it to past-the-end. This means that you can
    // treat the container circularly.
    cit_ = bucket_->end();
  } else {
    --cit_;
  }
}

// Advance to the first entry with a key >= target
void VectorRep::Iterator::Seek(const Slice& user_key,
                               const char* memtable_key) {
  DoSort();
  // Do binary search to find first value not less than the target
  const char* encoded_key =
      (memtable_key != nullptr) ? memtable_key : EncodeKey(&tmp_, user_key);
  cit_ = std::equal_range(bucket_->begin(), bucket_->end(), encoded_key,
                          [this](const char* a, const char* b) {
                            return compare_(a, b) < 0;
                          })
             .first;
}

Status VectorRep::Iterator::SeekAndValidate(
    const Slice& /* internal_key */, const char* /* memtable_key */,
    bool /* allow_data_in_errors */, bool /* detect_key_out_of_order */,
    const std::function<Status(const char*, bool)>&
    /* key_validation_callback */) {
  if (vrep_) {
    WriteLock l(&vrep_->rwlock_);
    if (bucket_->begin() == bucket_->end()) {
      // Memtable is empty
      return Status::OK();
    } else {
      return Status::NotSupported("SeekAndValidate() not implemented");
    }
  } else {
    return Status::NotSupported("SeekAndValidate() not implemented");
  }
}

// Advance to the first entry with a key <= target
void VectorRep::Iterator::SeekForPrev(const Slice& /*user_key*/,
                                      const char* /*memtable_key*/) {
  assert(false);
}

// Position at the first entry in collection.
// Final state of iterator is Valid() iff collection is not empty.
void VectorRep::Iterator::SeekToFirst() {
  DoSort();
  cit_ = bucket_->begin();
}

// Position at the last entry in collection.
// Final state of iterator is Valid() iff collection is not empty.
void VectorRep::Iterator::SeekToLast() {
  DoSort();
  cit_ = bucket_->end();
  if (bucket_->size() != 0) {
    --cit_;
  }
}

void VectorRep::Get(const LookupKey& k, void* callback_args,
                    bool (*callback_func)(void* arg, const char* entry)) {
  rwlock_.ReadLock();
  VectorRep* vector_rep;
  std::shared_ptr<Bucket> bucket;
  if (immutable_) {
    vector_rep = this;
  } else {
    vector_rep = nullptr;
    bucket.reset(new Bucket(*bucket_));  // make a copy
  }
  VectorRep::Iterator iter(vector_rep, immutable_ ? bucket_ : bucket, compare_);
  rwlock_.ReadUnlock();

  for (iter.Seek(k.user_key(), k.memtable_key().data());
       iter.Valid() && callback_func(callback_args, iter.key()); iter.Next()) {
  }
}

MemTableRep::Iterator* VectorRep::GetIterator(Arena* arena) {
  char* mem = nullptr;
  if (arena != nullptr) {
    mem = arena->AllocateAligned(sizeof(Iterator));
  }
  ReadLock l(&rwlock_);
  // Do not sort here. The sorting would be done the first time
  // a Seek is performed on the iterator.
  if (immutable_) {
    if (arena == nullptr) {
      return new Iterator(this, bucket_, compare_);
    } else {
      return new (mem) Iterator(this, bucket_, compare_);
    }
  } else {
    std::shared_ptr<Bucket> tmp;
    tmp.reset(new Bucket(*bucket_));  // make a copy
    if (arena == nullptr) {
      return new Iterator(nullptr, tmp, compare_);
    } else {
      return new (mem) Iterator(nullptr, tmp, compare_);
    }
  }
}

char* VectorGCRep::Segment::Allocate(size_t len) {
  auto entry = std::make_unique<char[]>(sizeof(EntryHeader) + len);
  auto* header = reinterpret_cast<EntryHeader*>(entry.get());
  header->segment = this;
  header->len = len;
  char* raw = entry.get() + sizeof(EntryHeader);
  entry_bytes += sizeof(EntryHeader) + len;
  entries.push_back(std::move(entry));
  return raw;
}

VectorGCRep::VectorGCRep(const KeyComparator& compare, Allocator* allocator,
                         size_t count)
    : MemTableRep(allocator),
      bucket_size_(0),
      entry_bytes_(0),
      active_(std::make_shared<Segment>(count)),
      immutable_(false),
      gc_in_progress_(false),
      count_(count),
      compare_(compare),
      tl_writes_(DeleteTlBucket) {
  segments_.push_back(active_);
}

KeyHandle VectorGCRep::Allocate(const size_t len, char** buf) {
  char* raw = nullptr;
  size_t allocated = 0;
  {
    WriteLock l(&rwlock_);
    assert(!immutable_);
    raw = active_->Allocate(len);
    allocated = sizeof(EntryHeader) + len;
  }
  entry_bytes_.FetchAddRelaxed(allocated);
  *buf = raw;
  return static_cast<KeyHandle>(raw);
}

void VectorGCRep::Insert(KeyHandle handle) {
  const char* key = nullptr;
  {
    WriteLock l(&rwlock_);
    assert(!immutable_);
    key = ResolveHandleForInsert(handle);
    active_->bucket->push_back(key);
  }
  bucket_size_.FetchAddRelaxed(1);
}

void VectorGCRep::InsertConcurrently(KeyHandle handle) {
  auto* v = static_cast<TlBucket*>(tl_writes_.Get());
  if (!v) {
    v = new TlBucket();
    tl_writes_.Reset(v);
  }
  v->push_back(static_cast<char*>(handle));
}

bool VectorGCRep::Contains(const char* key) const {
  auto snapshot = MakeSnapshot();
  return std::find(snapshot->bucket->begin(), snapshot->bucket->end(), key) !=
         snapshot->bucket->end();
}

void VectorGCRep::MarkReadOnly() {
  WriteLock l(&rwlock_);
  immutable_ = true;
  RebuildImmutableBucketLocked();
}

size_t VectorGCRep::ApproximateMemoryUsage() {
  return entry_bytes_.LoadRelaxed() +
         bucket_size_.LoadRelaxed() * sizeof(Bucket::value_type);
}

void VectorGCRep::BatchPostProcess() {
  auto* v = static_cast<TlBucket*>(tl_writes_.Get());
  if (v) {
    {
      WriteLock l(&rwlock_);
      assert(!immutable_);
      for (auto& key : *v) {
        active_->bucket->push_back(ResolveHandleForInsert(
            const_cast<char*>(static_cast<const char*>(key))));
      }
    }
    bucket_size_.FetchAddRelaxed(v->size());
    delete v;
    tl_writes_.Reset(nullptr);
  }
}

VectorGCRep::EntryHeader* VectorGCRep::GetEntryHeader(const char* entry) {
  return reinterpret_cast<EntryHeader*>(const_cast<char*>(entry) -
                                        sizeof(EntryHeader));
}

size_t VectorGCRep::GetEntryLength(const char* entry) {
  return GetEntryHeader(entry)->len;
}

const char* VectorGCRep::ResolveHandleForInsert(KeyHandle handle) {
  auto* key = static_cast<char*>(handle);
  EntryHeader* header = GetEntryHeader(key);
  if (header->segment->accepts_inserts) {
    return key;
  }

  char* copy = active_->Allocate(header->len);
  memcpy(copy, key, header->len);
  entry_bytes_.FetchAddRelaxed(sizeof(EntryHeader) + header->len);
  return copy;
}

void VectorGCRep::RecomputeStatsLocked() {
  size_t bucket_size = 0;
  size_t entry_bytes = 0;
  for (const auto& segment : segments_) {
    bucket_size += segment->bucket->size();
    entry_bytes += segment->entry_bytes;
  }
  bucket_size_.StoreRelaxed(bucket_size);
  entry_bytes_.StoreRelaxed(entry_bytes);
}

void VectorGCRep::RebuildImmutableBucketLocked() {
  if (!immutable_) {
    immutable_bucket_.reset();
    return;
  }

  immutable_bucket_.reset(new Bucket());
  immutable_bucket_->reserve(bucket_size_.LoadRelaxed());
  for (const auto& segment : segments_) {
    immutable_bucket_->insert(immutable_bucket_->end(),
                              segment->bucket->begin(), segment->bucket->end());
  }
  std::sort(immutable_bucket_->begin(), immutable_bucket_->end(),
            stl_wrappers::Compare(compare_));
}

MemTableRep::GarbageCollectionStats VectorGCRep::GetStats(
    const Bucket& bucket) {
  GarbageCollectionStats stats;
  for (const char* entry : bucket) {
    Slice key_slice = GetLengthPrefixedSlice(entry);
    ParsedInternalKey ikey;
    Status parse_s = ParseInternalKey(key_slice, &ikey, true /* log_err_key */);
    if (parse_s.ok() &&
        (ikey.type == kTypeDeletion || ikey.type == kTypeSingleDeletion ||
         ikey.type == kTypeDeletionWithTimestamp)) {
      ++stats.deletes;
    }
    ++stats.entries;
    stats.data_size += GetEntryLength(entry);
  }
  return stats;
}

class VectorGCRep::GCContext : public MemTableRep::GarbageCollectionContext {
 public:
  GCContext(VectorGCRep* rep, std::shared_ptr<Snapshot> input,
            std::shared_ptr<Segment> output, GarbageCollectionStats input_stats)
      : rep_(rep),
        input_(std::move(input)),
        output_(std::move(output)),
        input_stats_(input_stats),
        finished_(false) {
    output_->accepts_inserts = false;
  }

  ~GCContext() override { Abort(); }

  Iterator* NewIterator(Arena* arena = nullptr) override {
    char* mem = nullptr;
    if (arena != nullptr) {
      mem = arena->AllocateAligned(sizeof(VectorGCRep::Iterator));
    }
    if (arena == nullptr) {
      return new VectorGCRep::Iterator(input_, rep_->compare_);
    }
    return new (mem) VectorGCRep::Iterator(input_, rep_->compare_);
  }

  KeyHandle Allocate(const size_t len, char** buf) override {
    char* raw = output_->Allocate(len);
    *buf = raw;
    return static_cast<KeyHandle>(raw);
  }

  void Insert(KeyHandle handle) override {
    output_->bucket->push_back(static_cast<const char*>(handle));
  }

  const GarbageCollectionStats& InputStats() const override {
    return input_stats_;
  }

  Status Finish() override {
    if (finished_) {
      return Status::OK();
    }

    WriteLock l(&rep_->rwlock_);
    if (!rep_->gc_in_progress_) {
      finished_ = true;
      return Status::Aborted("vectorgc memtable GC was already finished");
    }

    std::unordered_set<Segment*> input_segments;
    input_segments.reserve(input_->segments.size());
    for (const auto& segment : input_->segments) {
      input_segments.insert(segment.get());
    }

    std::vector<std::shared_ptr<Segment>> new_segments;
    if (!output_->bucket->empty()) {
      new_segments.push_back(output_);
    }
    for (const auto& segment : rep_->segments_) {
      if (input_segments.find(segment.get()) == input_segments.end()) {
        new_segments.push_back(segment);
      }
    }
    rep_->segments_ = std::move(new_segments);
    rep_->gc_in_progress_ = false;
    rep_->RecomputeStatsLocked();
    rep_->RebuildImmutableBucketLocked();
    finished_ = true;
    return Status::OK();
  }

  void Abort() override {
    if (finished_) {
      return;
    }
    WriteLock l(&rep_->rwlock_);
    if (rep_->gc_in_progress_) {
      rep_->gc_in_progress_ = false;
      rep_->RebuildImmutableBucketLocked();
    }
    finished_ = true;
  }

 private:
  VectorGCRep* rep_;
  std::shared_ptr<Snapshot> input_;
  std::shared_ptr<Segment> output_;
  GarbageCollectionStats input_stats_;
  bool finished_;
};

std::unique_ptr<MemTableRep::GarbageCollectionContext>
VectorGCRep::StartGarbageCollection() {
  WriteLock l(&rwlock_);
  if (immutable_ || gc_in_progress_) {
    return nullptr;
  }

  auto snapshot = std::make_shared<Snapshot>();
  snapshot->segments = segments_;
  snapshot->bucket->reserve(bucket_size_.LoadRelaxed());
  for (const auto& segment : snapshot->segments) {
    segment->accepts_inserts = false;
    snapshot->bucket->insert(snapshot->bucket->end(), segment->bucket->begin(),
                             segment->bucket->end());
  }

  auto output = std::make_shared<Segment>(snapshot->bucket->size());
  GarbageCollectionStats input_stats = GetStats(*snapshot->bucket);
  active_ = std::make_shared<Segment>(count_);
  segments_.push_back(active_);
  gc_in_progress_ = true;

  return std::unique_ptr<GarbageCollectionContext>(
      new GCContext(this, std::move(snapshot), std::move(output), input_stats));
}

std::shared_ptr<VectorGCRep::Snapshot> VectorGCRep::MakeSnapshot() const {
  auto snapshot = std::make_shared<Snapshot>();
  ReadLock l(&rwlock_);
  snapshot->segments = segments_;
  if (immutable_ && immutable_bucket_ != nullptr) {
    snapshot->bucket = immutable_bucket_;
    snapshot->sorted = true;
  } else {
    snapshot->bucket->reserve(bucket_size_.LoadRelaxed());
    for (const auto& segment : snapshot->segments) {
      snapshot->bucket->insert(snapshot->bucket->end(),
                               segment->bucket->begin(),
                               segment->bucket->end());
    }
  }
  return snapshot;
}

void VectorGCRep::Get(const LookupKey& k, void* callback_args,
                      bool (*callback_func)(void* arg, const char* entry)) {
  VectorGCRep::Iterator iter(MakeSnapshot(), compare_);
  for (iter.Seek(k.user_key(), k.memtable_key().data());
       iter.Valid() && callback_func(callback_args, iter.key()); iter.Next()) {
  }
}

void VectorGCRep::UniqueRandomSample(const uint64_t num_entries,
                                     const uint64_t target_sample_size,
                                     std::unordered_set<const char*>* entries) {
  (void)num_entries;
  entries->clear();
  if (target_sample_size == 0) {
    return;
  }

  auto snapshot = MakeSnapshot();
  const uint64_t nentries = static_cast<uint64_t>(snapshot->bucket->size());
  if (nentries == 0) {
    return;
  }
  if (target_sample_size >= nentries) {
    entries->insert(snapshot->bucket->begin(), snapshot->bucket->end());
    return;
  }

  Random* rnd = Random::GetTLSInstance();
  const uint64_t max_attempts = target_sample_size * 5;
  for (uint64_t attempts = 0;
       attempts < max_attempts && entries->size() < target_sample_size;
       ++attempts) {
    const uint64_t index = rnd->Next() % nentries;
    entries->insert((*snapshot->bucket)[index]);
  }

  for (auto it = snapshot->bucket->begin();
       entries->size() < target_sample_size && it != snapshot->bucket->end();
       ++it) {
    entries->insert(*it);
  }
}

VectorGCRep::Iterator::Iterator(std::shared_ptr<Snapshot> snapshot,
                                const KeyComparator& compare)
    : snapshot_(std::move(snapshot)),
      cit_(snapshot_->bucket->end()),
      compare_(compare),
      sorted_(snapshot_->sorted) {}

void VectorGCRep::Iterator::DoSort() const {
  if (!sorted_) {
    std::sort(snapshot_->bucket->begin(), snapshot_->bucket->end(),
              stl_wrappers::Compare(compare_));
    cit_ = snapshot_->bucket->begin();
    sorted_ = true;
  }
}

bool VectorGCRep::Iterator::Valid() const {
  DoSort();
  return cit_ != snapshot_->bucket->end();
}

const char* VectorGCRep::Iterator::key() const {
  assert(sorted_);
  return *cit_;
}

void VectorGCRep::Iterator::Next() {
  assert(sorted_);
  if (cit_ != snapshot_->bucket->end()) {
    ++cit_;
  }
}

void VectorGCRep::Iterator::Prev() {
  assert(sorted_);
  if (cit_ == snapshot_->bucket->begin()) {
    cit_ = snapshot_->bucket->end();
  } else {
    --cit_;
  }
}

void VectorGCRep::Iterator::Seek(const Slice& user_key,
                                 const char* memtable_key) {
  DoSort();
  const char* encoded_key =
      (memtable_key != nullptr) ? memtable_key : EncodeKey(&tmp_, user_key);
  cit_ =
      std::equal_range(
          snapshot_->bucket->begin(), snapshot_->bucket->end(), encoded_key,
          [this](const char* a, const char* b) { return compare_(a, b) < 0; })
          .first;
}

void VectorGCRep::Iterator::SeekForPrev(const Slice& user_key,
                                        const char* memtable_key) {
  DoSort();
  const char* encoded_key =
      (memtable_key != nullptr) ? memtable_key : EncodeKey(&tmp_, user_key);
  cit_ = std::upper_bound(
      snapshot_->bucket->begin(), snapshot_->bucket->end(), encoded_key,
      [this](const char* a, const char* b) { return compare_(a, b) < 0; });
  if (cit_ == snapshot_->bucket->begin()) {
    cit_ = snapshot_->bucket->end();
  } else {
    --cit_;
  }
}

void VectorGCRep::Iterator::SeekToFirst() {
  DoSort();
  cit_ = snapshot_->bucket->begin();
}

void VectorGCRep::Iterator::SeekToLast() {
  DoSort();
  cit_ = snapshot_->bucket->end();
  if (!snapshot_->bucket->empty()) {
    --cit_;
  }
}

MemTableRep::Iterator* VectorGCRep::GetIterator(Arena* arena) {
  char* mem = nullptr;
  if (arena != nullptr) {
    mem = arena->AllocateAligned(sizeof(Iterator));
  }
  auto snapshot = MakeSnapshot();
  if (arena == nullptr) {
    return new Iterator(std::move(snapshot), compare_);
  } else {
    return new (mem) Iterator(std::move(snapshot), compare_);
  }
}
}  // namespace

static std::unordered_map<std::string, OptionTypeInfo> vector_rep_table_info = {
    {"count",
     {0, OptionType::kSizeT, OptionVerificationType::kNormal,
      OptionTypeFlags::kNone}},
};

VectorRepFactory::VectorRepFactory(size_t count) : count_(count) {
  RegisterOptions("VectorRepFactoryOptions", &count_, &vector_rep_table_info);
}

MemTableRep* VectorRepFactory::CreateMemTableRep(
    const MemTableRep::KeyComparator& compare, Allocator* allocator,
    const SliceTransform*, Logger* /*logger*/) {
  return new VectorRep(compare, allocator, count_);
}

VectorGCRepFactory::VectorGCRepFactory(size_t count) : count_(count) {
  RegisterOptions("VectorGCRepFactoryOptions", &count_, &vector_rep_table_info);
}

MemTableRep* VectorGCRepFactory::CreateMemTableRep(
    const MemTableRep::KeyComparator& compare, Allocator* allocator,
    const SliceTransform*, Logger* /*logger*/) {
  return new VectorGCRep(compare, allocator, count_);
}
}  // namespace ROCKSDB_NAMESPACE
