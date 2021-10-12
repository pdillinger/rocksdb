//  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "db_stress_tool/db_stress_listener.h"

#include <cstdint>

#include "rocksdb/file_system.h"
#include "util/coding_lean.h"

namespace ROCKSDB_NAMESPACE {

#ifndef ROCKSDB_LITE

// TODO: consider using expected_values_dir instead, but this is more
// convenient for now.
UniqueIdVerifier::UniqueIdVerifier(const std::string& db_name)
    : path_(db_name + "/.unique_ids") {
  // We expect such a small number of files generated during this test
  // (thousands?), checking full 192-bit IDs for uniqueness is a very
  // weak check. For a stronger check, we pick a specific 64-bit
  // subsequence from the ID to check for uniqueness. All bits of the
  // ID should be high quality, and 64 bits should be unique with
  // very good probability for the quantities in this test.
  offset_ = Random::GetTLSInstance()->Uniform(17);  // 0 to 16

  // Use default FileSystem to avoid fault injection, etc.
  FileSystem& fs = *FileSystem::Default();
  IOOptions opts;

  std::unique_ptr<FSSequentialFile> reader;
  Status s =
      fs.NewSequentialFile(path_, FileOptions(), &reader, /*dbg*/ nullptr);
  if (s.ok()) {
    // Load from file
    UniqueIdVerifier::FullID id;
    Slice result;
    for (;;) {
      s = reader->Read(id.size(), opts, &result, id.data(), /*dbg*/ nullptr);
      if (!s.ok()) {
        fprintf(stderr, "Error reading unique id file: %s\n",
                s.ToString().c_str());
        assert(false);
      }
      if (result.size() < id.size()) {
        // EOF
        if (result.size() != 0) {
          // Corrupt file. Not a DB bug but could happen if OS doesn't provide
          // good guarantees on process crash.
          fprintf(stdout, "Warning: clearing corrupt unique id file\n");
          id_set_.clear();
          reader.reset();
          s = fs.DeleteFile(path_, opts, /*dbg*/ nullptr);
          assert(s.ok());
        }
        break;
      }
      VerifyNoWrite(id);
    }
  } else {
    // Newly created is ok.
    // But FileSystem doesn't tell us whether non-existence was the cause of
    // the failure. (Issue #9021)
    Status s2 = fs.FileExists(path_, opts, /*dbg*/ nullptr);
    if (!s2.IsNotFound()) {
      fprintf(stderr, "Error opening unique id file: %s\n",
              s.ToString().c_str());
      assert(false);
    }
  }
  fprintf(stdout, "(Re-)verified %zu unique IDs\n", id_set_.size());
  s = fs.ReopenWritableFile(path_, FileOptions(), &data_file_writer_,
                            /*dbg*/ nullptr);
  if (!s.ok()) {
    fprintf(stderr, "Error opening unique id file for append: %s\n",
            s.ToString().c_str());
    assert(false);
  }
}

UniqueIdVerifier::~UniqueIdVerifier() {
  data_file_writer_->Close(IOOptions(), /*dbg*/ nullptr);
}

void UniqueIdVerifier::VerifyNoWrite(const FullID& id) {
  bool is_new = id_set_.insert(DecodeFixed64(&id[offset_])).second;
  if (!is_new) {
    fprintf(stderr,
            "Duplicate partial unique ID found (offset=%zu, count=%zu)\n",
            offset_, id_set_.size());
    assert(false);
  }
}

void UniqueIdVerifier::Verify(const FullID& id) {
  std::lock_guard<std::mutex> lock(mutex_);
  // If we accumulate more than ~4 million IDs, there would be > 1 in 1M
  // natural chance of collision. Thus, simply stop checking at that point.
  if (id_set_.size() >= 4294967) {
    return;
  }
  IOStatus s = data_file_writer_->Append(Slice(id.data(), id.size()),
                                         IOOptions(), /*dbg*/ nullptr);
  if (!s.ok()) {
    fprintf(stderr, "Error writing to unique id file: %s\n",
            s.ToString().c_str());
    assert(false);
  }
  s = data_file_writer_->Flush(IOOptions(), /*dbg*/ nullptr);
  if (!s.ok()) {
    fprintf(stderr, "Error flushing unique id file: %s\n",
            s.ToString().c_str());
    assert(false);
  }
  VerifyNoWrite(id);
}

void DbStressListener::VerifyTableFileUniqueId(
    const TableProperties& new_file_properties) {
  // Verify unique ID
  UniqueIdVerifier::FullID id;
  GetUniqueIdFromTableProperties(new_file_properties, &id);
  unique_ids_.Verify(id);
}

#endif  // !ROCKSDB_LITE

}  // namespace ROCKSDB_NAMESPACE
