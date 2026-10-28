#!/bin/bash
#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under both the GPLv2 (found in the
#  COPYING file in the root directory) and Apache 2.0 License
#  (found in the LICENSE.Apache file in the root directory).

# TEMPORARY (not for commit). Builds pc_phase_sim against three variants of
# PeriodicCompactionPhaser::TriggerTime, each linked against its own release
# static library:
#   current : db/periodic_compaction_phaser.cc at OLD_REV (before the change)
#   khalf   : at NEW_REV (the change, k = 1/2)
#   kmax    : NEW_REV with k = max(q, 1/2)
# The rest of the source tree comes from NEW_REV.
#
# Usage (from the repo root or anywhere inside the checkout):
#   pc_phase_sim/build_variants.sh BUILD_DIR [OLD_REV] [NEW_REV]
# Produces BUILD_DIR/pc_phase_sim_{current,khalf,kmax}.
set -euo pipefail
BUILD=$(realpath -m "${1:?usage: build_variants.sh BUILD_DIR [OLD_REV] [NEW_REV]}")
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
NEW_REV=${3:-$(git -C "$REPO" log -1 --format=%H \
  --grep='^Spread periodic compaction catch-up')}
OLD_REV=${2:-${NEW_REV}~1}
PHASER=db/periodic_compaction_phaser.cc
KHALF_LINE='  const uint64_t k_pct = 50;'
KMAX_LINE='  const uint64_t k_pct = std::max(q_pct, uint64_t{50});'

mkdir -p "$BUILD/src" "$BUILD/variants"
git -C "$REPO" archive "$NEW_REV" | tar -x -C "$BUILD/src"
git -C "$REPO" show "$OLD_REV:$PHASER" > "$BUILD/variants/current.cc"
git -C "$REPO" show "$NEW_REV:$PHASER" > "$BUILD/variants/khalf.cc"
grep -qxF "$KHALF_LINE" "$BUILD/variants/khalf.cc"
sed "s/^$KHALF_LINE\$/$KMAX_LINE/" "$BUILD/variants/khalf.cc" > "$BUILD/variants/kmax.cc"
grep -qxF "$KMAX_LINE" "$BUILD/variants/kmax.cc"

# PLATFORM_CXXFLAGS can carry -nostdlib (e.g. fbcode toolchains), so compile
# and link separately, as the RocksDB Makefile does.
cat > "$BUILD/pc_sim.mk" <<EOF
include make_config.mk
pc_phase_sim_bin: $HERE/pc_phase_sim.cc librocksdb.a
	\$(CXX) \$(PLATFORM_CXXFLAGS) -O2 -fno-rtti -Iinclude -I. -c $HERE/pc_phase_sim.cc -o pc_phase_sim.o
	\$(CXX) pc_phase_sim.o librocksdb.a \$(EXEC_LDFLAGS) \$(PLATFORM_LDFLAGS) -o \$(OUT)
EOF

cd "$BUILD/src"
for v in current kmax khalf; do
  cp "$BUILD/variants/$v.cc" "$PHASER"
  DEBUG_LEVEL=0 AUTO_CLEAN=1 make -j"$(nproc)" static_lib > "$BUILD/build_$v.log" 2>&1
  make -f "$BUILD/pc_sim.mk" -B pc_phase_sim_bin OUT="$BUILD/pc_phase_sim_$v" \
    >> "$BUILD/build_$v.log" 2>&1
  echo "built $BUILD/pc_phase_sim_$v"
done
