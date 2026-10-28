#!/bin/bash
#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under both the GPLv2 (found in the
#  COPYING file in the root directory) and Apache 2.0 License
#  (found in the LICENSE.Apache file in the root directory).

# TEMPORARY (not for commit). Runs every pc_phase_sim scenario against each
# TriggerTime variant, one process per (variant, scenario), all concurrently.
# current_rep is a second run of the current code, as a noise reference.
#
# Preferred phases are defined in absolute (unix) time, so all variants of a
# scenario start at the same absolute time (PC_SIM_START_AT) and, with
# identical DB IDs, are paired. Scenarios are staggered by 2s to spread the
# seeding load. Results used for pc_phase_switchover.html: N=100, M=1000
# (76 processes, ~20 minutes, a few GB of /dev/shm).
#
# Usage: run_all.sh BUILD_DIR OUT_DIR [N] [M]
#   BUILD_DIR holds pc_phase_sim_{current,kmax,khalf} from build_variants.sh.
#   PC_SIM_SCENARIOS (space-separated) overrides the scenario list.
#   PC_SIM_SHM (default /dev/shm/pcsim_run) is where the DBs live.
# Then: python3 pc_compare.py OUT_DIR ../pc_phase_switchover.html page.html
set -u
BIN=$(realpath "${1:?usage: run_all.sh BUILD_DIR OUT_DIR [N] [M]}")
OUT=$(realpath -m "${2:?usage: run_all.sh BUILD_DIR OUT_DIR [N] [M]}")
N=${3:-100}
M=${4:-1000}
SHM=${PC_SIM_SHM:-/dev/shm/pcsim_run}
SCENARIOS=${PC_SIM_SCENARIOS:-"upgrade_dynamic_r33 upgrade_dynamic_r50 upgrade_dynamic_r100 upgrade_restart_r33
midlife_dynamic_r33 midlife_dynamic_r50 midlife_dynamic_r100 staggered_phased_r33
turndown_r33 turnup_r33 natural_sync_r0 natural_sync_r33 natural_sync_r50 natural_stag_r33
burst_r0 burst_r25 burst_r33 burst_r50 burst_r100"}
rm -rf "$SHM"
base=$(( $(date +%s) + 15 ))
pids=()
i=0
for s in $SCENARIOS; do
  start=$(( base + 2 * i ))
  i=$((i + 1))
  for v in current kmax khalf current_rep; do
    b=$v
    [ "$v" = current_rep ] && b=current
    mkdir -p "$OUT/$v"
    PC_SIM_ONLY=$s PC_SIM_REP=0 PC_SIM_START_AT=$start PC_SIM_DIR="$SHM/$v/$s" \
      "$BIN/pc_phase_sim_$b" "$N" "$M" "$OUT/$v/$s.csv" > "$OUT/$v/$s.log" 2>&1 &
    pids+=($!)
  done
done
echo "launched ${#pids[@]} processes at $(date); first start at $base"
fail=0
for p in "${pids[@]}"; do
  wait "$p" || fail=$((fail + 1))
done
rm -rf "$SHM"
echo "done at $(date); failures=$fail"
