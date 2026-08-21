//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

// TEMPORARY simulator (not for commit) for periodic-compaction preferred-phase
// scheduling. Real DBs, short periodic_compaction_seconds, real periodic-task
// scheduler, per-DB EventListener recording EVERY compaction with its kind
// (P=periodic, W=write/manual). Optionally drives write-driven "natural" full
// compactions (every ~natural_period) and random write "bursts" (~burst_mean).
// Emits events.csv for badness analysis + visualization.
//
// Build: make -f pc_sim.mk
// Run:   LD_LIBRARY_PATH=. ./pc_phase_sim [N_seconds] [num_dbs] [out.csv]

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "rocksdb/db.h"
#include "rocksdb/listener.h"
#include "rocksdb/options.h"

using namespace ROCKSDB_NAMESPACE;

namespace {

struct Event {
  std::string scenario;
  int db;
  double t_rel_s;
  char kind;  // 'P' periodic, 'W' write/manual (incl. seed)
};

std::mutex g_mu;
std::vector<Event> g_events;
std::string g_scenario;
std::atomic<double> g_start_ns{0};
std::vector<double>
    g_last_comp;  // per-db last compaction time (rel), under g_mu

double NowSec() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
double NowRel() { return NowSec() - g_start_ns.load(); }

class PhaseListener : public EventListener {
 public:
  explicit PhaseListener(int db_index) : db_index_(db_index) {}
  void OnCompactionCompleted(DB* /*db*/, const CompactionJobInfo& ci) override {
    const char kind =
        (ci.compaction_reason == CompactionReason::kPeriodicCompaction) ? 'P'
                                                                        : 'W';
    const double t = NowRel();
    std::lock_guard<std::mutex> l(g_mu);
    if (db_index_ >= 0 && db_index_ < static_cast<int>(g_last_comp.size())) {
      g_last_comp[db_index_] = t;
    }
    g_events.push_back({g_scenario, db_index_, t, kind});
  }

 private:
  int db_index_;
};

void Check(const Status& s, const char* what) {
  if (!s.ok()) {
    fprintf(stderr, "FATAL %s: %s\n", what, s.ToString().c_str());
    abort();
  }
}

Options MakeOptions(int db_index, uint64_t n_seconds, int recovery_percent) {
  Options o;
  o.create_if_missing = true;
  o.compaction_style = kCompactionStyleLevel;
  o.num_levels = 7;
  o.level0_file_num_compaction_trigger = 8;  // avoid incidental L0 compactions
  o.periodic_compaction_seconds = n_seconds;
  o.compaction_schedule_seed = kDbNameForScheduleSeed;  // distinct per DB
  o.periodic_compaction_phase_recovery_percent = recovery_percent;
  o.max_compaction_trigger_wakeup_seconds = 1;  // check every ~1s
  o.stats_dump_period_sec = 0;
  o.stats_persist_period_sec = 0;
  o.listeners.push_back(std::make_shared<PhaseListener>(db_index));
  return o;
}

// Force a full (re)compaction into L1 with a fresh file_creation_time. Recorded
// as a 'W' compaction. Overwrites a bounded keyspace so DB size stays tiny.
void WriteCompact(DB* db, int nkeys) {
  WriteOptions wo;
  for (int k = 0; k < nkeys; ++k) {
    Check(db->Put(wo, "key" + std::to_string(k), std::string(40, 'v')), "Put");
  }
  Check(db->Flush(FlushOptions()), "Flush");
  CompactRangeOptions cro;
  cro.change_level = true;
  cro.target_level =
      1;  // keep off the last level so periodic rewrites in place
  Check(db->CompactRange(cro, nullptr, nullptr), "CompactRange");
}

struct Db {
  std::unique_ptr<DB> db;
  std::string path;
};

// staggered: seed DBs at offsets N/M apart.
// upgrade_after_s>=0: SetDBOptions(recovery=upgrade_recovery) at that time.
// restart_after_s>=0: Close+reopen all DBs with recovery=restart_recovery.
// natural_period>0: each DB gets a full write-compaction natural_period after
//   its last compaction (writes reset the periodic countdown).
// burst_mean>0: each DB gets random write bursts ~every burst_mean (uniform
//   [2/3, 4/3]*burst_mean), independent of periodic.
void RunScenario(const std::string& name, const std::string& base_dir,
                 int num_dbs, uint64_t n_seconds, double run_seconds,
                 int recovery_percent, bool staggered, double upgrade_after_s,
                 int upgrade_recovery, double restart_after_s = -1,
                 int restart_recovery = 0, double natural_period = 0,
                 double burst_mean = 0) {
  fprintf(stderr, "\n=== %s (N=%llu, M=%d, run=%.0fs, rp=%d%s%s%s) ===\n",
          name.c_str(), (unsigned long long)n_seconds, num_dbs, run_seconds,
          recovery_percent, staggered ? ", staggered" : "",
          natural_period > 0 ? ", natural" : "",
          burst_mean > 0 ? ", burst" : "");
  g_scenario = name;

  std::vector<Db> dbs(num_dbs);
  for (int i = 0; i < num_dbs; ++i) {
    dbs[i].path = base_dir + "/" + name + "_db" + std::to_string(i);
    (void)system(("rm -rf '" + dbs[i].path + "'").c_str());
  }
  {
    std::lock_guard<std::mutex> l(g_mu);
    g_last_comp.assign(num_dbs, 0.0);
  }
  g_start_ns.store(NowSec());

  double stagger_gap = staggered ? double(n_seconds) / num_dbs : 0.0;
  for (int i = 0; i < num_dbs; ++i) {
    Options o = MakeOptions(i, n_seconds, recovery_percent);
    Check(DB::Open(o, dbs[i].path, &dbs[i].db), "Open");
    WriteCompact(dbs[i].db.get(), 2000);  // seed (recorded as W)
    if (staggered && i + 1 < num_dbs) {
      std::this_thread::sleep_for(std::chrono::duration<double>(stagger_gap));
    }
  }

  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  std::vector<double> next_burst(num_dbs, 0.0);
  if (burst_mean > 0) {
    for (int i = 0; i < num_dbs; ++i) next_burst[i] = u01(rng) * burst_mean;
  }

  bool upgraded = false, restarted = false;
  while (NowRel() < run_seconds) {
    const double now = NowRel();
    if (upgrade_after_s >= 0 && !upgraded && now >= upgrade_after_s) {
      for (auto& d : dbs) {
        Check(d.db->SetDBOptions({{"periodic_compaction_phase_recovery_percent",
                                   std::to_string(upgrade_recovery)}}),
              "SetDBOptions");
      }
      upgraded = true;
      fprintf(stderr, "  [t=%.1fs] DYNAMIC enable rp=%d\n", now,
              upgrade_recovery);
    }
    if (restart_after_s >= 0 && !restarted && now >= restart_after_s) {
      for (int i = 0; i < num_dbs; ++i) {
        Check(dbs[i].db->Close(), "Close(restart)");
        dbs[i].db.reset();
        Options o = MakeOptions(i, n_seconds, restart_recovery);
        Check(DB::Open(o, dbs[i].path, &dbs[i].db), "Reopen(restart)");
      }
      restarted = true;
      fprintf(stderr, "  [t=%.1fs] RESTART enable rp=%d\n", NowRel(),
              restart_recovery);
    }
    if (natural_period > 0) {
      std::vector<double> last;
      {
        std::lock_guard<std::mutex> l(g_mu);
        last = g_last_comp;
      }
      for (int i = 0; i < num_dbs; ++i) {
        if (now - last[i] >= natural_period) WriteCompact(dbs[i].db.get(), 400);
      }
    }
    if (burst_mean > 0) {
      for (int i = 0; i < num_dbs; ++i) {
        if (now >= next_burst[i]) {
          WriteCompact(dbs[i].db.get(), 400);
          next_burst[i] =
              NowRel() + burst_mean * (2.0 / 3.0 + (2.0 / 3.0) * u01(rng));
        }
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  for (auto& d : dbs) {
    Check(d.db->Close(), "Close");
    d.db.reset();
    (void)system(("rm -rf '" + d.path + "'").c_str());
  }
}

}  // namespace

int main(int argc, char** argv) {
  uint64_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 12;
  int M = argc > 2 ? std::atoi(argv[2]) : 40;
  std::string out_csv = argc > 3 ? argv[3] : "events.csv";
  const char* base_env = getenv("PC_SIM_DIR");
  std::string base = base_env ? base_env : "/dev/shm/pc_sim";
  (void)system(("mkdir -p '" + base + "'").c_str());
  const double N = double(n);
  const double P075 = 0.75 * N, B15 = 1.5 * N;

  // --- Pure periodic (no external writes) ---
  // Dynamic enable at 3 recovery rates (shows the fix + transient badness).
  RunScenario("upgrade_dynamic_r33", base, M, n, 6.2 * N, 0, false, 2.4 * N,
              33);
  RunScenario("upgrade_dynamic_r50", base, M, n, 6.2 * N, 0, false, 2.4 * N,
              50);
  RunScenario("upgrade_dynamic_r100", base, M, n, 6.2 * N, 0, false, 2.4 * N,
              100);
  // Restart (reopen) enable at 33 -- should look like the dynamic r33 case.
  RunScenario("upgrade_restart_r33", base, M, n, 6.2 * N, 0, false, -1, 0,
              2.4 * N, 33);
  // Staggered fresh times, phasing on -> rearrange to seed phases (re-seed).
  RunScenario("staggered_phased_r33", base, M, n, 6.2 * N, 33, true, -1, 0);

  // --- Natural write-driven full compaction every 0.75N ---
  RunScenario("natural_sync_r0", base, M, n, 5.2 * N, 0, false, -1, 0, -1, 0,
              P075, 0);
  RunScenario("natural_sync_r33", base, M, n, 6.2 * N, 33, false, -1, 0, -1, 0,
              P075, 0);
  RunScenario("natural_sync_r50", base, M, n, 6.2 * N, 50, false, -1, 0, -1, 0,
              P075, 0);
  RunScenario("natural_stag_r33", base, M, n, 6.2 * N, 33, true, -1, 0, -1, 0,
              P075, 0);

  // --- Random write bursts ~every 1.5N: recovery-rate sweep ---
  RunScenario("burst_r0", base, M, n, 5.2 * N, 0, false, -1, 0, -1, 0, 0, B15);
  RunScenario("burst_r25", base, M, n, 6.2 * N, 25, false, -1, 0, -1, 0, 0,
              B15);
  RunScenario("burst_r33", base, M, n, 6.2 * N, 33, false, -1, 0, -1, 0, 0,
              B15);
  RunScenario("burst_r50", base, M, n, 6.2 * N, 50, false, -1, 0, -1, 0, 0,
              B15);
  RunScenario("burst_r100", base, M, n, 6.2 * N, 100, false, -1, 0, -1, 0, 0,
              B15);

  FILE* f = fopen(out_csv.c_str(), "w");
  fprintf(f, "scenario,db,t_rel_s,kind,N,M\n");
  {
    std::lock_guard<std::mutex> l(g_mu);
    for (const auto& e : g_events) {
      fprintf(f, "%s,%d,%.3f,%c,%llu,%d\n", e.scenario.c_str(), e.db, e.t_rel_s,
              e.kind, (unsigned long long)n, M);
    }
  }
  fclose(f);
  fprintf(stderr, "\nWrote %zu events to %s\n", g_events.size(),
          out_csv.c_str());
  return 0;
}
