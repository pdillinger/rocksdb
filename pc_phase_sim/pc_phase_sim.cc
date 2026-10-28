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
// Ported from branch phased_periodic_compaction_pre to current main:
// * compaction_schedule_seed no longer exists (dropped).
// * Each DB gets a deterministic DB ID (scenario, PC_SIM_REP, db index), hence
// a
//   deterministic preferred phase, so runs against different TriggerTime
//   variants are paired (same phases), reducing comparison noise.
// * midlife_dynamic_r{33,50,100}: staggered (uniform-age) fleet, phasing off,
//   dynamic enable at 2.4N (the production switchover case).
// * PC_SIM_ONLY is a comma-separated list of exact scenario names.
// * Staggered seeding is paced against absolute target times; lockstep
//   seeding and restart are parallelized so larger M keeps a tight herd.
// * The shared LOW (compaction) pool is enlarged so a herd of tiny periodic
//   compactions is not serialized behind 2 threads (queueing artifact).
//
// Run: ./pc_phase_sim [N_seconds] [num_dbs] [out.csv]
//   env: PC_SIM_DIR (DB base dir), PC_SIM_ONLY, PC_SIM_REP (DB ID salt),
//        PC_SIM_START_AT (unix seconds at which to start the scenario)

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "rocksdb/db.h"
#include "rocksdb/env.h"
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
std::string g_rep = "0";

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

// Fixed DB ID (hence fixed preferred phase) per (scenario, rep, db index).
class FixedIdEnv : public EnvWrapper {
 public:
  FixedIdEnv(Env* base, std::string id)
      : EnvWrapper(base), id_(std::move(id)) {}
  const char* Name() const override { return "FixedIdEnv"; }
  std::string GenerateUniqueId() override { return id_; }

 private:
  std::string id_;
};

void Check(const Status& s, const char* what) {
  if (!s.ok()) {
    fprintf(stderr, "FATAL %s: %s\n", what, s.ToString().c_str());
    abort();
  }
}

Options MakeOptions(Env* env, int db_index, uint64_t n_seconds,
                    int recovery_percent) {
  Options o;
  o.env = env;
  o.create_if_missing = true;
  o.compaction_style = kCompactionStyleLevel;
  o.num_levels = 7;
  o.level0_file_num_compaction_trigger = 8;  // avoid incidental L0 compactions
  o.periodic_compaction_seconds = n_seconds;
  o.periodic_compaction_phase_recovery_percent = recovery_percent;
  o.max_compaction_trigger_wakeup_seconds = 1;  // check every ~1s
  o.stats_dump_period_sec = 0;
  o.stats_persist_period_sec = 0;
  o.keep_log_file_num = 1;
  o.max_log_file_size = 1 << 20;
  o.info_log_level = WARN_LEVEL;
  // Many concurrent DBs on tmpfs: avoid ~70MB WAL (and MANIFEST/SST)
  // preallocation per DB, which is real memory there.
  o.allow_fallocate = false;
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
  std::unique_ptr<Env> env;
  std::unique_ptr<DB> db;
  std::string path;
};

// Run fn(i) for i in [0, n) on a few threads.
template <typename F>
void ParallelFor(int n, F fn) {
  const int kThreads = 16;
  std::atomic<int> next{0};
  std::vector<std::thread> ts;
  for (int t = 0; t < kThreads; ++t) {
    ts.emplace_back([&]() {
      for (int i = next.fetch_add(1); i < n; i = next.fetch_add(1)) {
        fn(i);
      }
    });
  }
  for (auto& t : ts) {
    t.join();
  }
}

bool Selected(const std::string& name) {
  const char* only = getenv("PC_SIM_ONLY");
  if (only == nullptr || *only == '\0') {
    return true;
  }
  std::stringstream ss(only);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item == name) {
      return true;
    }
  }
  return false;
}

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
                 double burst_mean = 0, double period_change_after_s = -1,
                 uint64_t new_n_seconds = 0, int period_change_recovery = -1) {
  if (!Selected(name)) {
    return;
  }
  fprintf(stderr, "\n=== %s (N=%llu, M=%d, run=%.0fs, rp=%d%s%s%s%s) ===\n",
          name.c_str(), (unsigned long long)n_seconds, num_dbs, run_seconds,
          recovery_percent, staggered ? ", staggered" : "",
          natural_period > 0 ? ", natural" : "",
          burst_mean > 0 ? ", burst" : "",
          period_change_after_s >= 0 ? ", period-change" : "");
  g_scenario = name;

  std::vector<Db> dbs(num_dbs);
  for (int i = 0; i < num_dbs; ++i) {
    dbs[i].path = base_dir + "/" + name + "_db" + std::to_string(i);
    (void)system(("rm -rf '" + dbs[i].path + "'").c_str());
    dbs[i].env.reset(new FixedIdEnv(
        Env::Default(),
        "pcsim-" + name + "-" + g_rep + "-" + std::to_string(i)));
  }
  {
    std::lock_guard<std::mutex> l(g_mu);
    g_last_comp.assign(num_dbs, 0.0);
  }
  // Preferred phases are defined in absolute (unix) time, so runs being
  // compared (e.g. against different TriggerTime variants) must start at the
  // same absolute time to be paired.
  const char* start_at_env = getenv("PC_SIM_START_AT");
  if (start_at_env != nullptr && *start_at_env != '\0') {
    const double start_at = std::strtod(start_at_env, nullptr);
    for (;;) {
      const double now_unix =
          std::chrono::duration<double>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
      if (now_unix >= start_at) {
        break;
      }
      std::this_thread::sleep_for(
          std::chrono::duration<double>(std::min(start_at - now_unix, 0.5)));
    }
  }
  g_start_ns.store(NowSec());

  auto open_and_seed = [&](int i) {
    Options o = MakeOptions(dbs[i].env.get(), i, n_seconds, recovery_percent);
    Check(DB::Open(o, dbs[i].path, &dbs[i].db), "Open");
    WriteCompact(dbs[i].db.get(), 2000);  // seed (recorded as W)
  };
  if (staggered) {
    const double stagger_gap = double(n_seconds) / num_dbs;
    for (int i = 0; i < num_dbs; ++i) {
      const double target = i * stagger_gap;
      const double now = NowRel();
      if (target > now) {
        std::this_thread::sleep_for(
            std::chrono::duration<double>(target - now));
      }
      open_and_seed(i);
    }
  } else {
    ParallelFor(num_dbs, open_and_seed);
  }
  fprintf(stderr, "  [t=%.1fs] seeded %d DBs\n", NowRel(), num_dbs);

  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  std::vector<double> next_burst(num_dbs, 0.0);
  if (burst_mean > 0) {
    for (int i = 0; i < num_dbs; ++i) next_burst[i] = u01(rng) * burst_mean;
  }

  bool upgraded = false, restarted = false, period_changed = false;
  while (NowRel() < run_seconds) {
    const double now = NowRel();
    if (upgrade_after_s >= 0 && !upgraded && now >= upgrade_after_s) {
      for (auto& d : dbs) {
        Check(d.db->SetDBOptions({{"periodic_compaction_phase_recovery_percent",
                                   std::to_string(upgrade_recovery)}}),
              "SetDBOptions");
      }
      upgraded = true;
      fprintf(stderr, "  [t=%.1fs..%.1fs] DYNAMIC enable rp=%d\n", now,
              NowRel(), upgrade_recovery);
    }
    if (restart_after_s >= 0 && !restarted && now >= restart_after_s) {
      ParallelFor(num_dbs, [&](int i) {
        Check(dbs[i].db->Close(), "Close(restart)");
        dbs[i].db.reset();
        Options o =
            MakeOptions(dbs[i].env.get(), i, n_seconds, restart_recovery);
        Check(DB::Open(o, dbs[i].path, &dbs[i].db), "Reopen(restart)");
      });
      restarted = true;
      fprintf(stderr, "  [t=%.1fs..%.1fs] RESTART enable rp=%d\n", now,
              NowRel(), restart_recovery);
    }
    if (period_change_after_s >= 0 && !period_changed &&
        now >= period_change_after_s) {
      // Turn periodic_compaction_seconds up or down for all DBs. A turn-down
      // makes the past-due portion of the (age-spread) fleet eligible at once;
      // the re-anchor + newN/4 grid spreads that cohort instead of bursting. A
      // turn-up just extends deadlines (no burst). Report the age mix at the
      // change (relative to the smaller/tighter period) to characterize it.
      const uint64_t ref = new_n_seconds > n_seconds
                               ? n_seconds / 2  // turn-up: "recent" vs half old
                               : std::min(n_seconds, new_n_seconds);
      int recent = 0, past_due = 0;
      {
        std::lock_guard<std::mutex> l(g_mu);
        for (int i = 0; i < num_dbs; ++i) {
          if (now - g_last_comp[i] < static_cast<double>(ref)) {
            recent++;
          } else {
            past_due++;
          }
        }
      }
      for (auto& d : dbs) {
        Check(d.db->SetOptions({{"periodic_compaction_seconds",
                                 std::to_string(new_n_seconds)}}),
              "SetOptions(period)");
        if (period_change_recovery >= 0) {
          // Also switch recovery_percent at the change (e.g. 100 -> 33): start
          // fully spread (rp=100), then let the turn play out at the default.
          Check(
              d.db->SetDBOptions({{"periodic_compaction_phase_recovery_percent",
                                   std::to_string(period_change_recovery)}}),
              "SetDBOptions(rp@change)");
        }
      }
      period_changed = true;
      fprintf(stderr,
              "  [t=%.1fs] PERIOD change -> %llu (rp -> %d); at change "
              "(ref=%llus): recent(age<ref)=%d past_due(age>=ref)=%d of %d\n",
              NowRel(), (unsigned long long)new_n_seconds,
              period_change_recovery, (unsigned long long)ref, recent, past_due,
              num_dbs);
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

  ParallelFor(num_dbs, [&](int i) {
    Check(dbs[i].db->Close(), "Close");
    dbs[i].db.reset();
    (void)system(("rm -rf '" + dbs[i].path + "'").c_str());
  });
}

}  // namespace

int main(int argc, char** argv) {
  uint64_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 12;
  int M = argc > 2 ? std::atoi(argv[2]) : 40;
  std::string out_csv = argc > 3 ? argv[3] : "events.csv";
  const char* base_env = getenv("PC_SIM_DIR");
  std::string base = base_env ? base_env : "/dev/shm/pc_sim";
  const char* rep_env = getenv("PC_SIM_REP");
  if (rep_env != nullptr) {
    g_rep = rep_env;
  }
  (void)system(("mkdir -p '" + base + "'").c_str());
  Env::Default()->SetBackgroundThreads(16, Env::Priority::LOW);
  Env::Default()->SetBackgroundThreads(4, Env::Priority::HIGH);
  const double N = double(n);
  const double P075 = 0.75 * N, B15 = 1.5 * N;

  // --- Pure periodic (no external writes) ---
  // Dynamic enable at 3 recovery rates (shows the de-herd + transient badness).
  // Interval = N (the base periodic-compaction interval). Run 8N so ~4-5
  // post-enable cycles are visible.
  RunScenario("upgrade_dynamic_r33", base, M, n, 8.0 * N, 0, false, 2.4 * N,
              33);
  RunScenario("upgrade_dynamic_r50", base, M, n, 8.0 * N, 0, false, 2.4 * N,
              50);
  RunScenario("upgrade_dynamic_r100", base, M, n, 8.0 * N, 0, false, 2.4 * N,
              100);
  // Restart (reopen) enable at 33 -- should look like the dynamic r33 case.
  RunScenario("upgrade_restart_r33", base, M, n, 8.0 * N, 0, false, -1, 0,
              2.4 * N, 33);
  // Staggered fresh times (uniform file ages), phasing off, then dynamic
  // enable at 2.4N: the production switchover case.
  for (int rp : {33, 50, 100}) {
    RunScenario("midlife_dynamic_r" + std::to_string(rp), base, M, n, 8.0 * N,
                0, true, 2.4 * N, rp);
  }
  // Staggered fresh times, phasing on -> rearrange to seed phases (re-seed).
  RunScenario("staggered_phased_r33", base, M, n, 6.2 * N, 33, true, -1, 0);

  // --- Interval change: turn the periodic-compaction interval DOWN then UP by
  //     a factor of 3. Start at rp=100 so the fleet is fully spread within ~1
  //     cycle, then the config change flips BOTH the interval and rp->33.
  RunScenario("turndown_r33", base, M, 3 * n, 9.0 * N, 100, false, -1, 0, -1, 0,
              0, 0, 3.5 * N, n, 33);
  RunScenario("turnup_r33", base, M, n, 12.0 * N, 100, false, -1, 0, -1, 0, 0,
              0, 2.0 * N, 3 * n, 33);

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
