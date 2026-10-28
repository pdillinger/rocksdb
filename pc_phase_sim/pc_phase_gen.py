#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under both the GPLv2 (found in the
#  COPYING file in the root directory) and Apache 2.0 License
#  (found in the LICENSE.Apache file in the root directory).

# Regenerates pc_phase_sim.html from an events.csv emitted by pc_phase_sim.
# Reuses the existing pc_phase_sim.html as the plotting template (SVG plot /
# table / button JS + CSS are kept verbatim); only the embedded DATA blob, the
# per-scenario DESC text, and the intro line are replaced. The "badness" metric
# is recomputed here to match the committed analyzer (calibrated to within
# ~0.005 across the original scenarios).
#
# Usage: python3 pc_phase_gen.py events.csv [pc_phase_sim.html]

import csv
import json
import math
import random
import sys

DT = 0.5  # badness time-bucket (seconds), matching the original arrays

# Wall-clock time (seconds) of the enable / interval-change event per scenario,
# used to draw the dashed marker. Keyed by name; value is a multiple of N.
UPGRADE_MULT = {
    "upgrade_dynamic_r33": 2.4,
    "upgrade_dynamic_r50": 2.4,
    "upgrade_dynamic_r100": 2.4,
    "upgrade_restart_r33": 2.4,
    "turndown_r33": 3.5,
    "turnup_r33": 2.0,
}

# Left edge of the plot view (multiple of N). Crops a long pre-change window so
# the plot focuses on the change and after; 0 (default) shows from the start.
X0_MULT = {
    "turndown_r33": 2.5,  # ~1 new-interval before the change (change at 3.5N)
}

# Per-scenario interval as a multiple of the base N (from the CSV), used for the
# badness reference and grid-line spacing. For the turn scenarios the interval
# changes mid-run; use the POST-change interval since it dominates the (cropped)
# view. Default 1 (== base N).
PERIOD_MULT = {
    "turnup_r33": 3.0,  # ends at 3N after the turn-up
}

# Scenarios whose badness is force-held at 0 until the config change: we make the
# interval change from a quiescent, on-phase fleet and only want to measure the
# cost attributable to the change itself (not a pre-change reference mismatch).
HOLD_BADNESS = {"turndown_r33", "turnup_r33"}

DESC = {
    "upgrade_dynamic_r33": "Pure periodic, interval N. Dynamic SetDBOptions enable to rp=33% at t=2.4N. The lockstep fleet de-herds toward a spread phase distribution over the following cycles (geometric recovery). Badness spikes at the transient, then decays. (At the small default N the recovery step is close to the sim's ~1s compaction/timestamp latency, so it spreads by circulating rather than locking exactly -- run with a larger N, e.g. 36, for a crisper convergence.)",
    "upgrade_dynamic_r50": "Pure periodic, interval N, dynamic enable to rp=50% at t=2.4N. De-herds faster than 33%; larger transient.",
    "upgrade_dynamic_r100": "Pure periodic, interval N, dynamic enable to rp=100% at t=2.4N. Pulls the whole gap in one cycle -> biggest transient spike, then on-phase.",
    "upgrade_restart_r33": "Pure periodic, interval N, RESTART (reopen) enable to rp=33% at t=2.4N. Fresh VersionSet re-anchors at open; behaves like the dynamic r33 case.",
    "naive_upgrade_r25": "NAIVE baseline (synthetic, not a real DB): instead of phase recovery, each DB compacts uniformly 0-25% EARLY at random every cycle -- one-sided random jitter on the interval -- enabled at t=2.4N from the same lockstep herd. It de-herds, BUT it never stops jittering, so it pays a PERMANENT over-compaction badness (steady ~12-14%, flat) and only spreads diffusively (variance grows ~sqrt(cycles), never fully uniform). Contrast upgrade_dynamic_r33: the phase approach de-herds AND its badness DECAYS toward ~0 (it converges to a fixed phase and then compacts on time). Same anti-herding, far lower steady-state cost -- the case for phase-based recovery over random jitter.",
    "staggered_phased_r33": "Pure periodic, staggered fresh times, phasing 33% -> rearrange to seed phases (equivalent to a fresh re-seed). Small transient badness.",
    "turndown_r33": "Interval TURN-DOWN 3x: 3N -> N (36s -> 12s at N=12s). Starts at rp=100 so the fleet is fully spread within ~1 cycle; at t=3.5N the config change flips both the interval (3N->N) and rp (100->33). A fully-spread fleet has uniform ages, so ~2/3 are past the new N-second deadline at the change -> the re-anchor + N/4 grid spreads that cohort while ~1/3 (recently compacted) carry on. (Plot cropped to ~1 interval before the change. Badness is force-held at 0 until the change -- the fleet is quiescent/on-phase there -- so the curve shows only the cost attributable to the turn-down, scored against the post-change interval N.)",
    "turnup_r33": "Interval TURN-UP 3x: N -> 3N (12s -> 36s at N=12s). Starts at rp=100 (fully spread within ~1 cycle); at t=2N the config change flips the interval (N->3N) and rp (100->33). Deadlines only grow, so nothing becomes past-due and there is no burst -- just a lull (a real absence of compactions, not missing data) up to one new interval, then the sparser 3N cadence resumes. Run shows ~4-5 of the new 3N cycles; badness is force-held at 0 until the change and then scored against the new 3N interval.",
    "natural_sync_r0": "Writes fully compact every 0.75N (sync). Phasing OFF -> periodic never fires (0.75N<N); badness ~0. Baseline.",
    "natural_sync_r33": "Same 0.75N writes, phasing 33%. Periodic sometimes preempts a write early -> sustained (not just transient) badness as the phase drifts vs the 0.75N cadence.",
    "natural_sync_r50": "Same 0.75N writes, phasing 50%. Higher sustained badness than 33%.",
    "natural_stag_r33": "0.75N writes, staggered start, phasing 33%.",
    "burst_r0": "Random write bursts ~every 1.5N, plus periodic (needed for the N SLO since 1.5N>N). Phasing OFF -> baseline badness ~0.",
    "burst_r25": "Bursts re-randomize the phase every ~1.5N; phasing 25% chases it. Lowest sustained badness of the phased sweep.",
    "burst_r33": "Bursts re-randomize phase; phasing 33% (the default) -> modest sustained badness.",
    "burst_r50": "Bursts re-randomize phase; phasing 50% -> more chasing cost.",
    "burst_r100": "Bursts re-randomize phase; phasing 100% -> largest sustained badness (chases hardest).",
}

INTRO = (
    '<p style="font-size:13px;margin:2px 0">Real DBs (%(M)d), '
    "periodic_compaction_seconds N=%(N)ds, real periodic-task scheduler, "
    "per-DB listeners recording every compaction (periodic vs "
    "write-driven). Scenarios: pure periodic (dynamic/restart enable), "
    "interval turn-down/turn-up (SetOptions), natural writes (full "
    "compaction every 0.75N), and random write bursts (~every 1.5N) with a "
    "recovery-rate sweep.</p>"
)


def compute_badness(events, N, tmax, grace=1.5, start=0.0):
    """events: list of (t, db, kind_char 'P'/'W'). Each compaction amortizes at
    1/N over the next N; while overdue (past N+grace) cost accrues at 2/N
    (double); an EARLY periodic compaction charges the previous file's
    un-amortized remainder immediately (over-compaction); an early WRITE
    replacement is free. `grace` absorbs scheduler/queueing latency.

    `start` force-holds badness at 0 for t < start and begins the accounting
    there (used for interval-change scenarios: we make the change from a
    quiescent state and only want the cost attributable to the change itself)."""
    nb = max(1, int(math.ceil(tmax / DT)))
    realized = [0.0] * nb
    ideal = [0.0] * nb
    meas = start  # measurement floor: nothing is counted before this

    def add_rate(arr, t0, t1, rate):
        t0 = max(t0, meas)
        if t1 <= t0:
            return
        b0 = max(0, int(t0 / DT))
        b1 = min(nb - 1, int((t1 - 1e-9) / DT))
        for b in range(b0, b1 + 1):
            lo = max(t0, b * DT)
            hi = min(t1, (b + 1) * DT)
            if hi > lo:
                arr[b] += rate * (hi - lo)

    def add_point(arr, t, amt):
        if t < meas:
            return
        b = int(t / DT)
        if 0 <= b < nb:
            arr[b] += amt

    bydb = {}
    for t, db, k in events:
        bydb.setdefault(db, []).append((t, k))

    for evs in bydb.values():
        evs.sort()
        seed_t = evs[0][0]
        last_t = evs[-1][0]
        eval_end = min(tmax, last_t + N)
        add_rate(ideal, max(seed_t, start), eval_end, 1.0 / N)
        prev_t = seed_t
        for i in range(1, len(evs)):
            t, k = evs[i]
            cov_end = prev_t + N
            add_rate(realized, prev_t, min(t, cov_end), 1.0 / N)
            overdue_start = cov_end + grace
            if t > overdue_start:
                add_rate(realized, cov_end, overdue_start, 1.0 / N)
                add_rate(realized, overdue_start, t, 2.0 / N)
            elif t > cov_end:
                add_rate(realized, cov_end, t, 1.0 / N)
            if t < cov_end and k == "P":
                add_point(realized, t, (cov_end - t) / N)
            prev_t = t
        add_rate(realized, prev_t, eval_end, 1.0 / N)

    inst, cum = [], []
    rc = ic = 0.0
    for b in range(nb):
        inst.append(realized[b] / ideal[b] - 1 if ideal[b] > 1e-9 else 0.0)
        rc += realized[b]
        ic += ideal[b]
        cum.append(rc / ic - 1 if ic > 1e-9 else 0.0)
    half = nb // 2
    return {
        "badness_t": [round((b + 1) * DT, 3) for b in range(nb)],
        "badness_inst": [round(x, 3) for x in inst],
        "badness_cum": [round(x, 3) for x in cum],
        "peak_inst": round(max(inst) if inst else 0.0, 2),
        "steady_inst": round(sum(inst[half:]) / max(1, len(inst[half:])), 3),
        "end_cum": round(cum[-1], 3) if cum else 0.0,
    }


def synth_naive(N, M, jitter_frac=0.25, enable_mult=2.4, run_mult=8.0, rng_seed=98765):
    """Synthetic (non-DB) baseline modelling the naive alternative to phase
    recovery: after 'enable', every DB compacts uniformly 0..jitter_frac EARLY
    at random each cycle (one-sided jitter on the interval) instead of steering
    toward a per-DB phase. Before enable the fleet is a lockstep herd (compacts
    exactly at the deadline). Returns events as (t, db, 'P'/'W')."""
    rng = random.Random(rng_seed)
    enable_t = enable_mult * N
    run_t = run_mult * N
    events = []
    for db in range(M):
        events.append((0.0, db, "W"))  # seed
        last = 0.0
        t = float(N)
        while t <= enable_t:  # lockstep at the deadline (herd) pre-enable
            events.append((t, db, "P"))
            last = t
            t += N
        while True:  # naive one-sided early jitter post-enable
            fire = last + N - rng.uniform(0.0, jitter_frac * N)
            if fire >= run_t:
                break
            events.append((fire, db, "P"))
            last = fire
    return events


def build_data(csv_path):
    order = []
    evs = {}
    N = M = None
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            s = r["scenario"]
            if s not in evs:
                evs[s] = []
                order.append(s)
            evs[s].append((float(r["t_rel_s"]), int(r["db"]), r["kind"]))
            N = int(r["N"])
            M = int(r["M"])
    scenarios = []
    for name in order:
        e = evs[name]
        tmax = max(x[0] for x in e)
        events = [[round(t, 2), db, 1 if k == "P" else 0] for (t, db, k) in e]
        # Per-scenario interval as a multiple of the base N (from the CSV), used
        # for the badness reference and grid-line spacing. Turn scenarios also
        # force-hold badness at 0 until the config change (quiescent baseline).
        period = int(round(PERIOD_MULT.get(name, 1.0) * N))
        up = UPGRADE_MULT.get(name)
        hold = round(up * N, 3) if (up is not None and name in HOLD_BADNESS) else 0.0
        b = compute_badness(e, period, tmax, start=hold)
        sc = {
            "name": name,
            "events": events,
            "tmax": round(tmax, 2),
            "nP": sum(1 for x in e if x[2] == "P"),
            "nW": sum(1 for x in e if x[2] == "W"),
            "upgrade_t": round(up * N, 1) if up is not None else None,
            "x0": round(X0_MULT.get(name, 0.0) * N, 1),
            "gridN": period,
        }
        sc.update(b)
        scenarios.append(sc)
    # Synthetic (non-DB) naive-jitter baseline, inserted next to the upgrade
    # group for easy A/B against phase recovery.
    naive = synth_naive(N, M)
    ntmax = max(t for t, _, _ in naive)
    nsc = {
        "name": "naive_upgrade_r25",
        "events": [[round(t, 2), db, 1 if k == "P" else 0] for (t, db, k) in naive],
        "tmax": round(ntmax, 2),
        "nP": sum(1 for x in naive if x[2] == "P"),
        "nW": sum(1 for x in naive if x[2] == "W"),
        "upgrade_t": round(2.4 * N, 1),
        "x0": 0.0,
        "gridN": N,
    }
    nsc.update(compute_badness(naive, N, ntmax))
    idx = next(
        (i for i, s in enumerate(scenarios) if s["name"] == "upgrade_restart_r33"),
        len(scenarios) - 1,
    )
    scenarios.insert(idx + 1, nsc)
    return {"N": N, "M": M, "scenarios": scenarios}


def regenerate(csv_path, html_path):
    data = build_data(csv_path)
    html = open(html_path).read()

    # Replace the DATA blob (single line: const DATA={...};).
    a = html.index("const DATA=")
    b = html.index(";", a)
    data_js = "const DATA=" + json.dumps(data, separators=(",", ":"))
    html = html[:a] + data_js + html[b:]

    # Replace the DESC object (const DESC={ ... };).
    a = html.index("const DESC={")
    b = html.index("\n};", a) + len("\n};")
    desc_js = (
        "const DESC={\n"
        + ",\n".join(" %s:%s" % (json.dumps(k), json.dumps(v)) for k, v in DESC.items())
        + "\n};"
    )
    html = html[:a] + desc_js + html[b:]

    # Replace the intro paragraph.
    marker = '<p style="font-size:13px;margin:2px 0">'
    a = html.index(marker)
    b = html.index("</p>", a) + len("</p>")
    html = html[:a] + (INTRO % {"M": data["M"], "N": data["N"]}) + html[b:]

    open(html_path, "w").write(html)
    return data


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "events.csv"
    html_path = sys.argv[2] if len(sys.argv) > 2 else "pc_phase_sim.html"
    data = regenerate(csv_path, html_path)
    print(
        "Regenerated %s: %d scenarios, N=%d, M=%d"
        % (html_path, len(data["scenarios"]), data["N"], data["M"])
    )
    for s in data["scenarios"]:
        print(
            "  %-22s nP=%-4d nW=%-4d peak=%.2f steady=%.3f end=%.3f"
            % (
                s["name"],
                s["nP"],
                s["nW"],
                s["peak_inst"],
                s["steady_inst"],
                s["end_cum"],
            )
        )


if __name__ == "__main__":
    main()
