#!/usr/bin/env python3
"""Regenerate the hardening-pass numbers cited in paper.tex (the ones not covered by
paper_generality.py / paper_tail_statistic.py), straight from the archived eval CSVs in
data. Emits outputs/hardening_numbers.json + a printed summary. Re-runnable; this is the
reproducibility anchor for the hand-stated text claims (delta-sweep, symlog generality,
categorical support boundary, mod_5_4 out-of-sample).

Seed-level convention (matches paper_generality.py): a (dir,arm,seed) is solved if any
eval-seed row solved; the seed's T is the median over its solved rows; the reported figure is
solved-seeds / total-seeds and the median T over solved seeds.
"""
import csv, glob, json, re, statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
ARM_RE = re.compile(r"eval_arm-(?P<arm>.+?)_target-.+?_seed(?P<seed>\d+)")


def _bool(x):
    return str(x).strip().lower() in ("1", "true", "yes")


def agg(dirs, arm=None):
    """Aggregate seed-level solve/T over one or more result dirs (glob eval/**/*.csv)."""
    seed_any, seed_costs = {}, {}
    for d in dirs:
        for p in glob.glob(str(Path(d) / "eval" / "**" / "*.csv"), recursive=True):
            m = ARM_RE.search(Path(p).name)
            if not m:
                continue
            if arm is not None and m.group("arm") != arm:
                continue
            seed = f"{Path(d).name}:{m.group('seed')}"  # dedupe seeds across dirs
            try:
                rows = list(csv.DictReader(open(p)))
            except OSError:
                continue
            for r in rows:
                seed_any[seed] = True
                if _bool(r.get("solved")) and r.get("cost") not in (None, "", "None"):
                    seed_costs.setdefault(seed, []).append(float(r["cost"]))
    n = len(seed_any)
    solved = [s for s in seed_any if seed_costs.get(s)]
    per_seed_T = [statistics.median(seed_costs[s]) for s in solved]
    medT = round(statistics.median(per_seed_T)) if per_seed_T else None
    return {"n": n, "solved": len(solved), "medT": medT}


def main():
    out = {}

    # 1. delta-sweep (barenco_tof_3, 5 seeds): robust to the Huber threshold.
    dsw = DATA / "results_delta_sweep_20260628"
    out["delta_sweep_barenco_tof_3"] = {
        arm: agg([dsw], arm) for arm in
        ["scalar_mse", "scalar_huber_d0.5", "scalar_huber_d1", "scalar_huber_d2", "scalar_huber_d5"]
    }

    # 2. gf_2pow3_mult Huber at the matched 1500-step budget (6 seeds) -> the soft rescue.
    gf3 = DATA / "results_gf3_mm48_full_20260628"
    out["gf_2pow3_mult"] = {arm: agg([gf3], arm) for arm in
                            ["scalar_mse", "scalar_huber_d1", "quantile_risk_neutral"]}

    # 3. symlog target-transform generality (arm is scalar_mse; the symlog flag is the dir).
    H = DATA / "results_harden_20260628"
    out["symlog"] = {
        "barenco_tof_3": agg([DATA / "results_scalar_mse_symlog_barenco3"]),
        "cuccaro_adder_n3": agg(sorted(H.glob("results_harden_symlog_cuc_*"))),
        "gf_2pow3_mult": agg(sorted(H.glob("results_harden_symlog15_gf3_*"))),
    }

    # 4. categorical support-floor boundary on barenco_tof_3 (the floor must clear the tail ~-72).
    out["categorical_support_boundary"] = {
        "[-80,0]": agg(sorted(H.glob("results_support_categorical_81_*"))),
        "[-120,0]": agg(sorted(H.glob("results_support_categorical_121_*"))),
    }

    # 5. mod_5_4 out-of-sample predict-then-confirm (predicted neutral; all heads solve).
    m54 = DATA / "results_mod_5_4_20260628"
    out["mod_5_4_out_of_sample"] = {arm: agg([m54], arm) for arm in
                                    ["scalar_mse", "scalar_huber_d1", "quantile_risk_neutral"]}

    (REPO / "outputs" / "hardening_numbers.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
