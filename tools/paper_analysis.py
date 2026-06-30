#!/usr/bin/env python3
"""Aggregate ATQ frozen-decode eval data into the paper's tables and figures.

Re-runnable: scans every results_*/eval directory it is given (or all local ones),
aggregates at TRAIN-SEED granularity (the audit established that the real unit of
replication is the train seed, not the eval seed -- the solved (not T-count) flag is deterministic
across eval seeds at gumbel_scale=0.1), and emits:
  - outputs/figures/*.pdf        (matplotlib figures for the manuscript)
  - outputs/numbers.json         (every number the manuscript cites)

Usage: python3 tools/paper_analysis.py [--eval_dirs d1 d2 ...] [--out outputs]
"""
from __future__ import annotations
import argparse, csv, json, math, re, statistics
from collections import defaultdict
from pathlib import Path

# Primary naming scheme used by every paper-data eval dir:
#   eval_arm-<arm>_target-<target>_seed<seed>.csv
ARM_RE = re.compile(r'eval_arm-(?P<arm>.+?)_target-(?P<target>.+?)_seed(?P<seed>\d+)')
# Secondary scheme used by the compute-grid sweep (config encodes the arm, e.g.
#   eval_cfg-gumbel_huber_sims32_net-default_target-barenco_tof_3_seed2024.csv).
# Without this, these 71 CSVs all fall through to arm='NA' and the CSV's own
# target column (audit 2026-06-27). Not in the table allowlist, but parsed so the
# no-match count below is honest.
CFG_RE = re.compile(r'eval_cfg-(?P<arm>.+?)_target-(?P<target>.+?)_seed(?P<seed>\d+)')

# Explicit allowlist of paper-data eval dirs (mirrors the pinned-DIRS approach in
# tools/paper_mechanism_figure.py). Globbing every results_* unions the SAME
# (target,arm,train_seed) seeds across the longbudget / independent-hardware / value-control runs,
# which would silently contaminate the headline solve-rates. The mechanism number
# (barenco_tof_3 value-control matrix) needs the first two dirs; generality needs
# the rest. Override with --eval_dirs.
MECHANISM_DIRS = [
    'results_a2_value_controls_20260626_163106',
    'results_catwide_barenco3',
]
GENERALITY_DIRS = [
    'results_gen_cuccaro',
    'results_a2_generality_nc3_20260627',
    'results_robust_generality_gf_2pow2_mult_20260627',
    'results_robust_generality_barenco_tof_4_20260627',
    # New GF(2^3) multiply-matrix generality run (mm48); only present on machines
    # where it has been pulled. Included automatically when the dir exists so the
    # allowlist need not be re-edited.
    'results_robust_generality_gf_2pow3_mult_mm48_20260627',
]
ALLOWLIST_DIRS = MECHANISM_DIRS + GENERALITY_DIRS


def _bool(x): return str(x).strip().lower() in ('1', 'true', 'yes')


def parse_name(p: Path):
    """(arm, target, seed, matched) for one eval CSV.

    Tries the eval_arm- scheme, then the eval_cfg- scheme, then falls back to the
    CSV-internal arm/target/train_seed columns with a path-based target (the
    parent dir under .../eval/<target>/...). `matched` is True only when one of
    the two filename regexes fired -- used to count/flag stragglers.
    """
    for rx in (ARM_RE, CFG_RE):
        m = rx.search(p.name)
        if m:
            return m.group('arm'), m.group('target'), int(m.group('seed')), True
    # Path-based fallback: .../<dir>/eval/<target>/<file>.csv  -> <target>.
    parts = p.parts
    path_target = None
    if 'eval' in parts:
        i = parts.index('eval')
        if i + 1 < len(parts) - 1:  # there IS a subdir between eval/ and the file
            path_target = parts[i + 1]
    return None, path_target, None, False


def read_eval_dir(d: Path):
    """Returns (rows, n_unmatched) for control==orbit, k==1 only.

    n_unmatched counts CSVs whose filename matched NEITHER naming regex (so arm
    came from the CSV column or 'NA'); the caller asserts this is 0 for the
    allowlisted table dirs and warns loudly otherwise.
    """
    rows = []
    n_unmatched = 0
    for p in sorted(d.glob('**/*.csv')):
        f_arm, f_target, f_seed, matched = parse_name(p)
        if not matched:
            n_unmatched += 1
        try:
            with p.open(newline='') as f:
                for r in csv.DictReader(f):
                    if r.get('control', 'orbit') != 'orbit':
                        continue
                    if str(r.get('k', '1')) != '1':
                        continue
                    arm = f_arm if f_arm is not None else r.get('arm', 'NA')
                    target = f_target if f_target is not None else r.get('target', 'NA')
                    seed = f_seed if f_seed is not None else int(r.get('train_seed', -1))
                    rows.append(dict(arm=arm, target=target, train_seed=seed,
                                     solved=_bool(r.get('solved')),
                                     cost=float(r.get('cost', 'nan'))))
        except Exception as e:
            print(f'  ! skip {p}: {e}')
    return rows, n_unmatched


def seed_level(rows):
    """Collapse eval-seeds: a (target,arm,train_seed) is 'solved' if any eval-seed solved
    (solved flag deterministic in practice; T may vary slightly across eval seeds); decoded T = median over solved eval rows.

    Buckets are keyed by (source_dir, target, arm, train_seed) so that the same
    train seed living in two results_* dirs does NOT silently union into one row.
    A hard assertion then verifies that no (target,arm,train_seed) is sourced from
    more than one dir -- i.e. the caller passed a clean (e.g. single-dir or
    non-overlapping) set; cross-dir collisions fail loudly instead of surviving
    'by accident'."""
    by = defaultdict(list)
    for r in rows:
        src = r.get('source_dir', '')
        by[(src, r['target'], r['arm'], r['train_seed'])].append(r)
    # Detect cross-dir collisions: same (target,arm,train_seed) from >1 source_dir.
    srcs_per_key = defaultdict(set)
    for (src, target, arm, seed) in by:
        srcs_per_key[(target, arm, seed)].add(src)
    collisions = {k: sorted(v) for k, v in srcs_per_key.items() if len(v) > 1}
    assert not collisions, (
        'cross-dir train-seed collision: the same (target,arm,train_seed) was '
        'drawn from multiple results_* dirs, which would silently union '
        'incomparable runs. Offenders:\n  ' +
        '\n  '.join(f'{k} <- {v}' for k, v in sorted(collisions.items())))
    out = []
    for (src, target, arm, seed), v in by.items():
        solved_costs = [x['cost'] for x in v if x['solved']]
        out.append(dict(target=target, arm=arm, train_seed=seed,
                        solved=len(solved_costs) > 0,
                        median_T=statistics.median(solved_costs) if solved_costs else None))
    return out


def arm_summary(seed_rows):
    """Per (target, arm): #solved train-seeds / #train-seeds, median T over solved seeds."""
    by = defaultdict(list)
    for r in seed_rows:
        by[(r['target'], r['arm'])].append(r)
    out = {}
    for (target, arm), v in by.items():
        n = len(v)
        s = sum(1 for x in v if x['solved'])
        med = [x['median_T'] for x in v if x['median_T'] is not None]
        out[(target, arm)] = dict(n_seeds=n, n_solved=s,
                                  solve_rate=s / n if n else float('nan'),
                                  median_T=statistics.median(med) if med else None)
    return out


def fisher_two_sided(a, b, c, d):
    """2x2 Fisher exact, two-sided (sum of all 2x2 tables with prob <= observed). a,b / c,d."""
    n = a + b + c + d
    def comb(n, k): return math.comb(n, k)
    def p(a, b, c, d):
        return (comb(a + b, a) * comb(c + d, c)) / comb(n, a + c)
    r1, c1 = a + b, a + c
    obs = p(a, b, c, d)
    tot = 0.0
    for x in range(0, min(r1, c1) + 1):
        bb, cc, dd = r1 - x, c1 - x, n - r1 - c1 + x
        if bb < 0 or cc < 0 or dd < 0:
            continue
        pr = p(x, bb, cc, dd)
        if pr <= obs + 1e-12:
            tot += pr
    return min(1.0, tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='.')
    ap.add_argument('--eval_dirs', nargs='*', default=None)
    ap.add_argument('--out', default='outputs')
    args = ap.parse_args()
    repo = Path(args.repo)
    outdir = repo / args.out
    (outdir / 'figures').mkdir(parents=True, exist_ok=True)

    if args.eval_dirs:
        eval_dirs = [Path(d) for d in args.eval_dirs]
        allowlisted = False
    else:
        # Explicit allowlist (skip dirs not pulled to this machine). Pinning this
        # is the whole point of the fix: globbing every results_* unions the same
        # train seeds across runs and silently contaminates the headline numbers.
        eval_dirs = [repo / d for d in ALLOWLIST_DIRS if (repo / d / 'eval').is_dir()]
        allowlisted = True

    all_rows = []
    per_dir = {}
    total_unmatched = 0
    for d in eval_dirs:
        rows, n_unmatched = read_eval_dir(d / 'eval')
        if n_unmatched:
            total_unmatched += n_unmatched
            print(f'  WARNING: {d.name}: {n_unmatched} CSV(s) matched NEITHER '
                  f'eval_arm- nor eval_cfg- naming (arm taken from CSV column or NA)')
            # The allowlisted table dirs are all eval_arm-; any straggler there is
            # a data/regex bug that would mis-bucket a paper row -> fail loudly.
            assert not allowlisted, (
                f'{n_unmatched} unnamed CSV(s) in allowlisted dir {d.name}; refusing '
                'to emit paper numbers from un-self-describing rows')
        if rows:
            # Tag every row with its source dir so seed_level keys buckets per-dir
            # and can detect cross-dir collisions.
            for r in rows:
                r['source_dir'] = d.name
            per_dir[d.name] = arm_summary(seed_level(rows))
            all_rows.extend(rows)
    if total_unmatched:
        print(f'  WARNING: {total_unmatched} CSV(s) total fell through both naming '
              f'regexes (see lines above)')
    summ = arm_summary(seed_level(all_rows))

    numbers = {'datasets': {}, 'mechanism': {}, 'gumbel_vs_muzero': {}}
    for name, s in per_dir.items():
        numbers['datasets'][name] = {f'{t}|{a}': v for (t, a), v in s.items()}

    # ---- Mechanism table (barenco_tof_3, the merged value-control matrix) ----
    arms_order = ['scalar_mse', 'scalar_huber_d1', 'categorical_61', 'categorical_wide',
                  'quantile_risk_neutral', 'quantile_q075']
    mech = {}
    for a in arms_order:
        v = summ.get(('barenco_tof_3', a))
        if v:
            mech[a] = v
    numbers['mechanism'] = mech
    print('=== MECHANISM (barenco_tof_3, frozen decode, train-seed level) ===')
    for a in arms_order:
        if a in mech:
            v = mech[a]
            print(f'  {a:24s} solved={v["n_solved"]}/{v["n_seeds"]} '
                  f'median_T={v["median_T"]}')
    # Fisher exact mse vs huber (solved/unsolved train seeds)
    if 'scalar_mse' in mech and 'scalar_huber_d1' in mech:
        m, h = mech['scalar_mse'], mech['scalar_huber_d1']
        a, b = h['n_solved'], h['n_seeds'] - h['n_solved']
        c, dd = m['n_solved'], m['n_seeds'] - m['n_solved']
        pval = fisher_two_sided(a, b, c, dd)
        numbers['mechanism']['_fisher_huber_vs_mse_p'] = pval
        print(f'  Fisher exact (huber vs mse), two-sided p = {pval:.4g}')

    # ---- Gumbel vs MuZero baseline table ----
    gm = repo / 'results_algebraic_prior_complete_20260624/analysis/baseline_gumbel_vs_muzero.csv'
    if gm.exists():
        with gm.open() as f:
            for r in csv.DictReader(f):
                numbers['gumbel_vs_muzero'][r['target']] = {
                    'muzero_mean': float(r['muzero_mean']),
                    'gumbel_mean': float(r['gumbel_mean']),
                    'pct_reduction': float(r['mean_pct_reduction_vs_muzero']),
                    'n_paired': int(r['n_paired']),
                    'p': r['sign_test_p_two_sided'],
                }

    # The binary solve-rate bar chart was superseded by the richer fig_mech_curves
    # / fig_mech_strip pair (tools/paper_mechanism_figure.py); this tool now only
    # emits numbers.json, not a figure.

    (outdir / 'numbers.json').write_text(json.dumps(numbers, indent=2, default=str))
    print(f'wrote {outdir}/numbers.json  ({len(per_dir)} eval datasets)')


if __name__ == '__main__':
    main()
