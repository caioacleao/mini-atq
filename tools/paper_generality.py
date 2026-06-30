#!/usr/bin/env python3
"""Aggregate the generality-zoo results into the paper's Table V rows.

For each benchmark target and the three reported value heads (MSE, Huber,
quantile-risk-neutral), computes the train-seed-level solve rate and median
frozen-decode T-count, classifies the regime, and emits ready-to-paste LaTeX
rows plus outputs/generality_numbers.json. Re-runnable; pending targets (no eval
data yet) render as em-dashes.

Regimes:
  rescue  -- MSE solves 0 seeds, a robust head solves > 0 (reliability rescue).
  quality -- all solve, but a robust head lowers the median T-count.
  neutral -- all solve at the same T (no rescue, no T gain).
  pending -- no eval data yet.

Usage: python3 tools/paper_generality.py [--repo .]
"""
from __future__ import annotations
import argparse, csv, glob, json, re, statistics
from collections import defaultdict
from pathlib import Path

ARM_RE = re.compile(r'eval_arm-(?P<arm>.+?)_target-.+?_seed(?P<seed>\d+)')
ARMS = ['scalar_mse', 'scalar_huber_d1', 'quantile_risk_neutral']

# (display_tag, family, result_dir)
TARGETS = [
    (r'barenco\_tof\_3',    'Toffoli', 'results_a2_value_controls_20260626_163106'),
    (r'cuccaro\_adder\_n3', 'Adder',   'results_gen_cuccaro'),
    (r'nc\_toff\_3',        'Toffoli', 'results_a2_generality_nc3_20260627'),
    (r'gf\_2pow2\_mult',    'Mult.\\', 'results_robust_generality_gf_2pow2_mult_20260627'),
    (r'barenco\_tof\_4',    'Toffoli', 'results_robust_generality_barenco_tof_4_20260627'),
    (r'gf\_2pow3\_mult',    'Mult.\\', 'results_gf3_mm48_full_20260628'),  # fair mm=48 budget
]


def _bool(x): return str(x).strip().lower() in ('1', 'true', 'yes')


def arm_result(d: Path, arm: str):
    """Train-seed level, matching tools/paper_analysis.py: a (target,arm,seed) is
    solved if any eval-seed solved; the per-seed decoded T is the median over that
    seed's solved eval rows, and the reported T is the median over solved seeds.

    The CSV for each file is fully parsed before any row is committed, so a
    malformed/truncated file is skipped whole (with a notice) rather than
    contributing a half-read, silently-wrong tally."""
    seed_any = defaultdict(bool)         # seed -> has >=1 orbit/k=1 eval row
    seed_solved_costs = defaultdict(list)  # seed -> costs of its solved eval rows
    for p in glob.glob(str(d / 'eval' / '**' / '*.csv'), recursive=True):
        m = ARM_RE.search(Path(p).name)
        if not m or m.group('arm') != arm:
            continue
        seed = m.group('seed')
        try:
            with open(p, newline='') as f:
                rows = list(csv.DictReader(f))
        except (csv.Error, OSError, UnicodeDecodeError) as e:
            print(f'  ! skip {p}: {e}')
            continue
        for r in rows:
            if r.get('control', 'orbit') != 'orbit' or str(r.get('k', '1')) != '1':
                continue
            seed_any[seed] = True
            if _bool(r['solved']):
                seed_solved_costs[seed].append(float(r['cost']))
    n = len(seed_any)
    s = sum(1 for sd in seed_any if seed_solved_costs.get(sd))
    per_seed_med = [statistics.median(v) for v in seed_solved_costs.values() if v]
    medT = int(statistics.median(per_seed_med)) if per_seed_med else None
    return dict(n=n, solved=s, medT=medT)


def regime(res):
    mse, hub, qn = res['scalar_mse'], res['scalar_huber_d1'], res['quantile_risk_neutral']
    if mse['n'] == 0:
        return 'pending'
    robust_solved = hub['solved'] > 0 or qn['solved'] > 0
    if mse['solved'] == 0 and robust_solved:
        return 'rescue'
    if mse['solved'] > 0:
        robust_Ts = [t for t in (hub['medT'], qn['medT']) if t is not None]
        best_robust = min(robust_Ts) if robust_Ts else None
        if mse['medT'] is not None and best_robust is not None and best_robust < mse['medT']:
            return 'quality'
        return 'neutral'
    # mse solves 0 and no robust head solves either: every head fails at this
    # budget. These are targets whose minimal no-gadget decomposition exceeds the
    # compute-light move/search budget (the regime boundary), not a loss-specific
    # collapse the robust head could rescue.
    return 'ceiling'


def cell(r):
    if r['n'] == 0:
        return '---'
    if r['solved'] == 0:
        return f"$0/{r['n']}$"
    t = f"\\,{{\\scriptsize(${r['medT']}$)}}" if r['medT'] is not None else ''
    return f"$\\mathbf{{{r['solved']}/{r['n']}}}${t}" if False else f"${r['solved']}/{r['n']}${t}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='.')
    args = ap.parse_args()
    repo = Path(args.repo)
    out = {}
    print('=== Generality table (train-seed level) ===')
    print(f"{'target':18s} {'MSE':>10s} {'Huber':>10s} {'Quant':>10s}  regime")
    rows = []
    for tag, fam, dname in TARGETS:
        d = repo / dname
        res = {a: arm_result(d, a) for a in ARMS}
        reg = regime(res)
        out[tag.replace('\\', '')] = {a: res[a] for a in ARMS} | {'regime': reg}

        def fmt(a):
            r = res[a]
            if r['n'] == 0:
                return '--'
            return f"{r['solved']}/{r['n']}" + (f"(T{r['medT']})" if r['medT'] else '')
        print(f"  {tag.replace(chr(92),''):16s} {fmt('scalar_mse'):>10s} "
              f"{fmt('scalar_huber_d1'):>10s} {fmt('quantile_risk_neutral'):>10s}  {reg}")

        regtex = reg if reg not in ('pending',) else '---'
        row = (f"{tag:18s} & {fam:8s} & {cell(res['scalar_mse'])} & "
               f"{cell(res['scalar_huber_d1'])} & {cell(res['quantile_risk_neutral'])} "
               f"& {regtex} \\\\")
        rows.append(row)

    (repo / 'outputs' / 'generality_numbers.json').write_text(json.dumps(out, indent=2))
    print('\n=== LaTeX rows (paste into tab:generality) ===')
    for r in rows:
        print(r)
    print(f"\nwrote {repo}/outputs/generality_numbers.json")


if __name__ == '__main__':
    main()
