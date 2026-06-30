#!/usr/bin/env python3
"""Richer mechanism figure for the Mini AT-Q paper (replaces the binary bar).

Two panels, both carrying continuous data the binary solve-rate threw away:
  (a) Training dynamics: running-average return vs step (per value head).
      MSE / narrow-categorical plateau near the failure floor; robust heads
      descend toward a valid decomposition.
  (b) Frozen-decode decoded T-count, one point per train seed (jittered), with
      the move cap (unsolved) and the published no-gadget optimum as reference
      lines. Shows the magnitude of the gap (cap vs ~14) AND the per-seed spread.

Also prints the enriched per-arm table (final return, solved seeds, decoded-T
distribution) the manuscript's Table cites.

Usage: python3 tools/paper_mechanism_figure.py [--target barenco_tof_3]
"""
from __future__ import annotations
import argparse, csv, glob, json, re, statistics
from collections import defaultdict, Counter
from pathlib import Path

STEP_RE = re.compile(r'Step:\s*(\d+)\s*\.\..*Running Average Returns:\s*\[\s*([-\d.eE+]+)\s*\]')
ARM_RE = re.compile(r'eval_arm-(?P<arm>.+?)_target-(?P<target>.+?)_seed(?P<seed>\d+)')
CAP = 72.0
OPTIMUM = 13  # published no-gadget T-count for barenco_tof_3

# (key, label, color, marker)  -- failures first, then robust successes
ARMS = [
    ('scalar_mse',            'MSE (demo)',           '#c0392b', 'X'),
    ('categorical_61',        'Categ. $[-60,0]$',     '#e67e22', 'P'),
    ('scalar_huber_d1',       'Huber (Mini AT-Q)',    '#27ae60', 'o'),
    ('categorical_wide',      'Categ. $[-160,0]$',    '#8e44ad', 'D'),
    ('quantile_risk_neutral', 'Quantile (neutral)',   '#2980b9', '^'),
    ('quantile_q075',         'Quantile (q0.75)',     '#16a085', 'v'),
]


def _bool(x): return str(x).strip().lower() in ('1', 'true', 'yes')


def parse_curves(repo, target):
    # Restrict to the pinned value-control run (+ catwide for the wide-categorical
    # arm), matching parse_decoded. Globbing results_* would also pull the
    # longbudget (3000-step) and independent-hardware (1000-step) scalar_mse seeds, contaminating
    # the step-1000 mean used for the "final return" column (audit 2026-06-27).
    DIRS = ['results_a2_value_controls_20260626_163106', 'results_catwide_barenco3']
    curves = defaultdict(lambda: defaultdict(list))
    for arm, *_ in ARMS:
        for d in DIRS:
            for lp in glob.glob(str(repo / d / 'logs' / target / f'arm-{arm}' / '*.log')):
                for line in Path(lp).read_text(errors='ignore').splitlines():
                    m = STEP_RE.search(line)
                    if m:
                        curves[arm][int(m.group(1))].append(float(m.group(2)))
    return curves


def parse_decoded(repo, target):
    """Per arm -> {seed: (decoded_T, distinct_factors, solved)}.

    decoded_T = min solved cost (else CAP); distinct_factors = number of unique
    actions in the (deterministic) frozen decode -- 1 for the degenerate
    single-action MSE collapse, ~T for a real decomposition.
    """
    seed_solved = defaultdict(lambda: defaultdict(bool))
    seed_cost = defaultdict(lambda: defaultdict(list))
    seed_distinct = defaultdict(dict)
    for root in ['results_a2_value_controls_20260626_163106', 'results_catwide_barenco3']:
        for p in glob.glob(str(repo / f'{root}/eval/**/*.csv'), recursive=True):
            m = ARM_RE.search(p)
            for row in csv.DictReader(open(p)):
                if row.get('control', 'orbit') != 'orbit' or str(row.get('k', '1')) != '1':
                    continue
                arm = m.group('arm') if m else row.get('arm')
                tgt = m.group('target') if m else row.get('target')
                seed = m.group('seed') if m else row.get('train_seed')
                if target.split('_')[0] not in (tgt or ''):
                    continue
                seed_cost[arm][seed].append(float(row['cost']))
                if _bool(row['solved']):
                    seed_solved[arm][seed] = True
                if seed not in seed_distinct[arm]:
                    seq = row.get('canonical_action_sequence') or row.get('action_sequence') or ''
                    seed_distinct[arm][seed] = len(set(seq.split()))
    decoded = {}
    for arm in seed_cost:
        decoded[arm] = {s: (min(seed_cost[arm][s]) if seed_solved[arm][s] else CAP,
                            seed_distinct[arm].get(s, 0), seed_solved[arm][s])
                        for s in seed_cost[arm]}
    return decoded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='.')
    ap.add_argument('--target', default='barenco_tof_3')
    ap.add_argument('--out', default='outputs')
    args = ap.parse_args()
    repo = Path(args.repo); out = repo / args.out
    (out / 'figures').mkdir(parents=True, exist_ok=True)

    curves = parse_curves(repo, args.target)
    decoded = parse_decoded(repo, args.target)

    # ---- enriched table numbers ----
    table = {}
    print(f'=== ENRICHED MECHANISM TABLE ({args.target}) ===')
    print(f'{"arm":24s} {"final_ret":>9s} {"solved":>7s}  decoded-T (count)')
    for arm, *_ in ARMS:
        steps = sorted(curves.get(arm, {}))
        final = round(statistics.mean(curves[arm][steps[-1]]), 1) if steps else None
        d = decoded.get(arm, {})
        solved = sum(1 for s in d if d[s][2])
        n = len(d)
        solved_T = sorted(int(d[s][0]) for s in d if d[s][2])
        dist = dict(sorted(Counter(solved_T).items()))
        table[arm] = dict(final_return=final, solved=solved, n=n,
                          decoded_T=solved_T, dist=dist,
                          median_T=statistics.median(solved_T) if solved_T else None)
        ds = ', '.join(f'{t}$\\times${c}' if c > 1 else f'{t}' for t, c in dist.items()) or 'cap'
        print(f'  {arm:22s} {str(final):>9s} {solved:>4d}/{n}  {ds}')
    (out / 'mechanism_table.json').write_text(json.dumps(table, indent=2))

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f'matplotlib unavailable: {e}'); return

    # --- Figure (a): training dynamics (single axis; legend OUTSIDE, below) ---
    figA, axL = plt.subplots(figsize=(3.4, 2.6))
    for arm, label, color, mk in ARMS:
        if arm not in curves:
            continue
        steps = sorted(curves[arm])
        means = [statistics.mean(curves[arm][s]) for s in steps]
        axL.plot(steps, means, color=color, lw=1.7, label=label)
    axL.set_xlim(0, 1050)
    axL.set_xlabel('Training step', fontsize=8)
    axL.set_ylabel('Running average return', fontsize=8)
    axL.tick_params(labelsize=7)
    axL.set_title('(a) Training dynamics', fontsize=8.5)
    axL.legend(loc='upper center', bbox_to_anchor=(0.5, -0.26), ncol=3,
               fontsize=5.7, frameon=False, columnspacing=1.0, handlelength=1.4,
               handletextpad=0.4)
    figA.tight_layout()
    figA.savefig(out / 'figures/fig_mech_curves.pdf')
    print(f'  wrote {out}/figures/fig_mech_curves.pdf')

    # --- Figure (b): decode degeneracy -- distinct factors used (single axis) ---
    figB, axR = plt.subplots(figsize=(3.4, 2.6))
    rng = np.random.default_rng(0)
    for i, (arm, label, color, mk) in enumerate(ARMS):
        d = decoded.get(arm, {})
        for s in d:
            _cost, distinct, solved = d[s]
            x = i + (rng.random() - 0.5) * 0.30
            if solved:
                axR.scatter([x], [distinct], color=color, marker=mk, s=36, zorder=3,
                            edgecolors='white', linewidths=0.4)
            else:
                axR.scatter([x], [distinct], c=color, marker='x', s=46, zorder=3,
                            linewidths=1.6)
    axR.axhline(OPTIMUM, color='green', lw=0.8, ls='--', alpha=0.7)
    axR.text(-0.4, OPTIMUM + 0.4, f'$\\approx$ valid decomposition ({OPTIMUM})',
             va='bottom', ha='left', fontsize=6, color='green')
    axR.axhline(1, color='gray', lw=0.8, ls=':', alpha=0.8)
    axR.text(-0.4, 1.5, 'single action (degenerate)', va='bottom', ha='left',
             fontsize=6, color='gray')
    axR.set_xticks(range(len(ARMS)))
    axR.set_xticklabels(['MSE', 'C[-60,0]', 'Huber', 'C[-160,0]', 'Q-neut', 'Q-0.75'],
                        fontsize=6.3, rotation=30, ha='right')
    axR.set_xlim(-0.6, len(ARMS) - 0.4)
    axR.set_ylabel('Distinct factors in decode', fontsize=8)
    axR.set_ylim(-1, 22)
    axR.tick_params(labelsize=7)
    axR.set_title('(b) Decode degeneracy', fontsize=8.5)
    figB.tight_layout()
    figB.savefig(out / 'figures/fig_mech_strip.pdf')
    print(f'  wrote {out}/figures/fig_mech_strip.pdf')


if __name__ == '__main__':
    main()
