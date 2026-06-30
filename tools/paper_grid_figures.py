#!/usr/bin/env python3
"""Compute-efficiency ("mini AT-Q") figures from the compute grid.

Parses results from run_compute_grid_task.sh
(eval_cfg-<config>_sims<N>_net-<net>_target-<t>_seed<s>.csv), aggregates at
train-seed granularity, and emits two figures:
  - fig_eff_sims.pdf : solve rate vs MCTS simulations, by search/value config
  - fig_eff_net.pdf  : solve rate vs torso size, by search/value config
plus outputs/grid_numbers.json. Re-runnable; produces whatever the data supports.

Usage: python3 tools/paper_grid_figures.py --grid_eval <dir>/eval [--out outputs]
"""
from __future__ import annotations
import argparse, csv, json, re, statistics
from collections import defaultdict
from pathlib import Path

GRID_RE = re.compile(
    r'eval_cfg-(?P<cfg>.+?)_sims(?P<sims>\d+)_net-(?P<net>.+?)_target-(?P<target>.+?)_seed(?P<seed>\d+)')

CONFIG_ORDER = ['muzero_mse', 'muzero_huber', 'gumbel_mse', 'gumbel_huber']
CONFIG_LABEL = {'muzero_mse': 'MuZero+MSE (demo)', 'muzero_huber': 'MuZero+Huber',
                'gumbel_mse': 'Gumbel+MSE', 'gumbel_huber': 'Gumbel+Huber (ours)'}
CONFIG_STYLE = {'muzero_mse': ('#c0392b', 'o', '-'), 'muzero_huber': ('#e67e22', 's', '--'),
                'gumbel_mse': ('#2980b9', '^', '--'), 'gumbel_huber': ('#27ae60', 'D', '-')}


def _bool(x): return str(x).strip().lower() in ('1', 'true', 'yes')


def read_grid(eval_dir: Path):
    rows = []
    for p in sorted(eval_dir.glob('**/*.csv')):
        m = GRID_RE.search(p.name)
        if not m:
            continue
        with p.open(newline='') as f:
            for r in csv.DictReader(f):
                if r.get('control', 'orbit') != 'orbit' or str(r.get('k', '1')) != '1':
                    continue
                rows.append(dict(cfg=m.group('cfg'), sims=int(m.group('sims')),
                                 net=m.group('net'), target=m.group('target'),
                                 seed=int(m.group('seed')),
                                 solved=_bool(r.get('solved')), cost=float(r.get('cost', 'nan'))))
    return rows


def aggregate(rows):
    # train-seed level: a (cfg,sims,net,target,seed) solved if any eval row solved
    by_seed = defaultdict(list)
    for r in rows:
        by_seed[(r['cfg'], r['sims'], r['net'], r['target'], r['seed'])].append(r)
    cell = {}
    for k, v in by_seed.items():
        costs = [x['cost'] for x in v if x['solved']]
        cell[k] = (len(costs) > 0, statistics.median(costs) if costs else None)
    # per (cfg,sims,net,target): solve rate over seeds, median T
    by_cond = defaultdict(list)
    for (cfg, sims, net, target, seed), (sol, med) in cell.items():
        by_cond[(cfg, sims, net, target)].append((sol, med))
    out = {}
    for k, v in by_cond.items():
        n = len(v); s = sum(1 for sol, _ in v if sol)
        meds = [m for _, m in v if m is not None]
        out[k] = dict(n=n, solved=s, rate=s / n,
                      median_T=statistics.median(meds) if meds else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid_eval', required=True)
    ap.add_argument('--out', default='outputs')
    ap.add_argument('--target', default=None, help='restrict to one target (default: first found)')
    args = ap.parse_args()
    out = Path(args.out); (out / 'figures').mkdir(parents=True, exist_ok=True)

    rows = read_grid(Path(args.grid_eval))
    if not rows:
        print('no grid eval rows found'); return
    agg = aggregate(rows)
    targets = sorted({k[3] for k in agg})
    target = args.target or targets[0]
    print(f'targets present: {targets}; using {target}')

    # serialize
    grid_numbers = {}
    for (cfg, sims, net, tgt), v in sorted(agg.items()):
        grid_numbers[f'{tgt}|{cfg}|sims{sims}|{net}'] = v
    (out / 'grid_numbers.json').write_text(json.dumps(grid_numbers, indent=2, default=str))

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    except Exception as e:
        print(f'matplotlib unavailable: {e}'); return

    sims_levels = sorted({k[1] for k in agg if k[3] == target})
    nets = sorted({k[2] for k in agg if k[3] == target})

    # Fig: solve rate vs sims, at the largest net, lines by config
    big_net = 'default' if 'default' in nets else (nets[-1] if nets else None)
    if big_net and len(sims_levels) >= 1:
        fig, ax = plt.subplots(figsize=(3.4, 2.3))
        for cfg in CONFIG_ORDER:
            xs, ys = [], []
            for sims in sims_levels:
                v = agg.get((cfg, sims, big_net, target))
                if v:
                    xs.append(sims); ys.append(v['rate'] * 100)
            if xs:
                c, mk, ls = CONFIG_STYLE.get(cfg, ('gray', 'o', '-'))
                ax.plot(xs, ys, marker=mk, ls=ls, color=c, label=CONFIG_LABEL.get(cfg, cfg), ms=5)
        ax.set_xlabel('MCTS simulations', fontsize=8); ax.set_ylabel('Solve rate (%)', fontsize=8)
        ax.set_title(f'{target} (net={big_net})', fontsize=8); ax.set_ylim(-5, 108)
        ax.tick_params(labelsize=7); ax.legend(fontsize=6, loc='best')
        fig.tight_layout(); fig.savefig(out / 'figures/fig_eff_sims.pdf'); print('wrote fig_eff_sims.pdf')

    # Fig: solve rate vs net size, at the largest sims, lines by config
    big_sims = max(sims_levels) if sims_levels else None
    if big_sims and len(nets) >= 1:
        fig, ax = plt.subplots(figsize=(3.4, 2.3))
        netx = {n: i for i, n in enumerate(nets)}
        for cfg in CONFIG_ORDER:
            xs, ys = [], []
            for n in nets:
                v = agg.get((cfg, big_sims, n, target))
                if v:
                    xs.append(netx[n]); ys.append(v['rate'] * 100)
            if xs:
                c, mk, ls = CONFIG_STYLE.get(cfg, ('gray', 'o', '-'))
                ax.plot(xs, ys, marker=mk, ls=ls, color=c, label=CONFIG_LABEL.get(cfg, cfg), ms=5)
        ax.set_xticks(list(netx.values())); ax.set_xticklabels(nets, fontsize=7)
        ax.set_xlabel('Torso size', fontsize=8); ax.set_ylabel('Solve rate (%)', fontsize=8)
        ax.set_title(f'{target} (sims={big_sims})', fontsize=8); ax.set_ylim(-5, 108)
        ax.tick_params(labelsize=7); ax.legend(fontsize=6, loc='best')
        fig.tight_layout(); fig.savefig(out / 'figures/fig_eff_net.pdf'); print('wrote fig_eff_net.pdf')

    print(f'wrote {out}/grid_numbers.json')


if __name__ == '__main__':
    main()
