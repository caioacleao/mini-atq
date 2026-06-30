#!/usr/bin/env python3
"""Compute-grid heatmap for the Mini AT-Q paper (replaces the flat line plots).

Reads outputs/grid_numbers.json (produced by paper_grid_figures.py) and renders a
single compact heatmap: rows = search x loss, columns = torso x simulations,
cells colored by frozen-decode solve rate with median T-count printed inside.
This shows the whole 16-cell structure at a glance: the MuZero zero-block, the
Gumbel+MSE fragility island, and the all-solving Gumbel+Huber row.

Usage: python3 tools/paper_grid_heatmap.py [--out outputs] [--target barenco_tof_3]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

# Lookup keys ('muzero'/'gumbel') match grid_numbers.json; display labels name the
# engine honestly: the baseline is the standard PUCT MCTS used by AlphaZero/AT-Q.
ROWS = [('muzero', 'mse', 'AlphaZero MCTS + MSE'),
        ('muzero', 'huber', 'AlphaZero MCTS + Huber'),
        ('gumbel', 'mse', 'Gumbel + MSE'),
        ('gumbel', 'huber', 'Gumbel + Huber')]
COLS = [('small', 8), ('small', 32), ('default', 8), ('default', 32)]
COL_LABELS = ['small\n8 sim', 'small\n32 sim', 'default\n8 sim', 'default\n32 sim']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='.')
    ap.add_argument('--out', default='outputs')
    ap.add_argument('--target', default='barenco_tof_3')
    args = ap.parse_args()
    repo = Path(args.repo); out = repo / args.out
    g = json.loads((out / 'grid_numbers.json').read_text())

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    import numpy as np

    rate = np.full((len(ROWS), len(COLS)), np.nan)
    txt = [['' for _ in COLS] for _ in ROWS]
    for i, (cfg_s, cfg_l, _) in enumerate(ROWS):
        for j, (net, sims) in enumerate(COLS):
            v = g.get(f'{args.target}|{cfg_s}_{cfg_l}|sims{sims}|{net}')
            if not v:
                continue
            rate[i, j] = v['rate']
            if v['solved'] > 0:
                txt[i][j] = f"{v['solved']}/{v['n']}\nT={int(v['median_T'])}"
            else:
                txt[i][j] = f"0/{v['n']}"

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    im = ax.imshow(rate, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    for i in range(len(ROWS)):
        for j in range(len(COLS)):
            if not np.isnan(rate[i, j]):
                ax.text(j, i, txt[i][j], ha='center', va='center', fontsize=6.5,
                        color='black')
    ax.set_xticks(range(len(COLS))); ax.set_xticklabels(COL_LABELS, fontsize=6.5)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([r[2] for r in ROWS], fontsize=6.8)
    # group separators
    ax.axhline(1.5, color='white', lw=2); ax.axvline(1.5, color='white', lw=2)
    ax.set_title('Frozen-decode solve rate (median $T$)', fontsize=8)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(out / 'figures/fig_grid_heatmap.pdf')
    print(f'wrote {out}/figures/fig_grid_heatmap.pdf')


if __name__ == '__main__':
    main()
