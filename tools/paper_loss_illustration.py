#!/usr/bin/env python3
"""Conceptual loss-shape illustration for the Mini AT-Q paper.

Visualizes why a robust value loss matters: under squared error the loss (and
its gradient) on a rare, large-magnitude failure grows without bound, so a few
deep failures dominate training; Huber and pinball (quantile) losses grow only
linearly in the tail, bounding the influence of those outliers.

Two panels (single-column figure):
  (a) loss vs prediction error
  (b) |gradient| (influence) vs prediction error
in units of the Huber transition scale delta.

Usage: python3 tools/paper_loss_illustration.py [--out outputs]
"""
from __future__ import annotations
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='outputs')
    args = ap.parse_args()
    out = Path(args.out); (out / 'figures').mkdir(parents=True, exist_ok=True)

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    import numpy as np

    e = np.linspace(-6, 6, 1201)
    d = 1.0  # Huber transition scale
    mse = 0.5 * e**2
    huber = np.where(np.abs(e) <= d, 0.5 * e**2, d * (np.abs(e) - 0.5 * d))
    pinball = 0.5 * np.abs(e)            # tau=0.5 quantile (pinball) loss
    g_mse = np.abs(e)
    g_huber = np.minimum(np.abs(e), d)
    g_pin = np.full_like(e, 0.5)

    C = {'mse': '#c0392b', 'huber': '#27ae60', 'pin': '#2980b9'}
    figT, (a, b) = plt.subplots(1, 2, figsize=(3.5, 1.85))

    a.plot(e, mse, color=C['mse'], lw=1.8, label='MSE')
    a.plot(e, huber, color=C['huber'], lw=1.8, label='Huber')
    a.plot(e, pinball, color=C['pin'], lw=1.6, ls='--', label='Pinball')
    a.axvspan(-6, -3, color='gray', alpha=0.10)
    a.axvspan(3, 6, color='gray', alpha=0.10)
    a.text(-4.5, 1.3, 'rare large\nfailures', ha='center', va='bottom',
           fontsize=5.4, color='dimgray')
    a.set_xlabel('prediction error  $e$', fontsize=7.5)
    a.set_ylabel('loss', fontsize=7.5)
    a.set_ylim(0, 18); a.tick_params(labelsize=6.5)
    a.legend(fontsize=5.8, loc='upper center', framealpha=0.9, handlelength=1.3)
    a.set_title('(a) loss', fontsize=8)

    b.plot(e, g_mse, color=C['mse'], lw=1.8, label='MSE')
    b.plot(e, g_huber, color=C['huber'], lw=1.8, label='Huber')
    b.plot(e, g_pin, color=C['pin'], lw=1.6, ls='--', label='Pinball')
    b.axhline(d, color='gray', lw=0.6, ls=':')
    b.text(-5.8, d + 0.15, r'bounded ($\delta$)', fontsize=5.6, color='gray',
           va='bottom', ha='left')
    b.set_xlabel('prediction error  $e$', fontsize=7.5)
    b.set_ylabel(r'$|\partial\,\mathrm{loss}/\partial e|$', fontsize=7.5)
    b.set_ylim(0, 6); b.tick_params(labelsize=6.5)
    b.set_title('(b) influence', fontsize=8)

    figT.tight_layout(w_pad=1.2)
    figT.savefig(out / 'figures/fig_robust_loss.pdf')
    print(f'wrote {out}/figures/fig_robust_loss.pdf')


if __name__ == '__main__':
    main()
