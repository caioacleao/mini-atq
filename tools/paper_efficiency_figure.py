#!/usr/bin/env python3
"""Wall-clock timing data for the Mini AT-Q paper (Table 'tab:timing').

Combines per-config training time (median seconds/step from the compute-grid logs)
and decode time (median seconds from the eval CSVs) with the frozen-decode solve
rate (grid_numbers.json), and writes outputs/timing_numbers.json. The story: Mini
AT-Q (Gumbel+Huber, small torso, 8 sims) trains in minutes and solves, whereas the
AlphaZero-style MCTS engine never solves even after far more compute.

Usage: python3 tools/paper_efficiency_figure.py [--out outputs]
"""
from __future__ import annotations
import argparse, glob, json, re, statistics
from collections import defaultdict
from pathlib import Path

TIME = re.compile(r'Time taken:\s*([\d.]+)\s*seconds/step')
CFG = re.compile(r'cfg-(?P<cfg>.+?)_sims(?P<sims>\d+)_net-(?P<net>.+?)_target')
STEPS = 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='.')
    ap.add_argument('--out', default='outputs')
    ap.add_argument('--target', default='barenco_tof_3')
    args = ap.parse_args()
    repo = Path(args.repo); out = repo / args.out
    g = json.loads((out / 'grid_numbers.json').read_text())

    import csv
    DEC = re.compile(r'eval_cfg-(?P<cfg>.+?)_sims(?P<sims>\d+)_net-(?P<net>.+?)_target')
    tstep = defaultdict(list)
    dec = defaultdict(list)
    for lp in glob.glob(str(repo / 'results_compute_grid_20260627/logs/*.log')):
        m = CFG.search(lp)
        if not m:
            continue
        key = (m.group('cfg'), int(m.group('sims')), m.group('net'))
        for line in Path(lp).read_text(errors='ignore').splitlines():
            t = TIME.search(line)
            if t:
                tstep[key].append(float(t.group(1)))
    for p in glob.glob(str(repo / 'results_compute_grid_20260627/eval/**/*.csv'), recursive=True):
        m = DEC.search(p)
        if not m:
            continue
        key = (m.group('cfg'), int(m.group('sims')), m.group('net'))
        for r in csv.DictReader(open(p)):
            if r.get('control', 'orbit') != 'orbit' or str(r.get('k', '1')) != '1':
                continue
            try:
                dec[key].append(float(r['seconds']))
            except Exception:
                pass

    # timing_numbers.json: the reproducible source for the wall-clock table
    timing = {}
    for (cfg, sims, net), secs in tstep.items():
        gv = g.get(f'{args.target}|{cfg}|sims{sims}|{net}')
        ss = statistics.median(secs)
        timing[f'{cfg}|sims{sims}|{net}'] = dict(
            s_per_step=round(ss, 2), train_min=round(ss * STEPS / 60, 1),
            decode_s=round(statistics.median(dec[(cfg, sims, net)]), 2) if dec.get((cfg, sims, net)) else None,
            solved=gv['solved'] if gv else None, n=gv['n'] if gv else None,
            median_T=gv['median_T'] if gv else None)
    (out / 'timing_numbers.json').write_text(json.dumps(timing, indent=2, default=str))
    print(f'wrote {out}/timing_numbers.json')

    # report the table rows (the manuscript's Table 'tab:timing')
    def row(cfg, sims, net):
        v = tstep.get((cfg, sims, net))
        if not v:
            return None
        ss = statistics.median(v)
        gv = g.get(f'{args.target}|{cfg}|sims{sims}|{net}', {})
        dd = statistics.median(dec[(cfg, sims, net)]) if dec.get((cfg, sims, net)) else None
        return f"{cfg} {net}/{sims}: {ss:.2f} s/step, {ss*STEPS/60:.0f} min, " \
               f"decode {dd:.2f} s, solved {gv.get('solved')}/{gv.get('n')} T={gv.get('median_T')}"
    print('=== WALL-CLOCK TABLE (barenco_tof_3) ===')
    for cfg, sims, net in [('gumbel_huber', 8, 'small'), ('gumbel_huber', 32, 'default'),
                           ('muzero_huber', 8, 'small'), ('muzero_huber', 32, 'default')]:
        print('  ' + str(row(cfg, sims, net)))


if __name__ == '__main__':
    main()
