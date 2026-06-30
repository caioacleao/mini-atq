# Mini AT-Q

**Predicting value-head collapse before training in AlphaTensor-Quantum.**

This repository is the reproducibility companion to the paper:

> Caio Almeida Carneiro Leão and João Victor Moreira Cardoso,
> *Mini AT-Q: Predicting Value-Head Collapse Before Training in AlphaTensor-Quantum.*

---

## What the paper shows 

AlphaTensor-Quantum (AT-Q) minimizes a circuit's **T-count** by decomposing its
GF(2) signature tensor with an AlphaZero-style RL agent. The publicly released
*demo* is fragile: on some circuits it solves 0/5. We trace this to a
**value-head collapse** — the search has heavy-tailed returns (rare, large-magnitude
failures), and the demo's squared-error value head collapses to a near-constant,
giving a degenerate single-action policy. We (i) propose a **training-free, per-target
statistic** that *predicts which circuits collapse before any training*; (ii) **fix**
the collapse with a value head that accommodates the tail (a one-line Huber loss, or
adequate distributional support); and (iii) make the agent compute-light by swapping
the AlphaZero-style MCTS for a **Gumbel** search. We package the result as **Mini AT-Q**.

---

## Repository layout

```
alphatensor_quantum/   Modified copy of DeepMind's AT-Q demo agent (Apache-2.0).
                       Adds the value-head/loss variants (Huber, categorical,
                       quantile, symlog), the Gumbel engine, and the eval harness.
                       See NOTICE for the list of modifications.
tools/                 Reproduction scripts:
                         paper_analysis.py            mechanism table + tests
                         paper_generality.py          generality table (Table IV)
                         paper_hardening_numbers.py   delta-sweep, symlog, support, mod_5_4
                         paper_tail_statistic.py      the tail predictor + Fig. 4
                         paper_mechanism_figure.py    mechanism curves/strip
                         paper_grid_figures.py        compute-grid numbers
                         paper_grid_heatmap.py        compute grid (Fig. 5)
                         paper_efficiency_figure.py   timing figure
                         paper_loss_illustration.py   loss/influence figure
                         build_benchmark_manifests.py target manifests
                         comparison/qiskit_compare.py classical ZX (PyZX) baseline
                         run_a2_value_controls_task.sh per-job training harness
data/                  Frozen-decode evaluation CSVs (the unit of record) +
                       restore.sh + README (the data -> table/figure mapping).
outputs/               Regenerated numbers and figures land here (gitignored).
notebooks/             reproduce.ipynb — a guided, end-to-end reproduction run.
third_party/           circuit-to-tensor (git submodule) — target-tensor encoding.
```

`data/` is the unit of record: each CSV row is one frozen-decode attempt
(`solved`, T-count, moves). The paper's numbers regenerate from these CSVs via the
`tools/paper_*.py` scripts (regenerated artifacts go to `outputs/`). See
[`data/README.md`](data/README.md) for the data → table/figure mapping.

---

## Quick start

```bash
git clone --recurse-submodules <this-repo-url>
cd mini-atq

# Analysis / figure layer (regenerate numbers and figures from data/):
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For from-scratch **training** (a GPU is recommended), install the pinned agent stack:

```bash
pip install -r alphatensor_quantum/src/demo/requirements.txt
```

If you cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

---

## Reproduce the paper

### A) Verify the numbers and figures from the released data (fast, no GPU)

```bash
bash data/restore.sh                  # stage data/results_* at the repo root
python tools/paper_analysis.py        # -> outputs/numbers.json (mechanism, isolation)
python tools/paper_generality.py      # -> outputs/generality_numbers.json (Table IV)
python tools/paper_hardening_numbers.py  # -> outputs/hardening_numbers.json
python tools/paper_tail_statistic.py  # -> outputs/tail_statistic.json + outputs/figures/
python tools/paper_mechanism_figure.py
python tools/paper_loss_illustration.py
python tools/paper_grid_figures.py --grid_eval results_compute_grid_20260627/eval
python tools/paper_grid_heatmap.py    # compute-grid heatmap (Fig. 5)
```

Every script writes its regenerated numbers and figures under `outputs/`. The notebook
`notebooks/reproduce.ipynb` runs this pipeline end to end and displays the results.

### B) Train from scratch (reproduces the data in A)

`tools/run_a2_value_controls_task.sh` is the per-job training body: it runs
`alphatensor_quantum.src.demo.run_demo` for a chosen value-head/loss arm and target,
then evaluates the checkpoint under frozen decode. The paper's sweeps (value-head
variants, delta-sweep, support sweep, compute grid, generality families) are
orchestrations of this body across arms, seeds, and targets. Typical settings:
`16` self-play games/step, `32` MCTS simulations, `1000` training steps, Gumbel with
`8` considered actions, no gadgets — about an order of magnitude fewer simulations
per move than the published AT-Q configuration, on a single GPU.

### C) PyZX reference (optional)

```bash
pip install qiskit pyzx
python tools/comparison/qiskit_compare.py   # PyZX full_reduce T-counts (Table I)
```

---

## Attribution and license

This repository is released under the **Apache License 2.0** (see `LICENSE`).

The `alphatensor_quantum/` directory is a **modified** copy of Google DeepMind's
[AlphaTensor-Quantum](https://github.com/google-deepmind/alphatensor_quantum)
(Copyright 2025 Google LLC, Apache-2.0); its original license is kept at
`alphatensor_quantum/LICENSE`, and the modifications are listed in `NOTICE`.
The `third_party/circuit-to-tensor` submodule is a separate work under its own license.
