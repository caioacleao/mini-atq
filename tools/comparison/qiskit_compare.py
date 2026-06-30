#!/usr/bin/env python3
"""
Reproducible T-count comparison: Qiskit transpile vs PyZX vs AlphaTensor-Quantum / Mini AT-Q.

For each benchmark circuit we measure T-count on TWO objects:
  (A) the FULL logical circuit (<target>.qasm), with every Toffoli decomposed into
      Clifford+T (7 T-gates each). This is what Qiskit/PyZX actually optimize.
  (B) the CNOT+phase block (<target>.cnotphase.qasm) emitted by the circuit-to-tensor
      pipeline -- the phase-polynomial / signature-tensor part that AT-Q optimizes.

We run:
  * baseline T-count (as-loaded)
  * Qiskit transpile(optimization_level=3)  -> T-count
  * PyZX full_reduce + extract (+ basic_optimization, phase_teleport) -> T-count

Interpreter:
  /private/tmp/claude-501/.../scratchpad/qkvenv/bin/python
Run:
  <python> scratchpad/qiskit_compare.py
"""

import os
import sys
import tempfile
import warnings
warnings.filterwarnings("ignore")

from qiskit import QuantumCircuit, transpile
import pyzx as zx


def normalize_qasm_for_pyzx(path: str) -> str:
    """PyZX's QASM2 parser requires `include "qelib1.inc";` to appear before the
    first qreg. Some benchmark files put qreg first. Rewrite into a temp file with
    OPENQASM line, then the include, then everything else (minus duplicate include)."""
    with open(path) as f:
        lines = [ln.rstrip("\n") for ln in f]
    openqasm = None
    rest = []
    for ln in lines:
        st = ln.strip()
        if st.startswith("OPENQASM"):
            openqasm = ln
            continue
        if st.startswith("include"):
            # drop -- we re-insert one canonical include
            continue
        rest.append(ln)
    if openqasm is None:
        openqasm = "OPENQASM 2.0;"
    out = [openqasm, 'include "qelib1.inc";'] + rest
    tf = tempfile.NamedTemporaryFile("w", suffix=".qasm", delete=False)
    tf.write("\n".join(out) + "\n")
    tf.close()
    return tf.name


def zx_from_qasm(path: str):
    """Robust pyzx load: try directly, else normalize header ordering."""
    try:
        return zx.Circuit.from_qasm_file(path)
    except Exception:
        norm = normalize_qasm_for_pyzx(path)
        try:
            return zx.Circuit.from_qasm_file(norm)
        finally:
            try:
                os.unlink(norm)
            except OSError:
                pass

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH = os.environ.get(
    "ATQ_BENCH", os.path.join(_REPO, "third_party", "circuit-to-tensor", "benchmarks")
)

# (name, subdir, qubits-hint) -- subdir is arithmetic/ or applications/
TARGETS = [
    ("barenco_tof_3",    "arithmetic"),
    ("barenco_tof_4",    "arithmetic"),
    ("nc_tof_3",         "arithmetic"),
    ("gf_2pow2_mult",    "arithmetic"),
    ("gf_2pow3_mult",    "arithmetic"),
    ("cuccaro_adder_n3", "applications"),
    ("mod_5_4",          "arithmetic"),
    ("hamming_weight_n4","applications"),
    ("hamming_weight_n5","applications"),
]

# Published / our numbers from paper Table I + tonight's runs (no-gadget regime).
# These optimize the SIGNATURE TENSOR (CCZ/phase-poly part), not the full circuit.
PUBLISHED = {
    # name: (AT-Q T, Mini AT-Q T, known optimum)
    "barenco_tof_3":    (13, 14, 13),
    "nc_tof_3":         (13, 17, 13),   # Mini AT-Q quantile 14
    "gf_2pow2_mult":    (17, 17, 17),
    "gf_2pow3_mult":    (29, 36, 29),   # Mini AT-Q tonight, mm=48
    "cuccaro_adder_n3": (None, 14, None),
    "mod_5_4":          (None, None, None),
    "hamming_weight_n4":(None, None, None),
    "hamming_weight_n5":(None, None, None),
}


def tcount_qiskit(qc: QuantumCircuit) -> int:
    ops = qc.count_ops()
    return int(ops.get("t", 0) + ops.get("tdg", 0))


def qiskit_t_after_opt3(path: str):
    """Decompose Toffolis to Clifford+T, transpile opt level 3, count T."""
    qc = QuantumCircuit.from_qasm_file(path)
    # Basis that forces ccx -> Clifford+T (7 T each) so we have a real T-count target.
    basis = ["h", "cx", "t", "tdg", "s", "sdg", "z", "x", "rz"]
    tqc = transpile(qc, basis_gates=basis, optimization_level=3, seed_transpiler=0)
    return tcount_qiskit(tqc), qc.num_qubits


def baseline_t_full(path: str):
    """Baseline T of the full circuit: decompose Toffolis, opt level 0, count T."""
    qc = QuantumCircuit.from_qasm_file(path)
    basis = ["h", "cx", "t", "tdg", "s", "sdg", "z", "x", "rz"]
    tqc = transpile(qc, basis_gates=basis, optimization_level=0, seed_transpiler=0)
    return tcount_qiskit(tqc), qc.num_qubits


def pyzx_t(path: str):
    """PyZX T-count reduction. We track each route's T-count AND whether pyzx's
    verify_equality confirms the optimized circuit equals the input.

    IMPORTANT: we report the best *verified* T-count (full_reduce verifies cleanly).
    full_optimize sometimes returns a lower count but verify_equality is False, so
    we do NOT claim those as PyZX's achieved T-count -- they are listed separately
    as unverified/heuristic."""
    c = zx_from_qasm(path)
    base_t = c.tcount()
    routes = {}  # name -> (tcount, verified_bool)

    # Route 1: full_reduce on ZX graph -> extract -> basic_optimization (standard).
    g = c.to_graph()
    zx.simplify.full_reduce(g)
    c_extracted = zx.extract_circuit(g.copy())
    c_opt = zx.basic_optimization(c_extracted.to_basic_gates()).to_basic_gates()
    routes["full_reduce"] = (c_opt.tcount(), bool(c.verify_equality(c_opt)))

    # Route 2: full_optimize (stronger heuristic; equivalence often unverifiable).
    try:
        c3 = zx_from_qasm(path)
        c3o = zx.full_optimize(c3.to_basic_gates())
        routes["full_optimize"] = (c3o.tcount(), bool(c.verify_equality(c3o)))
    except Exception:
        routes["full_optimize"] = (None, None)

    verified = [t for (t, v) in routes.values() if v is True and isinstance(t, int)]
    best_verified = min(verified) if verified else None
    all_counts = [t for (t, v) in routes.values() if isinstance(t, int)]
    best_any = min(all_counts) if all_counts else None
    return base_t, best_verified, best_any, routes


def find_path(name, subdir, kind):
    """kind: 'full' -> <name>.qasm ; 'cnotphase' -> <name>.cnotphase.qasm"""
    suffix = ".qasm" if kind == "full" else ".cnotphase.qasm"
    p = os.path.join(BENCH, subdir, name, name + suffix)
    if os.path.exists(p):
        return p
    # fall back: search both subdirs
    for sd in ("arithmetic", "applications"):
        p2 = os.path.join(BENCH, sd, name, name + suffix)
        if os.path.exists(p2):
            return p2
    return None


def main():
    rows = []
    for name, subdir in TARGETS:
        full_path = find_path(name, subdir, "full")
        cp_path = find_path(name, subdir, "cnotphase")

        # ---- FULL circuit ----
        base_full, nq = baseline_t_full(full_path)
        qk_full, _ = qiskit_t_after_opt3(full_path)
        pz_base_full, pz_full, pz_any_full, pz_routes_full = pyzx_t(full_path)

        # ---- CNOTPHASE block ----
        if cp_path:
            base_cp, _ = baseline_t_full(cp_path)
            qk_cp, _ = qiskit_t_after_opt3(cp_path)
            pz_base_cp, pz_cp, pz_any_cp, pz_routes_cp = pyzx_t(cp_path)
        else:
            base_cp = qk_cp = pz_cp = pz_any_cp = None
            pz_routes_cp = {}

        atq, miniatq, opt = PUBLISHED[name]

        rows.append(dict(
            name=name, nq=nq,
            base_full=base_full, qk_full=qk_full, pz_full=pz_full, pz_any_full=pz_any_full,
            pz_routes_full=pz_routes_full,
            base_cp=base_cp, qk_cp=qk_cp, pz_cp=pz_cp, pz_any_cp=pz_any_cp,
            pz_routes_cp=pz_routes_cp,
            atq=atq, miniatq=miniatq, opt=opt,
        ))

    # ---- print ----
    def s(x):
        return "-" if x is None else str(x)

    print("\n================ FULL CIRCUIT (Toffoli -> Clifford+T) ================")
    print("PyZX = best VERIFIED (full_reduce); PyZX* = best incl. unverified full_optimize")
    print(f"{'circuit':<18}{'q':>3}{'Tbase':>7}{'Qiskit3':>9}{'PyZX':>7}{'PyZX*':>7}"
          f"{'ATQ':>6}{'MiniATQ':>9}{'opt':>6}")
    for r in rows:
        print(f"{r['name']:<18}{r['nq']:>3}{s(r['base_full']):>7}{s(r['qk_full']):>9}"
              f"{s(r['pz_full']):>7}{s(r['pz_any_full']):>7}"
              f"{s(r['atq']):>6}{s(r['miniatq']):>9}{s(r['opt']):>6}")

    print("\n================ CNOT+PHASE BLOCK (.cnotphase.qasm) ================")
    print(f"{'circuit':<18}{'Tbase':>7}{'Qiskit3':>9}{'PyZX':>7}{'PyZX*':>7}"
          f"{'ATQ':>6}{'MiniATQ':>9}{'opt':>6}")
    for r in rows:
        print(f"{r['name']:<18}{s(r['base_cp']):>7}{s(r['qk_cp']):>9}"
              f"{s(r['pz_cp']):>7}{s(r['pz_any_cp']):>7}"
              f"{s(r['atq']):>6}{s(r['miniatq']):>9}{s(r['opt']):>6}")

    print("\n================ PyZX route detail (full circuit) ================")
    for r in rows:
        print(f"{r['name']:<18} {r['pz_routes_full']}")
    print("\n================ PyZX route detail (cnotphase) ================")
    for r in rows:
        print(f"{r['name']:<18} {r['pz_routes_cp']}")

    # markdown (PyZX columns are best VERIFIED via full_reduce)
    print("\n================ MARKDOWN (verified PyZX; cnotphase = fair AT-Q comparison) ===")
    print("| circuit | qubits | T baseline (full) | Qiskit-opt3 T (full) | PyZX T (full) | "
          "T baseline (cnotphase) | PyZX T (cnotphase) | AT-Q T | Mini AT-Q T | optimum |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['name']} | {r['nq']} | {s(r['base_full'])} | {s(r['qk_full'])} | "
              f"{s(r['pz_full'])} | {s(r['base_cp'])} | {s(r['pz_cp'])} | "
              f"{s(r['atq'])} | {s(r['miniatq'])} | {s(r['opt'])} |")


if __name__ == "__main__":
    main()
