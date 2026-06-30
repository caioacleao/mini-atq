#!/usr/bin/env bash
# Restore the archived evaluation data to the repo root so the analysis tools
# (which read results_*/eval at the root) can regenerate the paper's numbers.
# Idempotent: copies data/results_* -> <repo-root>/results_*.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/.." && pwd)"
n=0
for d in "$here"/results_*; do
  [ -d "$d" ] || continue
  name="$(basename "$d")"
  mkdir -p "$root/$name"
  cp -R "$d/." "$root/$name/"
  n=$((n+1))
done
echo "restored $n result dirs to $root"
echo "now run: python3 tools/paper_analysis.py && python3 tools/paper_generality.py"
