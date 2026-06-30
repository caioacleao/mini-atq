#!/usr/bin/env python3
"""Builds imported-target manifests for curated ATQ benchmarks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from alphatensor_quantum.src import tensors
from tools import import_qasm_targets


ATQ_CORE_N14_TARGETS = (
    'arithmetic/mod_5_4/mod_5_4.tensor.npy',
    'arithmetic/gf_2pow2_mult/gf_2pow2_mult.tensor.npy',
    'arithmetic/nc_tof_3/nc_tof_3.tensor.npy',
    'arithmetic/barenco_tof_3/barenco_tof_3.tensor.npy',
    'applications/cuccaro_adder_n3/cuccaro_adder_n3.tensor.npy',
    'arithmetic/gf_2pow3_mult/gf_2pow3_mult.tensor.npy',
    'applications/hamming_weight_n4/hamming_weight_n4.tensor.npy',
    'applications/hamming_weight_n5/hamming_weight_n5.tensor.npy',
    'arithmetic/mod_mult_55/mod_mult_55.tensor.npy',
    'arithmetic/nc_tof_4/nc_tof_4.tensor.npy',
    'arithmetic/gf_2pow4_mult/gf_2pow4_mult_comp1.tensor.npy',
    'applications/cuccaro_adder_n4/cuccaro_adder_n4.tensor.npy',
    'applications/hamming_weight_n6/hamming_weight_n6.tensor.npy',
    'applications/hamming_weight_n7/hamming_weight_n7.tensor.npy',
    'arithmetic/barenco_tof_4/barenco_tof_4.tensor.npy',
    'arithmetic/vbe_adder_3/vbe_adder_3.tensor.npy',
)


def _repo_root() -> Path:
  return Path(__file__).resolve().parents[1]


def _benchmark_root() -> Path:
  return _repo_root() / 'third_party' / 'circuit-to-tensor' / 'benchmarks'


def _target_name(path: Path) -> str:
  name = path.name
  if name.endswith('.tensor.npy'):
    return name[:-len('.tensor.npy')]
  return path.stem


def _relative(path: Path, base: Path) -> str:
  return os.path.relpath(path.resolve(), base.resolve())


def _optional_relative(manifest_dir: Path, path: Path) -> str | None:
  if not path.exists():
    return None
  return _relative(path, manifest_dir)


def _related_path(tensor_path: Path, suffix: str) -> Path:
  base = tensor_path.name[:-len('.tensor.npy')]
  return tensor_path.with_name(f'{base}{suffix}')


def _manifest_entry(manifest_dir: Path, tensor_path: Path) -> dict[str, object]:
  tensor = tensors.validate_signature_tensor(np.load(tensor_path), tensor_path)
  entry = {
      'name': _target_name(tensor_path),
      'tensor_path': _relative(tensor_path, manifest_dir),
      'tensor_size': int(tensor.shape[0]),
      'tensor_entries': int(tensor.sum()),
  }
  related_paths = {
      'matrix_path': _related_path(tensor_path, '.matrix.npy'),
      'mapping_path': _related_path(tensor_path, '.mapping.txt'),
      'block_qasm_path': _related_path(tensor_path, '.cnotphase.qasm'),
      'block_qc_path': _related_path(tensor_path, '.cnotphase.qc'),
  }
  for key, path in related_paths.items():
    value = _optional_relative(manifest_dir, path)
    if value is not None:
      entry[key] = value
  return entry


def _write_manifest(output_dir: Path, tensor_path: Path) -> dict[str, object]:
  target_name = _target_name(tensor_path)
  manifest_dir = output_dir / target_name
  manifest_dir.mkdir(parents=True, exist_ok=True)
  entry = _manifest_entry(manifest_dir, tensor_path)
  manifest = {
      'format': import_qasm_targets.MANIFEST_FORMAT,
      'source': 'circuit-to-tensor-benchmark',
      'source_repo': import_qasm_targets.CIRCUIT_TO_TENSOR_REPO,
      'targets': [entry],
  }
  manifest_path = manifest_dir / 'targets.json'
  with manifest_path.open('w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2)
    f.write('\n')
  return {
      'name': target_name,
      'manifest_path': _relative(manifest_path, output_dir),
      'tensor_path': str(tensor_path),
      'tensor_size': entry['tensor_size'],
      'tensor_entries': entry['tensor_entries'],
      'num_actions': 2 ** int(entry['tensor_size']) - 1,
  }


def _parse_csv(text: str) -> list[str]:
  return [item.strip() for item in text.split(',') if item.strip()]


def _write_plan(
    output_dir: Path,
    rows: list[dict[str, object]],
    seeds: list[str],
    policies: list[str],
    expected_logs_per_target: int | None,
) -> Path:
  expected_logs = (
      expected_logs_per_target
      if expected_logs_per_target is not None
      else len(seeds) * len(policies) if seeds and policies else 0
  )
  rows = [dict(row, expected_logs=expected_logs) for row in rows]
  plan = {
      'format': 'atq_benchmark_plan_v1',
      'benchmark': 'atq_core_n14',
      'description': (
          'Curated N<=14 ATQ/circuit-to-tensor benchmark spanning arithmetic '
          'and application-centered Clifford+T targets.'
      ),
      'seeds': seeds,
      'policies': policies,
      'num_seeds': len(seeds),
      'expected_logs_per_target': expected_logs,
      'targets': rows,
  }
  plan_path = output_dir / 'benchmark_plan.json'
  with plan_path.open('w', encoding='utf-8') as f:
    json.dump(plan, f, indent=2)
    f.write('\n')
  return plan_path


def parse_args(argv: list[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--benchmark',
      default='atq_core_n14',
      choices=['atq_core_n14'],
      help='Curated benchmark to materialize.',
  )
  parser.add_argument(
      '--output_dir',
      required=True,
      type=Path,
      help='Directory that will receive one manifest per target.',
  )
  parser.add_argument(
      '--seeds',
      default='',
      help='Optional comma-separated seeds for dashboard expected counts.',
  )
  parser.add_argument(
      '--policies',
      default='muzero,gumbel',
      help='Optional comma-separated policies for dashboard expected counts.',
  )
  parser.add_argument(
      '--expected_logs_per_target',
      type=int,
      help='Optional explicit expected log count per target for dashboards.',
  )
  parser.add_argument(
      '--targets',
      default='',
      help='Optional comma-separated target names to include.',
  )
  return parser.parse_args(argv)


def main(argv: list[str]) -> int:
  args = parse_args(argv)
  root = _benchmark_root()
  selected_targets = set(_parse_csv(args.targets))
  args.output_dir.mkdir(parents=True, exist_ok=True)
  rows = []
  for relative_path in ATQ_CORE_N14_TARGETS:
    tensor_path = root / relative_path
    target_name = _target_name(tensor_path)
    if selected_targets and target_name not in selected_targets:
      continue
    if not tensor_path.exists():
      raise FileNotFoundError(f'Missing benchmark tensor: {tensor_path}')
    rows.append(_write_manifest(args.output_dir, tensor_path))
  if selected_targets and len(rows) != len(selected_targets):
    found = {str(row['name']) for row in rows}
    missing = sorted(selected_targets - found)
    raise ValueError(f'Unknown benchmark targets: {missing}')
  plan_path = _write_plan(
      args.output_dir,
      rows,
      _parse_csv(args.seeds),
      _parse_csv(args.policies),
      args.expected_logs_per_target,
  )
  print(f'Wrote {len(rows)} target manifests to {args.output_dir}')
  print(f'Wrote benchmark plan: {plan_path}')
  for row in rows:
    print(
        f"{row['name']}: N={row['tensor_size']} "
        f"actions={row['num_actions']} manifest={row['manifest_path']}"
    )
  return 0


if __name__ == '__main__':
  raise SystemExit(main(sys.argv[1:]))
