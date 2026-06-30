#!/usr/bin/env python3
"""Converts OpenQASM circuits into ATQ imported-target manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from alphatensor_quantum.src import tensors


MANIFEST_FORMAT = 'atq_imported_targets_v1'
CIRCUIT_TO_TENSOR_REPO = 'https://github.com/tlaakkonen/circuit-to-tensor'


def _repo_root() -> Path:
  return Path(__file__).resolve().parents[1]


def _default_submodule_binary() -> Path:
  return (
      _repo_root()
      / 'third_party'
      / 'circuit-to-tensor'
      / 'target'
      / 'release'
      / 'circuit-to-tensor'
  )


def _resolve_command(command: str | None) -> list[str]:
  if command:
    return command.split()
  binary = shutil.which('circuit-to-tensor')
  if binary:
    return [binary]
  submodule_binary = _default_submodule_binary()
  if submodule_binary.exists():
    return [str(submodule_binary)]
  cargo = shutil.which('cargo')
  manifest = _repo_root() / 'third_party' / 'circuit-to-tensor' / 'Cargo.toml'
  if cargo and manifest.exists():
    return [cargo, 'run', '--release', '--manifest-path', str(manifest), '--']
  raise FileNotFoundError(
      'Could not find circuit-to-tensor. Install it with '
      f'`cargo install --git {CIRCUIT_TO_TENSOR_REPO}` or build the imported '
      'submodule with `cd third_party/circuit-to-tensor && cargo build --release`.'
  )


def _compile_qasm(args: argparse.Namespace) -> None:
  command = _resolve_command(args.circuit_to_tensor_cmd)
  compile_args = command + ['compile', '-e', args.emit]
  if args.qubits is not None:
    compile_args += ['--qubits', str(args.qubits)]
  if args.ancilla is not None:
    compile_args += ['--ancilla', str(args.ancilla)]
  if args.zx_preopt:
    compile_args.append('--zx-preopt')
  compile_args += ['--split-iters', str(args.split_iters)]
  compile_args += [str(args.output_dir)]
  compile_args += [str(path) for path in args.qasm_files]
  subprocess.run(compile_args, check=True)


def _relative(path: Path, base: Path) -> str:
  return path.resolve().relative_to(base.resolve()).as_posix()


def _target_name(tensor_path: Path) -> str:
  name = tensor_path.name
  if name.endswith('.tensor.npy'):
    return name[:-len('.tensor.npy')]
  return tensor_path.stem


def _optional_relative(output_dir: Path, path: Path) -> str | None:
  return _relative(path, output_dir) if path.exists() else None


def _manifest_entry(output_dir: Path, tensor_path: Path) -> dict[str, object]:
  tensor = tensors.validate_signature_tensor(np.load(tensor_path), tensor_path)
  base = tensor_path.name[:-len('.tensor.npy')]
  entry = {
      'name': _target_name(tensor_path),
      'tensor_path': _relative(tensor_path, output_dir),
      'tensor_size': int(tensor.shape[0]),
      'tensor_entries': int(tensor.sum()),
  }
  related_paths = {
      'matrix_path': output_dir / f'{base}.matrix.npy',
      'mapping_path': output_dir / f'{base}.mapping.txt',
      'block_qasm_path': output_dir / f'{base}.cnotphase.qasm',
      'block_qc_path': output_dir / f'{base}.cnotphase.qc',
  }
  for key, path in related_paths.items():
    value = _optional_relative(output_dir, path)
    if value is not None:
      entry[key] = value
  return entry


def _write_manifest(output_dir: Path, qasm_files: list[Path]) -> Path:
  tensor_paths = sorted(output_dir.glob('*.tensor.npy'))
  if not tensor_paths:
    raise FileNotFoundError(
        f'No `.tensor.npy` files were produced in {output_dir}.'
    )
  manifest = {
      'format': MANIFEST_FORMAT,
      'source': 'circuit-to-tensor',
      'source_repo': CIRCUIT_TO_TENSOR_REPO,
      'qasm_files': [str(path) for path in qasm_files],
      'targets': [
          _manifest_entry(output_dir, tensor_path)
          for tensor_path in tensor_paths
      ],
  }
  manifest_path = output_dir / 'targets.json'
  with manifest_path.open('w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2)
    f.write('\n')
  return manifest_path


def _existing_qasm_files(values: list[str]) -> list[Path]:
  paths = [Path(value) for value in values]
  missing = [str(path) for path in paths if not path.is_file()]
  if missing:
    raise FileNotFoundError(f'QASM file(s) not found: {", ".join(missing)}')
  return paths


def parse_args(argv: list[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description='Import OpenQASM v2 circuits as ATQ tensor targets.'
  )
  parser.add_argument('qasm_files', nargs='+', help='Input .qasm files.')
  parser.add_argument(
      '--output_dir',
      required=True,
      type=Path,
      help='Directory for circuit-to-tensor outputs and ATQ manifest.',
  )
  parser.add_argument(
      '--circuit_to_tensor_cmd',
      help='Optional command for circuit-to-tensor, e.g. "cargo run --release --".',
  )
  parser.add_argument(
      '--emit',
      default='tensor,matrix,block-qasm,log',
      help='circuit-to-tensor output types. Defaults avoid feynver verification.',
  )
  parser.add_argument('--qubits', type=int, help='Maximum qubits per block.')
  parser.add_argument('--ancilla', type=int, help='Maximum ancilla per block.')
  parser.add_argument(
      '--split_iters',
      type=int,
      default=10_000,
      help='Hadamard gadgetization split iterations.',
  )
  parser.add_argument(
      '--zx_preopt',
      action='store_true',
      help='Enable circuit-to-tensor QuiZX pre-optimization.',
  )
  args = parser.parse_args(argv)
  args.qasm_files = _existing_qasm_files(args.qasm_files)
  args.output_dir.mkdir(parents=True, exist_ok=True)
  return args


def main(argv: list[str]) -> int:
  args = parse_args(argv)
  _compile_qasm(args)
  manifest_path = _write_manifest(args.output_dir, args.qasm_files)
  print(f'Wrote ATQ target manifest: {manifest_path}')
  print('Use it with:')
  print(
      '  PYTHONPATH=. .venv/bin/python -m '
      'alphatensor_quantum.src.demo.run_demo '
      f'--target_manifest={manifest_path}'
  )
  return 0


if __name__ == '__main__':
  raise SystemExit(main(sys.argv[1:]))
