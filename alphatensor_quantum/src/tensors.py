# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tensor utilities for AlphaTensor-Quantum."""

import dataclasses
import enum
import json
from pathlib import Path
from typing import Any

import immutabledict

import jax.numpy as jnp
import jaxtyping as jt
import numpy as np


_SMALL_TCOUNT_3 = np.array(
    [
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[1, 1, 1], [1, 0, 0], [1, 0, 0]],
        [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
    ],
    dtype=np.int32,
)

_BARENCO_TOFF_3 = np.array(
    [
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 1, 0, 1],
            [0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 1, 0, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 1, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 1, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
    ],
    dtype=np.int32,
)

_MOD_5_4 = np.array(
    [
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 1, 1, 1, 0],
        ],
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [1, 0, 1, 1, 0],
        ],
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
        ],
        [
            [0, 1, 1, 1, 0],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
    ],
    dtype=np.int32,
)

_NC_TOFF_3 = np.array(
    [
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 1],
            [0, 1, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 1, 0, 0],
            [0, 1, 1, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0],
        ],
        [
            [0, 0, 1, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 1, 0, 1, 1, 0, 0],
            [1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0],
            [1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 1, 1, 0, 1, 0, 0],
            [1, 0, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 1, 1, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 1, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
    ],
    dtype=np.int32,
)


class CircuitType(enum.Enum):
  """Types of circuits."""
  # Some circuits taken from the "Benchmarks" section of the paper.
  BARENCO_TOFF_3 = 1
  MOD_5_4 = 2
  NC_TOFF_3 = 3
  # A small 3-qubit circuit with optimal T-count of 3, useful for testing.
  SMALL_TCOUNT_3 = 4


@dataclasses.dataclass(frozen=True, kw_only=True)
class ImportedTensorTarget:
  """Signature tensor loaded from an external file."""

  name: str
  tensor_path: str


TensorTarget = CircuitType | ImportedTensorTarget


_TENSORS_DICT = immutabledict.immutabledict({
    CircuitType.BARENCO_TOFF_3: _BARENCO_TOFF_3,
    CircuitType.MOD_5_4: _MOD_5_4,
    CircuitType.NC_TOFF_3: _NC_TOFF_3,
    CircuitType.SMALL_TCOUNT_3: _SMALL_TCOUNT_3,
})


def _assert_symmetric(tensor: np.ndarray, path: Path | None = None) -> None:
  for perm in ((0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
    if not np.array_equal(tensor, np.transpose(tensor, perm)):
      location = '' if path is None else f' in {path}'
      raise ValueError(f'Signature tensor{location} must be symmetric.')


def validate_signature_tensor(
    tensor: np.ndarray, path: Path | None = None
) -> np.ndarray:
  """Validates and normalizes an imported signature tensor."""
  tensor = np.asarray(tensor)
  location = '' if path is None else f' in {path}'
  if tensor.ndim != 3 or len(set(tensor.shape)) != 1:
    raise ValueError(
        f'Signature tensor{location} must have shape (N, N, N). '
        f'Got: {tensor.shape}.'
    )
  values = np.unique(tensor)
  if not np.all(np.isin(values, [0, 1, False, True])):
    raise ValueError(f'Signature tensor{location} must contain only 0/1 values.')
  _assert_symmetric(tensor.astype(np.int32), path)
  return tensor.astype(np.int32)


def _reroot_to_repo(path: Path) -> Path:
  """Re-roots a stale absolute path under the current repo (portability).

  Checkpoints and manifests created on another machine can store absolute paths
  (e.g. an absolute `.../third_party/...` path from a transient run directory) that do not exist after the run is
  moved to a different host. If such a path is missing but the same
  `third_party/...` suffix exists under this repo, use the local copy instead.
  """
  if path.exists() or 'third_party' not in path.parts:
    return path
  repo_root = Path(__file__).resolve().parents[2]
  index = path.parts.index('third_party')
  candidate = repo_root.joinpath(*path.parts[index:])
  return candidate if candidate.exists() else path


def load_signature_tensor_file(
    tensor_path: str | Path
) -> jt.Integer[jt.Array, 'size size size']:
  """Loads a `.tensor.npy` signature tensor from disk."""
  path = _reroot_to_repo(Path(tensor_path))
  return jnp.array(validate_signature_tensor(np.load(path), path))


def _resolve_manifest_path(manifest_path: Path, value: str) -> str:
  path = Path(value)
  if path.is_absolute():
    return str(path)
  return str((manifest_path.parent / path).resolve())


def load_imported_targets_manifest(
    manifest_path: str | Path
) -> list[ImportedTensorTarget]:
  """Loads imported targets generated by `tools/import_qasm_targets.py`."""
  path = Path(manifest_path)
  with path.open(encoding='utf-8') as f:
    manifest: dict[str, Any] = json.load(f)
  if manifest.get('format') != 'atq_imported_targets_v1':
    raise ValueError(
        f'Unsupported imported-target manifest format: '
        f'{manifest.get("format")!r}.'
    )
  targets = []
  for entry in manifest.get('targets', []):
    targets.append(ImportedTensorTarget(
        name=entry['name'],
        tensor_path=_resolve_manifest_path(path, entry['tensor_path']),
    ))
  if not targets:
    raise ValueError(f'No targets found in imported-target manifest: {path}.')
  return targets


def zero_pad_tensor(
    tensor: jt.Integer[jt.Array, 'size size size'],
    pad_to_size: int
) -> jt.Integer[jt.Array, '{pad_to_size} {pad_to_size} {pad_to_size}']:
  """Zero-pads the given tensor to the given size.

  Args:
    tensor: The tensor to pad.
    pad_to_size: The size to pad to. It must be at least as large as the tensor
      size.

  Returns:
    The padded tensor, such that the original tensor can be recovered by keeping
    the first `size` entries of each dimension.
  """
  size = tensor.shape[0]
  padding_width = pad_to_size - size
  return jnp.pad(tensor, (0, padding_width))


def get_signature_tensor(
    circuit_type: TensorTarget
) -> jt.Integer[jt.Array, 'size size size']:
  """Returns the signature tensor for the given quantum circuit.

  Args:
    circuit_type: The circuit type or imported tensor target.

  Returns:
    The (symmetric) target signature tensor, with entries in {0, 1}.
  """
  if isinstance(circuit_type, ImportedTensorTarget):
    return load_signature_tensor_file(circuit_type.tensor_path)
  if circuit_type not in _TENSORS_DICT:
    raise ValueError(f'Unsupported circuit type: {circuit_type}')
  return jnp.array(_TENSORS_DICT[circuit_type])
