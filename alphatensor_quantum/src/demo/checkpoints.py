# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Small checkpoint helpers for demo experiments."""

from __future__ import annotations

import dataclasses
import datetime
import enum
import json
from pathlib import Path
import pickle
import platform
import subprocess
import sys
from typing import Any, NamedTuple

import jax
import numpy as np


CHECKPOINT_FORMAT = 'atq_demo_checkpoint_v1'


class EvalRunState(NamedTuple):
  """Minimal checkpoint state needed for eval-only decoding."""
  params: Any


def _jsonable(value: Any) -> Any:
  """Converts config-like objects to JSON-serializable metadata."""
  if dataclasses.is_dataclass(value):
    return {
        field.name: _jsonable(getattr(value, field.name))
        for field in dataclasses.fields(value)
    }
  if isinstance(value, enum.Enum):
    return value.name
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, dict):
    return {str(key): _jsonable(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_jsonable(item) for item in value]
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, np.ndarray):
    return value.tolist()
  return value


def git_commit(repo_dir: str | Path = '.') -> str:
  """Returns the current git commit hash, or `unknown` outside git."""
  try:
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'],
        cwd=repo_dir,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
  except (OSError, subprocess.CalledProcessError):
    return 'unknown'


def checkpoint_path(
    checkpoint_dir: str | Path,
    *,
    target_tag: str,
    seed: int,
    step: int,
) -> Path:
  """Returns a stable checkpoint path for one run."""
  filename = f'checkpoint_target-{target_tag}_seed{seed}_step{step}.pkl'
  return Path(checkpoint_dir) / filename


def save_checkpoint(
    path: str | Path,
    *,
    run_state: Any,
    config: Any,
    step: int,
    seed: int,
    command: list[str] | None = None,
    change_of_basis_frame: Any | None = None,
    extra_metadata: dict[str, Any] | None = None,
    checkpoint_payload: str = 'full',
    repo_dir: str | Path = '.',
) -> Path:
  """Saves a demo run checkpoint and a readable JSON sidecar."""
  if checkpoint_payload not in ('full', 'eval'):
    raise ValueError(
        'checkpoint_payload must be one of: full, eval. '
        f'Got: {checkpoint_payload}.'
    )
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  metadata = {
      'format': CHECKPOINT_FORMAT,
      'step': int(step),
      'seed': int(seed),
      'command': list(command or sys.argv),
      'git_commit': git_commit(repo_dir),
      'checkpoint_payload': checkpoint_payload,
      'created_at_utc': datetime.datetime.now(
          datetime.timezone.utc
      ).isoformat(),
      'runtime': {
          'python_version': sys.version,
          'platform': platform.platform(),
          'jax_version': jax.__version__,
      },
      'config': _jsonable(config),
  }
  if extra_metadata:
    metadata.update(_jsonable(extra_metadata))
  checkpoint_run_state = (
      jax.device_get(run_state)
      if checkpoint_payload == 'full'
      else EvalRunState(params=jax.device_get(run_state.params))
  )
  payload = {
      'format': CHECKPOINT_FORMAT,
      'metadata': metadata,
      'config': config,
      'run_state': checkpoint_run_state,
      'change_of_basis_frame': (
          None if change_of_basis_frame is None
          else np.asarray(jax.device_get(change_of_basis_frame))
      ),
  }
  with path.open('wb') as checkpoint_file:
    pickle.dump(payload, checkpoint_file, protocol=pickle.HIGHEST_PROTOCOL)
  with path.with_suffix(path.suffix + '.json').open(
      'w', encoding='utf-8'
  ) as metadata_file:
    json.dump(metadata, metadata_file, indent=2, sort_keys=True)
    metadata_file.write('\n')
  return path


def load_checkpoint(path: str | Path) -> dict[str, Any]:
  """Loads a checkpoint saved with `save_checkpoint`."""
  with Path(path).open('rb') as checkpoint_file:
    payload = pickle.load(checkpoint_file)
  if payload.get('format') != CHECKPOINT_FORMAT:
    raise ValueError(
        f'Unsupported checkpoint format: {payload.get("format")!r}.'
    )
  return payload
