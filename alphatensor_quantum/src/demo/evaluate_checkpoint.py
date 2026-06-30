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

"""Evaluates a saved demo checkpoint under explicit basis controls."""

from __future__ import annotations

import csv
from pathlib import Path
import time

from absl import app
from absl import flags
import jax
import jax.numpy as jnp
import numpy as np

from alphatensor_quantum.src import change_of_basis
from alphatensor_quantum.src.demo import agent as agent_lib
from alphatensor_quantum.src.demo import checkpoints


FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint', None, 'Checkpoint `.pkl` to evaluate.')
flags.DEFINE_string('output_csv', None, 'CSV path for eval records.')
flags.DEFINE_string(
    'controls',
    'orbit,same_base_restarts',
    'Comma-separated controls: orbit,same_base_restarts,placebo.',
)
flags.DEFINE_string('k_values', '1,4,8', 'Comma-separated k values.')
flags.DEFINE_string('eval_seeds', '0', 'Comma-separated eval seeds.')
flags.DEFINE_integer('target_index', 0, 'Target index inside the config.')
flags.DEFINE_integer(
    'max_eval_steps',
    0,
    'Maximum eval decode steps. If 0, uses env_config.max_num_moves.',
)


_FIELDNAMES = [
    'checkpoint',
    'checkpoint_step',
    'train_seed',
    'target',
    'target_index',
    'control',
    'k',
    'attempt',
    'basis_id',
    'is_conjugate',
    'eval_seed',
    'cost',
    'solved',
    'num_moves',
    'sum_rewards',
    'seconds',
    'action_sequence',
    'canonical_action_sequence',
    'canonical_factor_sequence',
]


def _parse_csv_ints(text: str) -> list[int]:
  return [int(item.strip()) for item in text.split(',') if item.strip()]


def _parse_csv_strings(text: str) -> list[str]:
  return [item.strip() for item in text.split(',') if item.strip()]


def _stable_int(text: str) -> int:
  return sum((index + 1) * ord(char) for index, char in enumerate(text))


def _target_name(target) -> str:
  return target.name.lower()


def _action_to_factor(action: int, tensor_size: int) -> np.ndarray:
  value = action + 1
  factor = np.zeros((tensor_size,), dtype=np.int32)
  for index in range(tensor_size):
    factor[index] = value % 2
    value //= 2
  return factor


def _factor_to_action(factor: np.ndarray) -> int:
  powers = 2 ** np.arange(factor.shape[0])
  return int(np.sum(factor * powers) - 1)


def _canonical_sequence(
    action_sequence: list[int],
    basis_matrix: np.ndarray,
    *,
    is_conjugate: bool,
) -> tuple[list[int], list[str]]:
  if not is_conjugate:
    return [], []
  inverse = np.asarray(
      change_of_basis.invert_matrix_gf2(jnp.asarray(basis_matrix)),
      dtype=np.int32,
  )
  canonical_actions = []
  canonical_factors = []
  for action in action_sequence:
    factor = _action_to_factor(action, inverse.shape[0])
    canonical_factor = np.mod(inverse @ factor, 2).astype(np.int32)
    canonical_actions.append(_factor_to_action(canonical_factor))
    canonical_factors.append(''.join(map(str, canonical_factor.tolist())))
  return canonical_actions, canonical_factors


def _orbit_matrices(
    frame: np.ndarray,
    k: int,
) -> tuple[np.ndarray, list[str], list[bool]]:
  size = frame.shape[-1]
  if k < 1:
    raise ValueError('k must be positive.')
  if frame.shape[0] < k - 1:
    raise ValueError(
        f'Frame has {frame.shape[0]} matrices; need at least {k - 1}.'
    )
  matrices = [np.eye(size, dtype=np.int32)]
  basis_ids = ['identity']
  for index in range(k - 1):
    matrices.append(np.asarray(frame[index], dtype=np.int32))
    basis_ids.append(f'frame_{index:03d}')
  return np.stack(matrices), basis_ids, [True] * k


def _same_base_matrices(
    size: int,
    k: int,
) -> tuple[np.ndarray, list[str], list[bool]]:
  matrices = np.broadcast_to(
      np.eye(size, dtype=np.int32), (k, size, size)
  ).copy()
  basis_ids = [f'identity_restart_{index:03d}' for index in range(k)]
  return matrices, basis_ids, [True] * k


def _placebo_matrices(
    frame: np.ndarray,
    k: int,
) -> tuple[np.ndarray, list[str], list[bool]]:
  base, _, _ = _orbit_matrices(frame, k)
  matrices = base.copy()
  matrices[:, 0, :] = matrices[:, 1, :]
  basis_ids = [f'singular_placebo_{index:03d}' for index in range(k)]
  return matrices, basis_ids, [False] * k


def _control_matrices(
    control: str,
    frame: np.ndarray,
    k: int,
) -> tuple[np.ndarray, list[str], list[bool]]:
  if control == 'orbit':
    return _orbit_matrices(frame, k)
  if control == 'same_base_restarts':
    return _same_base_matrices(frame.shape[-1], k)
  if control == 'placebo':
    return _placebo_matrices(frame, k)
  raise ValueError(f'Unknown control: {control}.')


def _run_episode(
    agent: agent_lib.Agent,
    params,
    env_state,
    rng: jax.Array,
    max_eval_steps: int,
) -> tuple[object, list[list[int]], float]:
  """Runs one batched eval episode and returns final state/action sequences."""
  action_steps = []
  start = time.time()
  for step in range(max_eval_steps):
    if bool(np.all(np.asarray(env_state.is_terminal))):
      break
    env_state, actions = agent.run_eval_step(
        params, jax.random.fold_in(rng, step), env_state
    )
    action_steps.append(np.asarray(actions))
  seconds = time.time() - start
  if action_steps:
    action_array = np.stack(action_steps, axis=0)
  else:
    batch_size = int(env_state.num_moves.shape[0])
    action_array = np.zeros((0, batch_size), dtype=np.int32)
  sequences = []
  for batch_index in range(action_array.shape[1]):
    sequences.append([
        int(action)
        for action in action_array[:, batch_index]
        if int(action) >= 0
    ])
  return env_state, sequences, seconds


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    controls: list[str],
    k_values: list[int],
    eval_seeds: list[int],
    target_index: int,
    max_eval_steps: int | None,
) -> list[dict[str, object]]:
  """Evaluates one checkpoint and returns long-form result rows."""
  payload = checkpoints.load_checkpoint(checkpoint_path)
  config = payload['config']
  metadata = payload['metadata']
  train_seed = int(metadata['seed'])
  checkpoint_step = int(metadata['step'])

  agent = agent_lib.Agent(config)
  agent.init_run_state(
      jax.random.PRNGKey(train_seed), allocate_replay=False
  )
  frame = payload.get('change_of_basis_frame')
  if frame is None:
    frame = np.asarray(agent.change_of_basis)
  else:
    frame = np.asarray(frame)
  max_steps = (
      int(config.env_config.max_num_moves)
      if max_eval_steps is None else int(max_eval_steps)
  )
  target = config.env_config.target_circuit_types[target_index]
  target_name = _target_name(target)
  rows = []
  for eval_seed in eval_seeds:
    for control in controls:
      for k in k_values:
        matrices, basis_ids, conjugate_flags = _control_matrices(
            control, frame, k
        )
        env_state = agent._env.init_state_with_basis(  # pylint: disable=protected-access
            target_index,
            jnp.asarray(matrices, dtype=jnp.int32),
        )
        rng = jax.random.PRNGKey(eval_seed)
        rng = jax.random.fold_in(rng, _stable_int(control))
        rng = jax.random.fold_in(rng, k)
        final_state, sequences, seconds = _run_episode(
            agent, payload['run_state'].params, env_state, rng, max_steps
        )
        tensor = np.asarray(final_state.tensor)
        sum_rewards = np.asarray(final_state.sum_rewards)
        is_terminal = np.asarray(final_state.is_terminal)
        num_moves = np.asarray(final_state.num_moves)
        for attempt, basis_id in enumerate(basis_ids):
          solved = bool(is_terminal[attempt] and np.all(tensor[attempt] == 0))
          canonical_actions, canonical_factors = _canonical_sequence(
              sequences[attempt],
              matrices[attempt],
              is_conjugate=conjugate_flags[attempt],
          )
          rows.append({
              'checkpoint': str(checkpoint_path),
              'checkpoint_step': checkpoint_step,
              'train_seed': train_seed,
              'target': target_name,
              'target_index': target_index,
              'control': control,
              'k': k,
              'attempt': attempt,
              'basis_id': basis_id,
              'is_conjugate': conjugate_flags[attempt],
              'eval_seed': eval_seed,
              'cost': float(-sum_rewards[attempt]),
              'solved': solved,
              'num_moves': int(num_moves[attempt]),
              'sum_rewards': float(sum_rewards[attempt]),
              'seconds': seconds / max(1, k),
              'action_sequence': ' '.join(map(str, sequences[attempt])),
              'canonical_action_sequence': ' '.join(
                  map(str, canonical_actions)
              ),
              'canonical_factor_sequence': ';'.join(canonical_factors),
          })
  return rows


def main(argv: list[str]) -> None:
  del argv
  if FLAGS.checkpoint is None:
    raise ValueError('--checkpoint is required.')
  if FLAGS.output_csv is None:
    raise ValueError('--output_csv is required.')
  max_eval_steps = None if FLAGS.max_eval_steps <= 0 else FLAGS.max_eval_steps
  rows = evaluate_checkpoint(
      FLAGS.checkpoint,
      controls=_parse_csv_strings(FLAGS.controls),
      k_values=_parse_csv_ints(FLAGS.k_values),
      eval_seeds=_parse_csv_ints(FLAGS.eval_seeds),
      target_index=FLAGS.target_index,
      max_eval_steps=max_eval_steps,
  )
  output_csv = Path(FLAGS.output_csv)
  output_csv.parent.mkdir(parents=True, exist_ok=True)
  with output_csv.open('w', newline='', encoding='utf-8') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=_FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)
  print(f'Wrote {len(rows)} eval rows to {output_csv}')


if __name__ == '__main__':
  app.run(main)
