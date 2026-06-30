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

"""Algebraic root action priors for the AlphaTensor-Quantum demo."""

from typing import NamedTuple

import jax.numpy as jnp
import jaxtyping as jt


PRIOR_MODES = frozenset({'none', 'delta', 'density', 'hybrid'})
_EPS = 1e-6


class ActionPriorData(NamedTuple):
  """Precomputed action tensors used by algebraic priors."""
  rank_one_flat: jt.Float[jt.Array, 'num_actions tensor_volume']
  support_volume: jt.Float[jt.Array, 'num_actions']


def build_action_prior_data(tensor_size: int) -> ActionPriorData:
  """Builds flattened rank-one tensors for every nonzero factor action."""
  action_indices = jnp.arange(1, 2 ** tensor_size, dtype=jnp.uint32)
  bit_positions = jnp.arange(tensor_size, dtype=jnp.uint32)
  factors = (
      (action_indices[:, None] >> bit_positions[None, :]) & 1
  ).astype(jnp.float32)
  support = jnp.sum(factors, axis=-1)
  rank_one_flat = jnp.einsum(
      'ai,aj,ak->aijk', factors, factors, factors
  ).reshape((factors.shape[0], tensor_size ** 3))
  return ActionPriorData(
      rank_one_flat=rank_one_flat,
      support_volume=support ** 3,
  )


def _zscore(
    scores: jt.Float[jt.Array, 'batch_size num_actions']
) -> jt.Float[jt.Array, 'batch_size num_actions']:
  mean = jnp.mean(scores, axis=-1, keepdims=True)
  std = jnp.std(scores, axis=-1, keepdims=True)
  return (scores - mean) / (std + _EPS)


def action_prior_logits(
    residual_tensor: jt.Integer[jt.Array, 'batch_size size size size'],
    data: ActionPriorData,
    mode: str,
) -> jt.Float[jt.Array, 'batch_size num_actions']:
  """Returns normalized algebraic prior logits for root action selection."""
  if mode not in PRIOR_MODES:
    raise ValueError(f'Unknown algebraic prior mode: {mode}.')
  residual_flat = residual_tensor.reshape((residual_tensor.shape[0], -1))
  overlap = residual_flat.astype(jnp.float32) @ data.rank_one_flat.T
  if mode == 'none':
    return jnp.zeros_like(overlap)

  delta = 2.0 * overlap - data.support_volume[None, :]
  if mode == 'delta':
    return _zscore(delta)

  density = overlap / data.support_volume[None, :]
  if mode == 'density':
    return _zscore(density)

  hybrid = _zscore(delta) + _zscore(density)
  return _zscore(hybrid)
