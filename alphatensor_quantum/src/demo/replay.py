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

"""JAX-friendly replay buffer for demo actor/learner experiments."""

from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import jaxtyping as jt

from alphatensor_quantum.src import environment


class ReplayBatch(NamedTuple):
  """A flat batch of actor transitions."""
  observations: environment.Observation
  policy_targets: jt.Float[jt.Array, 'batch num_actions']
  value_targets: jt.Float[jt.Array, 'batch']
  valid: jt.Bool[jt.Array, 'batch']


class ReplayBuffer(NamedTuple):
  """Fixed-capacity circular replay storage."""
  observations: environment.Observation
  policy_targets: jt.Float[jt.Array, 'capacity num_actions']
  value_targets: jt.Float[jt.Array, 'capacity']
  valid: jt.Bool[jt.Array, 'capacity']
  insert_index: jt.Integer[jt.Array, '']
  size: jt.Integer[jt.Array, '']


def _zeros_like_unbatched_batch(
    value: chex.Array,
    capacity: int,
) -> chex.Array:
  return jnp.zeros((capacity,) + value.shape[1:], dtype=value.dtype)


def buffer_init(
    capacity: int,
    obs_template: environment.Observation,
    num_actions: int,
) -> ReplayBuffer:
  """Initializes replay storage from a batched observation template."""
  if capacity <= 0:
    raise ValueError('capacity must be positive.')
  if num_actions <= 0:
    raise ValueError('num_actions must be positive.')
  observations = jax.tree_util.tree_map(
      lambda value: _zeros_like_unbatched_batch(value, capacity),
      obs_template,
  )
  return ReplayBuffer(
      observations=observations,
      policy_targets=jnp.zeros((capacity, num_actions), dtype=jnp.float32),
      value_targets=jnp.zeros((capacity,), dtype=jnp.float32),
      valid=jnp.zeros((capacity,), dtype=jnp.bool_),
      insert_index=jnp.array(0, dtype=jnp.int32),
      size=jnp.array(0, dtype=jnp.int32),
  )


def buffer_insert(
    buffer: ReplayBuffer,
    transitions: ReplayBatch,
) -> ReplayBuffer:
  """Inserts a flat batch into replay using circular indexing."""
  num_items = transitions.policy_targets.shape[0]
  capacity = buffer.policy_targets.shape[0]
  indices = (buffer.insert_index + jnp.arange(num_items)) % capacity
  observations = jax.tree_util.tree_map(
      lambda stored, new: stored.at[indices].set(new),
      buffer.observations,
      transitions.observations,
  )
  return ReplayBuffer(
      observations=observations,
      policy_targets=buffer.policy_targets.at[indices].set(
          transitions.policy_targets
      ),
      value_targets=buffer.value_targets.at[indices].set(
          transitions.value_targets
      ),
      valid=buffer.valid.at[indices].set(transitions.valid),
      insert_index=(buffer.insert_index + num_items) % capacity,
      size=jnp.minimum(buffer.size + num_items, capacity),
  )


def valid_size(buffer: ReplayBuffer) -> jt.Integer[jt.Array, '']:
  """Returns the number of currently valid transitions."""
  return jnp.sum(buffer.valid.astype(jnp.int32))


def buffer_sample(
    buffer: ReplayBuffer,
    rng: chex.PRNGKey,
    batch_size: int,
) -> ReplayBatch:
  """Samples a minibatch from valid replay slots."""
  if batch_size <= 0:
    raise ValueError('batch_size must be positive.')
  valid = buffer.valid.astype(jnp.float32)
  num_valid = jnp.sum(valid)
  fallback = jnp.zeros_like(valid).at[0].set(1.0)
  probs = jnp.where(num_valid > 0, valid / jnp.maximum(num_valid, 1.0),
                    fallback)
  indices = jax.random.choice(
      rng,
      a=buffer.policy_targets.shape[0],
      shape=(batch_size,),
      p=probs,
  )
  observations = jax.tree_util.tree_map(
      lambda value: jnp.take(value, indices, axis=0),
      buffer.observations,
  )
  return ReplayBatch(
      observations=observations,
      policy_targets=jnp.take(buffer.policy_targets, indices, axis=0),
      value_targets=jnp.take(buffer.value_targets, indices, axis=0),
      valid=jnp.take(buffer.valid, indices, axis=0),
  )


def discountless_return_to_go(
    rewards: jt.Float[jt.Array, 'time batch'],
    valid: jt.Bool[jt.Array, 'time batch'],
) -> jt.Float[jt.Array, 'time batch']:
  """Computes reverse cumulative returns over valid rollout steps."""
  masked_rewards = jnp.where(valid, rewards, 0.0)
  return jnp.flip(jnp.cumsum(jnp.flip(masked_rewards, axis=0), axis=0), axis=0)
