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

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from alphatensor_quantum.src import environment
from alphatensor_quantum.src.demo import replay


def _observations(values):
  values = jnp.asarray(values, dtype=jnp.float32)
  batch_size = values.shape[0]
  return environment.Observation(
      tensor=jnp.reshape(values, (batch_size, 1, 1, 1)),
      past_factors_as_planes=jnp.reshape(values + 10, (batch_size, 1, 1, 1)),
      sqrt_played_fraction=values + 20,
  )


def _batch(values, *, valid=None):
  values = jnp.asarray(values, dtype=jnp.float32)
  if valid is None:
    valid = jnp.ones((values.shape[0],), dtype=jnp.bool_)
  return replay.ReplayBatch(
      observations=_observations(values),
      policy_targets=jnp.stack([values, values + 1], axis=-1),
      value_targets=values + 2,
      valid=jnp.asarray(valid, dtype=jnp.bool_),
  )


class ReplayTest(absltest.TestCase):

  def test_insert_preserves_observation_pytree(self):
    buffer = replay.buffer_init(4, _observations(jnp.arange(2)), 2)
    buffer = replay.buffer_insert(buffer, _batch([1, 2, 3]))

    self.assertEqual(buffer.size, 3)
    self.assertEqual(buffer.insert_index, 3)
    np.testing.assert_array_equal(
        buffer.observations.tensor[:3, 0, 0, 0], np.array([1, 2, 3])
    )
    np.testing.assert_array_equal(
        buffer.observations.past_factors_as_planes[:3, 0, 0, 0],
        np.array([11, 12, 13]),
    )
    np.testing.assert_array_equal(
        buffer.observations.sqrt_played_fraction[:3],
        np.array([21, 22, 23]),
    )

  def test_circular_insert_saturates_capacity(self):
    buffer = replay.buffer_init(5, _observations(jnp.arange(2)), 2)
    buffer = replay.buffer_insert(buffer, _batch([0, 1, 2]))
    buffer = replay.buffer_insert(buffer, _batch([3, 4, 5, 6]))

    self.assertEqual(buffer.size, 5)
    self.assertEqual(buffer.insert_index, 2)
    np.testing.assert_array_equal(
        buffer.value_targets, np.array([7, 8, 4, 5, 6])
    )

  def test_sample_only_returns_valid_entries(self):
    buffer = replay.buffer_init(4, _observations(jnp.arange(2)), 2)
    buffer = replay.buffer_insert(
        buffer,
        _batch([0, 1, 2, 3], valid=[True, False, True, False]),
    )

    sample = replay.buffer_sample(buffer, jax.random.PRNGKey(0), 32)

    self.assertTrue(np.all(np.asarray(sample.valid)))
    self.assertTrue(
        set(np.asarray(sample.value_targets).tolist()).issubset({2.0, 4.0})
    )

  def test_discountless_return_to_go_masks_invalid_steps(self):
    rewards = jnp.array([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0]])
    valid = jnp.array([[True, True], [True, False], [False, False]])

    returns = replay.discountless_return_to_go(rewards, valid)

    np.testing.assert_array_equal(
        returns,
        np.array([[3.0, 5.0], [2.0, 0.0], [0.0, 0.0]]),
    )


if __name__ == '__main__':
  absltest.main()
