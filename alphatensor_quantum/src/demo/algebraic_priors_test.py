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
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np

from alphatensor_quantum.src.demo import algebraic_priors


def _zscore(values):
  values = np.asarray(values, dtype=np.float32)
  return (values - np.mean(values)) / (np.std(values) + 1e-6)


class AlgebraicPriorsTest(parameterized.TestCase):

  def test_build_action_prior_data_matches_action_order(self):
    data = algebraic_priors.build_action_prior_data(2)

    self.assertEqual(data.rank_one_flat.shape, (3, 8))
    np.testing.assert_array_equal(data.support_volume, np.array([1, 1, 8]))

  @parameterized.parameters('delta', 'density', 'hybrid')
  def test_action_prior_logits_has_expected_shape_and_is_finite(self, mode):
    data = algebraic_priors.build_action_prior_data(2)
    tensor = jnp.zeros((2, 2, 2, 2), dtype=jnp.int32)

    logits = algebraic_priors.action_prior_logits(tensor, data, mode)

    self.assertEqual(logits.shape, (2, 3))
    self.assertTrue(np.all(np.isfinite(logits)))

  def test_none_prior_returns_zeros(self):
    data = algebraic_priors.build_action_prior_data(2)
    tensor = jnp.ones((2, 2, 2, 2), dtype=jnp.int32)

    logits = algebraic_priors.action_prior_logits(tensor, data, 'none')

    np.testing.assert_array_equal(logits, np.zeros((2, 3), dtype=np.float32))

  def test_delta_prior_matches_manual_calculation(self):
    data = algebraic_priors.build_action_prior_data(2)
    tensor = jnp.zeros((1, 2, 2, 2), dtype=jnp.int32).at[0, 0, 0, 0].set(1)

    logits = algebraic_priors.action_prior_logits(tensor, data, 'delta')

    # Actions are [1, 0], [0, 1], [1, 1]. Raw delta values are:
    # 2 * overlap - |factor|^3 = [1, -1, -6].
    np.testing.assert_allclose(logits[0], _zscore([1, -1, -6]), rtol=1e-5)

  def test_density_prior_matches_manual_calculation(self):
    data = algebraic_priors.build_action_prior_data(2)
    tensor = jnp.zeros((1, 2, 2, 2), dtype=jnp.int32).at[0, 0, 0, 0].set(1)

    logits = algebraic_priors.action_prior_logits(tensor, data, 'density')

    # Raw density values are overlap / |factor|^3 = [1, 0, 1/8].
    np.testing.assert_allclose(logits[0], _zscore([1, 0, 1 / 8]), rtol=1e-5)
    self.assertGreater(logits[0, 0], logits[0, 2])
    self.assertGreater(logits[0, 2], logits[0, 1])

  def test_delta_prefers_action_with_best_immediate_residual_reduction(self):
    data = algebraic_priors.build_action_prior_data(2)
    tensor = jnp.zeros((1, 2, 2, 2), dtype=jnp.int32).at[0, 0, 0, 0].set(1)

    logits = algebraic_priors.action_prior_logits(tensor, data, 'delta')

    self.assertEqual(int(jnp.argmax(logits[0])), 0)


if __name__ == '__main__':
  absltest.main()
