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
import numpy as np

from alphatensor_quantum.src.demo import agent as agent_lib
from alphatensor_quantum.src.demo import demo_config


def _small_config(**kwargs):
  base_kwargs = dict(
      use_gadgets=False,
      target_set='mod5',
      batch_size=1,
      num_mcts_simulations=1,
      num_training_steps=1,
      eval_frequency_steps=1,
      max_num_moves=5,
  )
  base_kwargs.update(kwargs)
  return demo_config.get_demo_config(**base_kwargs)


def _init_params(config):
  agent = agent_lib.Agent(config)
  return agent.init_run_state(jax.random.PRNGKey(0)).params


def _num_parameters(params):
  return sum(leaf.size for leaf in jax.tree_util.tree_leaves(params))


class DemoConfigTest(absltest.TestCase):

  def test_default_torso_params_match_explicit_legacy_values(self):
    default_config = _small_config()
    explicit_config = _small_config(
        num_layers_torso=4,
        num_heads=8,
        head_depth=8,
        mlp_widening_factor=2,
    )

    default_params = _init_params(default_config)
    explicit_params = _init_params(explicit_config)

    for default_leaf, explicit_leaf in zip(
        jax.tree_util.tree_leaves(default_params),
        jax.tree_util.tree_leaves(explicit_params),
    ):
      np.testing.assert_array_equal(default_leaf, explicit_leaf)

  def test_larger_torso_initializes_and_has_more_parameters(self):
    default_params = _init_params(_small_config())
    larger_params = _init_params(
        _small_config(num_layers_torso=8, head_depth=16)
    )

    self.assertGreater(_num_parameters(larger_params),
                       _num_parameters(default_params))

  def test_replay_defaults_are_derived_from_batch_and_max_moves(self):
    config = _small_config(batch_size=3, max_num_moves=11)

    self.assertEqual(config.exp_config.replay_capacity, 0)
    self.assertEqual(config.exp_config.train_batch_size, 3)
    self.assertEqual(config.exp_config.replay_min_size, 3)
    self.assertEqual(config.exp_config.actor_rollout_length, 11)
    self.assertEqual(config.exp_config.num_learner_steps_per_actor, 1)
    self.assertEqual(config.exp_config.value_target_mode, 'bootstrap')
    self.assertEqual(config.net_config.value_scalar_loss, 'mse')
    self.assertEqual(config.net_config.value_huber_delta, 1.0)
    self.assertEqual(config.net_config.num_value_categorical_bins, 0)

  def test_custom_replay_config(self):
    config = _small_config(
        replay_capacity=100,
        replay_min_size=17,
        train_batch_size=8,
        num_learner_steps_per_actor=3,
        actor_rollout_length=6,
        value_target_mode='mc_return',
    )

    self.assertEqual(config.exp_config.replay_capacity, 100)
    self.assertEqual(config.exp_config.replay_min_size, 17)
    self.assertEqual(config.exp_config.train_batch_size, 8)
    self.assertEqual(config.exp_config.num_learner_steps_per_actor, 3)
    self.assertEqual(config.exp_config.actor_rollout_length, 6)
    self.assertEqual(config.exp_config.value_target_mode, 'mc_return')

  def test_mc_return_requires_full_episode_rollout(self):
    with self.assertRaisesRegex(ValueError, 'actor_rollout_length'):
      _small_config(
          max_num_moves=30,
          actor_rollout_length=5,
          value_target_mode='mc_return',
      )

    config = _small_config(
        max_num_moves=30,
        actor_rollout_length=None,
        value_target_mode='mc_return',
    )

    self.assertEqual(config.exp_config.actor_rollout_length, 30)

  def test_rejects_non_positive_torso_params(self):
    with self.assertRaisesRegex(ValueError, 'num_layers_torso'):
      _small_config(num_layers_torso=0)
    with self.assertRaisesRegex(ValueError, 'num_heads'):
      _small_config(num_heads=0)
    with self.assertRaisesRegex(ValueError, 'head_depth'):
      _small_config(head_depth=0)
    with self.assertRaisesRegex(ValueError, 'mlp_widening_factor'):
      _small_config(mlp_widening_factor=0)

  def test_rejects_invalid_replay_params(self):
    with self.assertRaisesRegex(ValueError, 'replay_capacity'):
      _small_config(replay_capacity=-1)
    with self.assertRaisesRegex(ValueError, 'replay_min_size'):
      _small_config(replay_min_size=0)
    with self.assertRaisesRegex(ValueError, 'train_batch_size'):
      _small_config(train_batch_size=0)
    with self.assertRaisesRegex(ValueError, 'actor_rollout_length'):
      _small_config(actor_rollout_length=0)
    with self.assertRaisesRegex(ValueError, 'value_target_mode'):
      _small_config(value_target_mode='bad')

  def test_rejects_replay_sizes_larger_than_capacity(self):
    with self.assertRaisesRegex(ValueError, 'replay_min_size'):
      _small_config(replay_capacity=4, replay_min_size=8)
    with self.assertRaisesRegex(ValueError, 'train_batch_size'):
      _small_config(
          replay_capacity=4,
          replay_min_size=4,
          train_batch_size=8,
      )

  def test_custom_quantile_value_config(self):
    config = _small_config(
        replay_capacity=100,
        replay_min_size=1,
        train_batch_size=1,
        value_target_mode='mc_return',
        num_value_quantiles=8,
        value_risk_quantile=0.75,
    )

    self.assertEqual(config.net_config.num_value_quantiles, 8)
    self.assertEqual(config.net_config.value_risk_quantile, 0.75)

  def test_custom_scalar_huber_config(self):
    config = _small_config(
        value_scalar_loss='huber',
        value_huber_delta=0.5,
    )

    self.assertEqual(config.net_config.value_scalar_loss, 'huber')
    self.assertEqual(config.net_config.value_huber_delta, 0.5)

  def test_rejects_invalid_scalar_value_config(self):
    with self.assertRaisesRegex(ValueError, 'value_scalar_loss'):
      _small_config(value_scalar_loss='bad')
    with self.assertRaisesRegex(ValueError, 'value_huber_delta'):
      _small_config(value_huber_delta=0.0)

  def test_custom_categorical_value_config(self):
    config = _small_config(
        replay_capacity=100,
        replay_min_size=1,
        train_batch_size=1,
        max_num_moves=9,
        value_target_mode='mc_return',
        num_value_categorical_bins=51,
    )

    self.assertEqual(config.net_config.num_value_categorical_bins, 51)
    self.assertEqual(config.net_config.value_support_min, -18.0)
    self.assertEqual(config.net_config.value_support_max, 0.0)

  def test_custom_categorical_value_support(self):
    config = _small_config(
        replay_capacity=100,
        replay_min_size=1,
        train_batch_size=1,
        value_target_mode='mc_return',
        num_value_categorical_bins=7,
        value_support_min=-20.0,
        value_support_max=-2.0,
    )

    self.assertEqual(config.net_config.value_support_min, -20.0)
    self.assertEqual(config.net_config.value_support_max, -2.0)

  def test_rejects_invalid_categorical_value_config(self):
    with self.assertRaisesRegex(ValueError, 'num_value_categorical_bins'):
      _small_config(num_value_categorical_bins=-1)
    with self.assertRaisesRegex(ValueError, 'num_value_categorical_bins'):
      _small_config(
          replay_capacity=100,
          value_target_mode='mc_return',
          num_value_categorical_bins=1,
      )
    with self.assertRaisesRegex(ValueError, 'at most one'):
      _small_config(
          replay_capacity=100,
          value_target_mode='mc_return',
          num_value_quantiles=8,
          num_value_categorical_bins=51,
      )
    with self.assertRaisesRegex(ValueError, 'value_target_mode=mc_return'):
      _small_config(
          replay_capacity=100,
          value_target_mode='bootstrap',
          num_value_categorical_bins=51,
      )
    with self.assertRaisesRegex(ValueError, 'replay_capacity>0'):
      _small_config(
          value_target_mode='mc_return',
          num_value_categorical_bins=51,
      )
    with self.assertRaisesRegex(ValueError, 'value_support_min'):
      _small_config(
          replay_capacity=100,
          value_target_mode='mc_return',
          num_value_categorical_bins=51,
          value_support_min=0.0,
          value_support_max=0.0,
      )

  def test_rejects_invalid_quantile_value_config(self):
    with self.assertRaisesRegex(ValueError, 'num_value_quantiles'):
      _small_config(num_value_quantiles=-1)
    with self.assertRaisesRegex(ValueError, 'value_risk_quantile'):
      _small_config(value_risk_quantile=-0.1)
    with self.assertRaisesRegex(ValueError, 'value_risk_quantile'):
      _small_config(value_risk_quantile=1.0)
    with self.assertRaisesRegex(ValueError, 'value_target_mode=mc_return'):
      _small_config(
          replay_capacity=100,
          num_value_quantiles=8,
          value_target_mode='bootstrap',
      )
    with self.assertRaisesRegex(ValueError, 'replay_capacity>0'):
      _small_config(
          num_value_quantiles=8,
          value_target_mode='mc_return',
      )


if __name__ == '__main__':
  absltest.main()
