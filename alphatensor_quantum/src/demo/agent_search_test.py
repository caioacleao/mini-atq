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

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from alphatensor_quantum.src.demo import agent as agent_lib
from alphatensor_quantum.src.demo import demo_config


def _num_parameters(params):
  return sum(leaf.size for leaf in jax.tree_util.tree_leaves(params))


class AgentSearchTest(parameterized.TestCase):

  @parameterized.parameters('muzero', 'gumbel')
  def test_run_agent_env_interaction_with_search_policy(
      self, search_policy: str
  ):
    config = demo_config.get_demo_config(
        use_gadgets=False,
        target_set='mod5',
        search_policy=search_policy,
        batch_size=2,
        num_mcts_simulations=2,
        num_training_steps=1,
        eval_frequency_steps=1,
        max_num_moves=10,
        gumbel_max_num_considered_actions=2,
    )
    agent = agent_lib.Agent(config)
    run_state = agent.init_run_state(jax.random.PRNGKey(0))
    run_state = agent.run_agent_env_interaction(0, run_state)

    self.assertEqual(run_state.env_states.num_moves.shape, (2,))
    self.assertEqual(run_state.game_stats.best_return.shape, (1,))

  def test_run_agent_env_interaction_with_algebraic_prior(self):
    config = demo_config.get_demo_config(
        use_gadgets=False,
        target_set='mod5',
        search_policy='gumbel',
        batch_size=2,
        num_mcts_simulations=2,
        num_training_steps=1,
        eval_frequency_steps=1,
        max_num_moves=10,
        gumbel_max_num_considered_actions=2,
        algebraic_prior_mode='delta',
        algebraic_prior_beta=0.3,
    )
    agent = agent_lib.Agent(config)
    run_state = agent.init_run_state(jax.random.PRNGKey(0))
    run_state = agent.run_agent_env_interaction(0, run_state)

    self.assertEqual(run_state.env_states.num_moves.shape, (2,))
    self.assertEqual(run_state.game_stats.best_return.shape, (1,))

  def test_run_agent_env_interaction_with_replay_buffer(self):
    config = demo_config.get_demo_config(
        use_gadgets=False,
        target_set='mod5',
        search_policy='muzero',
        batch_size=1,
        num_mcts_simulations=1,
        num_training_steps=1,
        eval_frequency_steps=1,
        max_num_moves=7,
        replay_capacity=32,
        replay_min_size=2,
        train_batch_size=2,
        actor_rollout_length=7,
        value_target_mode='mc_return',
    )
    agent = agent_lib.Agent(config)
    run_state = agent.init_run_state(jax.random.PRNGKey(0))
    run_state = agent.run_agent_env_interaction(0, run_state)

    self.assertIsNotNone(run_state.replay)
    self.assertGreater(int(jnp.sum(run_state.replay.valid)), 0)
    self.assertGreater(int(jnp.sum(run_state.game_stats.num_games)), 0)
    self.assertEqual(run_state.env_states.num_moves.shape, (1,))
    self.assertEqual(run_state.game_stats.best_return.shape, (1,))

  def test_run_agent_env_interaction_with_quantile_replay_buffer(self):
    config = demo_config.get_demo_config(
        use_gadgets=False,
        target_set='mod5',
        search_policy='gumbel',
        batch_size=1,
        num_mcts_simulations=1,
        num_training_steps=1,
        eval_frequency_steps=1,
        max_num_moves=7,
        replay_capacity=64,
        replay_min_size=2,
        train_batch_size=2,
        actor_rollout_length=7,
        value_target_mode='mc_return',
        num_value_quantiles=8,
        value_risk_quantile=0.75,
        gumbel_max_num_considered_actions=1,
    )
    agent = agent_lib.Agent(config)
    run_state = agent.init_run_state(jax.random.PRNGKey(0))
    run_state = agent.run_agent_env_interaction(0, run_state)

    self.assertIsNotNone(run_state.replay)
    self.assertGreater(int(jnp.sum(run_state.replay.valid)), 0)
    self.assertEqual(run_state.env_states.num_moves.shape, (1,))

  def test_categorical_replay_buffer_smoke(self):
    config = demo_config.get_demo_config(
        use_gadgets=False,
        target_set='mod5',
        search_policy='gumbel',
        batch_size=1,
        num_mcts_simulations=1,
        num_training_steps=1,
        eval_frequency_steps=1,
        max_num_moves=7,
        replay_capacity=64,
        replay_min_size=2,
        train_batch_size=2,
        actor_rollout_length=7,
        value_target_mode='mc_return',
        num_value_categorical_bins=5,
        gumbel_max_num_considered_actions=1,
    )
    agent = agent_lib.Agent(config)
    run_state = agent.init_run_state(jax.random.PRNGKey(0))
    run_state = agent.run_agent_env_interaction(0, run_state)

    self.assertIsNotNone(run_state.replay)
    self.assertGreater(int(jnp.sum(run_state.replay.valid)), 0)
    self.assertEqual(run_state.env_states.num_moves.shape, (1,))

  def test_value_head_shapes_and_parameter_count(self):
    scalar_config = demo_config.get_demo_config(
        use_gadgets=False,
        target_set='mod5',
        batch_size=2,
        num_mcts_simulations=1,
        num_training_steps=1,
        eval_frequency_steps=1,
        max_num_moves=7,
    )
    quantile_config = demo_config.get_demo_config(
        use_gadgets=False,
        target_set='mod5',
        batch_size=2,
        num_mcts_simulations=1,
        num_training_steps=1,
        eval_frequency_steps=1,
        max_num_moves=7,
        replay_capacity=32,
        replay_min_size=2,
        train_batch_size=2,
        value_target_mode='mc_return',
        num_value_quantiles=8,
        value_risk_quantile=0.75,
    )
    categorical_config = demo_config.get_demo_config(
        use_gadgets=False,
        target_set='mod5',
        batch_size=2,
        num_mcts_simulations=1,
        num_training_steps=1,
        eval_frequency_steps=1,
        max_num_moves=7,
        replay_capacity=32,
        replay_min_size=2,
        train_batch_size=2,
        value_target_mode='mc_return',
        num_value_categorical_bins=5,
    )
    scalar_agent = agent_lib.Agent(scalar_config)
    quantile_agent = agent_lib.Agent(quantile_config)
    categorical_agent = agent_lib.Agent(categorical_config)
    scalar_state = scalar_agent.init_run_state(jax.random.PRNGKey(0))
    quantile_state = quantile_agent.init_run_state(jax.random.PRNGKey(0))
    categorical_state = categorical_agent.init_run_state(jax.random.PRNGKey(0))
    scalar_observations = scalar_agent._env.get_observation(
        scalar_state.env_states
    )
    quantile_observations = quantile_agent._env.get_observation(
        quantile_state.env_states
    )
    categorical_observations = categorical_agent._env.get_observation(
        categorical_state.env_states
    )

    _, scalar_value = scalar_agent._network.apply(
        scalar_state.params, jax.random.PRNGKey(1), scalar_observations
    )
    _, quantile_values = quantile_agent._network.apply(
        quantile_state.params, jax.random.PRNGKey(1), quantile_observations
    )
    _, categorical_logits = categorical_agent._network.apply(
        categorical_state.params,
        jax.random.PRNGKey(1),
        categorical_observations,
    )

    self.assertEqual(scalar_value.shape, (2,))
    self.assertEqual(quantile_values.shape, (2, 8))
    self.assertEqual(categorical_logits.shape, (2, 5))
    self.assertGreater(
        _num_parameters(quantile_state.params),
        _num_parameters(scalar_state.params),
    )
    self.assertGreater(
        _num_parameters(categorical_state.params),
        _num_parameters(scalar_state.params),
    )

  def test_default_value_controls_are_parameter_inert(self):
    default_config = demo_config.get_demo_config(
        use_gadgets=False,
        target_set='mod5',
        batch_size=2,
        num_mcts_simulations=1,
        num_training_steps=1,
        eval_frequency_steps=1,
        max_num_moves=7,
    )
    explicit_config = demo_config.get_demo_config(
        use_gadgets=False,
        target_set='mod5',
        batch_size=2,
        num_mcts_simulations=1,
        num_training_steps=1,
        eval_frequency_steps=1,
        max_num_moves=7,
        value_scalar_loss='mse',
        value_huber_delta=1.0,
        num_value_categorical_bins=0,
    )
    default_params = agent_lib.Agent(default_config).init_run_state(
        jax.random.PRNGKey(0)
    ).params
    explicit_params = agent_lib.Agent(explicit_config).init_run_state(
        jax.random.PRNGKey(0)
    ).params

    for default_leaf, explicit_leaf in zip(
        jax.tree_util.tree_leaves(default_params),
        jax.tree_util.tree_leaves(explicit_params),
    ):
      np.testing.assert_array_equal(default_leaf, explicit_leaf)

  def test_value_to_scalar_uses_upper_quantile_tail(self):
    config = demo_config.get_demo_config(
        use_gadgets=False,
        target_set='mod5',
        batch_size=1,
        num_mcts_simulations=1,
        num_training_steps=1,
        eval_frequency_steps=1,
        max_num_moves=7,
        replay_capacity=32,
        replay_min_size=1,
        train_batch_size=1,
        value_target_mode='mc_return',
        num_value_quantiles=8,
        value_risk_quantile=0.75,
    )
    agent = agent_lib.Agent(config)
    quantiles = jnp.arange(8, dtype=jnp.float32)[None, :]

    scalar = agent._value_to_scalar(quantiles)

    np.testing.assert_allclose(scalar, np.array([6.5], dtype=np.float32))

  def test_value_to_scalar_handles_mean_and_empty_tail(self):
    mean_config = demo_config.get_demo_config(
        use_gadgets=False,
        target_set='mod5',
        batch_size=1,
        num_mcts_simulations=1,
        num_training_steps=1,
        eval_frequency_steps=1,
        max_num_moves=7,
        replay_capacity=32,
        replay_min_size=1,
        train_batch_size=1,
        value_target_mode='mc_return',
        num_value_quantiles=8,
        value_risk_quantile=0.0,
    )
    high_config = demo_config.get_demo_config(
        use_gadgets=False,
        target_set='mod5',
        batch_size=1,
        num_mcts_simulations=1,
        num_training_steps=1,
        eval_frequency_steps=1,
        max_num_moves=7,
        replay_capacity=32,
        replay_min_size=1,
        train_batch_size=1,
        value_target_mode='mc_return',
        num_value_quantiles=8,
        value_risk_quantile=0.99,
    )
    quantiles = jnp.arange(8, dtype=jnp.float32)[None, :]

    np.testing.assert_allclose(
        agent_lib.Agent(mean_config)._value_to_scalar(quantiles),
        np.array([3.5], dtype=np.float32),
    )
    np.testing.assert_allclose(
        agent_lib.Agent(high_config)._value_to_scalar(quantiles),
        np.array([7.0], dtype=np.float32),
    )

  def test_value_to_scalar_is_optimistic_in_return_tail(self):
    config = demo_config.get_demo_config(
        use_gadgets=False,
        target_set='mod5',
        batch_size=1,
        num_mcts_simulations=1,
        num_training_steps=1,
        eval_frequency_steps=1,
        max_num_moves=7,
        replay_capacity=32,
        replay_min_size=1,
        train_batch_size=1,
        value_target_mode='mc_return',
        num_value_quantiles=8,
        value_risk_quantile=0.75,
    )
    agent = agent_lib.Agent(config)
    base = jnp.arange(8, dtype=jnp.float32)[None, :]
    better_tail = base.at[:, 6:].add(10.0)

    self.assertGreater(
        float(agent._value_to_scalar(better_tail)[0]),
        float(agent._value_to_scalar(base)[0]),
    )

  def test_quantile_value_loss_matches_pinball_formula(self):
    config = demo_config.get_demo_config(
        use_gadgets=False,
        target_set='mod5',
        batch_size=1,
        num_mcts_simulations=1,
        num_training_steps=1,
        eval_frequency_steps=1,
        max_num_moves=7,
        replay_capacity=32,
        replay_min_size=1,
        train_batch_size=1,
        value_target_mode='mc_return',
        num_value_quantiles=2,
        value_risk_quantile=0.0,
    )
    agent = agent_lib.Agent(config)
    value_output = jnp.array([[0.0, 2.0]], dtype=jnp.float32)
    targets = jnp.array([1.0], dtype=jnp.float32)

    loss = agent._per_example_value_loss(value_output, targets)

    np.testing.assert_allclose(loss, np.array([0.25], dtype=np.float32))

  def test_scalar_value_loss_matches_mse_default(self):
    config = demo_config.get_demo_config(
        use_gadgets=False,
        target_set='mod5',
        batch_size=1,
        num_mcts_simulations=1,
        num_training_steps=1,
        eval_frequency_steps=1,
        max_num_moves=7,
    )
    agent = agent_lib.Agent(config)
    value_output = jnp.array([1.0, 4.0], dtype=jnp.float32)
    targets = jnp.array([3.0, 1.0], dtype=jnp.float32)

    loss = agent._per_example_value_loss(value_output, targets)

    np.testing.assert_allclose(loss, np.array([4.0, 9.0], dtype=np.float32))

  def test_scalar_huber_value_loss(self):
    config = demo_config.get_demo_config(
        use_gadgets=False,
        target_set='mod5',
        batch_size=1,
        num_mcts_simulations=1,
        num_training_steps=1,
        eval_frequency_steps=1,
        max_num_moves=7,
        value_scalar_loss='huber',
        value_huber_delta=1.0,
    )
    agent = agent_lib.Agent(config)
    value_output = jnp.array([0.5, 3.0, -3.0], dtype=jnp.float32)
    targets = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)

    loss = agent._per_example_value_loss(value_output, targets)

    np.testing.assert_allclose(
        loss,
        np.array([0.125, 2.5, 2.5], dtype=np.float32),
    )

  def test_categorical_two_hot_targets(self):
    config = demo_config.get_demo_config(
        use_gadgets=False,
        target_set='mod5',
        batch_size=1,
        num_mcts_simulations=1,
        num_training_steps=1,
        eval_frequency_steps=1,
        max_num_moves=7,
        replay_capacity=32,
        replay_min_size=1,
        train_batch_size=1,
        value_target_mode='mc_return',
        num_value_categorical_bins=5,
        value_support_min=-4.0,
        value_support_max=0.0,
    )
    agent = agent_lib.Agent(config)
    targets = jnp.array([-2.0, -3.5, -10.0, 2.0], dtype=jnp.float32)

    two_hot = agent._categorical_two_hot_targets(targets, jnp.float32)

    expected = np.array([
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)
    np.testing.assert_allclose(two_hot, expected)
    np.testing.assert_allclose(
        jnp.sum(two_hot, axis=-1),
        np.ones((4,), dtype=np.float32),
    )

  def test_categorical_value_to_scalar_uses_expected_support(self):
    config = demo_config.get_demo_config(
        use_gadgets=False,
        target_set='mod5',
        batch_size=1,
        num_mcts_simulations=1,
        num_training_steps=1,
        eval_frequency_steps=1,
        max_num_moves=7,
        replay_capacity=32,
        replay_min_size=1,
        train_batch_size=1,
        value_target_mode='mc_return',
        num_value_categorical_bins=5,
        value_support_min=-4.0,
        value_support_max=0.0,
    )
    agent = agent_lib.Agent(config)
    low_logits = jnp.array([[20.0, 0.0, 0.0, 0.0, 0.0]], dtype=jnp.float32)
    high_logits = jnp.array([[0.0, 0.0, 0.0, 0.0, 20.0]], dtype=jnp.float32)

    low_value = agent._value_to_scalar(low_logits)
    high_value = agent._value_to_scalar(high_logits)

    np.testing.assert_allclose(low_value, np.array([-4.0]), atol=1e-5)
    np.testing.assert_allclose(high_value, np.array([0.0]), atol=1e-5)
    self.assertGreater(float(high_value[0]), float(low_value[0]))

  def test_categorical_value_loss_matches_two_hot_cross_entropy(self):
    config = demo_config.get_demo_config(
        use_gadgets=False,
        target_set='mod5',
        batch_size=1,
        num_mcts_simulations=1,
        num_training_steps=1,
        eval_frequency_steps=1,
        max_num_moves=7,
        replay_capacity=32,
        replay_min_size=1,
        train_batch_size=1,
        value_target_mode='mc_return',
        num_value_categorical_bins=3,
        value_support_min=-2.0,
        value_support_max=0.0,
    )
    agent = agent_lib.Agent(config)
    value_logits = jnp.log(
        jnp.array([[0.2, 0.3, 0.5]], dtype=jnp.float32)
    )
    targets = jnp.array([-0.5], dtype=jnp.float32)

    loss = agent._per_example_value_loss(value_logits, targets)

    expected = -(0.5 * np.log(0.3) + 0.5 * np.log(0.5))
    np.testing.assert_allclose(loss, np.array([expected]), rtol=1e-6)

  def test_loss_masks_invalid_replay_samples(self):
    config = demo_config.get_demo_config(
        use_gadgets=False,
        target_set='mod5',
        batch_size=2,
        num_mcts_simulations=1,
        num_training_steps=1,
        eval_frequency_steps=1,
        max_num_moves=7,
        replay_capacity=32,
        replay_min_size=2,
        train_batch_size=2,
        value_target_mode='mc_return',
        num_value_quantiles=2,
        value_risk_quantile=0.0,
    )
    config = dataclasses.replace(
        config,
        exp_config=dataclasses.replace(
            config.exp_config,
            loss=demo_config.LossParams(
                init_demonstrations_weight=0.0,
                demonstrations_boundaries_and_scales={},
            ),
        ),
    )
    agent = agent_lib.Agent(config)
    run_state = agent.init_run_state(jax.random.PRNGKey(0))
    observations = agent._env.get_observation(run_state.env_states)
    policy_targets = jnp.ones(
        (2, agent._num_actions), dtype=jnp.float32
    ) / agent._num_actions
    value_targets = jnp.array([1.0, -100.0], dtype=jnp.float32)
    rng = jax.random.PRNGKey(123)

    loss = agent._loss_fn(
        run_state.params,
        0,
        observations,
        policy_targets,
        value_targets,
        observations,
        policy_targets,
        value_targets,
        rng,
        jnp.array([True, False]),
    )

    rng_acting, _ = jax.random.split(rng, num=2)
    policy_logits, value_output = agent._network.apply(
        run_state.params, rng_acting, observations
    )
    policy_logprobs = jax.nn.log_softmax(policy_logits)
    policy_loss = jnp.sum(
        policy_targets * (jnp.log(policy_targets) - policy_logprobs), axis=-1
    )
    value_loss = agent._per_example_value_loss(value_output, value_targets)
    expected = policy_loss[0] + value_loss[0]

    np.testing.assert_allclose(loss, expected, rtol=1e-6)

  def test_algebraic_prior_changes_root_policy_logits(self):
    config = demo_config.get_demo_config(
        use_gadgets=False,
        target_set='mod5',
        search_policy='gumbel',
        batch_size=2,
        num_mcts_simulations=2,
        num_training_steps=1,
        eval_frequency_steps=1,
        max_num_moves=10,
        gumbel_max_num_considered_actions=2,
        algebraic_prior_mode='delta',
        algebraic_prior_beta=0.3,
    )
    agent = agent_lib.Agent(config)
    run_state = agent.init_run_state(jax.random.PRNGKey(0))
    tensor = jnp.zeros_like(run_state.env_states.tensor).at[:, 0, 0, 0].set(1)
    env_states = run_state.env_states._replace(tensor=tensor)
    policy_logits = jnp.zeros((2, agent._num_actions), dtype=jnp.float32)

    biased_logits = agent._apply_root_action_prior(policy_logits, env_states)

    self.assertEqual(biased_logits.shape, policy_logits.shape)
    self.assertTrue(np.any(np.asarray(biased_logits) != 0.0))

  def test_run_eval_step(self):
    config = demo_config.get_demo_config(
        use_gadgets=False,
        target_set='mod5',
        search_policy='gumbel',
        batch_size=2,
        num_mcts_simulations=2,
        num_training_steps=1,
        eval_frequency_steps=1,
        max_num_moves=10,
        gumbel_max_num_considered_actions=2,
    )
    agent = agent_lib.Agent(config)
    run_state = agent.init_run_state(jax.random.PRNGKey(0))

    new_env_states, actions = agent.run_eval_step(
        run_state.params, jax.random.PRNGKey(1), run_state.env_states
    )

    self.assertEqual(actions.shape, (2,))
    self.assertEqual(new_env_states.num_moves.shape, (2,))
    np.testing.assert_array_equal(
        new_env_states.num_moves, run_state.env_states.num_moves + 1
    )

  def test_muzero_rejects_algebraic_prior(self):
    with self.assertRaisesRegex(ValueError, 'only supported with Gumbel'):
      demo_config.get_demo_config(
          use_gadgets=False,
          target_set='mod5',
          search_policy='muzero',
          algebraic_prior_mode='delta',
          algebraic_prior_beta=0.3,
      )

  def test_algebraic_prior_requires_positive_beta(self):
    with self.assertRaisesRegex(ValueError, 'must be positive'):
      demo_config.get_demo_config(
          use_gadgets=False,
          target_set='mod5',
          search_policy='gumbel',
          algebraic_prior_mode='hybrid',
          algebraic_prior_beta=0.0,
      )


if __name__ == '__main__':
  absltest.main()
