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

from absl.testing import absltest
import jax
import tempfile
from unittest import mock

from alphatensor_quantum.src.demo import agent as agent_lib
from alphatensor_quantum.src.demo import checkpoints
from alphatensor_quantum.src.demo import demo_config
from alphatensor_quantum.src.demo import evaluate_checkpoint
from alphatensor_quantum.src.demo import replay as replay_lib


class EvaluateCheckpointTest(absltest.TestCase):

  def test_evaluate_checkpoint_outputs_expected_rows(self):
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
    tempdir = tempfile.TemporaryDirectory()
    self.addCleanup(tempdir.cleanup)
    checkpoint_path = tempdir.name + '/checkpoint.pkl'
    checkpoints.save_checkpoint(
        checkpoint_path,
        run_state=run_state,
        config=config,
        step=0,
        seed=2024,
        change_of_basis_frame=agent.change_of_basis,
    )

    rows = evaluate_checkpoint.evaluate_checkpoint(
        checkpoint_path,
        controls=['orbit', 'same_base_restarts'],
        k_values=[1, 2],
        eval_seeds=[0],
        target_index=0,
        max_eval_steps=1,
    )

    self.assertLen(rows, 6)
    self.assertEqual(rows[0]['target'], 'mod_5_4')
    self.assertIn('cost', rows[0])
    orbit_single = [
        row for row in rows
        if row['control'] == 'orbit' and row['k'] == 1
    ]
    self.assertLen(orbit_single, 1)
    self.assertEqual(orbit_single[0]['basis_id'], 'identity')
    self.assertTrue(orbit_single[0]['is_conjugate'])
    self.assertEqual(
        orbit_single[0]['canonical_action_sequence'],
        orbit_single[0]['action_sequence'],
    )
    self.assertIn('canonical_factor_sequence', orbit_single[0])

  def test_evaluate_replay_checkpoint_does_not_allocate_replay_buffer(self):
    config = demo_config.get_demo_config(
        use_gadgets=False,
        target_set='mod5',
        search_policy='gumbel',
        batch_size=2,
        num_mcts_simulations=2,
        num_training_steps=1,
        eval_frequency_steps=1,
        max_num_moves=10,
        replay_capacity=32,
        replay_min_size=2,
        train_batch_size=2,
        actor_rollout_length=10,
        value_target_mode='mc_return',
        num_value_quantiles=4,
        value_risk_quantile=0.5,
        gumbel_max_num_considered_actions=2,
    )
    agent = agent_lib.Agent(config)
    run_state = agent.init_run_state(jax.random.PRNGKey(0))
    tempdir = tempfile.TemporaryDirectory()
    self.addCleanup(tempdir.cleanup)
    checkpoint_path = tempdir.name + '/checkpoint.pkl'
    checkpoints.save_checkpoint(
        checkpoint_path,
        run_state=run_state,
        config=config,
        step=0,
        seed=2024,
        change_of_basis_frame=agent.change_of_basis,
    )

    with mock.patch.object(
        replay_lib, 'buffer_init',
        side_effect=AssertionError('eval allocated replay buffer'),
    ):
      rows = evaluate_checkpoint.evaluate_checkpoint(
          checkpoint_path,
          controls=['orbit'],
          k_values=[1],
          eval_seeds=[0],
          target_index=0,
          max_eval_steps=1,
      )

    self.assertLen(rows, 1)
    self.assertEqual(rows[0]['target'], 'mod_5_4')


if __name__ == '__main__':
  absltest.main()
