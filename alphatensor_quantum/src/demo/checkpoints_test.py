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
import numpy as np
import tempfile

from alphatensor_quantum.src.demo import agent as agent_lib
from alphatensor_quantum.src.demo import checkpoints
from alphatensor_quantum.src.demo import demo_config


def _small_config():
  return demo_config.get_demo_config(
      use_gadgets=False,
      target_set='mod5',
      search_policy='gumbel',
      batch_size=2,
      num_mcts_simulations=2,
      num_training_steps=2,
      eval_frequency_steps=1,
      max_num_moves=10,
      gumbel_max_num_considered_actions=2,
  )


class CheckpointsTest(absltest.TestCase):

  def test_save_load_checkpoint_round_trip(self):
    config = _small_config()
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
        command=['run_demo.py'],
        change_of_basis_frame=agent.change_of_basis,
        extra_metadata={'target_tag': 'mod_5_4'},
    )
    payload = checkpoints.load_checkpoint(checkpoint_path)

    self.assertEqual(payload['format'], checkpoints.CHECKPOINT_FORMAT)
    self.assertEqual(payload['metadata']['seed'], 2024)
    self.assertEqual(payload['metadata']['step'], 0)
    self.assertEqual(payload['metadata']['checkpoint_payload'], 'full')
    self.assertEqual(payload['metadata']['target_tag'], 'mod_5_4')
    self.assertIn('runtime', payload['metadata'])
    self.assertEqual(payload['config'].search_config.policy, 'gumbel')
    np.testing.assert_array_equal(
        payload['run_state'].env_states.tensor, run_state.env_states.tensor
    )
    self.assertEqual(payload['change_of_basis_frame'].shape[-2:], (5, 5))

  def test_save_eval_checkpoint_keeps_only_params(self):
    config = _small_config()
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
        checkpoint_payload='eval',
    )
    payload = checkpoints.load_checkpoint(checkpoint_path)

    self.assertEqual(payload['metadata']['checkpoint_payload'], 'eval')
    self.assertTrue(hasattr(payload['run_state'], 'params'))
    self.assertFalse(hasattr(payload['run_state'], 'opt_state'))

  def test_resume_matches_uninterrupted_run(self):
    config = _small_config()
    seed = jax.random.PRNGKey(0)
    agent = agent_lib.Agent(config)
    run_state = agent.init_run_state(seed)
    run_state_after_one = agent.run_agent_env_interaction(0, run_state)
    run_state_after_two = agent.run_agent_env_interaction(
        1, run_state_after_one
    )

    tempdir = tempfile.TemporaryDirectory()
    self.addCleanup(tempdir.cleanup)
    checkpoint_path = tempdir.name + '/checkpoint.pkl'
    checkpoints.save_checkpoint(
        checkpoint_path,
        run_state=run_state_after_one,
        config=config,
        step=1,
        seed=2024,
        change_of_basis_frame=agent.change_of_basis,
    )
    payload = checkpoints.load_checkpoint(checkpoint_path)
    resumed_agent = agent_lib.Agent(config)
    resumed_agent.init_run_state(seed)
    resumed_state = resumed_agent.run_agent_env_interaction(
        1, payload['run_state']
    )

    np.testing.assert_array_equal(
        resumed_state.env_states.tensor, run_state_after_two.env_states.tensor
    )
    np.testing.assert_array_equal(
        resumed_state.env_states.num_moves,
        run_state_after_two.env_states.num_moves,
    )
    np.testing.assert_array_equal(resumed_state.rng, run_state_after_two.rng)


if __name__ == '__main__':
  absltest.main()
