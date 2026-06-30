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

"""Configuration hyperparameters for the AlphaTensor-Quantum demo."""

import dataclasses

from alphatensor_quantum.src import config as config_lib
from alphatensor_quantum.src import tensors
from alphatensor_quantum.src.demo import algebraic_priors


@dataclasses.dataclass(frozen=True, kw_only=True)
class LossParams:
  """Hyperparameters for the loss.

  Attributes:
    init_demonstrations_weight: The initial weight of the loss corresponding to
      the episodes from synthetic demonstrations.
    demonstrations_boundaries_and_scales: The boundaries and scales for the
      synthetic demonstrations weight, to be used in a
      `piecewise_constant_schedule` Optax schedule.
  """
  init_demonstrations_weight: float
  demonstrations_boundaries_and_scales: dict[int, float]


@dataclasses.dataclass(frozen=True, kw_only=True)
class ExperimentParams:
  """Hyperparameters for the experiment.

  Attributes:
    batch_size: The batch size.
    num_mcts_simulations: The number of MCTS simulations to run per each action
      taken.
    num_training_steps: The total number of training steps.
    avg_return_smoothing: The smoothing factor for the average return, for
      reporting purposes only.
    eval_frequency_steps: The frequency (expressed in number of training steps)
      to report the running statistics. This is for reporting purposes only.
    replay_capacity: Capacity of the actor replay buffer. A value of 0 keeps the
      original online demo loop. Frozen invalid rollout slots still occupy this
      capacity, so fast-solving targets can have `valid_size` much smaller than
      `replay_capacity`; leave slack when sizing it.
    replay_min_size: Minimum number of valid replay entries before learner
      updates are run.
    train_batch_size: Number of replay samples used by each learner update.
    num_learner_steps_per_actor: Learner gradient steps after each actor
      rollout in replay mode. Learning-rate schedules and demonstration weights
      advance once per optimizer update, so values above 1 compress those
      schedules relative to actor rollouts.
    actor_rollout_length: Number of environment steps collected per actor
      rollout in replay mode. In replay mode, one reported training step is one
      actor rollout of `actor_rollout_length * batch_size` environment steps
      followed by `num_learner_steps_per_actor` learner updates.
    value_target_mode: Value targets for acting data: `bootstrap` preserves the
      original search-root value target; `mc_return` uses rollout returns.
    loss: The loss parameters.
  """
  batch_size: int = 2_048
  num_mcts_simulations: int = 800
  num_training_steps: int = 1_000_000
  avg_return_smoothing: float = 0.9
  eval_frequency_steps: int = 1_000
  replay_capacity: int = 0
  replay_min_size: int = 0
  train_batch_size: int = 2_048
  num_learner_steps_per_actor: int = 1
  actor_rollout_length: int = 250
  value_target_mode: str = 'bootstrap'
  loss: LossParams


@dataclasses.dataclass(frozen=True, kw_only=True)
class SearchParams:
  """Hyperparameters for the MCTS policy-improvement operator.

  Attributes:
    policy: Search policy used by MCTX. `muzero` preserves the original demo;
      `gumbel` uses MCTX's Gumbel MuZero policy improvement.
    gumbel_max_num_considered_actions: Maximum number of root actions expanded
      by Gumbel MuZero.
    gumbel_scale: Scale of the root Gumbel noise.
    algebraic_prior_mode: Root-only algebraic prior added to Gumbel policy
      logits. `none` preserves the original search behavior.
    algebraic_prior_beta: Scale of the algebraic prior logits.
  """
  policy: str = 'muzero'
  gumbel_max_num_considered_actions: int = 16
  gumbel_scale: float = 1.0
  algebraic_prior_mode: str = 'none'
  algebraic_prior_beta: float = 0.0


@dataclasses.dataclass(frozen=True, kw_only=True)
class DemoConfig:
  """All the hyperparameters for the demo."""
  exp_config: ExperimentParams
  env_config: config_lib.EnvironmentParams
  net_config: config_lib.NetworkParams
  opt_config: config_lib.OptimizerParams
  dem_config: config_lib.DemonstrationsParams
  search_config: SearchParams = dataclasses.field(default_factory=SearchParams)


def _target_circuit_types(
    use_gadgets: bool, target_set: str, target_manifest: str | None
) -> list[tensors.TensorTarget]:
  """Returns demo target circuits for a named target set."""
  if target_manifest is not None:
    return tensors.load_imported_targets_manifest(target_manifest)
  if target_set == 'default':
    if use_gadgets:
      return [
          # A tensor of size 5. The optimal decomposition has a single Toffoli
          # gadget, i.e., its equivalent T-count is 2.
          tensors.CircuitType.MOD_5_4,
      ]
    return [
        # A tensor of size 5 and rank 7.
        tensors.CircuitType.MOD_5_4,
        # A tensor of size 8 and rank 13.
        tensors.CircuitType.BARENCO_TOFF_3,
        # A tensor of size 7 and rank 13.
        tensors.CircuitType.NC_TOFF_3,
    ]
  if target_set == 'original':
    return [
        tensors.CircuitType.MOD_5_4,
        tensors.CircuitType.BARENCO_TOFF_3,
        tensors.CircuitType.NC_TOFF_3,
    ]
  if target_set == 'mod5':
    return [tensors.CircuitType.MOD_5_4]
  if target_set == 'barenco3':
    return [tensors.CircuitType.BARENCO_TOFF_3]
  if target_set == 'nc3':
    return [tensors.CircuitType.NC_TOFF_3]
  raise ValueError(f'Unknown target_set: {target_set}.')


def get_demo_config(
    use_gadgets: bool,
    *,
    target_set: str = 'default',
    target_manifest: str | None = None,
    search_policy: str = 'muzero',
    gumbel_max_num_considered_actions: int = 16,
    gumbel_scale: float = 1.0,
    algebraic_prior_mode: str = 'none',
    algebraic_prior_beta: float = 0.0,
    batch_size: int | None = None,
    num_mcts_simulations: int | None = None,
    num_training_steps: int | None = None,
    eval_frequency_steps: int | None = None,
    max_num_moves: int | None = None,
    replay_capacity: int = 0,
    replay_min_size: int | None = None,
    train_batch_size: int | None = None,
    num_learner_steps_per_actor: int = 1,
    actor_rollout_length: int | None = None,
    value_target_mode: str = 'bootstrap',
    num_value_quantiles: int = 0,
    value_risk_quantile: float = 0.0,
    value_scalar_loss: str = 'mse',
    value_huber_delta: float = 1.0,
    num_value_categorical_bins: int = 0,
    value_support_min: float | None = None,
    value_support_max: float | None = None,
    value_target_transform: str = 'none',
    num_layers_torso: int = 4,
    num_heads: int = 8,
    head_depth: int = 8,
    mlp_widening_factor: int = 2,
) -> DemoConfig:
  """Returns the config hyperparameters for the demo.

  Args:
    use_gadgets: Whether to consider gadgetization. This parameter affects not
      only the environment, but also the default target circuits.
    target_set: Named built-in target set to use when `target_manifest` is None.
    target_manifest: Optional manifest generated by `tools/import_qasm_targets.py`.

  Returns:
    The hyperparameters for the demo.
  """
  if search_policy not in ('muzero', 'gumbel'):
    raise ValueError(
        'search_policy must be one of: muzero, gumbel. '
        f'Got: {search_policy}.'
    )
  if gumbel_max_num_considered_actions < 1:
    raise ValueError('gumbel_max_num_considered_actions must be positive.')
  if algebraic_prior_mode not in algebraic_priors.PRIOR_MODES:
    raise ValueError(
        'algebraic_prior_mode must be one of: '
        f'{sorted(algebraic_priors.PRIOR_MODES)}. '
        f'Got: {algebraic_prior_mode}.'
    )
  if algebraic_prior_mode != 'none' and search_policy != 'gumbel':
    raise ValueError('Algebraic priors are only supported with Gumbel search.')
  if algebraic_prior_mode != 'none' and algebraic_prior_beta <= 0:
    raise ValueError(
        'algebraic_prior_beta must be positive when algebraic priors are used.'
    )
  if num_layers_torso < 1:
    raise ValueError('num_layers_torso must be positive.')
  if num_heads < 1:
    raise ValueError('num_heads must be positive.')
  if head_depth < 1:
    raise ValueError('head_depth must be positive.')
  if mlp_widening_factor < 1:
    raise ValueError('mlp_widening_factor must be positive.')
  if num_value_quantiles < 0:
    raise ValueError('num_value_quantiles must be non-negative.')
  if not 0.0 <= value_risk_quantile < 1.0:
    raise ValueError('value_risk_quantile must be in [0, 1).')
  if value_scalar_loss not in ('mse', 'huber'):
    raise ValueError('value_scalar_loss must be one of: mse, huber.')
  if value_target_transform not in ('none', 'symlog'):
    raise ValueError(
        'value_target_transform must be one of: none, symlog. '
        f'Got: {value_target_transform}.'
    )
  if value_target_transform != 'none' and (
      num_value_quantiles > 0 or num_value_categorical_bins > 0
  ):
    raise ValueError(
        'value_target_transform is only supported with the plain scalar value '
        'head (num_value_quantiles == 0 and num_value_categorical_bins == 0).'
    )
  if value_huber_delta <= 0:
    raise ValueError('value_huber_delta must be positive.')
  if num_value_categorical_bins < 0:
    raise ValueError('num_value_categorical_bins must be non-negative.')
  if replay_capacity < 0:
    raise ValueError('replay_capacity must be non-negative.')
  if num_learner_steps_per_actor < 1:
    raise ValueError('num_learner_steps_per_actor must be positive.')
  if value_target_mode not in ('bootstrap', 'mc_return'):
    raise ValueError(
        'value_target_mode must be one of: bootstrap, mc_return. '
        f'Got: {value_target_mode}.'
    )
  if num_value_quantiles > 0 and value_target_mode != 'mc_return':
    raise ValueError('Quantile value requires value_target_mode=mc_return.')
  if num_value_quantiles > 0 and replay_capacity == 0:
    raise ValueError('Quantile value requires replay_capacity>0.')
  if num_value_quantiles > 0 and num_value_categorical_bins > 0:
    raise ValueError(
        'Use at most one distributional value head: quantile or categorical.'
    )
  if num_value_categorical_bins > 0:
    if num_value_categorical_bins < 2:
      raise ValueError('num_value_categorical_bins must be at least 2.')
    if value_target_mode != 'mc_return':
      raise ValueError('Categorical value requires value_target_mode=mc_return.')
    if replay_capacity == 0:
      raise ValueError('Categorical value requires replay_capacity>0.')

  target_circuit_types = _target_circuit_types(
      use_gadgets, target_set, target_manifest
  )
  resolved_batch_size = 128 if batch_size is None else batch_size
  resolved_max_num_moves = 30 if max_num_moves is None else max_num_moves
  resolved_train_batch_size = (
      resolved_batch_size if train_batch_size is None else train_batch_size
  )
  resolved_replay_min_size = (
      resolved_train_batch_size
      if replay_min_size is None else replay_min_size
  )
  resolved_actor_rollout_length = (
      resolved_max_num_moves
      if actor_rollout_length is None else actor_rollout_length
  )
  if resolved_batch_size < 1:
    raise ValueError('batch_size must be positive.')
  if resolved_train_batch_size < 1:
    raise ValueError('train_batch_size must be positive.')
  if resolved_replay_min_size < 1:
    raise ValueError('replay_min_size must be positive.')
  if resolved_actor_rollout_length < 1:
    raise ValueError('actor_rollout_length must be positive.')
  if (
      value_target_mode == 'mc_return'
      and resolved_actor_rollout_length < resolved_max_num_moves
  ):
    raise ValueError(
        'actor_rollout_length must be >= max_num_moves when '
        'value_target_mode=mc_return (else returns are truncated and '
        'episodes span rollouts).'
    )
  if replay_capacity > 0:
    if resolved_replay_min_size > replay_capacity:
      raise ValueError(
          'replay_min_size must be <= replay_capacity when replay is enabled.'
      )
    if resolved_train_batch_size > replay_capacity:
      raise ValueError(
          'train_batch_size must be <= replay_capacity when replay is enabled.'
      )
  resolved_value_support_min = (
      -2.0 * resolved_max_num_moves
      if num_value_categorical_bins > 0 and value_support_min is None
      else -60.0 if value_support_min is None else value_support_min
  )
  resolved_value_support_max = (
      0.0 if value_support_max is None else value_support_max
  )
  if (
      num_value_categorical_bins > 0
      and resolved_value_support_min >= resolved_value_support_max
  ):
    raise ValueError('value_support_min must be < value_support_max.')

  exp_config = ExperimentParams(
      batch_size=resolved_batch_size,
      num_mcts_simulations=80
      if num_mcts_simulations is None else num_mcts_simulations,
      num_training_steps=50_000
      if num_training_steps is None else num_training_steps,
      eval_frequency_steps=50
      if eval_frequency_steps is None else eval_frequency_steps,
      replay_capacity=replay_capacity,
      replay_min_size=resolved_replay_min_size,
      train_batch_size=resolved_train_batch_size,
      num_learner_steps_per_actor=num_learner_steps_per_actor,
      actor_rollout_length=resolved_actor_rollout_length,
      value_target_mode=value_target_mode,
      loss=LossParams(
          init_demonstrations_weight=1.0,
          # Progressively reduce the weight of the demonstrations in favour of
          # the acting episodes.
          demonstrations_boundaries_and_scales={
              60: 0.99, 200: 0.5, 5_000: 0.2, 10_000: 0.1
          },
      ),
  )
  env_config = config_lib.EnvironmentParams(
      max_num_moves=resolved_max_num_moves,
      target_circuit_types=target_circuit_types,
      num_past_factors_to_observe=6,
      change_of_basis=config_lib.ChangeOfBasisParams(
          prob_zero_entry=0.9,
          num_change_of_basis_matrices=80,
          prob_canonical_basis=0.16,
      ),
      use_gadgets=use_gadgets,
  )
  net_config = config_lib.NetworkParams(
      num_layers_torso=num_layers_torso,
      num_value_quantiles=num_value_quantiles,
      value_risk_quantile=value_risk_quantile,
      value_scalar_loss=value_scalar_loss,
      value_huber_delta=value_huber_delta,
      num_value_categorical_bins=num_value_categorical_bins,
      value_support_min=resolved_value_support_min,
      value_support_max=resolved_value_support_max,
      value_target_transform=value_target_transform,
      attention_params=config_lib.AttentionParams(
          num_heads=num_heads,
          head_depth=head_depth,
          mlp_widening_factor=mlp_widening_factor,
      ),
  )
  opt_config = config_lib.OptimizerParams(
      init_lr=1e-3,
      lr_scheduler_transition_steps=5_000,
  )
  dem_config = config_lib.DemonstrationsParams(
      max_num_factors=30,
      max_num_gadgets=5,
      prob_include_gadget=0.9 if use_gadgets else 0.0,
  )
  return DemoConfig(
      exp_config=exp_config,
      env_config=env_config,
      net_config=net_config,
      opt_config=opt_config,
      dem_config=dem_config,
      search_config=SearchParams(
          policy=search_policy,
          gumbel_max_num_considered_actions=gumbel_max_num_considered_actions,
          gumbel_scale=gumbel_scale,
          algebraic_prior_mode=algebraic_prior_mode,
          algebraic_prior_beta=algebraic_prior_beta,
      ),
  )
