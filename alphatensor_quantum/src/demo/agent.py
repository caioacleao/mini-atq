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

"""Agent and states for the AlphaTensor-Quantum demo."""

import functools
from typing import NamedTuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import jaxtyping as jt
import mctx
import optax

from alphatensor_quantum.src import config as config_lib
from alphatensor_quantum.src import demonstrations as demonstrations_lib
from alphatensor_quantum.src import environment
from alphatensor_quantum.src import networks
from alphatensor_quantum.src.demo import algebraic_priors
from alphatensor_quantum.src.demo import demo_config
from alphatensor_quantum.src.demo import replay as replay_lib


class GameStats(NamedTuple):
  """Statistics of the played games.

  Attributes:
    num_games: The number of played games for each considered target. It
      includes a batch dimension.
    best_return: The best return (sum of rewards) for each considered target.
    avg_return: The average return (sum of rewards) for each considered target.
      Like `num_games`, `avg_return` includes a batch dimension; this is solely
      for convenience, as it makes it possible to filter out elements in the
      batch for which `num_games == 0` when computing the effective average
      return.
  """
  num_games: jt.Integer[jt.Array, 'batch_size num_target_tensors']
  best_return: jt.Float[jt.Array, 'num_target_tensors']
  avg_return: jt.Float[jt.Array, 'batch_size num_target_tensors']


class RunState(NamedTuple):
  """The state of the experiment run.

  Attributes:
    params: The network parameters.
    env_states: The environment states.
    demonstrations: The current synthetic demonstrations.
    demonstrations_states: The environment states for the synthetic
      demonstrations.
    opt_state: The optimizer state.
    game_stats: The game statistics.
    rng: A Jax random key.
    replay: Replay storage, or None when the original online loop is used.
  """
  params: chex.ArrayTree
  env_states: environment.EnvState
  demonstrations: demonstrations_lib.Demonstration
  demonstrations_states: environment.EnvState
  opt_state: optax.OptState
  game_stats: GameStats
  rng: chex.PRNGKey
  replay: replay_lib.ReplayBuffer | None = None


class ActorRollout(NamedTuple):
  """Transitions collected by a replay-mode actor rollout."""
  observations: environment.Observation
  policy_targets: jt.Float[jt.Array, 'time batch_size num_actions']
  bootstrap_value_targets: jt.Float[jt.Array, 'time batch_size']
  rewards: jt.Float[jt.Array, 'time batch_size']
  valid: jt.Bool[jt.Array, 'time batch_size']


class NeuralNetwork(hk.Module):
  """Neural network with policy and scalar-or-quantile value heads."""

  def __init__(
      self,
      num_actions: int,
      net_config: config_lib.NetworkParams,
      name: str = 'NeuralNetwork'
  ):
    """Initializes the module.

    Args:
      num_actions: The number of possible actions.
      net_config: The hyperparameters of the neural network.
      name: The name of the module.
    """
    super().__init__(name=name)
    self._num_actions = num_actions
    self._num_value_quantiles = getattr(net_config, 'num_value_quantiles', 0)
    self._num_value_categorical_bins = getattr(
        net_config, 'num_value_categorical_bins', 0
    )
    self._torso = networks.TorsoNetwork(net_config)

  def __call__(
      self, observations: environment.Observation
  ) -> tuple[jt.Float[jt.Array, 'batch_size num_actions'],
             jt.Float[jt.Array, 'batch_size ...']]:
    """Applies the network.

    Args:
      observations: The (batched) observed environment state.

    Returns:
      A 2-tuple:
      - The policy logits.
      - The output of the value head.
    """
    embeddings = self._torso(observations)
    batch_size = embeddings.shape[0]
    reshaped_embeddings = jnp.reshape(embeddings, (batch_size, -1))
    if self._num_value_quantiles > 0:
      num_value_outputs = self._num_value_quantiles
    elif self._num_value_categorical_bins > 0:
      num_value_outputs = self._num_value_categorical_bins
    else:
      num_value_outputs = 1
    outputs = hk.Linear(
        self._num_actions + num_value_outputs
    )(reshaped_embeddings)
    policy_logits = outputs[..., :self._num_actions]
    value_output = outputs[..., self._num_actions:]
    if (
        self._num_value_quantiles == 0
        and self._num_value_categorical_bins == 0
    ):
      value_output = value_output[..., 0]
    return policy_logits, value_output


def _broadcast_shapes(
    x: jt.Shaped[jt.Array, 'batch_size'],
    y: jt.Shaped[jt.Array, 'batch_size ...'],
) -> jt.Shaped[jt.Array, 'batch_size ...']:
  """Broadcasts `x` to a shape compatible with `y`.

  Args:
    x: The array to be broadcasted.
    y: The array whose shape is used as a reference for broadcasting.

  Returns:
    The array `x` reshaped to (batch_size, 1, ..., 1) so that it has the same
    number of dimensions as `y`.
  """
  batch_size = y.shape[0]
  return jnp.reshape(x, [batch_size] + [1] * (len(y.shape) - 1))


class Agent:
  """Simplified version of an AlphaTensor-Quantum agent."""

  def __init__(self, config: demo_config.DemoConfig):
    """Initializes the agent.

    Args:
      config: The config hyperparameters for the demo.
    """
    self._env: environment.Environment  # Initialized in `init_run_state`.
    self._config = config

    self._num_actions = 2 ** config.env_config.max_tensor_size - 1
    self._network = hk.transform(
        lambda obs: NeuralNetwork(self._num_actions, config.net_config)(obs)  # pylint: disable=unnecessary-lambda
    )
    self._search_policy = config.search_config.policy
    self._num_value_quantiles = getattr(
        config.net_config, 'num_value_quantiles', 0
    )
    self._value_risk_quantile = getattr(
        config.net_config, 'value_risk_quantile', 0.0
    )
    self._value_scalar_loss = getattr(
        config.net_config, 'value_scalar_loss', 'mse'
    )
    self._value_huber_delta = getattr(
        config.net_config, 'value_huber_delta', 1.0
    )
    self._num_value_categorical_bins = getattr(
        config.net_config, 'num_value_categorical_bins', 0
    )
    self._value_support_min = getattr(
        config.net_config, 'value_support_min', -60.0
    )
    self._value_support_max = getattr(
        config.net_config, 'value_support_max', 0.0
    )
    self._value_target_transform = getattr(
        config.net_config, 'value_target_transform', 'none'
    )
    self._algebraic_prior_mode = config.search_config.algebraic_prior_mode
    self._algebraic_prior_beta = config.search_config.algebraic_prior_beta
    self._algebraic_prior_data = (
        algebraic_priors.build_action_prior_data(config.env_config.max_tensor_size)
        if self._algebraic_prior_mode != 'none' else None
    )
    self._gumbel_max_num_considered_actions = min(
        config.search_config.gumbel_max_num_considered_actions,
        config.exp_config.num_mcts_simulations,
        self._num_actions,
    )
    # Inialize the optimizer.
    opt_scheduler = optax.exponential_decay(
        init_value=config.opt_config.init_lr,
        transition_steps=config.opt_config.lr_scheduler_transition_steps,
        decay_rate=config.opt_config.lr_scheduler_decay_factor,
        staircase=True,
    )
    self._opt = optax.chain(
        optax.adamw(
            learning_rate=opt_scheduler,
            weight_decay=config.opt_config.weight_decay
        ),
        optax.clip_by_global_norm(config.opt_config.clip_by_global_norm),
    )

  @property
  def change_of_basis(
      self
  ) -> jt.Integer[jt.Array, 'num_matrices size size']:
    """Returns the change-of-basis frame owned by the environment."""
    return self._env.change_of_basis

  def _apply_root_action_prior(
      self,
      policy_logits: jt.Float[jt.Array, 'batch_size num_actions'],
      env_states: environment.EnvState,
  ) -> jt.Float[jt.Array, 'batch_size num_actions']:
    """Adds the configured algebraic action prior to root policy logits."""
    if self._algebraic_prior_mode == 'none':
      return policy_logits
    assert self._algebraic_prior_data is not None
    prior = algebraic_priors.action_prior_logits(
        env_states.tensor,
        self._algebraic_prior_data,
        self._algebraic_prior_mode,
    )
    return policy_logits + self._algebraic_prior_beta * prior

  def _symlog(
      self, x: jt.Float[jt.Array, '...']
  ) -> jt.Float[jt.Array, '...']:
    """Symmetric log transform: sign(x) * log1p(|x|) (cf. DreamerV3)."""
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))

  def _symexp(
      self, x: jt.Float[jt.Array, '...']
  ) -> jt.Float[jt.Array, '...']:
    """Inverse of `_symlog`: sign(x) * expm1(|x|)."""
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))

  def _value_to_scalar(
      self, value_output: jt.Float[jt.Array, 'batch_size ...']
  ) -> jt.Float[jt.Array, 'batch_size']:
    """Reduces scalar or quantile value output to the scalar MCTX consumes."""
    if self._num_value_categorical_bins > 0:
      support = self._categorical_value_support(value_output.dtype)
      probabilities = jax.nn.softmax(value_output, axis=-1)
      return jnp.sum(probabilities * support, axis=-1)
    if self._num_value_quantiles == 0:
      # The scalar value head predicts in transform space (e.g. symlog) when a
      # `value_target_transform` is configured, because the target it is trained
      # against is transformed in `_per_example_value_loss`. Map back to raw
      # return space here so search/decode consumes the value on the same scale
      # as rewards. With `value_target_transform='none'` this is the identity,
      # keeping default behavior byte-identical.
      if self._value_target_transform == 'symlog':
        return self._symexp(value_output)
      return value_output
    dtype = value_output.dtype
    taus = (
        jnp.arange(self._num_value_quantiles, dtype=dtype) + 0.5
    ) / self._num_value_quantiles
    mask = taus >= self._value_risk_quantile
    mask = mask.astype(dtype)
    count = jnp.sum(mask)
    masked_mean = jnp.sum(value_output * mask, axis=-1) / jnp.maximum(
        count, jnp.asarray(1.0, dtype=dtype)
    )
    return jnp.where(count > 0, masked_mean, value_output[..., -1])

  def _categorical_value_support(self, dtype: jnp.dtype) -> jnp.ndarray:
    """Returns the static support for categorical value logits."""
    return jnp.linspace(
        jnp.asarray(self._value_support_min, dtype=dtype),
        jnp.asarray(self._value_support_max, dtype=dtype),
        self._num_value_categorical_bins,
        dtype=dtype,
    )

  def _categorical_two_hot_targets(
      self,
      value_targets: jt.Float[jt.Array, 'batch_size'],
      dtype: jnp.dtype,
  ) -> jt.Float[jt.Array, 'batch_size num_bins']:
    """Projects scalar value targets to adjacent categorical support bins."""
    support_min = jnp.asarray(self._value_support_min, dtype=dtype)
    support_max = jnp.asarray(self._value_support_max, dtype=dtype)
    clipped = jnp.clip(value_targets.astype(dtype), support_min, support_max)
    num_intervals = self._num_value_categorical_bins - 1
    position = (
        (clipped - support_min)
        / (support_max - support_min)
        * num_intervals
    )
    lower = jnp.floor(position).astype(jnp.int32)
    upper = jnp.ceil(position).astype(jnp.int32)
    lower = jnp.clip(lower, 0, self._num_value_categorical_bins - 1)
    upper = jnp.clip(upper, 0, self._num_value_categorical_bins - 1)
    upper_weight = position - jnp.floor(position)
    lower_weight = 1.0 - upper_weight
    targets = jax.nn.one_hot(
        lower, self._num_value_categorical_bins, dtype=dtype
    ) * lower_weight[..., None]
    targets += jax.nn.one_hot(
        upper, self._num_value_categorical_bins, dtype=dtype
    ) * upper_weight[..., None]
    return targets

  def _per_example_value_loss(
      self,
      value_output: jt.Float[jt.Array, 'batch_size ...'],
      value_targets: jt.Float[jt.Array, 'batch_size'],
  ) -> jt.Float[jt.Array, 'batch_size']:
    """Returns MSE or quantile pinball loss for scalar targets."""
    if self._num_value_categorical_bins > 0:
      dtype = value_output.dtype
      targets = self._categorical_two_hot_targets(value_targets, dtype)
      logprobs = jax.nn.log_softmax(value_output, axis=-1)
      return -jnp.sum(targets * logprobs, axis=-1)
    if self._num_value_quantiles == 0:
      # Transform ONLY the scalar target (e.g. symlog) to tame heavy-tailed
      # returns. The head output is left untransformed: it is trained to predict
      # in transform space, and `_value_to_scalar` applies the inverse when the
      # value is read back for search. With `value_target_transform='none'` this
      # is the identity, keeping default behavior byte-identical.
      if self._value_target_transform == 'symlog':
        value_targets = self._symlog(value_targets)
      errors = value_output - value_targets
      if self._value_scalar_loss == 'huber':
        abs_errors = jnp.abs(errors)
        delta = jnp.asarray(self._value_huber_delta, dtype=errors.dtype)
        quadratic = jnp.minimum(abs_errors, delta)
        linear = abs_errors - quadratic
        return 0.5 * jnp.square(quadratic) + delta * linear
      return jnp.square(errors)
    dtype = value_output.dtype
    taus = (
        jnp.arange(self._num_value_quantiles, dtype=dtype) + 0.5
    ) / self._num_value_quantiles
    errors = value_targets[..., None] - value_output
    pinball_loss = errors * (
        taus - (errors < 0).astype(dtype)
    )
    return jnp.mean(pinball_loss, axis=-1)

  def _run_search(
      self,
      params: chex.ArrayTree,
      rng: chex.PRNGKey,
      root: mctx.RootFnOutput,
  ) -> mctx.PolicyOutput:
    """Runs the configured MCTX policy-improvement operator."""
    if self._search_policy == 'muzero':
      return mctx.muzero_policy(
          params=params,
          rng_key=rng,
          root=root,
          recurrent_fn=self._recurrent_fn,
          num_simulations=self._config.exp_config.num_mcts_simulations,
          qtransform=mctx.qtransform_by_parent_and_siblings,
      )
    if self._search_policy == 'gumbel':
      return mctx.gumbel_muzero_policy(
          params=params,
          rng_key=rng,
          root=root,
          recurrent_fn=self._recurrent_fn,
          num_simulations=self._config.exp_config.num_mcts_simulations,
          max_num_considered_actions=self._gumbel_max_num_considered_actions,
          gumbel_scale=self._config.search_config.gumbel_scale,
      )
    raise ValueError(f'Unsupported search_policy: {self._search_policy}.')

  def init_run_state(
      self, rng: chex.PRNGKey, *, allocate_replay: bool = True
  ) -> RunState:
    """Initializes the run state.

    Args:
      rng: A Jax random key.
      allocate_replay: Whether to allocate the replay buffer when replay mode is
        configured. Eval-only callers can set this to false because they only
        need the environment and network shape before loading checkpoint params.

    Returns:
      A run state.
    """
    (
        rng_env,
        rng_env_states,
        rng_demonstrations,
        rng_params,
        rng_run_state
    ) = jax.random.split(rng, num=5)

    # Initialize the environment, the environment states, the synthetic
    # demonstrations, and the network parameters.
    self._env = environment.Environment(rng_env, self._config.env_config)
    env_states = self._env.init_state(
        jax.random.split(rng_env_states, self._config.exp_config.batch_size)
    )
    demonstrations = demonstrations_lib.generate_synthetic_demonstrations(
        self._config.env_config.max_tensor_size,
        self._config.dem_config,
        jax.random.split(
            rng_demonstrations, num=self._config.exp_config.batch_size
        )
    )
    initial_observations = self._env.get_observation(env_states)
    params = self._network.init(rng_params, initial_observations)
    replay = (
        replay_lib.buffer_init(
            self._config.exp_config.replay_capacity,
            initial_observations,
            self._num_actions,
        )
        if (
            allocate_replay
            and self._config.exp_config.replay_capacity > 0
        ) else None
    )
    # Initialize the game statistics.
    num_target_tensors = len(self._config.env_config.target_circuit_types)
    game_stats = GameStats(
        num_games=jnp.zeros(
            (self._config.exp_config.batch_size, num_target_tensors,),
            dtype=jnp.int32
        ),
        best_return=jnp.array([-jnp.inf] * num_target_tensors),
        avg_return=jnp.zeros(
            (self._config.exp_config.batch_size, num_target_tensors)
        ),
    )
    return RunState(
        params=params,
        env_states=env_states,
        demonstrations=demonstrations,
        demonstrations_states=self._env.init_state_from_demonstration(
            demonstrations
        ),
        opt_state=self._opt.init(params),
        game_stats=game_stats,
        rng=rng_run_state,
        replay=replay,
    )

  def _recurrent_fn(
      self,
      params: chex.ArrayTree,
      rng: chex.PRNGKey,
      actions: jt.Integer[jt.Array, 'batch_size'],
      env_states: environment.EnvState
  ) -> tuple[mctx.RecurrentFnOutput, environment.EnvState]:
    """Implements the recurrent policy.

    In AlphaTensor-Quantum, the environment is deterministic, so there is no
    need for a recurrent function that captures the environment dynamics.
    Instead of a neural network that predicts some embeddings representing the
    environment state, we return the environment state itself.

    Args:
      params: The network parameters.
      rng: A Jax random key.
      actions: The batched action indices.
      env_states: The batched environment states.

    Returns:
      A 2-tuple:
      - The output of the recurrent function.
      - The new environment states.
    """
    env_states = self._env.step(actions, env_states)
    observations = self._env.get_observation(env_states)
    policy_logits, value_output = self._network.apply(
        params, rng, observations
    )
    values = self._value_to_scalar(value_output)
    recurrent_fn_output = mctx.RecurrentFnOutput(
        prior_logits=policy_logits,
        value=values,
        reward=env_states.last_reward,
        discount=1.0 - env_states.is_terminal
    )
    return recurrent_fn_output, env_states

  def _loss_fn(
      self,
      params: chex.ArrayTree,
      global_step: int,
      acting_observations: environment.Observation,
      acting_policy_targets: jt.Float[jt.Array, 'batch_size num_actions'],
      acting_value_targets: jt.Float[jt.Array, 'batch_size'],
      demonstrations_observations: environment.Observation,
      demonstrations_policy_targets: jt.Float[jt.Array,
                                              'batch_size num_actions'],
      demonstrations_value_targets: jt.Float[jt.Array, 'batch_size'],
      rng: chex.PRNGKey,
      acting_valid: jt.Bool[jt.Array, 'batch_size'] | None = None,
  ) -> jt.Float[jt.Scalar, '']:
    """Obtains the loss.

    Args:
      params: The network parameters.
      global_step: The training step.
      acting_observations: The (batched) observed environment state.
      acting_policy_targets: The (batched) policy targets from the actors.
      acting_value_targets: The (batched) value targets from the actors.
      demonstrations_observations: The (batched) observed environment state from
        the synthetic demonstrations.
      demonstrations_policy_targets: The (batched) policy targets for the
        synthetic demonstrations.
      demonstrations_value_targets: The (batched) value targets for the
        synthetic demonstrations.
      rng: A Jax random key.
      acting_valid: Optional mask for replay samples.

    Returns:
      The sum of the policy and value losses.
    """
    rng_acting, rng_demonstrations = jax.random.split(rng, num=2)

    # Loss corresponding to the episodes from acting.
    acting_policy_logits, acting_value_output = self._network.apply(
        params, rng_acting, acting_observations
    )
    acting_policy_logprobs = jax.nn.log_softmax(acting_policy_logits)
    acting_policy_loss = jnp.sum(acting_policy_targets * (
        jnp.log(acting_policy_targets) - acting_policy_logprobs
    ), axis=-1)
    acting_value_loss = self._per_example_value_loss(
        acting_value_output, acting_value_targets
    )
    acting_per_example_loss = acting_policy_loss + acting_value_loss
    if acting_valid is None:
      acting_loss = jnp.mean(acting_per_example_loss)
    else:
      acting_valid = acting_valid.astype(jnp.float32)
      acting_loss = (
          jnp.sum(acting_per_example_loss * acting_valid)
          / jnp.maximum(jnp.sum(acting_valid), 1.0)
      )

    # Loss corresponding to the episodes from synthetic demonstrations.
    (
        demonstrations_policy_logits,
        demonstrations_value_output,
    ) = self._network.apply(
        params, rng_demonstrations, demonstrations_observations
    )
    demonstrations_policy_logprobs = jax.nn.log_softmax(
        demonstrations_policy_logits
    )
    demonstrations_policy_loss = -jnp.sum(
        demonstrations_policy_targets * demonstrations_policy_logprobs,
        axis=-1
    )
    demonstrations_value_loss = self._per_example_value_loss(
        demonstrations_value_output, demonstrations_value_targets
    )
    demonstrations_loss = jnp.mean(
        demonstrations_policy_loss + demonstrations_value_loss
    )

    # Obtain the weight for the two terms in the loss.
    demonstrations_weight = optax.piecewise_constant_schedule(
        init_value=self._config.exp_config.loss.init_demonstrations_weight,
        boundaries_and_scales=(
            self._config.exp_config.loss.demonstrations_boundaries_and_scales
        )
    )(global_step)
    return (
        (1.0 - demonstrations_weight) * acting_loss
        + demonstrations_weight * demonstrations_loss
    )

  def _update_game_stats(
      self, run_state: RunState, new_env_states: environment.EnvState
  ) -> GameStats:
    """Returns the new game statistics."""
    return self._update_game_stats_values(run_state.game_stats, new_env_states)

  def _update_game_stats_values(
      self, game_stats: GameStats, new_env_states: environment.EnvState
  ) -> GameStats:
    """Updates game statistics from newly stepped environment states."""
    is_terminal = new_env_states.is_terminal
    new_num_games_if_terminal = jax.vmap(
        lambda x, idx: x.at[idx].set(x[idx] + 1)
    )(game_stats.num_games, new_env_states.init_tensor_index)
    new_num_games = jnp.where(
        _broadcast_shapes(is_terminal, game_stats.num_games),
        new_num_games_if_terminal,
        game_stats.num_games
    )
    smoothing = self._config.exp_config.avg_return_smoothing
    new_avg_return_if_terminal = jax.vmap(
        lambda x, v, i: x.at[i].set(smoothing * x[i] + (1 - smoothing) * v)
    )(
        game_stats.avg_return,
        new_env_states.sum_rewards,
        new_env_states.init_tensor_index
    )
    new_avg_return = jnp.where(
        _broadcast_shapes(is_terminal, game_stats.avg_return),
        new_avg_return_if_terminal,
        game_stats.avg_return
    )
    num_target_tensors = len(self._config.env_config.target_circuit_types)
    negative_inf = -jnp.inf * jnp.ones(
        (self._config.exp_config.batch_size, num_target_tensors)
    )
    new_best_return_if_terminal = jax.vmap(lambda x, v, i: x.at[i].set(v))(
        negative_inf,
        new_env_states.sum_rewards,
        new_env_states.init_tensor_index
    )
    new_best_return = jnp.maximum(
        game_stats.best_return,
        jnp.max(jnp.where(
            _broadcast_shapes(is_terminal, new_best_return_if_terminal),
            new_best_return_if_terminal,
            negative_inf
        ), axis=0)
    )
    return GameStats(
        num_games=new_num_games,
        avg_return=new_avg_return,
        best_return=new_best_return,
    )

  def _update_demonstrations_and_states(
      self,
      demonstrations_actions: jt.Integer[jt.Array, 'batch_size'],
      run_state: RunState,
      rng: chex.PRNGKey
  ) -> tuple[demonstrations_lib.Demonstration, environment.EnvState]:
    """Updates the synthetic demonstrations and their states."""

    # Take a step for the environment states.
    new_demonstrations_states = self._env.step(
        demonstrations_actions, run_state.demonstrations_states
    )

    # Update the demonstrations if their corresponding episodes have terminated.
    new_demonstrations_if_terminal = (
        demonstrations_lib.generate_synthetic_demonstrations(
            self._config.env_config.max_tensor_size,
            self._config.dem_config,
            jax.random.split(rng, num=self._config.exp_config.batch_size),
        )
    )
    new_demonstrations = jax.tree_util.tree_map(
        lambda x, y: jnp.where(
            _broadcast_shapes(new_demonstrations_states.is_terminal, x), x, y
        ),
        new_demonstrations_if_terminal,
        run_state.demonstrations
    )

    # Update the demonstrations states for terminated episodes.
    new_demonstrations_states_if_terminal = (
        self._env.init_state_from_demonstration(new_demonstrations_if_terminal)
    )
    new_demonstrations_states = jax.tree_util.tree_map(
        lambda x, y: jnp.where(
            _broadcast_shapes(new_demonstrations_states.is_terminal, x), x, y
        ),
        new_demonstrations_states_if_terminal,
        new_demonstrations_states
    )
    return new_demonstrations, new_demonstrations_states

  def _run_iteration_agent_env_interaction(
      self, global_step: int, run_state: RunState
  ) -> RunState:
    """Runs one iteration of the agent-environment interaction loop.

    Args:
      global_step: The training step.
      run_state: The run state.

    Returns:
      The new run state.
    """
    rngs = jax.random.split(run_state.rng, num=7)

    acting_observations = self._env.get_observation(run_state.env_states)
    policy_logits, value_output = self._network.apply(
        run_state.params, rngs[0], acting_observations
    )
    values = self._value_to_scalar(value_output)
    policy_logits = self._apply_root_action_prior(
        policy_logits, run_state.env_states
    )
    root = mctx.RootFnOutput(
        prior_logits=policy_logits,
        value=values,
        embedding=run_state.env_states,
    )
    policy_output = self._run_search(run_state.params, rngs[1], root)
    search_value = policy_output.search_tree.node_values[
        :, policy_output.search_tree.ROOT_INDEX
    ]

    # Obtain the observations and the policy and value targets for the synthetic
    # demonstrations.
    demonstrations_observations = self._env.get_observation(
        run_state.demonstrations_states
    )
    (
        demonstrations_actions,
        demonstrations_value_targets
    ) = demonstrations_lib.get_action_and_value(
        run_state.demonstrations,
        run_state.demonstrations_states.num_moves,
    )

    # Compute the gradient of the loss and take a grad step.
    grads = jax.grad(self._loss_fn)(
        run_state.params,
        global_step,
        acting_observations,
        policy_output.action_weights,
        search_value,
        demonstrations_observations,
        jax.nn.one_hot(demonstrations_actions, self._num_actions),
        demonstrations_value_targets,
        rngs[2]
    )
    updates, new_opt_state = self._opt.update(
        grads, run_state.opt_state, run_state.params
    )
    new_params = optax.apply_updates(run_state.params, updates)

    # Select next action probabilistically based on visit counts.
    actions = jax.vmap(
        lambda r, p: jax.random.choice(r, a=self._num_actions, p=p)
    )(
        jax.random.split(rngs[3], self._config.exp_config.batch_size),
        policy_output.action_weights
    )
    new_env_states = self._env.step(actions, run_state.env_states)
    is_terminal = new_env_states.is_terminal

    # Update game statistics.
    new_game_stats = self._update_game_stats(run_state, new_env_states)

    # Reset the environment state if the episode has terminated.
    new_env_states = jax.tree_util.tree_map(
        lambda x, y: jnp.where(_broadcast_shapes(is_terminal, x), x, y),
        self._env.init_state(
            jax.random.split(rngs[4], num=self._config.exp_config.batch_size)
        ),
        new_env_states
    )

    # Reset the demonstrations and their states if the corresponding episodes
    # have terminated.
    (
        new_demonstrations, new_demonstrations_states
    ) = self._update_demonstrations_and_states(
        demonstrations_actions, run_state, rngs[5]
    )

    return RunState(
        params=new_params,
        env_states=new_env_states,
        demonstrations=new_demonstrations,
        demonstrations_states=new_demonstrations_states,
        opt_state=new_opt_state,
        game_stats=new_game_stats,
        rng=rngs[6],
        replay=run_state.replay,
    )

  def _run_replay_actor_rollout(self, run_state: RunState) -> RunState:
    """Collects one batched actor rollout and inserts it into replay."""
    assert run_state.replay is not None
    rng_rollout, rng_reset, rng_out = jax.random.split(run_state.rng, num=3)
    rollout_rngs = jax.random.split(
        rng_rollout, self._config.exp_config.actor_rollout_length
    )

    def scan_step(carry, rng):
      env_states, game_stats = carry
      rng_network, rng_search, rng_action = jax.random.split(rng, num=3)
      was_terminal = env_states.is_terminal
      valid = jnp.logical_not(was_terminal)

      observations = self._env.get_observation(env_states)
      policy_logits, value_output = self._network.apply(
          run_state.params, rng_network, observations
      )
      values = self._value_to_scalar(value_output)
      policy_logits = self._apply_root_action_prior(policy_logits, env_states)
      root = mctx.RootFnOutput(
          prior_logits=policy_logits,
          value=values,
          embedding=env_states,
      )
      policy_output = self._run_search(run_state.params, rng_search, root)
      search_value = policy_output.search_tree.node_values[
          :, policy_output.search_tree.ROOT_INDEX
      ]

      actions = jax.vmap(
          lambda r, p: jax.random.choice(r, a=self._num_actions, p=p)
      )(
          jax.random.split(rng_action, self._config.exp_config.batch_size),
          policy_output.action_weights,
      )
      stepped_env_states = self._env.step(actions, env_states)
      new_env_states = jax.tree_util.tree_map(
          lambda old, new: jnp.where(_broadcast_shapes(was_terminal, old),
                                     old, new),
          env_states,
          stepped_env_states,
      )

      stats_env_states = new_env_states._replace(
          is_terminal=jnp.logical_and(new_env_states.is_terminal, valid)
      )
      new_game_stats = self._update_game_stats_values(
          game_stats, stats_env_states
      )
      rollout_step = ActorRollout(
          observations=observations,
          policy_targets=policy_output.action_weights,
          bootstrap_value_targets=search_value,
          rewards=jnp.where(valid, stepped_env_states.last_reward, 0.0),
          valid=valid,
      )
      return (new_env_states, new_game_stats), rollout_step

    (env_states, game_stats), rollout = jax.lax.scan(
        scan_step,
        (run_state.env_states, run_state.game_stats),
        rollout_rngs,
    )
    if self._config.exp_config.value_target_mode == 'mc_return':
      value_targets = replay_lib.discountless_return_to_go(
          rollout.rewards, rollout.valid
      )
    else:
      value_targets = rollout.bootstrap_value_targets

    def flatten_time_batch(value):
      return jnp.reshape(
          value, (value.shape[0] * value.shape[1],) + value.shape[2:]
      )

    transitions = replay_lib.ReplayBatch(
        observations=jax.tree_util.tree_map(
            flatten_time_batch, rollout.observations
        ),
        policy_targets=flatten_time_batch(rollout.policy_targets),
        value_targets=flatten_time_batch(value_targets),
        valid=flatten_time_batch(rollout.valid),
    )
    new_replay = replay_lib.buffer_insert(run_state.replay, transitions)

    reset_env_states = self._env.init_state(
        jax.random.split(rng_reset, num=self._config.exp_config.batch_size)
    )
    is_terminal = env_states.is_terminal
    env_states = jax.tree_util.tree_map(
        lambda reset, current: jnp.where(
            _broadcast_shapes(is_terminal, current), reset, current
        ),
        reset_env_states,
        env_states,
    )
    return run_state._replace(
        env_states=env_states,
        game_stats=game_stats,
        replay=new_replay,
        rng=rng_out,
    )

  def _run_replay_learner_step(
      self, global_step: int, run_state: RunState
  ) -> RunState:
    """Runs one replay learner update when enough valid samples exist."""
    assert run_state.replay is not None

    def train(state):
      rng_sample, rng_grad, rng_demonstrations, rng_out = jax.random.split(
          state.rng, num=4
      )
      batch = replay_lib.buffer_sample(
          state.replay, rng_sample, self._config.exp_config.train_batch_size
      )
      demonstrations_observations = self._env.get_observation(
          state.demonstrations_states
      )
      (
          demonstrations_actions,
          demonstrations_value_targets
      ) = demonstrations_lib.get_action_and_value(
          state.demonstrations,
          state.demonstrations_states.num_moves,
      )
      grads = jax.grad(self._loss_fn)(
          state.params,
          global_step,
          batch.observations,
          batch.policy_targets,
          batch.value_targets,
          demonstrations_observations,
          jax.nn.one_hot(demonstrations_actions, self._num_actions),
          demonstrations_value_targets,
          rng_grad,
          batch.valid,
      )
      updates, new_opt_state = self._opt.update(
          grads, state.opt_state, state.params
      )
      new_params = optax.apply_updates(state.params, updates)
      (
          new_demonstrations, new_demonstrations_states
      ) = self._update_demonstrations_and_states(
          demonstrations_actions, state, rng_demonstrations
      )
      return state._replace(
          params=new_params,
          demonstrations=new_demonstrations,
          demonstrations_states=new_demonstrations_states,
          opt_state=new_opt_state,
          rng=rng_out,
      )

    def skip(state):
      _, rng_out = jax.random.split(state.rng)
      return state._replace(rng=rng_out)

    can_train = (
        replay_lib.valid_size(run_state.replay)
        >= self._config.exp_config.replay_min_size
    )
    return jax.lax.cond(can_train, train, skip, run_state)

  def _run_replay_iteration(
      self, global_step: int, run_state: RunState
  ) -> RunState:
    """Runs one actor rollout followed by replay learner updates."""
    run_state = self._run_replay_actor_rollout(run_state)

    def learner_body(learner_step, state):
      learner_global_step = (
          global_step * self._config.exp_config.num_learner_steps_per_actor
          + learner_step
      )
      return self._run_replay_learner_step(learner_global_step, state)

    return jax.lax.fori_loop(
        lower=0,
        upper=self._config.exp_config.num_learner_steps_per_actor,
        body_fun=learner_body,
        init_val=run_state,
    )

  def _run_eval_step(
      self,
      params: chex.ArrayTree,
      rng: chex.PRNGKey,
      env_states: environment.EnvState,
  ) -> tuple[environment.EnvState, jt.Integer[jt.Array, 'batch_size']]:
    """Runs one search-selected environment step without learning."""
    rng_network, rng_search = jax.random.split(rng)
    observations = self._env.get_observation(env_states)
    policy_logits, value_output = self._network.apply(
        params, rng_network, observations
    )
    values = self._value_to_scalar(value_output)
    policy_logits = self._apply_root_action_prior(policy_logits, env_states)
    root = mctx.RootFnOutput(
        prior_logits=policy_logits,
        value=values,
        embedding=env_states,
    )
    policy_output = self._run_search(params, rng_search, root)
    actions = policy_output.action
    stepped_env_states = self._env.step(actions, env_states)
    was_terminal = env_states.is_terminal
    new_env_states = jax.tree_util.tree_map(
        lambda old, new: jnp.where(_broadcast_shapes(was_terminal, old),
                                   old, new),
        env_states,
        stepped_env_states,
    )
    actions = jnp.where(was_terminal, -jnp.ones_like(actions), actions)
    return new_env_states, actions

  @functools.partial(jax.jit, static_argnums=(0,))
  def run_eval_step(
      self,
      params: chex.ArrayTree,
      rng: chex.PRNGKey,
      env_states: environment.EnvState,
  ) -> tuple[environment.EnvState, jt.Integer[jt.Array, 'batch_size']]:
    """Runs one eval-only decode step using the configured search policy."""
    return self._run_eval_step(params, rng, env_states)

  @functools.partial(jax.jit, static_argnums=(0,))
  def run_agent_env_interaction(
      self, global_step: int, run_state: RunState
  ) -> RunState:
    """Runs a few iterations of the agent-environment interaction loop.

    With replay enabled, one loop index is one actor rollout, not one gradient
    update: the actor collects `actor_rollout_length * batch_size` environment
    steps, then the learner applies `num_learner_steps_per_actor` optimizer
    updates. Schedules keyed by optimizer updates, such as the learning rate and
    demonstration weight, therefore advance once per learner update. Replay
    storage also includes frozen invalid rollout slots; `valid_size` can be much
    smaller than capacity for targets that terminate quickly.

    Args:
      global_step: The training step.
      run_state: The run state.

    Returns:
      The new run state, after running `eval_frequency_steps` tranining steps.
    """
    if self._config.exp_config.replay_capacity > 0:
      return jax.lax.fori_loop(
          lower=global_step,
          upper=self._config.exp_config.eval_frequency_steps + global_step,
          body_fun=self._run_replay_iteration,
          init_val=run_state,
      )
    return jax.lax.fori_loop(
        lower=global_step,
        upper=self._config.exp_config.eval_frequency_steps + global_step,
        body_fun=self._run_iteration_agent_env_interaction,
        init_val=run_state,
    )
