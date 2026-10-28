import copy
import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from gymnasium import spaces
from ray.rllib.algorithms.alpha_zero.alpha_zero_policy import AlphaZeroPolicy
from ray.rllib.evaluation import SampleBatch
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.postprocessing import Postprocessing, discount_cumsum
from ray.rllib.models import ActionDistribution, ModelV2
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.torch_mixins import EntropyCoeffSchedule, LearningRateSchedule
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.schedules import ConstantSchedule, PiecewiseSchedule, Schedule
from ray.rllib.utils.torch_utils import (
    apply_grad_clipping,
    explained_variance,
    sequence_mask,
)
from ray.rllib.utils.typing import AgentID, PolicyID, TensorStructType, TensorType
from ray.tune.registry import ENV_CREATOR, _global_registry
from torch import nn

from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.config import MbagConfigDict
from mbag.environment.goals import ALL_GOAL_GENERATORS
from mbag.environment.goals.embedding import embed_goal, get_goal_size
from mbag.environment.goals.fixed_pool import FixedPoolGoalGenerator
from mbag.environment.goals.possible_goals import (
    MinecraftStructureGoal,
    RandomStructurePossibleGoalSampler,
)
from mbag.environment.mbag_env import MbagEnv
from mbag.environment.state import MbagStateDict
from mbag.environment.types import (
    CURRENT_BLOCK_STATES,
    CURRENT_BLOCKS,
    GOAL_BLOCK_STATES,
    GOAL_BLOCKS,
    MbagInfoDict,
    WorldSize,
)

from ..kl_regularization import ANCHOR_POLICY_ACTION_DIST_INPUTS
from ..rllib_env import unwrap_mbag_env
from ..torch_models import (
    ACTION_MASK,
    MbagTorchModel,
    OptimizerMixin,
    OtherAgentActionPredictorMixin,
)
from .mcts import MbagMCTS, MbagMCTSNode, MbagRootParentNode
from .planning import MbagEnvModel
from .power.formulas import human_goal_ability_monte_carlo_targets
from .power.human_policy_prior import HeuristicMbagHumanPolicyPrior
from .power.progress import ProgressReference, StepsToGoalReference

ENV_STATES = "env_states"
PRIOR_POLICIES = "prior_policies"
MCTS_POLICIES = "mcts_policies"
OTHER_AGENT_ACTION_DIST_INPUTS = "other_agent_action_dist_inputs"
OWN_REWARDS = "own_rewards"
EXPECTED_REWARDS = "expected_rewards"
EXPECTED_OWN_REWARDS = "expected_own_rewards"
VALUE_ESTIMATES = "value_estimates"
GOAL_LOGITS = "goal_logits"
PREV_GOAL_KL_COEFF = "prev_goal_kl_coeff"
POSSIBLE_GOAL_BLOCKS = "possible_goal_blocks"
HUMAN_GOAL_ABILITY_TARGETS = "human_goal_ability_targets"
AGGREGATE_GOAL_ABILITY_TARGETS = "aggregate_goal_ability_targets"
FORCE_NOOP = "force_noop"
C_PUCT = "c_puct"
PREV_C_PUCT = "prev_c_puct"
FOR_TRAINING_MODEL = "for_training_model"


logger = logging.getLogger(__name__)


class MbagAlphaZeroPolicy(
    EntropyCoeffSchedule, LearningRateSchedule, OptimizerMixin, AlphaZeroPolicy
):
    mcts: MbagMCTS
    envs: List[MbagEnvModel]
    config: Dict[str, Any]

    def __init__(
        self,
        observation_space,
        action_space,
        config,
        **kwargs,
    ):
        TorchPolicy.__init__(
            self,
            observation_space,
            action_space,
            config,
            **kwargs,
        )

        self.set_training(False)
        # We default to setting policies as not training and only update this when
        # train() is actually called. This ensures that if policies are loaded for
        # evaluation then the shaped reward annealing is not used.

        model = self.model
        assert isinstance(model, MbagTorchModel)
        line_of_sight_masking = model.line_of_sight_masking

        self.use_power_reward: bool = config.get("use_power_reward", False)
        mcts_config = dict(config["mcts_config"])
        mcts_config.setdefault("power_xi", config.get("power_xi", 1.0))
        mcts_config.setdefault("power_eta", config.get("power_eta", 1.0))
        mcts_config.setdefault("power_x_epsilon", config.get("power_x_epsilon", 0.0))
        self.mcts = MbagMCTS(
            self.model,
            mcts_config,
            config["gamma"],
            use_critic=config["use_critic"],
            use_goal_predictor=config["use_goal_predictor"],
            use_power_reward=self.use_power_reward,
            use_other_agent_action_predictor=config.get(
                "use_other_agent_action_predictor", True
            ),
            _strict_mode=config.get("_strict_mode", False),
        )

        def env_creator():
            env_creator = _global_registry.get(ENV_CREATOR, config["env"])
            # We should never use Malmo in the env model.
            env_config: MbagConfigDict = copy.deepcopy(config["env_config"])
            env_config["malmo"]["use_malmo"] = False
            # Don't waste time generating goals in the env model.
            env_config["goal_generator"] = "basic"
            env_config["goal_generator_config"] = {}
            # Don't plan based on truncating early.
            env_config["truncate_on_no_progress_timesteps"] = None
            # If we're using a goal predictor, then we shouldn't end the episode when
            # the goal is completed because that leaks information about the goal.
            if self.mcts.use_goal_predictor:
                env_config["terminate_on_goal_completion"] = False
            # In case we're being fed observations from Malmo or from human data where
            # players might be overlapping.
            env_config["_check_for_overlapping_players"] = False
            env = env_creator(env_config)
            env_model = MbagEnvModel(
                env,
                env_config,
                line_of_sight_masking=line_of_sight_masking,
                expected_own_reward_scale=config.get("expected_own_reward_scale", 1.0),
                expected_reward_shift=config.get("expected_reward_shift", 0.0),
            )
            unwrap_mbag_env(env_model).update_global_timestep(
                self.global_timestep_for_envs
            )
            return env_model

        self.env_creator = env_creator
        self.envs = []
        self.obs_space = observation_space

        original_obs_space = self.obs_space
        if hasattr(original_obs_space, "original_space"):
            original_obs_space = original_obs_space.original_space
        assert isinstance(original_obs_space, spaces.Tuple)
        world_obs_space = original_obs_space.spaces[0]
        assert world_obs_space.shape is not None
        world_size = cast(WorldSize, world_obs_space.shape[1:])

        self.view_requirements[ACTION_MASK] = ViewRequirement(
            space=spaces.MultiBinary(action_space.n)
        )
        self.view_requirements[MCTS_POLICIES] = ViewRequirement(
            space=spaces.Box(low=0, high=1, shape=(action_space.n,))
        )
        self.view_requirements[PRIOR_POLICIES] = ViewRequirement(
            space=spaces.Box(low=0, high=1, shape=(action_space.n,))
        )
        self.view_requirements[SampleBatch.ACTION_DIST_INPUTS] = ViewRequirement(
            space=spaces.Box(low=-np.inf, high=np.inf, shape=(action_space.n,))
        )
        self.view_requirements[EXPECTED_REWARDS] = ViewRequirement(
            space=spaces.Box(low=-np.inf, high=np.inf, shape=())
        )
        self.view_requirements[EXPECTED_OWN_REWARDS] = ViewRequirement(
            space=spaces.Box(low=-np.inf, high=np.inf, shape=())
        )
        self.view_requirements[VALUE_ESTIMATES] = ViewRequirement(
            space=spaces.Box(low=-np.inf, high=np.inf, shape=())
        )
        if self.mcts.use_goal_predictor:
            self.view_requirements[GOAL_LOGITS] = ViewRequirement(
                space=spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(MinecraftBlocks.NUM_BLOCKS,) + world_size,
                )
            )

        # Defaults so loss() can reference these unconditionally; overridden
        # below (to nonzero defaults) only when the power objective is active.
        self.human_goal_ability_loss_coeff: float = 0.0
        self.aggregate_goal_ability_loss_coeff: float = 0.0

        if self.use_power_reward:
            env_config = config["env_config"]
            goal_generator_cls = env_config["goal_generator"]
            if isinstance(goal_generator_cls, str):
                goal_generator_cls = ALL_GOAL_GENERATORS[goal_generator_cls]
            goal_generator = goal_generator_cls(
                env_config.get("goal_generator_config", {})
            )
            # One possible goal g_h is sampled per trajectory fragment (not per
            # timestep), independently of the true goal, so that V_h^e(s_t, g_h)
            # forms a valid TD sequence across the fragment (see
            # postprocess_trajectory).
            self.power_goal_lenient: bool = (
                config.get("power_goal_predicate", "exact") == "lenient"
            )
            self.power_possible_goal_sampler = RandomStructurePossibleGoalSampler(
                goal_generator,
                world_size,
                inf_blocks=MbagEnv.get_config(env_config)["abilities"]["inf_blocks"],
                lenient=self.power_goal_lenient,
            )
            self.power_value_from_ur: bool = config.get("power_value_from_ur", False)
            self.power_rollout_goal: bool = config.get("power_rollout_goal", False)
            self.power_num_goals: float = float(config.get("power_num_goals", 1))
            if config.get("power_progress_baseline", False):
                self._set_up_power_progress_baseline(
                    goal_generator, world_size, env_config, config
                )
            self.power_gamma_h: float = config.get("power_gamma_h", config["gamma"])
            self.power_zeta: float = config.get("power_zeta", 2.0)
            self.human_goal_ability_loss_coeff = config.get(
                "human_goal_ability_loss_coeff", 1.0
            )
            self.aggregate_goal_ability_loss_coeff = config.get(
                "aggregate_goal_ability_loss_coeff", 1.0
            )
            self.power_human_policy_prior = HeuristicMbagHumanPolicyPrior(
                config["env_config"], beta_h=config.get("power_beta_h", 10.0)
            )
            # MbagMCTS was constructed above (before world_size/env_config were
            # available here); attach the human policy prior and possible-goal
            # sampler it needs to simulate the human during tree search.
            self.mcts.power_human_policy_prior = self.power_human_policy_prior
            self.mcts.power_possible_goal_sampler = self.power_possible_goal_sampler

            # These three columns are only ever written in
            # _add_power_training_columns (called from postprocess_trajectory,
            # after a full trajectory fragment is collected) -- never per-step
            # during compute_actions_from_input_dict, unlike e.g. GOAL_LOGITS.
            # used_for_compute_actions=False stops RLlib's action-time
            # connector from expecting a per-step buffered value for them.
            self.view_requirements[POSSIBLE_GOAL_BLOCKS] = ViewRequirement(
                space=spaces.Box(low=0, high=255, shape=world_size, dtype=np.uint8),
                used_for_compute_actions=False,
            )
            self.view_requirements[HUMAN_GOAL_ABILITY_TARGETS] = ViewRequirement(
                space=spaces.Box(low=0, high=1, shape=()),
                used_for_compute_actions=False,
            )
            self.view_requirements[AGGREGATE_GOAL_ABILITY_TARGETS] = ViewRequirement(
                space=spaces.Box(low=0, high=np.inf, shape=()),
                used_for_compute_actions=False,
            )

        self.view_requirements[C_PUCT] = ViewRequirement(
            space=np.nan,  # type: ignore
        )
        self.view_requirements[PREV_C_PUCT] = ViewRequirement(
            C_PUCT,
            space=spaces.Box(low=-np.inf, high=np.inf, shape=()),
            shift=-1,
            used_for_compute_actions=True,
            used_for_training=False,
            batch_repeat_value=self.config.get("model", {}).get("max_seq_len", 1),
        )

        EntropyCoeffSchedule.__init__(
            self, config["entropy_coeff"], config["entropy_coeff_schedule"]
        )
        LearningRateSchedule.__init__(
            self,
            config["lr"],
            config["lr_schedule"],
        )

        self._prev_goal_kl_coeff_schedule: Schedule
        prev_goal_kl_coeff_schedule = config.get("prev_goal_kl_coeff_schedule")
        if isinstance(prev_goal_kl_coeff_schedule, list):
            self._prev_goal_kl_coeff_schedule = PiecewiseSchedule(
                prev_goal_kl_coeff_schedule,
                outside_value=prev_goal_kl_coeff_schedule[-1][-1],
                framework=None,
            )
        else:
            self._prev_goal_kl_coeff_schedule = ConstantSchedule(
                config["prev_goal_kl_coeff"]
            )
        self.prev_goal_kl_coeff = self._prev_goal_kl_coeff_schedule.value(0)

    def set_training(self, training: bool):
        self._training = training

        if not self._training:
            # Set global timestep to a huge value so that we get whatever the reward
            # shaping schedule is at the end of training.
            self.global_timestep_for_envs = 2**63 - 1
        else:
            self.global_timestep_for_envs = getattr(self, "global_timestep", 0)

    def compute_actions(
        self,
        obs_batch: Union[List[TensorStructType], TensorStructType],
        state_batches: Optional[List[TensorType]] = None,
        prev_action_batch: Optional[
            Union[List[TensorStructType], TensorStructType]
        ] = None,
        prev_reward_batch: Optional[
            Union[List[TensorStructType], TensorStructType]
        ] = None,
        info_batch: Optional[Dict[str, list]] = None,
        episodes: Optional[List[Episode]] = None,
        explore: Optional[bool] = None,
        timestep: Optional[int] = None,
        *,
        force_noop=False,
        prev_c_puct: Optional[np.ndarray] = None,
        env_states: Optional[List[MbagStateDict]] = None,
        **kwargs,
    ):
        input_dict: Dict[str, Any] = {"obs": obs_batch}
        if prev_action_batch is not None:
            input_dict[SampleBatch.PREV_ACTIONS] = prev_action_batch
        if prev_reward_batch is not None:
            input_dict[SampleBatch.PREV_REWARDS] = prev_reward_batch
        for state_index, state_batch in enumerate(state_batches or []):
            input_dict[f"state_in_{state_index}"] = state_batch
        if prev_c_puct is not None:
            input_dict[PREV_C_PUCT] = prev_c_puct
        if env_states is not None:
            input_dict[ENV_STATES] = env_states

        return self.compute_actions_from_input_dict(
            input_dict=input_dict,
            episodes=episodes,
            state_batches=state_batches,
            force_noop=force_noop,
        )

    def _run_model_on_input_dict(self, input_dict):
        input_dict = self._lazy_tensor_dict(input_dict)
        state_batches = [
            input_dict[k] for k in input_dict.keys() if k[:8] == "state_in"
        ]
        seq_lens = (
            torch.tensor(
                [1] * len(state_batches[0]),
                dtype=torch.long,
                device=state_batches[0].device,
            )
            if state_batches
            else None
        )
        assert self.model is not None
        state_out: List[torch.Tensor]
        model_out, state_out = self.model(
            input_dict,
            state_batches,
            cast(torch.Tensor, seq_lens),
        )
        return model_out, state_out

    def _ensure_enough_envs(self, num_envs: int):
        while len(self.envs) < num_envs:
            env = self.env_creator()
            env.reset()
            self.envs.append(env)

    def _compute_actions_with_mcts(
        self,
        input_dict,
        obs,
    ) -> Tuple[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]:
        num_envs = obs[0].shape[0]
        model_state_len = sum(k.startswith("state_in") for k in input_dict.keys())

        if self.config.get("player_index") is not None:
            for env in self.envs:
                env.set_player_index(self.config["player_index"])
        else:
            for env_index, player_index in enumerate(
                input_dict[SampleBatch.AGENT_INDEX]
            ):
                self.envs[env_index].set_player_index(player_index)

        nodes: List[MbagMCTSNode] = []
        for env_index in range(num_envs):
            env_obs = tuple(obs_piece[env_index] for obs_piece in obs)
            if ENV_STATES in input_dict:
                env_state = input_dict[ENV_STATES][env_index]
                env_obs = self.envs[env_index].set_state(env_state)
            else:
                env_state, env_obs = self.envs[env_index].set_state_from_obs(env_obs)
            prev_action: Optional[int] = None
            if SampleBatch.PREV_ACTIONS in input_dict:
                prev_action = input_dict[SampleBatch.PREV_ACTIONS][env_index]
            model_state = [
                input_dict[f"state_in_{state_index}"][env_index]
                for state_index in range(model_state_len)
            ]
            nodes.append(
                MbagMCTSNode(
                    state=env_state,
                    obs=tuple(obs_piece[env_index] for obs_piece in obs),
                    obs_for_computing_valid_actions=env_obs,
                    reward=0,
                    done=False,
                    info=None,
                    action=None,
                    parent=MbagRootParentNode(env=self.envs[env_index]),
                    model_state_in=model_state,
                    mcts=self.mcts,
                    c_puct=(
                        input_dict[PREV_C_PUCT][env_index]
                        if PREV_C_PUCT in input_dict
                        else np.nan
                    ),
                    prev_action=prev_action,
                )
            )

        expected_rewards_list: List[float] = []
        expected_own_rewards_list: List[float] = []
        nodes_state_out: List[List[np.ndarray]] = []
        action_masks: List[np.ndarray] = []
        value_estimates_list: List[float] = []
        c_puct_list: List[float] = []
        goal_logits_list: List[Optional[np.ndarray]] = []
        prior_policies_list: List[np.ndarray] = []

        mcts_batch_size = self.config.get("mcts_batch_size")
        if mcts_batch_size is None:
            mcts_policies, actions = self.mcts.compute_actions(nodes)

            for node, action in zip(nodes, actions):
                if self.mcts.num_sims > 1:
                    expected_reward, expected_own_reward = node.get_expected_rewards(
                        action
                    )
                else:
                    expected_reward, expected_own_reward = 0, 0
                expected_rewards_list.append(expected_reward)
                expected_own_rewards_list.append(expected_own_reward)

                nodes_state_out.append(
                    [convert_to_numpy(state) for state in node.model_state_out]
                )
                action_masks.append(node.valid_actions)
                value_estimates_list.append(node.value_estimate)
                c_puct_list.append(node.c_puct)
                goal_logits_list.append(node.goal_logits)
                prior_policies_list.append(node.child_priors)
        else:
            mcts_policies_batches: List[np.ndarray] = []
            actions_batches: List[np.ndarray] = []
            while len(nodes) > 0:
                batch_nodes, nodes = nodes[:mcts_batch_size], nodes[mcts_batch_size:]
                mcts_policies_batch, actions_batch = self.mcts.compute_actions(
                    batch_nodes
                )

                mcts_policies_batches.append(mcts_policies_batch)
                actions_batches.append(actions_batch)

                for node, action in zip(batch_nodes, actions_batch):
                    if self.mcts.num_sims > 1:
                        expected_reward, expected_own_reward = (
                            node.get_expected_rewards(action)
                        )
                    else:
                        expected_reward, expected_own_reward = 0, 0
                    expected_rewards_list.append(expected_reward)
                    expected_own_rewards_list.append(expected_own_reward)

                    nodes_state_out.append(
                        [convert_to_numpy(state) for state in node.model_state_out]
                    )
                    action_masks.append(node.valid_actions)
                    value_estimates_list.append(node.value_estimate)
                    c_puct_list.append(node.c_puct)
                    goal_logits_list.append(node.goal_logits)
                    prior_policies_list.append(node.child_priors)

            mcts_policies = np.concatenate(mcts_policies_batches, axis=0)
            actions = np.concatenate(actions_batches, axis=0)

        del nodes

        expected_rewards = np.array(expected_rewards_list)
        expected_own_rewards = np.array(expected_own_rewards_list)

        state_out = []
        for state_index in range(model_state_len):
            state_out.append(
                np.stack(
                    [state_out[state_index] for state_out in nodes_state_out],
                    axis=0,
                )
            )

        action_mask = np.stack(action_masks, axis=0)
        prior_policies = np.stack(prior_policies_list, axis=0)
        value_estimates = np.array(value_estimates_list)
        c_puct = np.array(c_puct_list)

        extra_out = {
            ACTION_MASK: action_mask,
            PRIOR_POLICIES: prior_policies,
            MCTS_POLICIES: mcts_policies,
            EXPECTED_REWARDS: expected_rewards,
            EXPECTED_OWN_REWARDS: expected_own_rewards,
            VALUE_ESTIMATES: value_estimates,
            C_PUCT: c_puct,
        }

        if self.mcts.use_goal_predictor:

            def enforce_not_none(goal_logits: Optional[np.ndarray]) -> np.ndarray:
                assert goal_logits is not None
                return goal_logits

            extra_out[GOAL_LOGITS] = np.stack(
                [enforce_not_none(goal_logits) for goal_logits in goal_logits_list],
                axis=0,
            )

        return actions, state_out, extra_out

    def _compute_actions_noop(
        self,
        input_dict,
        obs,
    ) -> Tuple[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]:
        assert isinstance(self.action_space, spaces.Discrete)
        num_envs = obs[0].shape[0]

        actions = np.zeros(num_envs, dtype=int)
        expected_rewards = np.zeros(num_envs)
        expected_own_rewards = np.zeros(num_envs)
        mcts_policies = np.zeros((num_envs, self.action_space.n))
        mcts_policies[:, 0] = 1

        # Get action mask.
        assert isinstance(self.model, MbagTorchModel)
        action_mask = self.envs[0].get_valid_actions(obs, is_batch=True)
        input_dict[ACTION_MASK] = action_mask

        # Run inputs through the model to get state_out and value function.
        logits, state_out_torch = self._run_model_on_input_dict(input_dict)
        state_out = convert_to_numpy(state_out_torch)
        value_estimates = convert_to_numpy(self.model.value_function())

        extra_out = {
            ACTION_MASK: action_mask,
            MCTS_POLICIES: mcts_policies,
            PRIOR_POLICIES: convert_to_numpy(torch.softmax(logits, dim=1)),
            EXPECTED_REWARDS: expected_rewards,
            EXPECTED_OWN_REWARDS: expected_own_rewards,
            VALUE_ESTIMATES: value_estimates,
            C_PUCT: np.array([np.nan for _ in range(num_envs)]),
        }

        if self.mcts.use_goal_predictor:
            extra_out[GOAL_LOGITS] = convert_to_numpy(self.model.goal_predictor())

        return (
            actions,
            state_out,
            extra_out,
        )

    def _check_expected_rewards_and_store_in_episodes(
        self,
        episodes: List[Episode],
        compute_actions_extra_out: Dict[str, np.ndarray],
        prev_rewards: np.ndarray,
    ):
        expected_rewards = compute_actions_extra_out[EXPECTED_REWARDS]
        expected_own_rewards = compute_actions_extra_out[EXPECTED_OWN_REWARDS]

        for env_index, episode in enumerate(episodes):
            player_index = self.envs[env_index].player_index

            if (
                self.config.get("_strict_mode", False)
                and self._training
                and not (
                    self.config["use_goal_predictor"]
                    or self.envs[env_index].config["num_players"] > 1
                )
            ):
                # If there was an expected reward, make sure it matches the actual
                # reward given by the environment so we're not out of sync.
                episode_expected_rewards: Dict[int, float] = episode.user_data.get(
                    EXPECTED_REWARDS, {}
                )
                prev_expected_reward = episode_expected_rewards.get(player_index)
                if prev_expected_reward is not None:
                    assert np.isclose(
                        prev_rewards[env_index],
                        prev_expected_reward,
                    )

            episode_expected_rewards = episode.user_data.setdefault(
                EXPECTED_REWARDS, {}
            )
            episode_expected_rewards[player_index] = expected_rewards[env_index]
            episode_expected_own_rewards = episode.user_data.setdefault(
                EXPECTED_OWN_REWARDS, {}
            )
            episode_expected_own_rewards[player_index] = expected_own_rewards[env_index]

    def compute_actions_from_input_dict(
        self,
        input_dict,
        explore=None,
        timestep=None,
        episodes=None,
        force_noop=False,
        **kwargs,
    ):
        if logger.isEnabledFor(logging.DEBUG):
            if episodes is not None:
                info: MbagInfoDict = episodes[0].last_info_for("player_1")
                if info is not None:
                    reward = input_dict[SampleBatch.REWARDS][0]
                    own_reward = info["own_reward"]
                    goal_similarity = info["goal_similarity"]
                    logger.debug(f"{reward=} {own_reward=} {goal_similarity=}")

        assert self.mcts.model == self.model
        cast(nn.Module, self.model).eval()

        obs = input_dict[SampleBatch.OBS]
        obs = restore_original_dimensions(obs, self.obs_space, "numpy")

        num_envs = obs[0].shape[0]
        self._ensure_enough_envs(num_envs)

        with torch.no_grad():
            if self.config["pretrain"] or force_noop:
                actions, state_out, compute_actions_extra_out = (
                    self._compute_actions_noop(input_dict, obs)
                )
            else:
                actions, state_out, compute_actions_extra_out = (
                    self._compute_actions_with_mcts(
                        input_dict,
                        obs,
                    )
                )

        if episodes is not None:
            self._check_expected_rewards_and_store_in_episodes(
                episodes, compute_actions_extra_out, input_dict[SampleBatch.REWARDS]
            )

        action_mask = compute_actions_extra_out[ACTION_MASK]
        mcts_policies = compute_actions_extra_out[MCTS_POLICIES]
        prior_policies = compute_actions_extra_out[PRIOR_POLICIES]
        action_dist_inputs = np.log(mcts_policies)
        action_dist_inputs[mcts_policies == 0] = MbagTorchModel.MASK_LOGIT
        extra_out = {
            **self.extra_action_out(
                input_dict,
                kwargs.get("state_batches", []),
                cast(TorchModelV2, self.model),
                cast(Any, None),
            ),
            ACTION_MASK: action_mask,
            MCTS_POLICIES: mcts_policies,
            PRIOR_POLICIES: prior_policies,
            SampleBatch.ACTION_DIST_INPUTS: action_dist_inputs,
            EXPECTED_REWARDS: compute_actions_extra_out[EXPECTED_REWARDS],
            EXPECTED_OWN_REWARDS: compute_actions_extra_out[EXPECTED_OWN_REWARDS],
            VALUE_ESTIMATES: compute_actions_extra_out[VALUE_ESTIMATES],
            C_PUCT: compute_actions_extra_out[C_PUCT],
        }
        if GOAL_LOGITS in compute_actions_extra_out:
            extra_out[GOAL_LOGITS] = compute_actions_extra_out[GOAL_LOGITS]

        return np.array(actions), state_out, extra_out

    def _set_up_power_progress_baseline(
        self, goal_generator, world_size, env_config, config
    ) -> None:
        """
        Predict X_h on top of the mean progress toward the pool of possible goals
        (see mbag.rllib.alpha_zero.power.progress). Needs a fixed goal pool whose
        structures fill the buildable area, so each pool goal has exactly one
        placement in the world.
        """
        if not isinstance(goal_generator, FixedPoolGoalGenerator):
            raise ValueError("power_progress_baseline requires goal_pool_size > 0")
        inf_blocks = MbagEnv.get_config(env_config)["abilities"]["inf_blocks"]
        goal_size = get_goal_size(world_size, inf_blocks)
        usable_x = world_size[0] if inf_blocks else world_size[0] - 1
        goals = []
        for small_goal in goal_generator.get_pool(goal_size):
            if usable_x - small_goal.size[0] - 1 > 1:
                raise ValueError(
                    "power_progress_baseline needs pool goals that fill the build "
                    "area (their placement must not be random)"
                )
            goals.append(embed_goal(small_goal, world_size, inf_blocks).blocks)
        goals_tensor = torch.from_numpy(np.stack(goals).astype(np.int64))
        reference: Union[ProgressReference, StepsToGoalReference]
        if config.get("power_progress_kind", "progress") == "steps":
            reference = StepsToGoalReference(
                goals_tensor, config.get("power_gamma_h", config["gamma"])
            )
        else:
            initial_world = np.zeros(world_size, dtype=np.int64)
            initial_world[:, 0, :] = MinecraftBlocks.BEDROCK
            initial_world[:, 1, :] = MinecraftBlocks.NAME2ID["dirt"]
            reference = ProgressReference(
                goals_tensor, torch.from_numpy(initial_world)
            )
        models = getattr(self, "model_gpu_towers", None) or [self.model]
        for model in models:
            model.set_progress_reference(
                reference,
                config.get("power_zeta", 2.0),
                scale=float(config.get("power_num_goals", 1)),
            )

    def _add_power_training_columns(
        self,
        sample_batch: SampleBatch,
        last_input_dict: Optional[SampleBatch],
        terminated: bool,
    ) -> None:
        """
        Sample one possible goal g_h for this trajectory fragment (see
        mbag.rllib.alpha_zero.power) and compute Monte Carlo regression
        targets for V_h^e(s_t, g_h) and X_h(s_t) at each timestep, mirroring
        how VALUE_TARGETS above bootstraps V_r with the model's own value
        prediction at the truncation boundary.

        If the fragment ends in true termination (not just truncation), the
        last transition's achievement is read from the final observation
        (NEXT_OBS), and -- exactly like VALUE_TARGETS' last_r=0 convention -- the
        bootstrap value past it is 0.
        """
        assert self.model is not None

        # sample_batch[SampleBatch.OBS] may already be a torch tensor by this
        # point depending on the connector pipeline in use, even though
        # tensorlib="numpy" is requested below -- restore_original_dimensions
        # reshapes but doesn't necessarily cast, so force numpy explicitly.
        world_obs, _, _ = restore_original_dimensions(
            convert_to_numpy(sample_batch[SampleBatch.OBS]), self.obs_space, "numpy"
        )

        if self.power_rollout_goal:
            # As in the paper's Phase 2, the goal g_h is the one the human is pursuing
            # in this rollout (the episode's goal, which the observation carries even
            # though the model masks it), so V_h^e(s, g_h) is estimated for a goal the
            # human is actually working toward. The robot's networks never see it.
            goal_blocks = world_obs[0, GOAL_BLOCKS]
            if not np.any(goal_blocks):
                raise ValueError(
                    "power_rollout_goal needs the goal in the observation "
                    "(goal_visibility must be True)"
                )
            true_goal = MinecraftBlocks(goal_blocks.shape)
            true_goal.blocks[:] = goal_blocks
            true_goal.block_states[:] = world_obs[0, GOAL_BLOCK_STATES]
            possible_goal = MinecraftStructureGoal(
                true_goal, lenient=self.power_goal_lenient
            )
        else:
            possible_goal, _ = self.power_possible_goal_sampler.sample(None, 0)
        assert isinstance(possible_goal, MinecraftStructureGoal)
        target_blocks = possible_goal.target.blocks
        # achieved[t] for t = 0..T-2 is whether g_h is achieved in the state
        # reached after timestep t's action, i.e. OBS[t + 1].
        achieved = possible_goal.matches_blocks(
            world_obs[1:, CURRENT_BLOCKS], world_obs[1:, CURRENT_BLOCK_STATES]
        ).astype(np.float32)

        last_achieved: float
        last_human_goal_ability: float
        if terminated:
            # The last transition's successor state is the observation RLlib stores
            # as NEXT_OBS (the final observation returned with done=True), so we can
            # check whether g_h was achieved by the final action, e.g. the human
            # completing its house. Past that state there is nothing to bootstrap.
            last_achieved = 0.0
            if SampleBatch.NEXT_OBS in sample_batch:
                final_world_obs, _, _ = restore_original_dimensions(
                    convert_to_numpy(sample_batch[SampleBatch.NEXT_OBS][-1:]),
                    self.obs_space,
                    "numpy",
                )
                last_achieved = float(
                    possible_goal.matches_blocks(
                        final_world_obs[0, CURRENT_BLOCKS],
                        final_world_obs[0, CURRENT_BLOCK_STATES],
                    )
                )
            last_human_goal_ability = 0.0
        else:
            assert last_input_dict is not None
            last_world_obs, _, _ = restore_original_dimensions(
                convert_to_numpy(last_input_dict[SampleBatch.OBS]),
                self.obs_space,
                "numpy",
            )
            last_achieved = float(
                possible_goal.matches_blocks(
                    last_world_obs[0, CURRENT_BLOCKS],
                    last_world_obs[0, CURRENT_BLOCK_STATES],
                )
            )
            target_blocks_tensor = torch.from_numpy(target_blocks[None]).to(
                self.model.device
            )
            last_human_goal_ability = float(
                self.model.human_goal_ability(target_blocks_tensor)[0].item()
            )

        achieved = np.concatenate([achieved, [last_achieved]])
        v_h_e_targets = human_goal_ability_monte_carlo_targets(
            achieved,
            gamma_h=self.power_gamma_h,
            bootstrap_v_h_e=last_human_goal_ability,
        ).astype(np.float32)
        # Paper eq. (14): x_h(s) = |G_h| * V_h^e(s, g_h) ** zeta.
        x_h_targets = (self.power_num_goals * v_h_e_targets**self.power_zeta).astype(
            np.float32
        )

        sample_batch[POSSIBLE_GOAL_BLOCKS] = np.repeat(
            target_blocks[None], len(sample_batch), axis=0
        )
        sample_batch[HUMAN_GOAL_ABILITY_TARGETS] = v_h_e_targets
        sample_batch[AGGREGATE_GOAL_ABILITY_TARGETS] = x_h_targets

    def postprocess_trajectory(
        self,
        sample_batch: SampleBatch,
        other_agent_batches: Optional[
            Dict[AgentID, Tuple[PolicyID, Type[TorchPolicy], SampleBatch]]
        ] = None,
        episode: Optional[Episode] = None,
    ):
        with torch.no_grad():
            last_r: float
            terminated = sample_batch[SampleBatch.TERMINATEDS][-1]
            input_dict: Optional[SampleBatch] = None
            if terminated:
                last_r = 0
            else:
                input_dict = sample_batch.get_single_step_input_dict(
                    self.view_requirements, index="last"
                )
                input_dict = SampleBatch(input_dict)
                assert self.model is not None
                self._run_model_on_input_dict(input_dict)
                last_r = self.model.value_function()[0].item()
            rewards = sample_batch[SampleBatch.REWARDS]
            if self.use_power_reward and self.power_value_from_ur:
                # The robot has no extrinsic reward: V_r is the discounted sum of the
                # intrinsic reward U_r the search used, not the environment reward.
                rewards = np.asarray(sample_batch[EXPECTED_REWARDS], dtype=np.float32)
            rewards_plus_v = np.concatenate([rewards, np.array([last_r])])
            discounted_returns = discount_cumsum(rewards_plus_v, self.config["gamma"])[
                :-1
            ].astype(np.float32)
            sample_batch[Postprocessing.VALUE_TARGETS] = discounted_returns

            if self.use_power_reward:
                self._add_power_training_columns(sample_batch, input_dict, terminated)

        if other_agent_batches is not None:
            if len(other_agent_batches) > 1:
                raise RuntimeError(
                    "Training with multiple other agents is not supported."
                )
            elif len(other_agent_batches) == 1:
                other_agent_id, (_, _, other_agent_batch) = next(
                    iter(other_agent_batches.items())
                )
                if SampleBatch.ACTION_DIST_INPUTS in other_agent_batch:
                    sample_batch[OTHER_AGENT_ACTION_DIST_INPUTS] = other_agent_batch[
                        SampleBatch.ACTION_DIST_INPUTS
                    ]

                    assert len(sample_batch) == len(other_agent_batch)
                else:
                    logger.warn(
                        f"no action_dist_inputs in sample batch for {other_agent_id}"
                    )
            else:
                pass  # No need to include other agent batch for single player case.

        assert episode is not None
        infos: List[MbagInfoDict] = list(sample_batch[SampleBatch.INFOS][1:])
        agent_index = sample_batch[SampleBatch.AGENT_INDEX][0]
        agent_id = episode.get_agents()[agent_index]
        last_info = cast(MbagInfoDict, episode.last_info_for(agent_id))
        infos.append(last_info)
        sample_batch[OWN_REWARDS] = np.array([info["own_reward"] for info in infos])

        sample_batch[PREV_GOAL_KL_COEFF] = np.full(
            len(sample_batch), self.prev_goal_kl_coeff
        )

        # Remove state_out_* entries from the sample batch since they aren't needed
        # for training and they take up a lot of space.
        for key in list(sample_batch.keys()):
            if key.startswith("state_out_"):
                del sample_batch[key]

        return sample_batch

    def learn_on_batch(self, postprocessed_batch):
        return TorchPolicy.learn_on_batch(self, postprocessed_batch)

    def loss(
        self,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        assert isinstance(model, MbagTorchModel)

        # Forward pass in model.
        logits, state = model(train_batch)
        values = model.value_function()
        logits, values = torch.squeeze(logits), torch.squeeze(values)
        dist = dist_class(cast(Any, logits), model=model)
        assert isinstance(dist, TorchCategorical)

        # RNN case: Mask away 0-padded chunks at end of time axis.
        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])  # noqa: N806
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major(),
            )
            assert isinstance(mask, torch.Tensor)
            mask = torch.reshape(mask, [-1])

        # non-RNN case: No masking.
        else:
            mask = None

        policy_mask: torch.Tensor
        model_mask: torch.Tensor
        if FOR_TRAINING_MODEL in train_batch:
            policy_mask = cast(torch.Tensor, ~train_batch[FOR_TRAINING_MODEL])
            model_mask = cast(torch.Tensor, train_batch[FOR_TRAINING_MODEL])
        else:
            policy_mask = torch.ones_like(values, dtype=torch.bool)
            model_mask = torch.ones_like(values, dtype=torch.bool)
        if mask is not None:
            policy_mask = policy_mask & mask
            model_mask = model_mask & mask
        num_policy = torch.sum(policy_mask)
        num_model = torch.sum(model_mask)

        def reduce_mean_policy(t):
            return torch.sum(t[policy_mask]) / num_policy

        def reduce_mean_model(t):
            return torch.sum(t[model_mask]) / num_model

        # Compute actor and critic losses.
        policy_loss = reduce_mean_policy(
            -torch.sum(train_batch[MCTS_POLICIES] * dist.dist.logits, dim=-1)
        )
        prev_policy_kl = reduce_mean_policy(
            F.kl_div(
                F.log_softmax(dist.dist.logits, dim=1),
                cast(torch.Tensor, train_batch[PRIOR_POLICIES]),
                log_target=False,
                reduction="none",
            ).sum(dim=1)
        )
        value_loss = reduce_mean_policy(
            (values - train_batch[Postprocessing.VALUE_TARGETS]) ** 2
        )

        entropy = reduce_mean_policy(dist.entropy())

        # Compute goal prediction loss.
        goal_logits: torch.Tensor = model.goal_predictor()
        world_obs, _, _ = restore_original_dimensions(
            train_batch[SampleBatch.OBS],
            obs_space=self.observation_space,
            tensorlib=torch,
        )
        goal = world_obs[:, GOAL_BLOCKS].long()
        ce = nn.CrossEntropyLoss(reduction="none")
        goal_ce: torch.Tensor = ce(goal_logits, goal)
        goal_loss = reduce_mean_model(goal_ce.flatten(start_dim=1).mean(dim=1))

        unplaced_blocks = (goal != MinecraftBlocks.AIR) & (
            world_obs[:, CURRENT_BLOCKS] == MinecraftBlocks.AIR
        )
        unplaced_blocks_goal_loss = goal_ce[unplaced_blocks].mean()

        prev_goal_kl: torch.Tensor = torch.tensor(0.0, device=policy_loss.device)
        weighted_prev_goal_kl: torch.Tensor = torch.tensor(
            0.0, device=policy_loss.device
        )
        if GOAL_LOGITS in train_batch:
            prev_goal_logits = cast(torch.Tensor, train_batch[GOAL_LOGITS])
            prev_goal_logits = prev_goal_logits.permute(0, 2, 3, 4, 1).flatten(
                end_dim=3
            )
            flat_goal_logits = goal_logits.permute(0, 2, 3, 4, 1).flatten(end_dim=3)
            prev_goal_kl = (
                F.kl_div(
                    F.log_softmax(flat_goal_logits, dim=1),
                    F.log_softmax(prev_goal_logits, dim=1),
                    reduction="none",
                    log_target=True,
                )
                .sum(dim=1)
                .reshape_as(goal_ce)
                .flatten(start_dim=1)
                .mean(dim=1)
            )
            weighted_prev_goal_kl = prev_goal_kl * cast(
                torch.Tensor, train_batch[PREV_GOAL_KL_COEFF]
            )
            prev_goal_kl = reduce_mean_model(prev_goal_kl)
            weighted_prev_goal_kl = reduce_mean_model(weighted_prev_goal_kl)

        # Compute human-power (empowerment) objective losses: TD regressions
        # for V_h^e and X_h against the Monte Carlo targets computed in
        # postprocess_trajectory (see mbag.rllib.alpha_zero.power). Absent
        # (POSSIBLE_GOAL_BLOCKS not in train_batch) whenever use_power_reward
        # is False, so this is a no-op for the CIRL/assistance-games objective.
        human_goal_ability_loss: torch.Tensor = torch.tensor(
            0.0, device=policy_loss.device
        )
        aggregate_goal_ability_loss: torch.Tensor = torch.tensor(
            0.0, device=policy_loss.device
        )
        x_h_stats = {
            name: torch.tensor(0.0, device=policy_loss.device)
            for name in [
                "x_h_pred_min",
                "x_h_pred_mean",
                "x_h_target_mean",
                "x_h_target_max",
                "u_r_min",
            ]
        }
        if POSSIBLE_GOAL_BLOCKS in train_batch:
            predicted_v_h_e = model.human_goal_ability(
                cast(torch.Tensor, train_batch[POSSIBLE_GOAL_BLOCKS])
            )
            human_goal_ability_loss = reduce_mean_model(
                (predicted_v_h_e - train_batch[HUMAN_GOAL_ABILITY_TARGETS]) ** 2
            )

            predicted_x_h = model.aggregate_goal_ability()
            aggregate_goal_ability_loss = reduce_mean_model(
                (predicted_x_h - train_batch[AGGREGATE_GOAL_ABILITY_TARGETS]) ** 2
            )

            # Diagnostics only: how small X_h gets, since U_r = -(X_h^-xi)^eta
            # blows up as X_h -> 0.
            x_h_targets = cast(torch.Tensor, train_batch[AGGREGATE_GOAL_ABILITY_TARGETS])
            x_h_stats = {
                "x_h_pred_min": predicted_x_h.min().detach(),
                "x_h_pred_mean": predicted_x_h.mean().detach(),
                "x_h_target_mean": x_h_targets.mean().detach(),
                "x_h_target_max": x_h_targets.max().detach(),
                "u_r_min": (
                    -(
                        (predicted_x_h.min().detach() + self.mcts.power_x_epsilon)
                        ** (-self.mcts.power_xi)
                    )
                    ** self.mcts.power_eta
                ),
            }

        # Compute total loss.
        total_loss: torch.Tensor = (
            self.config["vf_loss_coeff"] * value_loss
            + self.config["goal_loss_coeff"] * goal_loss
            + weighted_prev_goal_kl
            + self.human_goal_ability_loss_coeff * human_goal_ability_loss
            + self.aggregate_goal_ability_loss_coeff * aggregate_goal_ability_loss
            - self.entropy_coeff * entropy
        )
        if not self.config["pretrain"]:
            total_loss = (
                total_loss
                + self.config["policy_loss_coeff"] * policy_loss
                + self.config["prev_policy_kl_coeff"] * prev_policy_kl
            )

        if isinstance(model, OtherAgentActionPredictorMixin):
            # Compute other agent action prediction loss.
            predicted_other_agent_action_dist = dist_class(
                cast(Any, model.predict_other_agent_action()),
                model=model,
            )
            other_agent_action_dist_inputs = train_batch[OTHER_AGENT_ACTION_DIST_INPUTS]
            # Get rid of -inf action dist inputs to avoid numeric issues with
            # KL divergence.
            other_agent_action_dist_inputs[
                other_agent_action_dist_inputs == -np.inf
            ] = -1e4
            actual_other_agent_action_dist = dist_class(
                other_agent_action_dist_inputs,  # type: ignore
                model=model,
            )
            other_agent_action_predictor_loss = reduce_mean_model(
                actual_other_agent_action_dist.kl(predicted_other_agent_action_dist)
            )

            model.tower_stats["other_agent_action_predictor_loss"] = (
                other_agent_action_predictor_loss.detach()
            )
            total_loss = (
                total_loss
                + self.config["other_agent_action_predictor_loss_coeff"]
                * other_agent_action_predictor_loss
            )

        # KL regularization.
        if ANCHOR_POLICY_ACTION_DIST_INPUTS in train_batch:
            action_dist = dist_class(cast(Any, logits), model)
            anchor_policy_action_dist_inputs = train_batch[
                ANCHOR_POLICY_ACTION_DIST_INPUTS
            ]
            anchor_policy_action_dist = dist_class(
                cast(Any, anchor_policy_action_dist_inputs), model
            )

            anchor_policy_kl = reduce_mean_policy(
                action_dist.kl(anchor_policy_action_dist)
            )
            model.tower_stats["anchor_policy_kl"] = anchor_policy_kl.detach()
            total_loss = (
                total_loss + self.config["anchor_policy_kl_coeff"] * anchor_policy_kl
            )

        model.tower_stats["total_loss"] = total_loss.detach()
        model.tower_stats["policy_loss"] = policy_loss.detach()
        model.tower_stats["prev_policy_kl"] = prev_policy_kl.detach()
        model.tower_stats["vf_loss"] = value_loss.detach()
        model.tower_stats["vf_explained_var"] = cast(
            torch.Tensor,
            explained_variance(train_batch[Postprocessing.VALUE_TARGETS], values),
        ).detach()
        model.tower_stats["goal_loss"] = goal_loss.detach()
        model.tower_stats["prev_goal_kl"] = prev_goal_kl.detach()
        model.tower_stats["weighted_prev_goal_kl"] = weighted_prev_goal_kl.detach()
        model.tower_stats["unplaced_blocks_goal_loss"] = (
            unplaced_blocks_goal_loss.detach()
        )
        model.tower_stats["entropy"] = entropy.detach()
        model.tower_stats["human_goal_ability_loss"] = human_goal_ability_loss.detach()
        model.tower_stats["aggregate_goal_ability_loss"] = (
            aggregate_goal_ability_loss.detach()
        )
        for name, value in x_h_stats.items():
            model.tower_stats[name] = value

        return total_loss

    def extra_grad_process(self, optimizer, loss):
        return apply_grad_clipping(self, optimizer, loss)

    def extra_grad_info(self, train_batch: SampleBatch):
        grad_info: Dict[str, TensorType] = {
            "entropy_coeff": self.entropy_coeff,
            "cur_lr": self.cur_lr,
            "mcts/temperature": self.mcts.temperature,
        }
        for metric in [
            "total_loss",
            "policy_loss",
            "prev_policy_kl",
            "vf_loss",
            "vf_explained_var",
            "goal_loss",
            "prev_goal_kl",
            "weighted_prev_goal_kl",
            "unplaced_blocks_goal_loss",
            "entropy",
            "human_goal_ability_loss",
            "aggregate_goal_ability_loss",
            "x_h_pred_min",
            "x_h_pred_mean",
            "x_h_target_mean",
            "x_h_target_max",
            "u_r_min",
            "other_agent_action_predictor_loss",
            "anchor_policy_kl",
        ]:
            try:
                grad_info[metric] = torch.mean(
                    torch.stack(cast(List[torch.Tensor], self.get_tower_stats(metric)))
                )
            except AssertionError:
                pass
        return convert_to_numpy(grad_info)

    def on_global_var_update(self, global_vars):
        super().on_global_var_update(global_vars)

        if self._training:
            self.global_timestep_for_envs = global_vars["timestep"]
        for env in self.envs:
            unwrap_mbag_env(env).update_global_timestep(self.global_timestep_for_envs)

        self.prev_goal_kl_coeff = self._prev_goal_kl_coeff_schedule.value(
            global_vars["timestep"]
        )
