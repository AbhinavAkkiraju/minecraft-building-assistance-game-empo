"""
Model-based X_h targets from simulated human rollouts (paper Section 3.1: the robot
learns from rollouts of its world model with the human behavior prior pi_h).

For a state s and each possible goal g in a small fixed goal set, simulate the human
prior pursuing g from s, with a passive robot, for up to ``horizon`` steps:

    V_h^e(s, g) ~= gamma_h ** (k - 1)  if g is first achieved after k steps, else 0,

and X_h(s) = sum_g V_h^e(s, g) ** zeta (paper eq. 7). A goal with bedrock on a cell it
needs is unreachable for a human who can't break bedrock, so its value is 0 without
simulating. Each value is a single rollout, i.e. an unbiased sample of the achievement
probability under the prior.

Approximation: the human is simulated alone (the robot neither acts nor occupies
space during the rollout), so V_h^e reflects the current state of the world rather
than the robot's future behavior.
"""

import copy
from typing import List

import numpy as np

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.config import MbagConfigDict
from mbag.environment.goals.possible_goals import MinecraftStructureGoal
from mbag.environment.mbag_env import MbagEnv

from .human_policy_prior import HumanPolicyPrior

class SimulatedGoalAbility:
    def __init__(
        self,
        env_config: MbagConfigDict,
        goals: List[MinecraftStructureGoal],
        prior: HumanPolicyPrior,
        gamma_h: float,
        zeta: float,
        horizon: int,
        seed: int = 0,
    ):
        config = copy.deepcopy(MbagEnv.get_config(env_config))
        config["num_clutter_blocks"] = 0
        config["randomize_first_episode_length"] = False
        config["horizon"] = horizon + 1
        config["malmo"]["use_malmo"] = False
        # Simulate the human alone: a passive robot standing still could occupy a
        # cell the goal needs and make it unreachable for the whole rollout.
        config["num_players"] = 1
        config["players"] = config["players"][:1]
        self.env = MbagEnv(config)
        self.env.reset()
        self.config = config
        self.goals = goals
        self.prior = prior
        self.gamma_h = gamma_h
        self.zeta = zeta
        self.horizon = horizon
        self.human_cannot_break_bedrock = bool(config["players"][0]["is_human"])
        self.action_mapping = MbagActionDistribution.get_action_mapping(config)
        self.rng = np.random.default_rng(seed)

    def _blocked(self, current: MinecraftBlocks, goal: MinecraftStructureGoal) -> bool:
        target = goal.target.blocks
        return bool(
            (
                (current.blocks == MinecraftBlocks.BEDROCK)
                & (target != MinecraftBlocks.AIR)
                & (target != MinecraftBlocks.BEDROCK)
            ).any()
        )

    def goal_ability(
        self, current: MinecraftBlocks, goal: MinecraftStructureGoal
    ) -> float:
        """A one-rollout sample of V_h^e(s, g) for the world ``current``."""
        if goal.matches_blocks(current.blocks, current.block_states):
            return 1.0
        if self.human_cannot_break_bedrock and self._blocked(current, goal):
            return 0.0
        state = self.env.get_state()
        state["current_blocks"] = current.copy()
        state["goal_blocks"] = goal.target.copy()
        state["timestep"] = 0
        obs = self.env.set_state(state)
        for k in range(1, self.horizon + 1):
            probs = self.prior.action_distribution(obs[0], goal)
            flat = MbagActionDistribution.to_flat(self.config, probs[None])[0]
            flat = flat / flat.sum()
            action = tuple(self.action_mapping[self.rng.choice(len(flat), p=flat)])
            obs, _, done, _ = self.env.step([action])
            current_blocks = self.env.current_blocks
            if goal.matches_blocks(current_blocks.blocks, current_blocks.block_states):
                return self.gamma_h ** (k - 1)
            if all(done):
                break
        return 0.0

    def goal_abilities(self, current: MinecraftBlocks) -> np.ndarray:
        """V_h^e(s, g) for every goal, in the order of ``self.goals``."""
        return np.array(
            [self.goal_ability(current, goal) for goal in self.goals], dtype=np.float32
        )

    def aggregate_goal_ability(self, current: MinecraftBlocks) -> float:
        """X_h(s) = sum over goals of V_h^e(s, g) ** zeta (paper eq. 7)."""
        return float((self.goal_abilities(current) ** self.zeta).sum())
