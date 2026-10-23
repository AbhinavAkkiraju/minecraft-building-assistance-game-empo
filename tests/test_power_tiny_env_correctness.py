"""
Phase 6 correctness tests: run the *real* possible-goal abstraction (Phase 1)
and heuristic human policy prior (Phase 2 Milestone 1) through an actual,
tiny, hand-solvable MbagEnv instance, and check the resulting V_h^e Monte
Carlo targets (Phase 3's formulas) against hand-computed expected values.

This complements test_power_formulas.py (which validates the formulas against
synthetic arrays) by validating the *integration* of the environment, the
possible-goal abstraction, and the human policy prior together -- mirroring
empo's own test_phase2.py pattern of checking exact values on tiny,
brute-force-solvable environments.
"""

import numpy as np
import pytest

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.goals.possible_goals import MinecraftStructureGoal
from mbag.environment.mbag_env import MbagEnv
from mbag.environment.types import CURRENT_BLOCK_STATES, CURRENT_BLOCKS
from mbag.rllib.alpha_zero.power.formulas import human_goal_ability_monte_carlo_targets
from mbag.rllib.alpha_zero.power.human_policy_prior import (
    HeuristicMbagHumanPolicyPrior,
)

WORLD_SIZE = (3, 3, 3)


def _make_env_config():
    return {
        "world_size": WORLD_SIZE,
        "num_players": 1,
        "horizon": 10,
        "goal_generator": "basic",
        "abilities": {
            "teleportation": True,
            "flying": True,
            "inf_blocks": True,
        },
    }


def _greedy_heuristic_action(prior, env, obs, possible_goal):
    action_mapping = MbagActionDistribution.get_action_mapping(env.config)
    action_dist = prior.action_distribution(obs, possible_goal)
    flat_dist = MbagActionDistribution.to_flat(env.config, action_dist[None])[0]
    flat_action = int(np.argmax(flat_dist))
    action_type, block_location_index, block_id = action_mapping[flat_action]
    return (int(action_type), int(block_location_index), int(block_id))


def _achieved(possible_goal, obs) -> float:
    world_obs, _, _ = obs
    return float(
        possible_goal.matches_blocks(
            world_obs[CURRENT_BLOCKS], world_obs[CURRENT_BLOCK_STATES]
        )
    )


def _target_from_initial(obs) -> MinecraftBlocks:
    """
    MbagEnv's default world isn't empty air -- it has a bedrock+dirt ground
    layer (see world_obs[CURRENT_BLOCKS] at reset). Possible-goal targets in
    these tests are built as a copy of that actual initial layout, with only
    specific (already-air) cells changed, so "differs by exactly one/two
    blocks" is accurate rather than assumed.
    """
    world_obs, _, _ = obs
    target = MinecraftBlocks(WORLD_SIZE)
    target.blocks[:] = world_obs[CURRENT_BLOCKS]
    target.block_states[:] = world_obs[CURRENT_BLOCK_STATES]
    return target


def test_single_block_difference_achieved_in_one_deterministic_step():
    """
    Target differs from the actual initial build by exactly one (currently
    air) block. With a high beta_h, the heuristic human policy prior
    deterministically places the correct block immediately, so the goal is
    achieved after exactly one action.

    Hand-computed: since achieved[0] = 1 (the goal is achieved in the very
    first successor state), V_h^e(s0, g_h) = 1.0 regardless of gamma_h or the
    bootstrap value -- achievement is absorbing.
    """
    config = _make_env_config()
    env = MbagEnv(config)
    obs_list, _ = env.reset()
    obs = obs_list[0]

    target = _target_from_initial(obs)
    target.blocks[1, 2, 1] = MinecraftBlocks.NAME2ID["cobblestone"]
    possible_goal = MinecraftStructureGoal(target)

    prior = HeuristicMbagHumanPolicyPrior(config, beta_h=50.0)
    action = _greedy_heuristic_action(prior, env, obs, possible_goal)

    obs_list, _, _, _ = env.step([action])
    obs = obs_list[0]

    achieved = _achieved(possible_goal, obs)
    assert achieved == 1.0

    for gamma_h, bootstrap in [(0.9, 0.0), (0.5, 0.7), (0.99, 1.0)]:
        targets = human_goal_ability_monte_carlo_targets(
            [achieved], gamma_h=gamma_h, bootstrap_v_h_e=bootstrap
        )
        assert targets[0] == pytest.approx(1.0)


def test_two_block_difference_achieved_in_two_deterministic_steps():
    """
    Target differs from the initial build by exactly two blocks, at
    locations far enough apart that fixing one doesn't affect the other.
    The heuristic prior should fix one mismatched cell per step (whichever
    has positive advantage; ties broken deterministically by argmax), so the
    goal is achieved only after the second action, not the first.

    Hand-computed (gamma_h = 0.9): achieved = [0, 1], so
        V_h^e(s1, g_h) = 1.0                     (achieved at s2)
        V_h^e(s0, g_h) = 0.9 * 1.0 = 0.9          (not achieved at s1)
    """
    config = _make_env_config()
    env = MbagEnv(config)
    obs_list, _ = env.reset()
    obs = obs_list[0]

    target = _target_from_initial(obs)
    target.blocks[0, 2, 0] = MinecraftBlocks.NAME2ID["cobblestone"]
    target.blocks[2, 2, 2] = MinecraftBlocks.NAME2ID["wool"]
    possible_goal = MinecraftStructureGoal(target)

    prior = HeuristicMbagHumanPolicyPrior(config, beta_h=50.0)

    achieved_sequence = []
    for _ in range(2):
        action = _greedy_heuristic_action(prior, env, obs, possible_goal)
        obs_list, _, _, _ = env.step([action])
        obs = obs_list[0]
        achieved_sequence.append(_achieved(possible_goal, obs))

    assert achieved_sequence == [0.0, 1.0]

    gamma_h = 0.9
    targets = human_goal_ability_monte_carlo_targets(
        achieved_sequence, gamma_h=gamma_h, bootstrap_v_h_e=0.0
    )
    assert targets[1] == pytest.approx(1.0)
    assert targets[0] == pytest.approx(gamma_h)


def test_already_achieved_goal_gives_noop_and_immediate_value_one():
    """
    If the current build already matches the possible goal exactly, the
    heuristic prior should select NOOP (no valid PLACE/BREAK actions have
    positive advantage), and stepping the env leaves the goal achieved.
    """
    config = _make_env_config()
    env = MbagEnv(config)
    obs_list, _ = env.reset()
    obs = obs_list[0]

    target = _target_from_initial(obs)
    possible_goal = MinecraftStructureGoal(target)

    prior = HeuristicMbagHumanPolicyPrior(config, beta_h=50.0)
    action = _greedy_heuristic_action(prior, env, obs, possible_goal)
    assert action[0] == 0  # NOOP action type.

    obs_list, _, _, _ = env.step([action])
    obs = obs_list[0]
    assert _achieved(possible_goal, obs) == 1.0
