import copy

import numpy as np
import pytest

from mbag.environment.mbag_env import DEFAULT_CONFIG
from mbag.rllib.alpha_zero.power.formulas import robot_intrinsic_reward
from mbag.rllib.alpha_zero.power.planning import get_power_reward

try:
    import torch
    from ray.rllib.models.catalog import ModelCatalog

    from mbag.agents.action_distributions import MbagActionDistribution
    from mbag.environment.goals.possible_goals import RandomStructurePossibleGoalSampler
    from mbag.environment.goals.simple import BasicGoalGenerator
    from mbag.rllib.alpha_zero.mcts import MbagMCTS, MbagMCTSNode, MbagRootParentNode
    from mbag.rllib.alpha_zero.planning import create_mbag_env_model
    from mbag.rllib.alpha_zero.power.human_policy_prior import (
        HeuristicMbagHumanPolicyPrior,
    )
except ImportError:
    pass


WORLD_SIZE = (5, 5, 5)

MCTS_PARAM = {
    "temperature": 1.5,
    "temperature_schedule": None,
    "dirichlet_epsilon": 0.25,
    "dirichlet_noise": 0.25,
    "num_simulations": 1,
    "argmax_tree_policy": False,
    "add_dirichlet_noise": False,
    "puct_coefficient": 1.0,
}


def test_get_power_reward_matches_robot_intrinsic_reward():
    # get_power_reward is a single-human (|H|=1) convenience wrapper.
    assert get_power_reward(0.6, xi=1.0, eta=1.1) == robot_intrinsic_reward(
        [0.6], xi=1.0, eta=1.1
    )


def test_get_power_reward_is_always_negative():
    for x_h in [0.01, 0.5, 0.99]:
        assert get_power_reward(x_h, xi=1.0, eta=1.0) < 0


def _build_model(env_config, *, use_power_heads: bool):
    env_model = create_mbag_env_model(env_config)
    return ModelCatalog.get_model_v2(
        env_model.observation_space.spaces[env_model.agent_id],
        env_model.action_space,
        num_outputs=1,
        model_config={
            "custom_model": "mbag_convolutional_model",
            "custom_action_dist": "mbag_bilevel_categorical",
            "custom_model_config": {
                "env_config": env_config,
                "embedding_size": 4,
                "hidden_size": 8,
                "hidden_channels": 8,
                "num_layers": 1,
                "mask_goal": True,
                "use_power_heads": use_power_heads,
            },
            "vf_share_layers": True,
        },
        framework="torch",
    )


@pytest.mark.uses_rllib
def test_expand_uses_power_reward_when_aggregate_goal_ability_given():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["world_size"] = WORLD_SIZE
    config["goal_generator"] = "basic"
    config["num_players"] = 1

    env = create_mbag_env_model(config)
    obs, info = env.reset()

    model = _build_model(config, use_power_heads=True)
    xi, eta = 1.0, 1.1
    mcts = MbagMCTS(
        model,
        {**MCTS_PARAM, "power_xi": xi, "power_eta": eta},
        gamma=0.95,
        use_goal_predictor=False,
        use_power_reward=True,
    )

    root = MbagRootParentNode(env)
    node0 = MbagMCTSNode(
        action=None,
        obs=obs,
        done=False,
        info=info,
        reward=0.0,
        state=env.get_state(),
        mcts=mcts,
        model_state_in=[],
        parent=root,
    )

    node0.expand(np.full(node0.action_total_value.shape, 1.0), value_estimate=0.0)
    node1 = node0.get_child(0)  # NOOP action.
    assert node1.info is not None

    child_priors = np.full(node1.action_total_value.shape, 1.0)
    aggregate_goal_ability = 0.42

    node1.expand(
        child_priors,
        value_estimate=0.0,
        aggregate_goal_ability=aggregate_goal_ability,
    )

    assert node1.reward == pytest.approx(get_power_reward(aggregate_goal_ability, xi, eta))
    assert node1.reward < 0


@pytest.mark.uses_rllib
def test_expand_leaves_reward_unchanged_when_no_power_or_goal_info():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["world_size"] = WORLD_SIZE
    config["goal_generator"] = "basic"
    config["num_players"] = 1

    env = create_mbag_env_model(config)
    obs, info = env.reset()

    model = _build_model(config, use_power_heads=False)
    mcts = MbagMCTS(
        model,
        MCTS_PARAM,
        gamma=0.95,
        use_goal_predictor=False,
        use_power_reward=False,
    )

    root = MbagRootParentNode(env)
    node0 = MbagMCTSNode(
        action=None,
        obs=obs,
        done=False,
        info=info,
        reward=0.0,
        state=env.get_state(),
        mcts=mcts,
        model_state_in=[],
        parent=root,
    )

    node0.expand(np.full(node0.action_total_value.shape, 1.0), value_estimate=0.0)
    node1 = node0.get_child(0)
    original_reward = node1.reward

    child_priors = np.full(node1.action_total_value.shape, 1.0)
    node1.expand(child_priors, value_estimate=0.0)

    assert node1.reward == original_reward


@pytest.mark.uses_rllib
def test_compute_actions_end_to_end_uses_power_reward():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["world_size"] = WORLD_SIZE
    config["goal_generator"] = "basic"
    config["num_players"] = 1

    env = create_mbag_env_model(config)
    obs, info = env.reset()

    model = _build_model(config, use_power_heads=True)
    model.eval()
    xi, eta = 1.0, 1.1
    mcts = MbagMCTS(
        model,
        {**MCTS_PARAM, "power_xi": xi, "power_eta": eta},
        gamma=0.95,
        use_goal_predictor=False,
        use_power_reward=True,
    )

    root = MbagRootParentNode(env)
    node0 = MbagMCTSNode(
        action=None,
        obs=obs,
        done=False,
        info=info,
        reward=0.0,
        state=env.get_state(),
        mcts=mcts,
        model_state_in=[],
        parent=root,
    )

    # First compute_actions call expands the root node (node0).
    mcts.compute_actions([node0])
    assert node0.is_expanded

    # Descend to a child and expand it too, exercising the same code path
    # MbagMCTSNode.select() would use inside a real simulation.
    child = node0.select()
    if child is node0:
        child = node0.get_child(0)
    mcts.compute_actions([child])
    assert child.is_expanded

    # The child's reward should be a valid U_r value predicted from its own
    # (freshly forward-passed) aggregate goal ability, i.e. always negative.
    assert child.reward < 0
    with torch.no_grad():
        model.compute_priors_and_value([child.obs])
        expected_x_h = float(model.aggregate_goal_ability()[0])
    assert child.reward == pytest.approx(
        get_power_reward(expected_x_h, xi, eta), rel=1e-3
    )


@pytest.mark.uses_rllib
def test_compute_actions_uses_human_policy_prior_for_other_agent_action_dist():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["world_size"] = WORLD_SIZE
    config["goal_generator"] = "basic"
    config["num_players"] = 1

    env = create_mbag_env_model(config)
    obs, info = env.reset()

    model = _build_model(config, use_power_heads=True)
    model.eval()
    human_policy_prior = HeuristicMbagHumanPolicyPrior(config, beta_h=10.0)
    possible_goal_sampler = RandomStructurePossibleGoalSampler(
        BasicGoalGenerator({}), WORLD_SIZE
    )
    mcts = MbagMCTS(
        model,
        MCTS_PARAM,
        gamma=0.95,
        use_goal_predictor=False,
        use_power_reward=True,
        power_human_policy_prior=human_policy_prior,
        power_possible_goal_sampler=possible_goal_sampler,
    )

    root = MbagRootParentNode(env)
    node0 = MbagMCTSNode(
        action=None,
        obs=obs,
        done=False,
        info=info,
        reward=0.0,
        state=env.get_state(),
        mcts=mcts,
        model_state_in=[],
        parent=root,
    )

    mcts.compute_actions([node0])

    assert node0.other_agent_action_dist is not None
    assert node0.other_agent_action_dist.shape == (node0.action_space_size,)
    assert np.all(node0.other_agent_action_dist >= 0)
    assert node0.other_agent_action_dist.sum() == pytest.approx(1.0)
