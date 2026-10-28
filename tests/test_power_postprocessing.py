import copy

import numpy as np
import pytest

from mbag.environment.mbag_env import DEFAULT_CONFIG
from mbag.rllib.alpha_zero.power.formulas import human_goal_ability_monte_carlo_targets

try:
    import torch
    from ray.rllib.models.catalog import ModelCatalog
    from ray.rllib.policy.sample_batch import SampleBatch

    from mbag.environment.goals.possible_goals import RandomStructurePossibleGoalSampler
    from mbag.environment.goals.simple import BasicGoalGenerator
    from mbag.environment.types import CURRENT_BLOCKS
    from mbag.rllib.alpha_zero.alpha_zero_policy import (
        AGGREGATE_GOAL_ABILITY_TARGETS,
        HUMAN_GOAL_ABILITY_TARGETS,
        POSSIBLE_GOAL_BLOCKS,
        MbagAlphaZeroPolicy,
    )
    from mbag.rllib.alpha_zero.planning import create_mbag_env_model
except ImportError:
    pass


WORLD_SIZE = (5, 5, 5)


def _make_config():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["world_size"] = WORLD_SIZE
    config["goal_generator"] = "basic"
    config["num_players"] = 1
    return config


def _build_model():
    env_model = create_mbag_env_model(_make_config())
    return ModelCatalog.get_model_v2(
        env_model.observation_space.spaces[env_model.agent_id],
        env_model.action_space,
        num_outputs=1,
        model_config={
            "custom_model": "mbag_convolutional_model",
            "custom_action_dist": "mbag_bilevel_categorical",
            "custom_model_config": {
                "env_config": _make_config(),
                "embedding_size": 4,
                "hidden_size": 8,
                "hidden_channels": 8,
                "num_layers": 1,
                "mask_goal": True,
                "use_power_heads": True,
            },
            "vf_share_layers": True,
        },
        framework="torch",
    )


class _FakePowerPolicy:
    """
    A minimal stand-in exposing only what
    MbagAlphaZeroPolicy._add_power_training_columns reads, so that real
    (unbound) method can be tested directly against a small, controllable
    trajectory fixture without needing a full TorchPolicy/AlphaZeroPolicy
    (which requires replicating a large default RLlib AlgorithmConfig).
    """

    def __init__(self, model, obs_space, gamma_h, zeta):
        self.model = model
        self.obs_space = obs_space
        self.power_gamma_h = gamma_h
        self.power_zeta = zeta
        self.power_rollout_goal = False
        self.power_goal_lenient = False
        self.power_num_goals = 1.0
        self.power_possible_goal_sampler = RandomStructurePossibleGoalSampler(
            BasicGoalGenerator({}), WORLD_SIZE
        )


@pytest.mark.uses_rllib
def test_add_power_training_columns_not_terminated():
    env_model = create_mbag_env_model(_make_config())
    model = _build_model()
    model.eval()
    preprocessor = ModelCatalog.get_preprocessor_for_space(
        env_model.observation_space.spaces[env_model.agent_id]
    )

    # Roll out a short trajectory of raw observations.
    obs, _ = env_model.reset()
    raw_obs_sequence = [obs]
    for _ in range(3):
        obs, _, _, _, _ = env_model.step(0)  # NOOP each time.
        raw_obs_sequence.append(obs)
    fragment_obs = raw_obs_sequence[:-1]  # T=3 timesteps in the fragment.
    last_obs = raw_obs_sequence[-1]  # The state past the fragment (s_T).

    preprocessed_fragment = np.stack(
        [preprocessor.transform(o) for o in fragment_obs]
    )
    sample_batch = SampleBatch(
        {
            SampleBatch.OBS: preprocessed_fragment,
            SampleBatch.TERMINATEDS: np.array([False, False, False]),
        }
    )
    last_input_dict = SampleBatch(
        {SampleBatch.OBS: preprocessor.transform(last_obs)[None]}
    )

    with torch.no_grad():
        model.compute_priors_and_value([last_obs])  # Populates _backbone_out for s_T.

    fake_policy = _FakePowerPolicy(
        model, env_model.observation_space.spaces[env_model.agent_id], gamma_h=0.9, zeta=2.0
    )

    MbagAlphaZeroPolicy._add_power_training_columns(
        fake_policy, sample_batch, last_input_dict, terminated=False
    )

    assert POSSIBLE_GOAL_BLOCKS in sample_batch
    assert HUMAN_GOAL_ABILITY_TARGETS in sample_batch
    assert AGGREGATE_GOAL_ABILITY_TARGETS in sample_batch

    possible_goal_blocks = sample_batch[POSSIBLE_GOAL_BLOCKS]
    assert possible_goal_blocks.shape == (3,) + WORLD_SIZE
    # The same possible goal is used across the whole fragment.
    assert np.all(possible_goal_blocks[0] == possible_goal_blocks[1])

    v_h_e_targets = sample_batch[HUMAN_GOAL_ABILITY_TARGETS]
    x_h_targets = sample_batch[AGGREGATE_GOAL_ABILITY_TARGETS]
    assert v_h_e_targets.shape == (3,)
    assert x_h_targets.shape == (3,)
    assert np.all((v_h_e_targets >= 0) & (v_h_e_targets <= 1))
    assert np.allclose(x_h_targets, v_h_e_targets**2.0)


@pytest.mark.uses_rllib
def test_add_power_training_columns_matches_formula_directly():
    # Cross-check against human_goal_ability_monte_carlo_targets computed by
    # hand from the same achieved sequence, to confirm the method wires the
    # formula together correctly rather than just producing plausible shapes.
    env_model = create_mbag_env_model(_make_config())
    model = _build_model()
    model.eval()
    preprocessor = ModelCatalog.get_preprocessor_for_space(
        env_model.observation_space.spaces[env_model.agent_id]
    )

    obs, _ = env_model.reset()
    raw_obs_sequence = [obs]
    for _ in range(3):
        obs, _, _, _, _ = env_model.step(0)
        raw_obs_sequence.append(obs)
    fragment_obs = raw_obs_sequence[:-1]
    last_obs = raw_obs_sequence[-1]

    preprocessed_fragment = np.stack(
        [preprocessor.transform(o) for o in fragment_obs]
    )
    sample_batch = SampleBatch(
        {
            SampleBatch.OBS: preprocessed_fragment,
            SampleBatch.TERMINATEDS: np.array([False, False, False]),
        }
    )
    last_input_dict = SampleBatch(
        {SampleBatch.OBS: preprocessor.transform(last_obs)[None]}
    )

    with torch.no_grad():
        model.compute_priors_and_value([last_obs])

    fake_policy = _FakePowerPolicy(
        model, env_model.observation_space.spaces[env_model.agent_id], gamma_h=0.8, zeta=1.5
    )

    # Sample the SAME possible goal the method will use internally by
    # monkeypatching the sampler to a deterministic one for this check.
    fixed_goal, _ = fake_policy.power_possible_goal_sampler.sample(None, 0)
    fake_policy.power_possible_goal_sampler = _DeterministicSampler(fixed_goal)

    MbagAlphaZeroPolicy._add_power_training_columns(
        fake_policy, sample_batch, last_input_dict, terminated=False
    )

    # Recompute achieved/bootstrap independently and check the formula matches.
    world_obs_fragment = np.stack(
        [o[0] for o in fragment_obs]
    )  # (T, NUM_CHANNELS, w, h, d)
    achieved_mid = fixed_goal.matches_blocks(
        world_obs_fragment[1:, CURRENT_BLOCKS], world_obs_fragment[1:, 1]
    ).astype(np.float32)

    last_world_obs = last_obs[0]
    last_achieved = float(
        fixed_goal.matches_blocks(last_world_obs[CURRENT_BLOCKS], last_world_obs[1])
    )
    with torch.no_grad():
        model.compute_priors_and_value([last_obs])
        target_blocks_tensor = torch.from_numpy(fixed_goal.target.blocks[None])
        last_v_h_e = float(model.human_goal_ability(target_blocks_tensor)[0].item())

    achieved = np.concatenate([achieved_mid, [last_achieved]])
    expected_targets = human_goal_ability_monte_carlo_targets(
        achieved, gamma_h=0.8, bootstrap_v_h_e=last_v_h_e
    )

    assert np.allclose(
        sample_batch[HUMAN_GOAL_ABILITY_TARGETS], expected_targets, atol=1e-5
    )


class _DeterministicSampler:
    def __init__(self, goal):
        self.goal = goal

    def sample(self, state, human_index):
        return self.goal, 1.0


@pytest.mark.uses_rllib
def test_aggregate_goal_ability_from_goal_set_matches_eq_7():
    env_model = create_mbag_env_model(_make_config())
    model = _build_model()
    model.eval()
    obs, _ = env_model.reset()
    goals = torch.randint(0, 5, (3,) + WORLD_SIZE)
    model.set_goal_set(goals, zeta=2.0)
    with torch.no_grad():
        model.compute_priors_and_value([obs, obs])
        per_goal = model.goal_set_abilities()  # (2, 3)
        x_h = model.aggregate_goal_ability()
        manual = torch.stack(
            [
                model.human_goal_ability(goal[None].expand(2, *goal.size()))
                for goal in goals
            ],
            dim=1,
        )
    assert per_goal.shape == (2, 3)
    assert torch.allclose(per_goal, manual)
    assert torch.allclose(x_h, (manual**2).sum(dim=1))


@pytest.mark.uses_rllib
def test_human_goal_ability_logits_match_probabilities():
    env_model = create_mbag_env_model(_make_config())
    model = _build_model()
    model.eval()
    obs, _ = env_model.reset()
    goals = torch.randint(0, 5, (2,) + WORLD_SIZE)
    with torch.no_grad():
        model.compute_priors_and_value([obs, obs])
        probs = model.human_goal_ability(goals)
        logits = model.human_goal_ability(goals, logits=True)
    assert torch.allclose(torch.sigmoid(logits), probs, atol=1e-6)
