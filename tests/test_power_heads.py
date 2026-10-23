from typing import Any, List, cast

import numpy as np
import pytest

from mbag.environment.mbag_env import MbagEnv

try:
    import torch
    from ray.rllib.models.catalog import ModelCatalog
    from ray.rllib.policy.sample_batch import SampleBatch

    from mbag.rllib.torch_models import MbagConvolutionalModel
except ImportError:
    pass


WORLD_SIZE = (5, 5, 5)


def _make_env() -> MbagEnv:
    return MbagEnv(
        {
            "goal_generator": "random",
            "horizon": 50,
            "num_players": 1,
            "abilities": {
                "teleportation": True,
                "flying": True,
                "inf_blocks": True,
            },
            "world_size": WORLD_SIZE,
        }
    )


def _build_model(env: MbagEnv, *, use_power_heads: bool):
    return ModelCatalog.get_model_v2(
        env.observation_space,
        env.action_space,
        num_outputs=1,
        model_config={
            "custom_model": "mbag_convolutional_model",
            "custom_action_dist": "mbag_bilevel_categorical",
            "custom_model_config": {
                "env_config": env.config,
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


def _obs_batch(env: MbagEnv, batch_size: int) -> "torch.Tensor":
    preprocessor = ModelCatalog.get_preprocessor_for_space(env.observation_space)
    obs_list = []
    for _ in range(batch_size):
        (obs,), _ = env.reset()
        obs_list.append(preprocessor.transform(cast(Any, obs)))
    return torch.from_numpy(np.stack(obs_list))


@pytest.mark.uses_rllib
def test_power_heads_constructed_only_when_enabled():
    env = _make_env()
    model_with = _build_model(env, use_power_heads=True)
    model_without = _build_model(env, use_power_heads=False)

    assert isinstance(model_with, MbagConvolutionalModel)
    assert hasattr(model_with, "human_goal_ability_head")
    assert hasattr(model_with, "aggregate_goal_ability_head")
    assert not hasattr(model_without, "human_goal_ability_head")
    assert not hasattr(model_without, "aggregate_goal_ability_head")


@pytest.mark.uses_rllib
def test_human_goal_ability_and_aggregate_goal_ability_shapes_and_ranges():
    env = _make_env()
    model = _build_model(env, use_power_heads=True)
    model.eval()

    batch_size = 4
    obs_batch = _obs_batch(env, batch_size)
    state_in = cast(List["torch.Tensor"], model.get_initial_state())
    model(
        {SampleBatch.OBS: obs_batch},
        [state_piece[None] for state_piece in state_in],
        torch.tensor([1] * batch_size),
    )

    possible_goal_blocks = torch.randint(
        0, 10, (batch_size,) + WORLD_SIZE, dtype=torch.long
    )
    v_h_e = model.human_goal_ability(possible_goal_blocks)
    assert v_h_e.shape == (batch_size,)
    assert torch.all((v_h_e > 0) & (v_h_e < 1))

    x_h = model.aggregate_goal_ability()
    assert x_h.shape == (batch_size,)
    assert torch.all(x_h > 0)


@pytest.mark.uses_rllib
def test_human_goal_ability_depends_on_the_possible_goal():
    env = _make_env()
    model = _build_model(env, use_power_heads=True)
    model.eval()

    obs_batch = _obs_batch(env, 1)
    state_in = cast(List["torch.Tensor"], model.get_initial_state())
    model(
        {SampleBatch.OBS: obs_batch},
        [state_piece[None] for state_piece in state_in],
        torch.tensor([1]),
    )

    all_air = torch.zeros((1,) + WORLD_SIZE, dtype=torch.long)
    all_stone = torch.full((1,) + WORLD_SIZE, 7, dtype=torch.long)  # stone

    v_h_e_air = model.human_goal_ability(all_air)
    v_h_e_stone = model.human_goal_ability(all_stone)
    assert not torch.allclose(v_h_e_air, v_h_e_stone)


@pytest.mark.uses_rllib
def test_power_heads_disabled_by_default_raise_on_use():
    env = _make_env()
    model = _build_model(env, use_power_heads=False)
    model.eval()

    obs_batch = _obs_batch(env, 1)
    state_in = cast(List["torch.Tensor"], model.get_initial_state())
    model(
        {SampleBatch.OBS: obs_batch},
        [state_piece[None] for state_piece in state_in],
        torch.tensor([1]),
    )

    with pytest.raises(AssertionError):
        model.aggregate_goal_ability()
