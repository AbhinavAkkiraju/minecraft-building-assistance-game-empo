import numpy as np

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.goals.possible_goals import MinecraftStructureGoal
from mbag.environment.mbag_env import MbagEnv
from mbag.environment.types import CURRENT_BLOCKS, MbagObs
from mbag.rllib.alpha_zero.power.human_policy_prior import (
    HeuristicMbagHumanPolicyPrior,
)

WORLD_SIZE = (3, 3, 3)
ENV_CONFIG = MbagEnv.get_config(
    {
        "world_size": WORLD_SIZE,
        "num_players": 1,
    }
)


def _obs_with_current_blocks(blocks: MinecraftBlocks) -> MbagObs:
    world_obs = np.zeros((6,) + WORLD_SIZE, dtype=np.uint8)
    world_obs[CURRENT_BLOCKS] = blocks.blocks
    inventory_obs = np.zeros((1, MinecraftBlocks.NUM_BLOCKS), dtype=np.int32)
    return world_obs, inventory_obs, 0


def test_prefers_placing_correct_block_at_mismatched_air_cell():
    target = MinecraftBlocks(WORLD_SIZE)
    target.blocks[0, 0, 0] = MinecraftBlocks.NAME2ID["cobblestone"]
    goal = MinecraftStructureGoal(target)

    current = MinecraftBlocks(WORLD_SIZE)  # all air
    prior = HeuristicMbagHumanPolicyPrior(ENV_CONFIG, beta_h=20.0)
    probs = prior.action_distribution(_obs_with_current_blocks(current), goal)

    assert probs.shape == (
        MbagActionDistribution.NUM_CHANNELS,
        *WORLD_SIZE,
    )
    assert abs(probs.sum() - 1.0) < 1e-9

    correct_channel = MbagActionDistribution.PLACE_BLOCK.start + MinecraftBlocks.NAME2ID[
        "cobblestone"
    ]
    assert probs[correct_channel, 0, 0, 0] > 0.9


def test_noop_only_when_goal_already_achieved():
    target = MinecraftBlocks(WORLD_SIZE)
    target.blocks[0, 0, 0] = MinecraftBlocks.NAME2ID["cobblestone"]
    goal = MinecraftStructureGoal(target)

    current = target.copy()
    prior = HeuristicMbagHumanPolicyPrior(ENV_CONFIG, beta_h=20.0)
    probs = prior.action_distribution(_obs_with_current_blocks(current), goal)

    assert probs[0, 0, 0, 0] == 1.0
    assert probs.sum() == 1.0


def test_uniform_over_valid_actions_when_beta_h_is_zero():
    target = MinecraftBlocks(WORLD_SIZE)
    target.blocks[0, 0, 0] = MinecraftBlocks.NAME2ID["cobblestone"]
    goal = MinecraftStructureGoal(target)

    current = MinecraftBlocks(WORLD_SIZE)
    prior = HeuristicMbagHumanPolicyPrior(ENV_CONFIG, beta_h=0.0)
    probs = prior.action_distribution(_obs_with_current_blocks(current), goal)

    nonzero_probs = probs[probs > 0]
    num_placeable_block_ids = len(MinecraftBlocks.PLACEABLE_BLOCK_IDS)
    # Only the (0,0,0) cell is a mismatch, so valid actions are: NOOP, plus
    # one PLACE_BLOCK option per placeable block id at that single cell.
    assert len(nonzero_probs) == 1 + num_placeable_block_ids
    assert np.allclose(nonzero_probs, nonzero_probs[0])


def test_break_wrong_block_is_valid_but_not_preferred_over_correct_fix_elsewhere():
    target = MinecraftBlocks(WORLD_SIZE)
    target.blocks[0, 0, 0] = MinecraftBlocks.NAME2ID["cobblestone"]
    target.blocks[1, 0, 0] = MinecraftBlocks.NAME2ID["wool"]
    goal = MinecraftStructureGoal(target)

    current = MinecraftBlocks(WORLD_SIZE)
    current.blocks[0, 0, 0] = MinecraftBlocks.NAME2ID["dirt"]  # wrong block placed
    # (1, 0, 0) is left as air -- a direct placement opportunity.

    prior = HeuristicMbagHumanPolicyPrior(ENV_CONFIG, beta_h=20.0)
    probs = prior.action_distribution(_obs_with_current_blocks(current), goal)

    assert probs[MbagActionDistribution.BREAK_BLOCK, 0, 0, 0] > 0
    correct_channel = MbagActionDistribution.PLACE_BLOCK.start + MinecraftBlocks.NAME2ID[
        "wool"
    ]
    assert probs[correct_channel, 1, 0, 0] > probs[MbagActionDistribution.BREAK_BLOCK, 0, 0, 0]
