import numpy as np

from mbag.environment.actions import MbagAction
from mbag.environment.blocks import MinecraftBlocks


def test_human_cannot_break_or_place_over_bedrock():
    blocks = MinecraftBlocks((3, 3, 3))
    location = (1, 1, 1)
    blocks.blocks[location] = MinecraftBlocks.BEDROCK

    for action_type in [MbagAction.BREAK_BLOCK, MbagAction.PLACE_BLOCK]:
        result = blocks.try_break_place(
            action_type,
            location,
            block_id=MinecraftBlocks.NAME2ID["dirt"],
            player_location=(1.0, 1.0, 0.0),
            is_human=True,
        )
        assert result is None
        assert blocks.blocks[location] == MinecraftBlocks.BEDROCK


def test_non_human_can_still_break_bedrock():
    blocks = MinecraftBlocks((3, 3, 3))
    location = (1, 1, 1)
    blocks.blocks[location] = MinecraftBlocks.BEDROCK

    result = blocks.try_break_place(MbagAction.BREAK_BLOCK, location)
    assert result is not None
    assert blocks.blocks[location] == MinecraftBlocks.AIR


def test_env_human_cannot_clear_bedrock_with_teleportation():
    from mbag.environment.goals.simple import RandomGoalGenerator
    from mbag.environment.mbag_env import MbagEnv

    env = MbagEnv(
        {
            "world_size": (6, 6, 6),
            "num_players": 2,
            "horizon": 30,
            "num_clutter_blocks": 20,
            "clutter_bedrock_fraction": 1.0,
            "goal_generator": RandomGoalGenerator,
            "goal_generator_config": {},
            "players": [{"is_human": True}, {}],
            "abilities": {"teleportation": True, "flying": True, "inf_blocks": True},
        }
    )
    env.reset()

    def num_clutter_bedrock() -> int:
        return int((env.current_blocks.blocks[:, 1:] == MinecraftBlocks.BEDROCK).sum())

    assert num_clutter_bedrock() == 20
    bedrock_location = tuple(
        int(i) for i in np.argwhere(env.current_blocks.blocks[:, 1:] == 1)[0]
    )
    bedrock_location = (bedrock_location[0], bedrock_location[1] + 1, bedrock_location[2])
    flat = int(np.ravel_multi_index(bedrock_location, (6, 6, 6)))

    # The human trying to break or place over bedrock does nothing.
    for action_type in [MbagAction.BREAK_BLOCK, MbagAction.PLACE_BLOCK]:
        env.step([(action_type, flat, 2), (MbagAction.NOOP, 0, 0)])
        assert num_clutter_bedrock() == 20

    # The non-human assistant can break it.
    env.step([(MbagAction.NOOP, 0, 0), (MbagAction.BREAK_BLOCK, flat, 0)])
    assert num_clutter_bedrock() == 19
