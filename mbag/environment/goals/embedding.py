import random
from typing import Optional

from ..blocks import MinecraftBlocks
from ..types import WorldSize


def get_goal_size(world_size: WorldSize, inf_blocks: bool) -> WorldSize:
    """
    The size a goal generator should generate at: the buildable area of the world,
    with a buffer of 1 on the sides, top, and bottom (and without the palette column
    when blocks are not infinite).
    """

    usable_x = world_size[0] if inf_blocks else world_size[0] - 1
    return (max(1, usable_x - 2), world_size[1] - 2, world_size[2] - 2)


def embed_goal(
    small_goal: MinecraftBlocks,
    world_size: WorldSize,
    inf_blocks: bool,
    x_offset: Optional[int] = None,
) -> MinecraftBlocks:
    """
    Embed a structure made by a goal generator into a full-size world, exactly as
    MbagEnv does for its true goal: on top of the bedrock and dirt floor, shifted up
    by one layer, at a random x offset if there is room (and with the palette column
    when blocks are not infinite).

    Pass x_offset to fix the offset instead of drawing it at random.
    """

    usable_x = world_size[0] if inf_blocks else world_size[0] - 1
    palette_x = world_size[0] - 1

    goal = MinecraftBlocks(world_size)
    goal.blocks[:, 0, :] = MinecraftBlocks.BEDROCK
    goal.blocks[:, 1, :] = MinecraftBlocks.NAME2ID["dirt"]

    shape = small_goal.size
    if x_offset is None:
        # Choose a uniformly random x offset in the valid range
        # [1, usable_x - shape[0] - 1].
        max_x_offset = usable_x - shape[0] - 1
        if max_x_offset >= 1:
            x_offset = random.randint(1, max_x_offset)
        else:
            x_offset = 1

    goal.blocks[
        x_offset : x_offset + shape[0], 1 : shape[1] + 1, 1 : shape[2] + 1
    ] = small_goal.blocks
    goal.block_states[
        x_offset : x_offset + shape[0], 1 : shape[1] + 1, 1 : shape[2] + 1
    ] = small_goal.block_states

    if not inf_blocks:
        for index, block in enumerate(MinecraftBlocks.PLACEABLE_BLOCK_IDS):
            if index >= world_size[2]:
                break
            goal.blocks[palette_x, 2, index] = block
            goal.block_states[palette_x, 2, index] = 0

    return goal
