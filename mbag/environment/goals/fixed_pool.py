import random
from typing import Dict, List, Tuple, Type, TypedDict, Union

import numpy as np

from ..blocks import MinecraftBlocks
from ..types import WorldSize
from .goal_generator import GoalGenerator, GoalGeneratorConfig


class FixedPoolGoalGeneratorConfig(TypedDict):
    goal_generator: Union[str, Type[GoalGenerator]]
    """The inner generator used to build the pool (a name or a class)."""

    goal_generator_config: GoalGeneratorConfig
    num_goals: int
    """Number of distinct goals in the pool."""

    pool_seed: int
    """Seed for building the pool, so every instance builds the same pool."""


class FixedPoolGoalGenerator(GoalGenerator):
    """
    Draws goals from a small fixed pool of structures made by an inner generator.

    The pool is built once per world size from ``pool_seed``, without touching the
    global random state, so any two instances with the same config hold the same
    pool. Used as the environment's goal generator, the true goal of each episode is
    one of the pool's ``num_goals`` structures. The human-power possible-goal sampler
    builds its own instance from the same config, so it samples from the same pool,
    which always contains the human's true goal.
    """

    config: FixedPoolGoalGeneratorConfig

    def __init__(self, config: GoalGeneratorConfig):
        super().__init__(config)
        if self.config["num_goals"] < 1:
            raise ValueError("num_goals must be at least 1")

        inner = self.config["goal_generator"]
        if isinstance(inner, str):
            from . import ALL_GOAL_GENERATORS

            inner = ALL_GOAL_GENERATORS[inner]
        self.inner_goal_generator = inner(self.config["goal_generator_config"])
        self._pools: Dict[Tuple[int, int, int], List[MinecraftBlocks]] = {}

    def get_pool(self, size: WorldSize) -> List[MinecraftBlocks]:
        key = tuple(size)
        if key not in self._pools:
            python_state = random.getstate()
            numpy_state = np.random.get_state()
            try:
                random.seed(self.config["pool_seed"])
                np.random.seed(self.config["pool_seed"])
                self._pools[key] = [
                    self._pad_to_width(
                        self.inner_goal_generator.generate_goal(size), size[0]
                    )
                    for _ in range(self.config["num_goals"])
                ]
            finally:
                random.setstate(python_state)
                np.random.set_state(numpy_state)
        return self._pools[key]

    @staticmethod
    def _pad_to_width(goal: MinecraftBlocks, width: int) -> MinecraftBlocks:
        """
        Pad a structure to the full goal width at a fixed (seeded) x offset. MbagEnv
        places narrower structures at a random x offset, which would make the same
        pool goal appear in different places; at full width the placement is
        deterministic. The padding keeps the ground (dirt) that the environment would
        leave untouched next to the structure.
        """
        if goal.size[0] >= width:
            return goal
        padded = MinecraftBlocks((width, goal.size[1], goal.size[2]))
        padded.blocks[:, 0, :] = MinecraftBlocks.NAME2ID["dirt"]
        offset = random.randint(0, width - goal.size[0])
        padded.blocks[offset : offset + goal.size[0]] = goal.blocks
        padded.block_states[offset : offset + goal.size[0]] = goal.block_states
        return padded

    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        return random.choice(self.get_pool(size)).copy()
