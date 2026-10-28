"""
Possible-goal abstraction for the human-power (empowerment) objective.

Implements ``G_h``, the space of possible goals the assistant's intrinsic
reward aggregates over, following the human-power theory of Heitzig & Potham
(arXiv:2508.00159) and mirroring the ``PossibleGoal``/``PossibleGoalSampler``/
``PossibleGoalGenerator`` split of the reference implementation in the
``empo`` repository (``empo/src/empo/possible_goal.py``).

A possible goal is not the environment's one true (usually hidden) goal
structure; it is one of many hypothetical structures a human *might* be
building. The assistant never infers which one is real -- it only estimates
the human's aggregate ability to achieve a variety of them (``X_h``).

Per the confirmed design decision for this port, possible goals are sampled
as fully independent candidate structures from an existing ``GoalGenerator``,
not conditioned on the human's in-progress build, and goal achievement
(``U_h``) is strictly binary (``1[s' in g_h]``), matching the paper's formula
exactly rather than MBAG's own graded block-similarity score.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional, Tuple

import numpy as np

from ..blocks import MinecraftBlocks
from ..state import MbagStateDict
from ..types import WorldSize
from .embedding import embed_goal, get_goal_size
from .goal_generator import GoalGenerator


class PossibleGoal(ABC):
    """
    A candidate goal structure, encoded as the 0/1 achievement indicator
    ``U_h(s, g_h) = 1[s' in g_h]``. Must be hashable so it can be used as a
    dictionary key when aggregating ``V_h^e``/``X_h``.
    """

    @abstractmethod
    def is_achieved(self, state: MbagStateDict) -> int:
        """Return 1 if this goal is achieved in ``state``, 0 otherwise."""

    @abstractmethod
    def __hash__(self) -> int: ...

    @abstractmethod
    def __eq__(self, other: object) -> bool: ...


class MinecraftStructureGoal(PossibleGoal):
    """
    A possible goal defined by a target structure.

    By default it is achieved iff the current build matches ``target``
    block-for-block over the whole world, including block states. With
    ``lenient=True`` it is achieved iff every cell where ``target`` has a block
    holds that block (with the same block state), and the rest of the world --
    e.g. bedrock clutter or stray blocks elsewhere -- is ignored. This treats a
    goal as the event "this structure is built" rather than a single world state.
    """

    def __init__(self, target: MinecraftBlocks, lenient: bool = False):
        self.target = target
        self.lenient = lenient
        self._hash = hash(
            (target.blocks.tobytes(), target.block_states.tobytes(), lenient)
        )

    def is_achieved(self, state: MbagStateDict) -> int:
        current = state["current_blocks"]
        return int(
            self.matches_blocks(current.blocks, current.block_states).all()
        )

    def matches_blocks(
        self, current_blocks: np.ndarray, current_block_states: np.ndarray
    ) -> np.ndarray:
        """
        Vectorized equivalent of ``is_achieved`` for raw block-id/block-state
        arrays -- e.g. ``world_obs[:, CURRENT_BLOCKS]``/
        ``world_obs[:, CURRENT_BLOCK_STATES]`` across a trajectory fragment --
        avoiding the need to construct a full ``MbagStateDict`` per timestep.

        Args:
            current_blocks: Array whose trailing dimensions match
                ``self.target.blocks.shape``; may have an arbitrary leading
                batch shape (e.g. a time dimension).
            current_block_states: Same shape as ``current_blocks``.

        Returns:
            Boolean array with the leading batch shape.
        """
        current_blocks = np.asarray(current_blocks)
        current_block_states = np.asarray(current_block_states)
        target_blocks = self.target.blocks
        target_block_states = self.target.block_states
        trailing_axes = tuple(range(-target_blocks.ndim, 0))
        cell_matches = (current_blocks == target_blocks) & (
            current_block_states == target_block_states
        )
        if self.lenient:
            needed = target_blocks != MinecraftBlocks.AIR
            cell_matches = cell_matches | ~needed
        return np.all(cell_matches, axis=trailing_axes)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, MinecraftStructureGoal)
            and self.lenient == other.lenient
            and self.target == other.target
        )


class PossibleGoalSampler(ABC):
    """Stochastic source of ``(goal, weight)`` pairs for Monte Carlo estimation of ``X_h``."""

    @abstractmethod
    def sample(
        self, state: MbagStateDict, human_index: int
    ) -> Tuple[PossibleGoal, float]:
        """
        Draw one possible goal for the human at ``human_index`` given
        ``state``, along with its weight in the
        ``X_h = E_g[weight * V_h^e(s,g)^zeta]`` aggregation.
        """


class PossibleGoalGenerator(ABC):
    """
    Finite source of ``(goal, weight)`` pairs to aggregate ``X_h`` over --
    either an exact enumeration (for small, hand-built test environments) or
    a fixed-size Monte Carlo sample (for the full Minecraft environment,
    where the goal space is too large to enumerate).
    """

    @abstractmethod
    def generate(
        self, state: MbagStateDict, human_index: int
    ) -> Iterator[Tuple[PossibleGoal, float]]:
        """Yield ``(goal, weight)`` pairs for the human at ``human_index``."""


class RandomStructurePossibleGoalSampler(PossibleGoalSampler):
    """
    Draws possible goals as fully independent candidate structures from an
    existing ``GoalGenerator`` (e.g. the ``craftassist``/``grabcraft``
    generators already used for the environment's one true goal), resampled
    fresh on every call rather than conditioned on the human's in-progress
    build. This is the literal reading of the paper's ``G_h`` and of
    ``empo``'s own possible-goal semantics, at the cost of a sparse ``V_h^e``
    signal for structures unrelated to what's already been built.
    """

    def __init__(
        self,
        goal_generator: GoalGenerator,
        size: WorldSize,
        weight: float = 1.0,
        inf_blocks: Optional[bool] = None,
        lenient: bool = False,
    ):
        """
        If inf_blocks is given, sampled structures are embedded into a full-size
        world exactly as MbagEnv does for its true goal (floor, shift, buffer), so a
        sampled goal can match the world state when the human builds it. Otherwise
        the generator's raw output at ``size`` is used.
        """
        self.goal_generator = goal_generator
        self.size = size
        self.weight = weight
        self.inf_blocks = inf_blocks
        self.lenient = lenient

    def sample(
        self, state: MbagStateDict, human_index: int
    ) -> Tuple[PossibleGoal, float]:
        if self.inf_blocks is None:
            target = self.goal_generator.generate_goal(self.size)
        else:
            small_goal = self.goal_generator.generate_goal(
                get_goal_size(self.size, self.inf_blocks)
            )
            target = embed_goal(small_goal, self.size, self.inf_blocks)
        return MinecraftStructureGoal(target, lenient=self.lenient), self.weight


class RandomStructurePossibleGoalGenerator(PossibleGoalGenerator):
    """
    Draws ``num_goals`` independent candidate structures via a
    ``RandomStructurePossibleGoalSampler`` and yields them with equal weight
    ``1/num_goals``, giving a fixed-size Monte Carlo estimate of
    ``X_h = E_g[V_h^e(s,g)^zeta]`` usable wherever exact enumeration (as in
    ``empo``'s tiny gridworld demos) isn't tractable.
    """

    def __init__(self, goal_generator: GoalGenerator, size: WorldSize, num_goals: int):
        if num_goals < 1:
            raise ValueError("num_goals must be at least 1")
        self.sampler = RandomStructurePossibleGoalSampler(goal_generator, size)
        self.num_goals = num_goals

    def generate(
        self, state: MbagStateDict, human_index: int
    ) -> Iterator[Tuple[PossibleGoal, float]]:
        weight = 1.0 / self.num_goals
        for _ in range(self.num_goals):
            goal, _ = self.sampler.sample(state, human_index)
            yield goal, weight
