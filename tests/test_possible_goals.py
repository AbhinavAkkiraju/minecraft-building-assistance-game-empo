import numpy as np

from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.goals.possible_goals import (
    MinecraftStructureGoal,
    RandomStructurePossibleGoalGenerator,
    RandomStructurePossibleGoalSampler,
)
from mbag.environment.goals.simple import BasicGoalGenerator
from mbag.environment.state import MbagStateDict

WORLD_SIZE = (3, 3, 3)


def _state_with_blocks(blocks: MinecraftBlocks) -> MbagStateDict:
    return {
        "current_blocks": blocks,
        "goal_blocks": MinecraftBlocks(WORLD_SIZE),
        "player_locations": [],
        "player_directions": [],
        "player_inventories": [],
        "last_interacted": None,  # type: ignore[typeddict-item]
        "timestep": 0,
    }


def test_minecraft_structure_goal_achieved_on_exact_match():
    target = MinecraftBlocks(WORLD_SIZE)
    target.blocks[0, 0, 0] = MinecraftBlocks.NAME2ID["cobblestone"]
    goal = MinecraftStructureGoal(target)

    matching_blocks = target.copy()
    assert goal.is_achieved(_state_with_blocks(matching_blocks)) == 1


def test_minecraft_structure_goal_not_achieved_on_mismatch():
    target = MinecraftBlocks(WORLD_SIZE)
    target.blocks[0, 0, 0] = MinecraftBlocks.NAME2ID["cobblestone"]
    goal = MinecraftStructureGoal(target)

    empty_blocks = MinecraftBlocks(WORLD_SIZE)
    assert goal.is_achieved(_state_with_blocks(empty_blocks)) == 0


def test_minecraft_structure_goal_hash_and_eq():
    target_a = MinecraftBlocks(WORLD_SIZE)
    target_a.blocks[0, 0, 0] = MinecraftBlocks.NAME2ID["cobblestone"]
    target_b = target_a.copy()
    target_c = MinecraftBlocks(WORLD_SIZE)

    goal_a = MinecraftStructureGoal(target_a)
    goal_b = MinecraftStructureGoal(target_b)
    goal_c = MinecraftStructureGoal(target_c)

    assert goal_a == goal_b
    assert hash(goal_a) == hash(goal_b)
    assert goal_a != goal_c
    assert {goal_a, goal_b, goal_c} == {goal_a, goal_c}


def test_matches_blocks_single_state_agrees_with_is_achieved():
    target = MinecraftBlocks(WORLD_SIZE)
    target.blocks[0, 0, 0] = MinecraftBlocks.NAME2ID["cobblestone"]
    goal = MinecraftStructureGoal(target)

    matching = target.copy()
    mismatching = MinecraftBlocks(WORLD_SIZE)

    assert bool(goal.matches_blocks(matching.blocks, matching.block_states)) == bool(
        goal.is_achieved(_state_with_blocks(matching))
    )
    assert bool(
        goal.matches_blocks(mismatching.blocks, mismatching.block_states)
    ) == bool(goal.is_achieved(_state_with_blocks(mismatching)))


def test_matches_blocks_batched_over_a_trajectory():
    target = MinecraftBlocks(WORLD_SIZE)
    target.blocks[0, 0, 0] = MinecraftBlocks.NAME2ID["cobblestone"]
    goal = MinecraftStructureGoal(target)

    empty = MinecraftBlocks(WORLD_SIZE)
    matching = target.copy()

    batched_blocks = np.stack([empty.blocks, matching.blocks, empty.blocks])
    batched_states = np.stack(
        [empty.block_states, matching.block_states, empty.block_states]
    )

    result = goal.matches_blocks(batched_blocks, batched_states)
    assert result.shape == (3,)
    assert list(result) == [False, True, False]


def test_random_structure_sampler_draws_from_goal_generator():
    goal_generator = BasicGoalGenerator({})
    sampler = RandomStructurePossibleGoalSampler(goal_generator, WORLD_SIZE)

    state = _state_with_blocks(MinecraftBlocks(WORLD_SIZE))
    goal, weight = sampler.sample(state, human_index=0)

    assert isinstance(goal, MinecraftStructureGoal)
    assert weight == 1.0
    # BasicGoalGenerator always produces the same structure, so achievement
    # can be checked directly against its known output.
    assert goal.is_achieved(_state_with_blocks(goal.target)) == 1


def test_random_structure_generator_yields_equal_weighted_samples():
    goal_generator = BasicGoalGenerator({})
    num_goals = 5
    generator = RandomStructurePossibleGoalGenerator(
        goal_generator, WORLD_SIZE, num_goals=num_goals
    )

    state = _state_with_blocks(MinecraftBlocks(WORLD_SIZE))
    pairs = list(generator.generate(state, human_index=0))

    assert len(pairs) == num_goals
    total_weight = sum(weight for _, weight in pairs)
    assert abs(total_weight - 1.0) < 1e-9
    assert all(isinstance(goal, MinecraftStructureGoal) for goal, _ in pairs)
