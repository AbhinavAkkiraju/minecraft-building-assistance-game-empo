import numpy as np
import torch

from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.goals import FixedPoolGoalGenerator
from mbag.environment.goals.embedding import embed_goal, get_goal_size
from mbag.rllib.alpha_zero.power.progress import ProgressReference

SIZE = (6, 6, 6)


def make_reference(num_goals: int = 3):
    generator = FixedPoolGoalGenerator(
        {
            "goal_generator": "random",
            "goal_generator_config": {},
            "num_goals": num_goals,
            "pool_seed": 0,
        }
    )
    pool = generator.get_pool(get_goal_size(SIZE, True))
    goals = np.stack([embed_goal(goal, SIZE, True).blocks for goal in pool])
    initial = np.zeros(SIZE, dtype=np.int64)
    initial[:, 0, :] = MinecraftBlocks.BEDROCK
    initial[:, 1, :] = MinecraftBlocks.NAME2ID["dirt"]
    return ProgressReference(torch.from_numpy(goals), torch.from_numpy(initial)), goals, initial


def progress(reference, world):
    return reference.progress(torch.from_numpy(world[None]).long())[0].numpy()


def test_progress_is_zero_at_start_and_one_at_goal():
    reference, goals, initial = make_reference()
    assert np.allclose(progress(reference, initial), 0.0)
    for index, goal in enumerate(goals):
        assert np.isclose(progress(reference, goal)[index], 1.0)


def test_building_correct_blocks_raises_progress_and_wrong_blocks_lower_it():
    reference, goals, initial = make_reference()
    goal = goals[0]

    # Start from the finished goal and remove blocks one by one: progress falls.
    world = goal.copy()
    previous = progress(reference, world)[0]
    removable = np.argwhere((goal != initial) & (goal != MinecraftBlocks.AIR))
    assert len(removable) > 3
    for x, y, z in removable[:3]:
        world[x, y, z] = MinecraftBlocks.AIR
        current = progress(reference, world)[0]
        assert current < previous
        previous = current

    # Adding a wrong block lowers progress too.
    air_cells = np.argwhere((world == MinecraftBlocks.AIR) & (goal == MinecraftBlocks.AIR))
    x, y, z = air_cells[0]
    world[x, y, z] = MinecraftBlocks.NAME2ID["dirt"]
    assert progress(reference, world)[0] < previous


def test_progress_is_clipped_to_unit_interval():
    reference, goals, initial = make_reference()
    world = np.full(SIZE, MinecraftBlocks.NAME2ID["dirt"], dtype=np.int64)
    values = progress(reference, world)
    assert np.all(values >= 0.0) and np.all(values <= 1.0)


def test_steps_to_goal_potential():
    from mbag.rllib.alpha_zero.power.progress import StepsToGoalReference

    _, goals, initial = make_reference()
    gamma = 0.9
    reference = StepsToGoalReference(torch.from_numpy(goals), gamma)
    goal = goals[0]

    def potential(world):
        return reference.progress(torch.from_numpy(world[None]).long())[0, 0].item()

    # A finished goal has potential 1.
    assert np.isclose(potential(goal), 1.0)

    # Removing a needed block costs one step; a wrong block in a needed cell costs two.
    needed_cells = np.argwhere((goal != MinecraftBlocks.AIR) & (goal != initial))
    x, y, z = needed_cells[0]
    missing = goal.copy()
    missing[x, y, z] = MinecraftBlocks.AIR
    assert np.isclose(potential(missing), gamma)
    wrong = goal.copy()
    wrong[x, y, z] = (
        MinecraftBlocks.NAME2ID["dirt"]
        if goal[x, y, z] != MinecraftBlocks.NAME2ID["dirt"]
        else MinecraftBlocks.NAME2ID["stone"]
    )
    assert np.isclose(potential(wrong), gamma**2)

    # Bedrock on a needed cell makes the goal unreachable; clearing it restores
    # a positive potential.
    blocked = missing.copy()
    blocked[x, y, z] = MinecraftBlocks.BEDROCK
    assert potential(blocked) == 0.0
    assert potential(missing) > 0.0

    # Bedrock on a cell the goal does not need is ignored.
    unneeded = np.argwhere(goal == MinecraftBlocks.AIR)[0]
    elsewhere = goal.copy()
    elsewhere[tuple(unneeded)] = MinecraftBlocks.BEDROCK
    assert np.isclose(potential(elsewhere), 1.0)
