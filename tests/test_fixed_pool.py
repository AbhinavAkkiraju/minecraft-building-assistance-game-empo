import random

import numpy as np

from mbag.environment.goals import FixedPoolGoalGenerator
from mbag.environment.goals.possible_goals import (
    MinecraftStructureGoal,
    RandomStructurePossibleGoalSampler,
)

SIZE = (6, 6, 6)


def make_generator(num_goals: int = 10, pool_seed: int = 0) -> FixedPoolGoalGenerator:
    return FixedPoolGoalGenerator(
        {
            "goal_generator": "random",
            "goal_generator_config": {},
            "num_goals": num_goals,
            "pool_seed": pool_seed,
        }
    )


def test_pool_has_requested_size_and_distinct_goals():
    pool = make_generator(10).get_pool(SIZE)
    assert len(pool) == 10
    assert len({MinecraftStructureGoal(goal) for goal in pool}) == 10


def test_pool_is_deterministic_across_instances_and_seed_dependent():
    pool_a = make_generator(10, pool_seed=3).get_pool(SIZE)
    pool_b = make_generator(10, pool_seed=3).get_pool(SIZE)
    pool_c = make_generator(10, pool_seed=4).get_pool(SIZE)
    assert all(a == b for a, b in zip(pool_a, pool_b))
    assert any(a != c for a, c in zip(pool_a, pool_c))


def test_building_pool_does_not_change_global_random_state():
    random.seed(123)
    np.random.seed(123)
    expected = (random.random(), np.random.rand())

    random.seed(123)
    np.random.seed(123)
    make_generator(10).get_pool(SIZE)
    assert (random.random(), np.random.rand()) == expected


def test_generated_goals_come_from_pool_and_are_copies():
    generator = make_generator(5)
    pool = generator.get_pool(SIZE)
    seen = set()
    for _ in range(100):
        goal = generator.generate_goal(SIZE)
        assert any(goal == member for member in pool)
        seen.add(goal.blocks.tobytes())
        goal.blocks[:] = 0  # Must not corrupt the pool.
    assert len(seen) == 5
    assert all(member.blocks.any() for member in pool)


def test_possible_goal_sampler_uses_same_pool_as_true_goal():
    true_goal_generator = make_generator(10)
    sampler = RandomStructurePossibleGoalSampler(make_generator(10), SIZE)
    true_goals = {
        MinecraftStructureGoal(true_goal_generator.generate_goal(SIZE))
        for _ in range(200)
    }
    possible_goals = {sampler.sample(None, 0)[0] for _ in range(200)}  # type: ignore
    assert len(true_goals) == 10
    assert true_goals == possible_goals


def test_sampled_goal_matches_environment_goal_and_is_achieved_by_human():
    """
    The possible-goal sampler must build goals the way MbagEnv does, so that a
    sampled goal equals the environment's true goal (when drawn from the same pool)
    and counts as achieved once the human finishes building it.
    """
    import numpy as np

    from mbag.agents.heuristic_agents import LowestBlockAgent
    from mbag.environment.mbag_env import MbagEnv

    pool_config = {
        "goal_generator": "random",
        "goal_generator_config": {},
        "num_goals": 1,
        "pool_seed": 0,
    }
    env_config = {
        "world_size": SIZE,
        "num_players": 2,
        "horizon": 150,
        "goal_generator": FixedPoolGoalGenerator,
        "goal_generator_config": pool_config,
        "players": [{"is_human": True}, {}],
        "abilities": {"teleportation": True, "flying": True, "inf_blocks": True},
    }
    env = MbagEnv(env_config)
    obs, _ = env.reset()

    sampler = RandomStructurePossibleGoalSampler(
        FixedPoolGoalGenerator(pool_config), SIZE, inf_blocks=True
    )
    possible_goal, _ = sampler.sample(None, 0)  # type: ignore
    assert np.array_equal(possible_goal.target.blocks, env.goal_blocks.blocks)

    agent = LowestBlockAgent({}, MbagEnv.get_config(env_config))
    agent.reset(seed=0)
    for _ in range(150):
        obs, _, done, _ = env.step([agent.get_action(obs[0]), (0, 0, 0)])
        if all(done):
            break
    assert possible_goal.matches_blocks(
        env.current_blocks.blocks, env.current_blocks.block_states
    )


def test_narrow_pool_goals_are_placed_identically_by_env_and_sampler():
    """
    Structures narrower than the build area would be placed at a random x offset by
    MbagEnv. In a pool they are padded to full width, so every reset places a pool
    goal exactly where the possible-goal sampler puts it.
    """
    from mbag.environment.goals.goal_generator import GoalGenerator
    from mbag.environment.blocks import MinecraftBlocks
    from mbag.environment.mbag_env import MbagEnv

    class NarrowGoalGenerator(GoalGenerator):
        def generate_goal(self, size):
            goal = MinecraftBlocks((2, 2, 2))
            goal.blocks[:] = np.random.randint(2, 9, size=goal.blocks.shape)
            return goal

    pool_config = {
        "goal_generator": NarrowGoalGenerator,
        "goal_generator_config": {},
        "num_goals": 3,
        "pool_seed": 1,
    }
    env = MbagEnv(
        {
            "world_size": SIZE,
            "num_players": 2,
            "horizon": 10,
            "goal_generator": FixedPoolGoalGenerator,
            "goal_generator_config": pool_config,
            "players": [{"is_human": True}, {}],
            "abilities": {"teleportation": True, "flying": True, "inf_blocks": True},
        }
    )
    sampler = RandomStructurePossibleGoalSampler(
        FixedPoolGoalGenerator(pool_config), SIZE, inf_blocks=True
    )
    possible = {sampler.sample(None, 0)[0].target.blocks.tobytes() for _ in range(100)}
    assert len(possible) == 3
    for _ in range(30):
        env.reset(force_regenerate_goal=True)
        assert env.goal_blocks.blocks.tobytes() in possible


def test_lenient_goal_ignores_unneeded_cells_but_not_needed_ones():
    from mbag.environment.blocks import MinecraftBlocks
    from mbag.environment.goals.possible_goals import MinecraftStructureGoal

    target = MinecraftBlocks((4, 4, 4))
    target.blocks[1, 1, 1] = MinecraftBlocks.NAME2ID["dirt"]
    target.blocks[2, 1, 1] = MinecraftBlocks.NAME2ID["dirt"]
    exact = MinecraftStructureGoal(target)
    lenient = MinecraftStructureGoal(target, lenient=True)

    world = target.copy()
    assert exact.matches_blocks(world.blocks, world.block_states)
    assert lenient.matches_blocks(world.blocks, world.block_states)

    # Bedrock on a cell the structure does not need: exact fails, lenient ignores it.
    world.blocks[3, 3, 3] = MinecraftBlocks.BEDROCK
    assert not exact.matches_blocks(world.blocks, world.block_states)
    assert lenient.matches_blocks(world.blocks, world.block_states)

    # Bedrock on a needed cell, or a needed block missing: lenient fails too.
    blocked = target.copy()
    blocked.blocks[1, 1, 1] = MinecraftBlocks.BEDROCK
    assert not lenient.matches_blocks(blocked.blocks, blocked.block_states)
    missing = target.copy()
    missing.blocks[2, 1, 1] = MinecraftBlocks.AIR
    assert not lenient.matches_blocks(missing.blocks, missing.block_states)

    # Batch dimension works, and the two modes are different goals.
    batch = np.stack([target.blocks, blocked.blocks])
    batch_states = np.stack([target.block_states, blocked.block_states])
    assert lenient.matches_blocks(batch, batch_states).tolist() == [True, False]
    assert exact != lenient
