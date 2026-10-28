import numpy as np

from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.goals import FixedPoolGoalGenerator
from mbag.environment.goals.embedding import embed_goal, get_goal_size
from mbag.environment.goals.possible_goals import MinecraftStructureGoal
from mbag.rllib.alpha_zero.power.human_policy_prior import (
    HeuristicMbagHumanPolicyPrior,
)
from mbag.rllib.alpha_zero.power.simulated_targets import SimulatedGoalAbility

SIZE = (6, 6, 6)
GAMMA = 0.99


def make_simulator(num_goals: int = 3):
    pool_config = {
        "goal_generator": "random",
        "goal_generator_config": {},
        "num_goals": num_goals,
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
    pool = FixedPoolGoalGenerator(pool_config).get_pool(get_goal_size(SIZE, True))
    goals = [
        MinecraftStructureGoal(embed_goal(g, SIZE, True), lenient=True) for g in pool
    ]
    prior = HeuristicMbagHumanPolicyPrior(
        env_config, beta_h=10.0, human_cannot_break_bedrock=True, lenient=True
    )
    simulator = SimulatedGoalAbility(
        env_config, goals, prior, gamma_h=GAMMA, zeta=2.0, horizon=100
    )
    return simulator, goals


def initial_world() -> MinecraftBlocks:
    world = MinecraftBlocks(SIZE)
    world.blocks[:, 0, :] = MinecraftBlocks.BEDROCK
    world.blocks[:, 1, :] = MinecraftBlocks.NAME2ID["dirt"]
    return world


def test_goal_ability_done_open_and_blocked():
    simulator, goals = make_simulator()
    goal = goals[0]
    done = goal.target.copy()
    assert simulator.goal_ability(done, goal) == 1.0

    needed = np.argwhere(
        (goal.target.blocks != MinecraftBlocks.AIR)
        & (goal.target.blocks != initial_world().blocks)
    )
    cell = tuple(needed[0])
    open_world = done.copy()
    open_world.blocks[cell] = MinecraftBlocks.AIR
    # One block from done: the prior places it almost immediately.
    assert simulator.goal_ability(open_world, goal) > 0.9

    blocked = done.copy()
    blocked.blocks[cell] = MinecraftBlocks.BEDROCK
    assert simulator.goal_ability(blocked, goal) == 0.0


def test_aggregate_goal_ability_drops_when_bedrock_blocks_goals():
    simulator, goals = make_simulator()
    start = initial_world()
    x_open = simulator.aggregate_goal_ability(start)
    assert 0.0 < x_open <= len(goals)

    # Put bedrock on a needed cell of every goal: nothing is reachable.
    blocked = start.copy()
    for goal in goals:
        needed = np.argwhere(
            (goal.target.blocks != MinecraftBlocks.AIR)
            & (goal.target.blocks != start.blocks)
        )
        blocked.blocks[tuple(needed[0])] = MinecraftBlocks.BEDROCK
    assert simulator.aggregate_goal_ability(blocked) == 0.0


def test_simulated_human_finishes_open_goals():
    # With the ability-respecting prior (valid actions only, bottom-up), the
    # simulated human should finish nearly every goal in an open world; a prior that
    # wastes steps or buries cells it still needs would make X_h targets all ~0.
    simulator, goals = make_simulator(num_goals=10)
    values = [simulator.goal_ability(initial_world(), goal) for goal in goals]
    assert sum(v > 0 for v in values) >= 8
