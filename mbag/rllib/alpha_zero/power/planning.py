"""
Power-based (U_r) reward computation for MCTS planning (see
mbag.rllib.alpha_zero.power), the drop-in replacement for the CIRL
belief-weighted reward computed by
mbag.rllib.alpha_zero.planning.MbagEnvModel.get_reward_with_other_agent_actions.

Unlike that function, this needs no goal-belief or own/joint-reward-blending
machinery at all: U_r(s') is a pure function of the successor state's
aggregate goal ability X_h(s'), already predicted by the model's
aggregate_goal_ability_head (see mbag.rllib.torch_models.MbagTorchModel).
"""

from mbag.rllib.alpha_zero.power.formulas import robot_intrinsic_reward


def get_power_reward(aggregate_goal_ability: float, xi: float, eta: float) -> float:
    """
    U_r(s') for a successor state, given the model's prediction of that
    state's aggregate goal ability X_h(s').

    MBAG's standard setup has a single human (|H|=1), so this is a thin
    wrapper around robot_intrinsic_reward: the Pigou-Dalton sum over humans
    reduces to the one human's aggregate goal ability. The general sum-over-
    humans form lives in robot_intrinsic_reward itself so this still behaves
    correctly if MBAG is ever run with more than one human.
    """
    return robot_intrinsic_reward([aggregate_goal_ability], xi=xi, eta=eta)
