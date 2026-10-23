import numpy as np
import pytest

from mbag.rllib.alpha_zero.power.formulas import (
    aggregate_goal_ability,
    expected_human_goal_ability,
    human_goal_ability_monte_carlo_targets,
    robot_intrinsic_reward,
    robot_value,
    soft_power_policy,
)


def test_human_goal_ability_on_deterministic_three_state_chain():
    """
    s0 -> s1 -> s2, deterministic transitions, goal g achieved only at s2.
    Hand-computed (gamma_h = 0.9):
        V_h^e(s1, g) = 1.0            (s2 is reached and achieves g)
        V_h^e(s0, g) = 0.9 * 1.0 = 0.9
    """
    gamma_h = 0.9

    v_h_e_s1 = expected_human_goal_ability(
        next_state_probs=[1.0], achieved=[1], next_v_h_e=[0.0], gamma_h=gamma_h
    )
    assert v_h_e_s1 == pytest.approx(1.0)

    v_h_e_s0 = expected_human_goal_ability(
        next_state_probs=[1.0], achieved=[0], next_v_h_e=[v_h_e_s1], gamma_h=gamma_h
    )
    assert v_h_e_s0 == pytest.approx(0.9)


def test_human_goal_ability_averages_over_stochastic_transition():
    """
    From s0, a 50/50 transition either directly achieves g (successor value 1)
    or reaches a dead-end state with V_h^e=0 (successor value gamma_h * 0 = 0).
    Expected V_h^e(s0, g) = 0.5 * 1 + 0.5 * 0 = 0.5.
    """
    v_h_e = expected_human_goal_ability(
        next_state_probs=[0.5, 0.5],
        achieved=[1, 0],
        next_v_h_e=[0.0, 0.0],
        gamma_h=0.9,
    )
    assert v_h_e == pytest.approx(0.5)


def test_human_goal_ability_monte_carlo_targets_hand_computed():
    # achieved at t=2 only; gamma_h=0.9, bootstrap=0.3 (irrelevant since the
    # goal is achieved within the fragment).
    targets = human_goal_ability_monte_carlo_targets(
        achieved=[0, 0, 1, 0, 0], gamma_h=0.9, bootstrap_v_h_e=0.3
    )
    expected = np.array([0.81, 0.9, 1.0, 0.243, 0.27])
    assert np.allclose(targets, expected)


def test_human_goal_ability_monte_carlo_targets_never_achieved_uses_bootstrap():
    targets = human_goal_ability_monte_carlo_targets(
        achieved=[0, 0, 0], gamma_h=0.9, bootstrap_v_h_e=0.5
    )
    # V[2] = 0.9*0.5, V[1] = 0.9*V[2], V[0] = 0.9*V[1]
    assert targets[2] == pytest.approx(0.9 * 0.5)
    assert targets[1] == pytest.approx(0.9 * targets[2])
    assert targets[0] == pytest.approx(0.9 * targets[1])


def test_human_goal_ability_monte_carlo_targets_matches_step_formula():
    # The batch backward-scan should agree with repeated application of
    # expected_human_goal_ability on a deterministic single-successor chain.
    achieved = [0, 1, 0, 0]
    gamma_h = 0.8
    bootstrap = 0.6
    targets = human_goal_ability_monte_carlo_targets(achieved, gamma_h, bootstrap)

    v_next = bootstrap
    for t in range(len(achieved) - 1, -1, -1):
        v_next = expected_human_goal_ability(
            next_state_probs=[1.0], achieved=[achieved[t]], next_v_h_e=[v_next], gamma_h=gamma_h
        )
        assert v_next == pytest.approx(targets[t])


def test_aggregate_goal_ability_weighted_sum_of_powers():
    # X_h = 0.5 * 0.8^2 + 0.5 * 0.2^2 = 0.5*0.64 + 0.5*0.04 = 0.34
    x_h = aggregate_goal_ability(
        v_h_e_samples=[0.8, 0.2], weights=[0.5, 0.5], zeta=2.0
    )
    assert x_h == pytest.approx(0.34)


def test_aggregate_goal_ability_zeta_one_reduces_to_weighted_mean():
    x_h = aggregate_goal_ability(v_h_e_samples=[0.8, 0.2], weights=[0.5, 0.5], zeta=1.0)
    assert x_h == pytest.approx(0.5)


def test_robot_intrinsic_reward_single_human_matches_direct_formula():
    # |H|=1: y = X_h^(-xi), U_r = -(X_h^(-xi))^eta.
    x_h = 0.34
    xi, eta = 1.0, 1.1
    u_r = robot_intrinsic_reward([x_h], xi=xi, eta=eta)
    assert u_r == pytest.approx(-((x_h**-xi) ** eta))
    assert u_r < 0


def test_robot_intrinsic_reward_is_pigou_dalton_inequality_averse():
    # Two humans with the same average X_h: an unequal split should yield a
    # more negative (worse) U_r than an equal split, since low-X_h humans
    # dominate the -xi-power sum for xi > 0.
    equal_u_r = robot_intrinsic_reward([0.5, 0.5], xi=1.0, eta=1.0)
    unequal_u_r = robot_intrinsic_reward([0.1, 0.9], xi=1.0, eta=1.0)
    assert unequal_u_r < equal_u_r


def test_aggregate_goal_ability_decreasing_in_zeta_for_heterogeneous_values():
    # For V_h^e in (0, 1), v^zeta shrinks as zeta grows, so a risk/reliability-
    # averse (heterogeneous-value) X_h should strictly decrease as zeta
    # increases.
    v_h_e_samples = [0.9, 0.3]
    weights = [0.5, 0.5]
    x_h_values = [
        aggregate_goal_ability(v_h_e_samples, weights, zeta=zeta)
        for zeta in [1.0, 2.0, 4.0, 8.0]
    ]
    assert x_h_values == sorted(x_h_values, reverse=True)
    assert x_h_values[0] > x_h_values[-1]


def test_aggregate_goal_ability_invariant_to_zeta_when_fully_achieved():
    # 1.0^zeta = 1.0 for any zeta, so a possible goal that's already fully
    # achieved (V_h^e = 1) contributes its full weight to X_h regardless of
    # the risk-aversion exponent.
    x_h_values = [
        aggregate_goal_ability([1.0], [1.0], zeta=zeta) for zeta in [1.0, 2.0, 5.0, 10.0]
    ]
    assert all(x_h == pytest.approx(1.0) for x_h in x_h_values)


def test_robot_intrinsic_reward_decreasing_in_eta_when_power_is_below_reference():
    # When X_h < 1 (the typical case, since V_h^e in [0,1]) and xi > 0, y =
    # mean(X_h^-xi) > 1, so U_r = -y^eta should get more negative (worse) as
    # eta increases -- a steeper penalty for being below the power "reference
    # point" of X_h=1.
    x_h = [0.3]
    u_r_values = [
        robot_intrinsic_reward(x_h, xi=1.0, eta=eta) for eta in [1.0, 1.5, 2.0, 3.0]
    ]
    assert u_r_values == sorted(u_r_values, reverse=True)
    assert u_r_values[0] > u_r_values[-1]


def test_soft_power_policy_sharpens_toward_best_action_as_beta_r_increases():
    # As beta_r grows, probability mass should concentrate on the action with
    # the least-negative (highest-power) Q_r, and the distribution's entropy
    # should strictly decrease.
    q_r_values = [-1.0, -2.0, -4.0]
    best_action_index = 0  # -1.0 is the least negative.

    probs_by_beta_r = [
        soft_power_policy(q_r_values, beta_r=beta_r)
        for beta_r in [0.0, 1.0, 2.0, 5.0, 20.0]
    ]
    best_action_probs = [probs[best_action_index] for probs in probs_by_beta_r]
    assert best_action_probs == sorted(best_action_probs)

    entropies = [
        float(-np.sum(probs * np.log(probs))) for probs in probs_by_beta_r
    ]
    assert entropies == sorted(entropies, reverse=True)
    assert entropies[0] > entropies[-1]


def test_soft_power_policy_matches_manual_softmax():
    # pi_r(a) propto (-Q_r)^{-beta_r}: Q_r=[-1,-2,-4], beta_r=1
    # -> weights [1, 0.5, 0.25] -> normalized [4/7, 2/7, 1/7].
    probs = soft_power_policy([-1.0, -2.0, -4.0], beta_r=1.0)
    expected = np.array([4 / 7, 2 / 7, 1 / 7])
    assert np.allclose(probs, expected)
    assert probs.sum() == pytest.approx(1.0)


def test_soft_power_policy_rejects_nonnegative_q_values():
    with pytest.raises(ValueError):
        soft_power_policy([-1.0, 0.0], beta_r=1.0)


def test_soft_power_policy_beta_r_zero_is_uniform():
    probs = soft_power_policy([-1.0, -5.0, -100.0], beta_r=0.0)
    assert np.allclose(probs, [1 / 3, 1 / 3, 1 / 3])


def test_robot_value_matches_manual_expectation():
    u_r = -0.5
    q_r_values = [-1.0, -2.0]
    pi_r = [0.75, 0.25]
    v_r = robot_value(u_r, q_r_values, pi_r)
    assert v_r == pytest.approx(-0.5 + (0.75 * -1.0 + 0.25 * -2.0))
