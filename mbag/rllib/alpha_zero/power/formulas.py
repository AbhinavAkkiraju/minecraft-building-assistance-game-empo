"""
Core formulas of the human-power objective, independent of any RL algorithm
or neural network architecture.

These mirror ``empo/src/empo/backward_induction/phase2.py`` (equations 6-9 of
Heitzig & Potham, arXiv:2508.00159) exactly, verified against that file's
actual implementation rather than paraphrased from it. Kept as small, pure
functions -- given transition probabilities and successor values/goal
achievements, they compute the corresponding TD target/aggregate -- so they
can be validated directly against hand-computed values on a tiny state graph
(see ``tests/test_power_formulas.py``) before any neural network or MCTS
integration is built on top of them.

Notation matches the paper and ``empo`` throughout: ``V_h^e`` (human's
expected goal-achievement value under the actual policies), ``X_h``
(aggregate goal ability), ``U_r`` (robot's intrinsic reward), ``Q_r``/``V_r``
(robot action-value/value), ``pi_r`` (robot's soft power-law policy).
"""

import numpy as np
import numpy.typing as npt


def human_goal_ability_successor_values(
    achieved: npt.ArrayLike, next_v_h_e: npt.ArrayLike, gamma_h: float
) -> np.ndarray:
    """
    Per-successor-state contribution to ``V_h^e(s, g_h)`` (eq. 6).

    For each successor state ``s'``, the value is exactly 1 if the goal is
    achieved in ``s'`` -- achievement is absorbing, not additive: this is a
    discounted probability of ever reaching the goal, not a reward-plus-
    future-value sum -- and otherwise ``gamma_h * V_h^e(s', g_h)``.

    Args:
        achieved: 0/1 array, whether ``g_h`` is achieved in each successor state.
        next_v_h_e: ``V_h^e(s', g_h)`` for each successor state.
        gamma_h: Human's discount factor.

    Returns:
        Array of per-successor-state values, to be averaged over the
        transition distribution (see ``expected_human_goal_ability``).
    """
    achieved_arr = np.asarray(achieved, dtype=float)
    next_v_h_e_arr = np.asarray(next_v_h_e, dtype=float)
    return np.where(achieved_arr > 0, 1.0, gamma_h * next_v_h_e_arr)


def expected_human_goal_ability(
    next_state_probs: npt.ArrayLike,
    achieved: npt.ArrayLike,
    next_v_h_e: npt.ArrayLike,
    gamma_h: float,
) -> float:
    """
    ``V_h^e(s, g_h)`` under one action profile: the transition-probability-
    weighted expectation of ``human_goal_ability_successor_values``.
    """
    successor_values = human_goal_ability_successor_values(
        achieved, next_v_h_e, gamma_h
    )
    return float(np.dot(np.asarray(next_state_probs, dtype=float), successor_values))


def human_goal_ability_monte_carlo_targets(
    achieved: npt.ArrayLike, gamma_h: float, bootstrap_v_h_e: float
) -> np.ndarray:
    """
    Monte Carlo regression targets for ``V_h^e(s_t, g_h)`` across a rollout
    fragment of length T, given (for each timestep t) whether ``g_h`` is
    achieved in the *successor* state reached after that timestep's action,
    and a bootstrap value ``V_h^e(s_T, g_h)`` for the state just past the end
    of the fragment (typically the model's own current prediction there --
    mirroring how RLlib's ``Postprocessing.VALUE_TARGETS`` bootstraps ``V_r``
    with the model's value prediction at the truncation point).

    Computed via the same backward recursion as
    ``human_goal_ability_successor_values``, applied once per timestep from
    the end of the fragment backward:
        ``V_h^e(s_T, g_h) = bootstrap_v_h_e``
        ``V_h^e(s_t, g_h) = 1.0``                       if ``achieved[t]``
                          ``= gamma_h * V_h^e(s_{t+1}, g_h)``  otherwise

    Args:
        achieved: 0/1 array of length T; ``achieved[t]`` is whether ``g_h``
            is achieved in the state reached after timestep t's action.
        gamma_h: Human's discount factor.
        bootstrap_v_h_e: ``V_h^e`` estimate for the state just past the last
            timestep in the fragment.

    Returns:
        Array of length T of regression targets for ``V_h^e(s_t, g_h)``.
    """
    achieved_arr = np.asarray(achieved, dtype=float)
    targets = np.empty(len(achieved_arr), dtype=float)
    next_value = float(bootstrap_v_h_e)
    for t in range(len(achieved_arr) - 1, -1, -1):
        next_value = 1.0 if achieved_arr[t] > 0 else gamma_h * next_value
        targets[t] = next_value
    return targets


def aggregate_goal_ability(
    v_h_e_samples: npt.ArrayLike, weights: npt.ArrayLike, zeta: float
) -> float:
    """
    ``X_h(s) = sum_g weight_g * V_h^e(s, g)^zeta`` (eq. 7): a human's
    risk/reliability-averse (``zeta > 1``) aggregate ability to achieve a
    variety of possible goals, over a weighted sample/enumeration of goals
    (see ``mbag.environment.goals.possible_goals``).
    """
    v_h_e_arr = np.asarray(v_h_e_samples, dtype=float)
    weights_arr = np.asarray(weights, dtype=float)
    return float(np.sum(weights_arr * v_h_e_arr**zeta))


def robot_intrinsic_reward(
    x_h_per_human: npt.ArrayLike, xi: float, eta: float
) -> float:
    """
    ``U_r(s) = -y(s)^eta``, ``y(s) = (1/|H|) * sum_h X_h(s)^(-xi)`` (eq. 8):
    the robot's intrinsic reward, an inequality- (``xi``) and intertemporal-
    inequality-averse (``eta``) aggregate over all humans' ``X_h``, always
    negative by construction. With a single human (``|H|=1``, MBAG's standard
    setup) this reduces to ``U_r = -(X_h^(-xi))^eta``.
    """
    x_h_arr = np.asarray(x_h_per_human, dtype=float)
    y = float(np.mean(x_h_arr ** (-xi)))
    return -(y**eta)


def soft_power_policy(q_r_values: npt.ArrayLike, beta_r: float) -> np.ndarray:
    """
    ``pi_r(a) propto (-Q_r(a))^{-beta_r}`` (eq. 9's policy, phase2.py:220-229):
    the robot's *soft* power-law policy, computed in log space for numerical
    stability. Not a hard argmax -- ``beta_r`` controls how sharply the robot
    prefers higher-power actions.

    Args:
        q_r_values: Q_r(s, a_r) for each candidate robot action. Must be
            strictly negative (an invariant of U_r/V_r/Q_r's construction).
        beta_r: Soft-maximization inverse temperature.

    Returns:
        Probability distribution over the candidate actions.
    """
    q_r_arr = np.asarray(q_r_values, dtype=float)
    if np.any(q_r_arr >= 0):
        raise ValueError(
            "Q_r values must be strictly negative for the soft power-law policy "
            f"pi_r(a) propto (-Q_r(a))^-beta_r; got {q_r_arr}."
        )
    log_neg_q_r = np.log(-q_r_arr)
    log_powers = -beta_r * log_neg_q_r
    log_powers -= np.max(log_powers)
    unnormalized = np.exp(log_powers)
    return unnormalized / unnormalized.sum()


def robot_value(u_r: float, q_r_values: npt.ArrayLike, pi_r: npt.ArrayLike) -> float:
    """
    ``V_r(s) = U_r(s) + E_{a_r ~ pi_r}[Q_r(s, a_r)]`` (matches phase2.py's
    ``vr = ur + dot(ps, Qr_values)``, undiscounted-duration case).
    """
    q_r_arr = np.asarray(q_r_values, dtype=float)
    pi_r_arr = np.asarray(pi_r, dtype=float)
    return float(u_r + np.dot(pi_r_arr, q_r_arr))
