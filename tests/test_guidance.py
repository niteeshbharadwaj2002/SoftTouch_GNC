"""
tests/test_guidance.py
Validates GuidanceOptimizer under normal and edge-case conditions.
"""

import numpy as np
import pytest

from src import optimiser as op
from config import parameters as prm

N_TEST = 120
DT_TEST = 0.5   # 30s fixed final time


@pytest.fixture
def optimizer():
    return op.GuidanceOptimizer(N=N_TEST, dt=DT_TEST)


def assert_optimal(result):
    assert result["status"] == "optimal", f"Solver did not converge: {result['status']}"


# ---------------------------------------------------------------------
# 1. Basic feasible descent
# ---------------------------------------------------------------------
def test_basic_descent_converges(optimizer):
    initial_state = [500.0, 1000.0, -20.0, -30.0, prm.WET_MASS]
    target_state = [0.0, 0.0, 0.0, 0.0]

    result = optimizer.solve(initial_state, target_state)
    assert_optimal(result)


def test_terminal_conditions_satisfied(optimizer):
    initial_state = [500.0, 1000.0, -20.0, -30.0, prm.WET_MASS]
    target_state = [0.0, 0.0, 0.0, 0.0]

    result = optimizer.solve(initial_state, target_state)
    assert_optimal(result)

    tol = 1e-2
    assert abs(result["x"][-1] - target_state[0]) < tol
    assert abs(result["y"][-1] - target_state[1]) < tol
    assert abs(result["vx"][-1] - target_state[2]) < tol
    assert abs(result["vy"][-1] - target_state[3]) < tol


def test_initial_conditions_preserved(optimizer):
    initial_state = [500.0, 1000.0, -20.0, -30.0, prm.WET_MASS]
    target_state = [0.0, 0.0, 0.0, 0.0]

    result = optimizer.solve(initial_state, target_state)
    assert_optimal(result)

    assert np.isclose(result["x"][0], initial_state[0])
    assert np.isclose(result["y"][0], initial_state[1])
    assert np.isclose(result["vx"][0], initial_state[2])
    assert np.isclose(result["vy"][0], initial_state[3])
    assert np.isclose(result["m"][0], initial_state[4])


# ---------------------------------------------------------------------
# 2. Constraint respect
# ---------------------------------------------------------------------
def test_thrust_within_bounds(optimizer):
    initial_state = [500.0, 1000.0, -20.0, -30.0, prm.WET_MASS]
    target_state = [0.0, 0.0, 0.0, 0.0]

    result = optimizer.solve(initial_state, target_state)
    assert_optimal(result)

    thrust_mag = np.hypot(result["Tx"], result["Ty"])
    assert np.all(thrust_mag <= prm.MAX_THRUST + 1e-1)
    assert np.all(thrust_mag >= prm.MIN_THRUST - 1e-1)


def test_mass_never_below_dry_mass(optimizer):
    initial_state = [500.0, 1000.0, -20.0, -30.0, prm.WET_MASS]
    target_state = [0.0, 0.0, 0.0, 0.0]

    result = optimizer.solve(initial_state, target_state)
    assert_optimal(result)

    assert np.all(result["m"] >= prm.DRY_MASS - 1e-2)


def test_mass_monotonically_decreasing(optimizer):
    initial_state = [500.0, 1000.0, -20.0, -30.0, prm.WET_MASS]
    target_state = [0.0, 0.0, 0.0, 0.0]

    result = optimizer.solve(initial_state, target_state)
    assert_optimal(result)

    assert np.all(np.diff(result["m"]) <= 1e-6)


def test_altitude_never_negative(optimizer):
    initial_state = [500.0, 1000.0, -20.0, -30.0, prm.WET_MASS]
    target_state = [0.0, 0.0, 0.0, 0.0]

    result = optimizer.solve(initial_state, target_state)
    assert_optimal(result)

    assert np.all(result["y"] >= -1e-2)


# ---------------------------------------------------------------------
# 3. Edge cases
# ---------------------------------------------------------------------
def test_already_at_target(optimizer):
    """
    Starting exactly at target with y=0 forces the rocket to hover for the
    full fixed duration (y >= 0 constraint prevents 'doing nothing'),
    so fuel used should be close to hover-fuel (thrust ~= weight), not zero
    and not bounded by MIN_THRUST.
    """
    initial_state = [0.0, 0.0, 0.0, 0.0, prm.WET_MASS]
    target_state = [0.0, 0.0, 0.0, 0.0]

    result = optimizer.solve(initial_state, target_state)
    assert_optimal(result)

    hover_thrust = prm.WET_MASS * prm.G  # thrust needed to exactly counter gravity
    hover_burn_rate = hover_thrust / (prm.ISP * prm.G0)
    expected_fuel = hover_burn_rate * N_TEST * DT_TEST

    fuel_used = result["m"][0] - result["m"][-1]
    # Allow generous margin — successive convexification + Euler discretization
    # won't match the continuous hover estimate exactly
    assert fuel_used < expected_fuel * 1.5
    assert fuel_used > expected_fuel * 0.5


def test_straight_vertical_descent(optimizer):
    initial_state = [0.0, 800.0, 0.0, -10.0, prm.WET_MASS]
    target_state = [0.0, 0.0, 0.0, 0.0]

    result = optimizer.solve(initial_state, target_state)
    assert_optimal(result)

    assert np.allclose(result["x"], 0.0, atol=1e-2)


def test_negative_downrange_start(optimizer):
    initial_state = [-500.0, 1000.0, 15.0, -30.0, prm.WET_MASS]
    target_state = [0.0, 0.0, 0.0, 0.0]

    result = optimizer.solve(initial_state, target_state)
    assert_optimal(result)
    assert abs(result["x"][-1] - target_state[0]) < 1e-2

# ---------------------------------------------------------------------
# 4. Infeasibility handling
# ---------------------------------------------------------------------
def test_insufficient_fuel_reports_infeasible(optimizer):
    initial_state = [500.0, 1000.0, -20.0, -30.0, prm.DRY_MASS + 5.0]
    target_state = [0.0, 0.0, 0.0, 0.0]

    result = optimizer.solve(initial_state, target_state)
    assert result["status"] != "optimal"


def test_unreachable_in_fixed_time_reports_infeasible(optimizer):
    initial_state = [50000.0, 1000.0, 0.0, -10.0, prm.WET_MASS]
    target_state = [0.0, 0.0, 0.0, 0.0]

    result = optimizer.solve(initial_state, target_state)
    assert result["status"] != "optimal"


# ---------------------------------------------------------------------
# 5. Output conversion helpers
# ---------------------------------------------------------------------
def test_control_profile_shape(optimizer):
    initial_state = [500.0, 1000.0, -20.0, -30.0, prm.WET_MASS]
    target_state = [0.0, 0.0, 0.0, 0.0]

    result = optimizer.solve(initial_state, target_state)
    assert_optimal(result)

    control_profile = optimizer.to_control_profile(result)
    assert control_profile.shape == (N_TEST, 2)


def test_state_trajectory_shape(optimizer):
    initial_state = [500.0, 1000.0, -20.0, -30.0, prm.WET_MASS]
    target_state = [0.0, 0.0, 0.0, 0.0]

    result = optimizer.solve(initial_state, target_state)
    assert_optimal(result)

    state_trajectory = optimizer.to_state_trajectory(result)
    assert state_trajectory.shape == (N_TEST + 1, 5)