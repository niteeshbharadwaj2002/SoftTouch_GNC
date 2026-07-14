import numpy as np
import pytest

from src.controls import LQRController
from src.dynamics import RocketDynamics
from src.optimiser import GuidanceOptimizer
from config import parameters as prm


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def controller():
    return LQRController()


@pytest.fixture
def dynamics():
    return RocketDynamics()


@pytest.fixture(scope="module")
def guidance_plan():
    N, dt = 120, 0.5
    optimizer = GuidanceOptimizer(N=N, dt=dt)
    initial_state = [500.0, 1000.0, -20.0, -30.0, prm.WET_MASS]
    target_state = [0.0, 0.0, 0.0, 0.0]

    result = optimizer.solve(initial_state, target_state)
    assert result["status"] in ("optimal", "optimal_inaccurate")

    planned_states = optimizer.to_state_trajectory(result)
    planned_controls = optimizer.to_control_profile(result)
    return planned_states, planned_controls, dt


# ---------------------------------------------------------------------
# 1. LQR gain computation — unit tests
# ---------------------------------------------------------------------
def test_gain_shape(controller):
    K = controller.compute_gain(m=prm.WET_MASS)
    assert K.shape == (2, 4)


def test_gain_finite_and_nonzero(controller):
    K = controller.compute_gain(m=prm.WET_MASS)
    assert np.all(np.isfinite(K))
    assert np.any(K != 0)


def test_gain_differs_with_mass(controller):
    """Since B scales with 1/m, K at wet mass and K near dry mass should
    differ meaningfully — this is the whole point of gain scheduling."""
    K_heavy = controller.compute_gain(m=prm.WET_MASS)
    K_light = controller.compute_gain(m=prm.DRY_MASS + 10.0)

    assert not np.allclose(K_heavy, K_light)
    # Lower mass -> larger 1/m -> same B-normalized effort needs smaller
    # raw gain magnitude (K roughly scales with m, since B ~ 1/m and the
    # Riccati solution compensates) — just check it's a real, finite change
    assert np.all(np.isfinite(K_light))


def test_gain_stable_near_dry_mass(controller):
    """Mass approaching (but not equal to) dry mass shouldn't blow up
    the Riccati solve — sanity check for the low-mass edge case."""
    K = controller.compute_gain(m=prm.DRY_MASS + 1.0)
    assert np.all(np.isfinite(K))


def test_gain_raises_or_handles_zero_mass(controller):
    """Zero/negative mass is physically invalid — B becomes singular
    (division by zero). Confirm this fails loudly rather than silently
    returning garbage."""
    with pytest.raises((ZeroDivisionError, np.linalg.LinAlgError, FloatingPointError, ValueError)):
        controller.compute_gain(m=0.0)


# ---------------------------------------------------------------------
# 2. Thrust clipping — unit tests
# ---------------------------------------------------------------------
def test_clip_within_bounds_unchanged(controller):
    T = np.array([1000.0, 15000.0])
    mag = np.linalg.norm(T)
    assert prm.MIN_THRUST < mag < prm.MAX_THRUST

    clipped = controller._clip_thrust(T)
    assert np.allclose(clipped, T)


def test_clip_above_max_scaled_down(controller):
    T = np.array([prm.MAX_THRUST, prm.MAX_THRUST])  # magnitude > MAX_THRUST
    clipped = controller._clip_thrust(T)

    assert np.isclose(np.linalg.norm(clipped), prm.MAX_THRUST, atol=1e-6)
    # Direction preserved
    assert np.isclose(np.arctan2(clipped[1], clipped[0]), np.arctan2(T[1], T[0]))


def test_clip_below_min_scaled_up(controller):
    T = np.array([100.0, 100.0])  # magnitude well below MIN_THRUST
    clipped = controller._clip_thrust(T)

    assert np.isclose(np.linalg.norm(clipped), prm.MIN_THRUST, atol=1e-6)
    assert np.isclose(np.arctan2(clipped[1], clipped[0]), np.arctan2(T[1], T[0]))


def test_clip_zero_vector_defaults_to_min_thrust_up(controller):
    T = np.array([0.0, 0.0])
    clipped = controller._clip_thrust(T)

    assert np.isclose(np.linalg.norm(clipped), prm.MIN_THRUST, atol=1e-6)
    assert clipped[0] == 0.0
    assert clipped[1] > 0  # defaults straight up


def test_clip_negative_direction_preserved(controller):
    """Thrust commanded in an unusual direction (e.g., negative x, negative
    y) should still only be rescaled, not redirected."""
    T = np.array([-prm.MAX_THRUST, -prm.MAX_THRUST])
    clipped = controller._clip_thrust(T)

    assert np.isclose(np.linalg.norm(clipped), prm.MAX_THRUST, atol=1e-6)
    assert clipped[0] < 0 and clipped[1] < 0


# ---------------------------------------------------------------------
# 3. compute_control — correction direction sanity
# ---------------------------------------------------------------------
def test_zero_error_no_correction(controller):
    """If actual == planned exactly, correction should be ~zero — output
    should equal planned_control (post-clip)."""
    state = [100.0, 500.0, -10.0, -20.0, prm.WET_MASS]
    planned_control = [0.0, prm.WET_MASS * prm.G]  # hover-ish thrust

    u = controller.compute_control(
        planned_state=state, actual_state=state, planned_control=planned_control
    )
    assert np.allclose(u, planned_control, atol=1.0)


def test_positive_position_error_increases_correction(controller):
    """If actual altitude is BELOW planned (rocket fell behind schedule),
    controller should command MORE upward thrust than the nominal plan."""
    planned_state = [0.0, 500.0, 0.0, -20.0, prm.WET_MASS]
    actual_state = [0.0, 450.0, 0.0, -20.0, prm.WET_MASS]  # 50m low
    planned_control = [0.0, prm.WET_MASS * prm.G]

    u = controller.compute_control(planned_state, actual_state, planned_control)
    assert u[1] > planned_control[1]


def test_output_respects_thrust_bounds(controller):
    """Even with a huge injected error, output must stay within actuator
    limits (clipping applied inside compute_control)."""
    planned_state = [0.0, 1000.0, 0.0, 0.0, prm.WET_MASS]
    actual_state = [0.0, -5000.0, 0.0, 500.0, prm.WET_MASS]  # absurd error
    planned_control = [0.0, prm.WET_MASS * prm.G]

    u = controller.compute_control(planned_state, actual_state, planned_control)
    mag = np.linalg.norm(u)
    assert prm.MIN_THRUST - 1e-6 <= mag <= prm.MAX_THRUST + 1e-6


# ---------------------------------------------------------------------
# 4. Closed-loop simulation — no disturbance (baseline tracking)
# ---------------------------------------------------------------------
def test_closed_loop_no_disturbance_tracks_plan(controller, dynamics, guidance_plan):
    planned_states, planned_controls, dt = guidance_plan

    actual_states, actual_controls = dynamics.simulate_closed_loop(
        planned_states, planned_controls, controller, dt, wind_force=None
    )

    # With no disturbance and correct dynamics, tracking error should stay small
    n = min(len(planned_states), len(actual_states))
    error = np.hypot(
        planned_states[:n, 0] - actual_states[:n, 0],
        planned_states[:n, 1] - actual_states[:n, 1],
    )
    assert np.max(error) < 5.0  # meters — tight tracking, no disturbance


def test_closed_loop_lands_near_target_no_disturbance(controller, dynamics, guidance_plan):
    planned_states, planned_controls, dt = guidance_plan

    actual_states, _ = dynamics.simulate_closed_loop(
        planned_states, planned_controls, controller, dt, wind_force=None
    )

    final_pos_error = np.hypot(actual_states[-1, 0] - prm.TARGET_X,
                                actual_states[-1, 1] - prm.TARGET_Y)
    assert final_pos_error < 10.0


def test_closed_loop_control_within_bounds(controller, dynamics, guidance_plan):
    planned_states, planned_controls, dt = guidance_plan

    _, actual_controls = dynamics.simulate_closed_loop(
        planned_states, planned_controls, controller, dt, wind_force=None
    )

    thrust_mag = np.hypot(actual_controls[:, 0], actual_controls[:, 1])
    assert np.all(thrust_mag <= prm.MAX_THRUST + 1e-3)
    assert np.all(thrust_mag >= prm.MIN_THRUST - 1e-3)


# ---------------------------------------------------------------------
# 5. Closed-loop simulation — wind disturbance rejection
# ---------------------------------------------------------------------
def test_constant_wind_still_lands_reasonably(controller, dynamics, guidance_plan):
    """Constant horizontal wind throughout flight — controller should
    still land within a reasonable radius, not diverge."""
    planned_states, planned_controls, dt = guidance_plan

    def constant_wind(step, state):
        return np.array([2000.0, 0.0])  # steady horizontal push, N

    actual_states, _ = dynamics.simulate_closed_loop(
        planned_states, planned_controls, controller, dt, wind_force=constant_wind
    )

    final_pos_error = np.hypot(actual_states[-1, 0] - prm.TARGET_X,
                                actual_states[-1, 1] - prm.TARGET_Y)
    assert final_pos_error < 50.0  # looser than no-disturbance case, but bounded


def test_wind_gust_causes_temporary_deviation_then_recovers(controller, dynamics, guidance_plan):
    """Gust active only in a mid-flight window — expect elevated tracking
    error during the gust, and reduced error by the end as LQR corrects."""
    planned_states, planned_controls, dt = guidance_plan
    n_steps = len(planned_controls)
    gust_start, gust_end = n_steps // 3, n_steps // 2

    def gust(step, state):
        if gust_start <= step < gust_end:
            return np.array([5000.0, 0.0])
        return np.array([0.0, 0.0])

    actual_states, _ = dynamics.simulate_closed_loop(
        planned_states, planned_controls, controller, dt, wind_force=gust
    )

    n = min(len(planned_states), len(actual_states))
    error = np.hypot(
        planned_states[:n, 0] - actual_states[:n, 0],
        planned_states[:n, 1] - actual_states[:n, 1],
    )

    error_during_gust = np.max(error[gust_start:min(gust_end, n)])
    error_at_end = error[-1]

    assert error_during_gust > 0  # gust actually perturbed the trajectory
    # Controller should have brought error back down significantly by landing
    assert error_at_end < error_during_gust


def test_wind_gust_thrust_stays_within_bounds(controller, dynamics, guidance_plan):

    planned_states, planned_controls, dt = guidance_plan
    n_steps = len(planned_controls)
    gust_start, gust_end = n_steps // 3, n_steps // 2

    def gust(step, state):
        if gust_start <= step < gust_end:
            return np.array([5000.0, 0.0])
        return np.array([0.0, 0.0])

    _, actual_controls = dynamics.simulate_closed_loop(
        planned_states, planned_controls, controller, dt, wind_force=gust
    )

    thrust_mag = np.hypot(actual_controls[:, 0], actual_controls[:, 1])
    assert np.all(thrust_mag <= prm.MAX_THRUST + 1e-3)
    assert np.all(thrust_mag >= prm.MIN_THRUST - 1e-3)


def test_strong_gust_does_not_crash_simulation(controller, dynamics, guidance_plan):

    planned_states, planned_controls, dt = guidance_plan
    n_steps = len(planned_controls)

    def strong_gust(step, state):
        if n_steps // 4 <= step < n_steps // 2:
            return np.array([15000.0, -3000.0])
        return np.array([0.0, 0.0])

    actual_states, actual_controls = dynamics.simulate_closed_loop(
        planned_states, planned_controls, controller, dt, wind_force=strong_gust
    )

    assert len(actual_states) > 1
    assert np.all(np.isfinite(actual_states))


# # ---------------------------------------------------------------------
# # 6. Ground termination in closed loop
# # ---------------------------------------------------------------------
# def test_closed_loop_terminates_on_ground_contact(controller, dynamics):
#     """If the plan and dynamics bring altitude to <= 0 early, the loop
#     should stop there, mirroring open-loop simulate() behavior."""
#     N, dt = 20, 0.5
#     total_time = N * dt  # 10s

#     # Self-consistent synthetic trajectory: constant descent velocity
#     # matching the position profile (y and vy must agree, or the plan
#     # is physically contradictory and the controller can't ever match it).
#     y_start, y_end = 50.0, 0.0
#     vy_constant = (y_end - y_start) / total_time  # -5.0 m/s

#     planned_states = np.zeros((N + 1, 5))
#     planned_states[:, 1] = np.linspace(y_start, y_end, N + 1)  # y
#     planned_states[:, 3] = vy_constant                          # vy, consistent with y
#     planned_states[:, 4] = prm.WET_MASS

#     # Constant velocity -> zero net acceleration -> thrust exactly cancels
#     # gravity (hover thrust), consistent with the velocity profile above
#     planned_controls = np.tile([0.0, prm.WET_MASS * prm.G], (N, 1))

#     actual_states, actual_controls = dynamics.simulate_closed_loop(
#         planned_states, planned_controls, controller, dt, wind_force=None
#     )

#     assert actual_states[-1, 1] <= 0.0 + 1e-6
#     assert len(actual_states) <= N + 1
#     assert len(actual_controls) == len(actual_states) - 1


# ---------------------------------------------------------------------
# 7. Output shape / consistency
# ---------------------------------------------------------------------
def test_closed_loop_output_shapes_consistent(controller, dynamics, guidance_plan):
    planned_states, planned_controls, dt = guidance_plan

    actual_states, actual_controls = dynamics.simulate_closed_loop(
        planned_states, planned_controls, controller, dt, wind_force=None
    )

    assert actual_states.shape[1] == 5
    assert actual_controls.shape[1] == 2
    assert len(actual_controls) == len(actual_states) - 1