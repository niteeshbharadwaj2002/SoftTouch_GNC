import numpy as np
import pytest
from src.dynamics import RocketDynamics
from config import parameters as prm


@pytest.fixture
def dynamics():
    return RocketDynamics(g=prm.G, isp=prm.ISP, g0=prm.G0)


def test_freefall_no_thrust(dynamics):
    """Zero thrust: rocket should fall under gravity alone, mass unchanged."""
    state = np.array([0.0, 1000.0, 0.0, 0.0, prm.WET_MASS])
    control = [0.0, 0.0]
    dt = 0.1
    n_steps = 50

    control_profile = np.tile(control, (n_steps, 1))
    states = dynamics.simulate(state, control_profile, dt, n_steps)

    # Altitude should decrease monotonically
    assert np.all(np.diff(states[:, 1]) < 0)
    # Downward velocity should increase in magnitude
    assert states[-1, 3] < states[0, 3]
    # Mass should not change (no thrust = no fuel burned)
    assert np.isclose(states[-1, 4], states[0, 4])


def test_constant_thrust_straight_down_decelerates(dynamics):
    """Thrust > weight: rocket should decelerate its downward velocity."""
    state = np.array([0.0, 1500.0, 0.0, -80.0, prm.WET_MASS])
    weight = prm.WET_MASS * prm.G
    control = [0.0, weight * 1.5]  # thrust > weight, should slow descent
    dt = 0.1
    n_steps = 100

    control_profile = np.tile(control, (n_steps, 1))
    states = dynamics.simulate(state, control_profile, dt, n_steps)

    # vy should become less negative (decelerating) over time, before landing
    assert states[10, 3] > states[0, 3]


def test_mass_depletes_with_thrust(dynamics):
    """Applying thrust should consume fuel — mass strictly decreases."""
    state = np.array([0.0, 1000.0, 0.0, -50.0, prm.WET_MASS])
    control = [0.0, 20000.0]
    dt = 0.1
    n_steps = 30

    control_profile = np.tile(control, (n_steps, 1))
    states = dynamics.simulate(state, control_profile, dt, n_steps)

    assert np.all(np.diff(states[:, 4]) < 0)
    assert states[-1, 4] < prm.WET_MASS


def test_mass_never_negative(dynamics):
    """Even with sustained high thrust, mass should clamp at zero, not go negative."""
    state = np.array([0.0, 500.0, 0.0, -20.0, 50.0])  # low starting mass
    control = [0.0, prm.MAX_THRUST]
    dt = 0.1
    n_steps = 200

    control_profile = np.tile(control, (n_steps, 1))
    states = dynamics.simulate(state, control_profile, dt, n_steps)

    assert np.all(states[:, 4] >= 0.0)


def test_simulation_stops_at_ground(dynamics):
    """Simulation should terminate once altitude <= 0, not continue underground."""
    state = np.array([0.0, 50.0, 0.0, -30.0, prm.WET_MASS])
    control = [0.0, 0.0]  # freefall, will hit ground fast
    dt = 0.1
    n_steps = 500  # way more than needed

    control_profile = np.tile(control, (n_steps, 1))
    states = dynamics.simulate(state, control_profile, dt, n_steps)

    assert states[-1, 1] <= 0.0
    assert len(states) < n_steps + 1  # terminated early


def test_zero_mass_raises_error(dynamics):
    """state_derivative should raise if mass is zero or negative — undefined dynamics."""
    state = np.array([0.0, 1000.0, 0.0, -50.0, 0.0])
    control = [0.0, 10000.0]

    with pytest.raises(ValueError):
        dynamics.state_derivative(state, control)


def test_horizontal_thrust_moves_x(dynamics):
    """Thrust in x should produce downrange acceleration/movement."""
    state = np.array([0.0, 1500.0, 0.0, 0.0, prm.WET_MASS])
    control = [5000.0, prm.WET_MASS * prm.G]  # cancel gravity in y, push in x
    dt = 0.1
    n_steps = 20

    control_profile = np.tile(control, (n_steps, 1))
    states = dynamics.simulate(state, control_profile, dt, n_steps)

    assert states[-1, 0] > states[0, 0]  # x increased
    assert states[-1, 2] > 0  # vx positive