import numpy as np
import pytest

from src.kalman import KalmanFilter, sensor_model
from config import parameters as prm


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def kf():
    initial_state = [500.0, 1000.0, -20.0, -30.0]
    return KalmanFilter(initial_state=initial_state)


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)  # reproducible noise


# ---------------------------------------------------------------------
# 1. Initialization
# ---------------------------------------------------------------------
def test_initial_state_matches_input(kf):
    assert np.allclose(kf.x_est, [500.0, 1000.0, -20.0, -30.0])


def test_initial_covariance_positive_definite(kf):
    eigvals = np.linalg.eigvalsh(kf.P)
    assert np.all(eigvals > 0)


# ---------------------------------------------------------------------
# 2. Predict step — pure kinematics + gravity, no correction
# ---------------------------------------------------------------------
def test_predict_moves_position_by_velocity(kf):
    dt = 0.1
    x0, y0 = kf.x_est[0], kf.x_est[1]
    vx0, vy0 = kf.x_est[2], kf.x_est[3]

    kf.predict(accel_measurement=[0.0, 0.0], dt=dt)

    # x should move purely by vx*dt (no accel commanded)
    assert np.isclose(kf.x_est[0], x0 + vx0 * dt, atol=1e-6)
    # y should move by vy*dt
    assert np.isclose(kf.x_est[1], y0 + vy0 * dt, atol=1e-6)


def test_predict_applies_gravity_to_vy_only(kf):
    dt = 0.1
    vx0, vy0 = kf.x_est[2], kf.x_est[3]

    kf.predict(accel_measurement=[0.0, 0.0], dt=dt)

    # vx unaffected by gravity (gravity only acts in -y)
    assert np.isclose(kf.x_est[2], vx0, atol=1e-6)
    # vy should decrease by g*dt (gravity, accelerometer doesn't sense it)
    assert np.isclose(kf.x_est[3], vy0 - prm.G * dt, atol=1e-6)


def test_predict_applies_commanded_accel(kf):
    dt = 0.1
    vx0, vy0 = kf.x_est[2], kf.x_est[3]
    ax, ay = 5.0, 10.0

    kf.predict(accel_measurement=[ax, ay], dt=dt)

    assert np.isclose(kf.x_est[2], vx0 + ax * dt, atol=1e-6)
    assert np.isclose(kf.x_est[3], vy0 + ay * dt - prm.G * dt, atol=1e-6)


def test_predict_increases_covariance(kf):
    """Prediction alone (no correction) should always increase uncertainty."""
    P_before = kf.P.copy()
    kf.predict(accel_measurement=[0.0, 0.0], dt=0.1)
    P_after = kf.P

    assert np.all(np.diag(P_after) >= np.diag(P_before))


# ---------------------------------------------------------------------
# 3. Update step — altimeter correction
# ---------------------------------------------------------------------
def test_update_pulls_y_toward_measurement(kf):
    kf.predict(accel_measurement=[0.0, 0.0], dt=0.1)
    y_before = kf.x_est[1]

    measured_y = y_before + 20.0  # measurement disagrees with prediction
    kf.update(altimeter_measurement=measured_y)

    # Estimate should move toward the measurement, not stay put or overshoot wildly
    assert kf.x_est[1] > y_before
    assert kf.x_est[1] < measured_y + 1e-6


def test_update_reduces_y_covariance(kf):
    kf.predict(accel_measurement=[0.0, 0.0], dt=0.1)
    P_before_y = kf.P[1, 1]

    kf.update(altimeter_measurement=kf.x_est[1])
    P_after_y = kf.P[1, 1]

    assert P_after_y < P_before_y


def test_update_does_not_change_x_estimate(kf):
    """Core design assertion: altimeter measures y only, so x estimate
    must be completely unaffected by the update step."""
    kf.predict(accel_measurement=[0.0, 0.0], dt=0.1)
    x_before = kf.x_est[0]

    kf.update(altimeter_measurement=kf.x_est[1] + 100.0)  # big y correction
    x_after = kf.x_est[0]

    assert np.isclose(x_before, x_after, atol=1e-9)


def test_update_does_not_reduce_x_covariance(kf):
    """Core design assertion: altimeter update must not shrink P[0,0]
    (x uncertainty) — there's no information about x in this measurement."""
    kf.predict(accel_measurement=[0.0, 0.0], dt=0.1)
    P_before_x = kf.P[0, 0]

    kf.update(altimeter_measurement=kf.x_est[1])
    P_after_x = kf.P[0, 0]

    assert np.isclose(P_after_x, P_before_x, atol=1e-9)


# ---------------------------------------------------------------------
# 4. The unobservable-mode demonstration (key design validation)
# ---------------------------------------------------------------------
def test_std_x_grows_unbounded_over_time(kf, rng):
    """x is never corrected — its uncertainty should grow monotonically
    (or at least never shrink) over many predict/update cycles."""
    dt = 0.1
    std_x_history = []

    true_state = np.array([500.0, 1000.0, -20.0, -30.0, prm.WET_MASS])
    for step in range(100):
        accel_meas, alt_meas = sensor_model(
            true_state, true_control=[0.0, prm.WET_MASS * prm.G],
            mass=true_state[4], rng=rng
        )
        kf.step(accel_meas, alt_meas, dt)
        std_x, _ = kf.position_std()
        std_x_history.append(std_x)

        # advance a crude true state in step with predict, just to keep
        # measurements roughly self-consistent (not testing dynamics here)
        true_state[0] += true_state[2] * dt
        true_state[1] += true_state[3] * dt

    # std_x should be non-decreasing (monotonic growth is the signature
    # of a state that's never corrected by any measurement)
    assert std_x_history[-1] > std_x_history[0]
    assert np.all(np.diff(std_x_history) >= -1e-9)  # never shrinks


def test_std_y_stays_bounded_over_time(kf, rng):
    """y IS corrected every step by the altimeter — its uncertainty
    should converge/stabilize, not grow unbounded like x."""
    dt = 0.1
    std_y_history = []

    true_state = np.array([500.0, 1000.0, -20.0, -30.0, prm.WET_MASS])
    for step in range(100):
        accel_meas, alt_meas = sensor_model(
            true_state, true_control=[0.0, prm.WET_MASS * prm.G],
            mass=true_state[4], rng=rng
        )
        kf.step(accel_meas, alt_meas, dt)
        _, std_y = kf.position_std()
        std_y_history.append(std_y)

        true_state[0] += true_state[2] * dt
        true_state[1] += true_state[3] * dt

    # y uncertainty should stabilize — last 20 steps shouldn't be growing
    late_window = std_y_history[-20:]
    assert max(late_window) - min(late_window) < 0.5  # roughly flat/converged
    # and should be much smaller than x's uncertainty by the same point
    assert std_y_history[-1] < 5.0  # bounded, not blown up


def test_std_x_diverges_from_std_y_over_time(kf, rng):
    """Direct comparison: by the end of a long run, x uncertainty should
    be dramatically larger than y uncertainty — the core unobservability
    signature side-by-side."""
    dt = 0.1
    true_state = np.array([500.0, 1000.0, -20.0, -30.0, prm.WET_MASS])

    for step in range(150):
        accel_meas, alt_meas = sensor_model(
            true_state, true_control=[0.0, prm.WET_MASS * prm.G],
            mass=true_state[4], rng=rng
        )
        kf.step(accel_meas, alt_meas, dt)
        true_state[0] += true_state[2] * dt
        true_state[1] += true_state[3] * dt

    std_x, std_y = kf.position_std()
    assert std_x > 5 * std_y  # x uncertainty dominates by a wide margin


def test_velocity_std_mirrors_position_pattern(kf, rng):
    """vx (unobserved) should also grow relative to vy (indirectly
    corrected via the y-vy coupling in the dynamics/covariance update)."""
    dt = 0.1
    true_state = np.array([500.0, 1000.0, -20.0, -30.0, prm.WET_MASS])

    for step in range(150):
        accel_meas, alt_meas = sensor_model(
            true_state, true_control=[0.0, prm.WET_MASS * prm.G],
            mass=true_state[4], rng=rng
        )
        kf.step(accel_meas, alt_meas, dt)
        true_state[0] += true_state[2] * dt
        true_state[1] += true_state[3] * dt

    std_vx, std_vy = kf.velocity_std()
    assert std_vx > std_vy


# ---------------------------------------------------------------------
# 5. sensor_model — noise generation sanity
# ---------------------------------------------------------------------
def test_sensor_model_output_shapes(rng):
    true_state = [0.0, 1000.0, 0.0, -30.0, prm.WET_MASS]
    accel_meas, alt_meas = sensor_model(
        true_state, true_control=[0.0, 20000.0], mass=prm.WET_MASS, rng=rng
    )
    assert np.asarray(accel_meas).shape == (2,)
    assert np.isscalar(alt_meas) or np.asarray(alt_meas).shape == ()


def test_sensor_model_noise_is_zero_mean(rng):
    """Over many samples, average noise should be close to zero (no bias)."""
    true_state = [0.0, 1000.0, 0.0, -30.0, prm.WET_MASS]
    control = [0.0, prm.WET_MASS * prm.G]

    alt_errors = []
    for _ in range(2000):
        _, alt_meas = sensor_model(true_state, control, prm.WET_MASS, rng=rng)
        alt_errors.append(alt_meas - true_state[1])

    assert abs(np.mean(alt_errors)) < 0.1  # should hover near 0


def test_sensor_model_noise_matches_configured_std(rng):
    """Sampled noise std should roughly match prm.ALTIMETER_NOISE_STD."""
    true_state = [0.0, 1000.0, 0.0, -30.0, prm.WET_MASS]
    control = [0.0, prm.WET_MASS * prm.G]

    alt_errors = []
    for _ in range(3000):
        _, alt_meas = sensor_model(true_state, control, prm.WET_MASS, rng=rng)
        alt_errors.append(alt_meas - true_state[1])

    sample_std = np.std(alt_errors)
    assert abs(sample_std - prm.ALTIMETER_NOISE_STD) < 0.1


def test_sensor_model_accel_reflects_true_specific_force(rng):
    """Accelerometer should measure Tx/m, Ty/m (NOT including gravity) —
    confirm no gravity term leaks into the accelerometer reading."""
    true_state = [0.0, 1000.0, 0.0, 0.0, prm.WET_MASS]
    Tx, Ty = 0.0, prm.WET_MASS * prm.G  # thrust exactly cancels gravity

    accel_samples = []
    for _ in range(1000):
        accel_meas, _ = sensor_model(true_state, [Tx, Ty], prm.WET_MASS, rng=rng)
        accel_samples.append(accel_meas)

    mean_accel = np.mean(accel_samples, axis=0)
    # Expected: ax~0, ay~g (since Ty/m = g exactly here) — NOT ay~0
    # (which would be the case if gravity were incorrectly subtracted)
    assert np.isclose(mean_accel[0], 0.0, atol=0.05)
    assert np.isclose(mean_accel[1], prm.G, atol=0.05)


# ---------------------------------------------------------------------
# 6. step() convenience method
# ---------------------------------------------------------------------
def test_step_returns_updated_estimate(kf):
    result = kf.step(accel_measurement=[0.0, 0.0], altimeter_measurement=990.0, dt=0.1)
    assert result.shape == (4,)
    assert np.allclose(result, kf.x_est)