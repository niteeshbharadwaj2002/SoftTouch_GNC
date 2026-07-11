import numpy as np

from config import parameters as prm

class KalmanFilter:
    def __init__(self, initial_state, initial_covariance=None, Q=None, R=None):
        self.x_est = np.array(initial_state, dtype=float)  # [x, y, vx, vy]

        self.P = initial_covariance if initial_covariance is not None else np.diag(
            [1.0, 1.0, 0.5, 0.5]
        )

        self.Q = Q if Q is not None else np.diag([0.01, 0.01, 0.05, 0.05])
        self.R = R if R is not None else np.array([[prm.ALTIMETER_NOISE_STD ** 2]])

        self.g = prm.G

    def predict(self, accel_measurement, dt):

        A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ])
        B = np.array([
            [0,  0],
            [0,  0],
            [dt, 0],
            [0,  dt],
        ])
        # Gravity acts only on vy, added as a constant affine term (accelerometer
        # doesn't sense it, so it must be added back in explicitly here)
        g_term = np.array([0.0, 0.0, 0.0, -self.g * dt])

        self.x_est = A @ self.x_est + B @ np.asarray(accel_measurement) + g_term
        self.P = A @ self.P @ A.T + self.Q

    def update(self, altimeter_measurement):

        H = np.array([[0, 1, 0, 0]])  # measures y only
        z = np.array([altimeter_measurement])

        y_innovation = z - H @ self.x_est
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x_est = self.x_est + (K @ y_innovation).flatten()
        self.P = (np.eye(4) - K @ H) @ self.P

    def step(self, accel_measurement, altimeter_measurement, dt):
        """Convenience: one full predict + update cycle. Returns current estimate."""
        self.predict(accel_measurement, dt)
        self.update(altimeter_measurement)
        return self.x_est.copy()

    def position_std(self):
        return np.sqrt(self.P[0, 0]), np.sqrt(self.P[1, 1])

    def velocity_std(self):
        """Same idea for velocity: std_vx grows, std_vy stays bounded."""
        return np.sqrt(self.P[2, 2]), np.sqrt(self.P[3, 3])


def sensor_model(true_state, true_control, mass, rng=None):

    if rng is None:
        rng = np.random.default_rng()

    Tx, Ty = true_control
    true_ax = Tx / mass
    true_ay = Ty / mass

    accel_noise = rng.normal(0, prm.ACCELEROMETER_NOISE_STD, size=2)
    accel_measurement = np.array([true_ax, true_ay]) + accel_noise

    altimeter_noise = rng.normal(0, prm.ALTIMETER_NOISE_STD)
    altimeter_measurement = true_state[1] + altimeter_noise

    return accel_measurement, altimeter_measurement