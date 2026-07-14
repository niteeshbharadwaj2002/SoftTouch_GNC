import numpy as np
from scipy.linalg import solve_continuous_are

from config import parameters as prm

class LQRController:
    def __init__(self, Q=None, R=None):
        self.Q = Q if Q is not None else np.diag([20.0, 10.0, 25.0, 15.0])
        self.R = R if R is not None else np.diag([0.001, 0.001])
        self.g = prm.G
        self.max_thrust = prm.MAX_THRUST
        self.min_thrust = prm.MIN_THRUST

    def _linearize(self, m):

        A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        B = np.array([
            [0,     0],
            [0,     0],
            [1 / m, 0],
            [0,     1 / m],
        ])
        return A, B

    def compute_gain(self, m):
        A, B = self._linearize(m)
        P = solve_continuous_are(A, B, self.Q, self.R)
        K = np.linalg.inv(self.R) @ B.T @ P
        return K

    def _clip_thrust(self, T):

        mag = np.linalg.norm(T)
        if mag < 1e-8:
            # No commanded direction at all — default to min thrust straight up
            return np.array([0.0, self.min_thrust])
        if mag > self.max_thrust:
            return T * (self.max_thrust / mag)
        if mag < self.min_thrust:
            return T * (self.min_thrust / mag)
        return T

    def compute_control(self, planned_state, actual_state, planned_control):

        m_current = actual_state[4]
        K = self.compute_gain(m_current)

        error = np.array(planned_state[:4]) - np.array(actual_state[:4])
        correction = K @ error

        u = np.array(planned_control) + correction
        return self._clip_thrust(u)