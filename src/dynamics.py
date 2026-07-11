import numpy as np
from config import parameters as prm

class RocketDynamics:
    def __init__(self, g=prm.G, isp=prm.ISP, g0=prm.G0):
        self.g = g
        self.isp = isp
        self.g0 = g0

    def state_derivative(self, state, control):

        x, y, vx, vy, m = state
        Tx, Ty = control

        if m <= 0:
            raise ValueError("Mass has depleted to zero or below.")

        # Accelerations
        ax = Tx / m
        ay = (Ty / m) - self.g

        # Mass depletion via rocket equation: mdot = -|T| / (Isp * g0)
        thrust_mag = np.hypot(Tx, Ty)
        mdot = -thrust_mag / (self.isp * self.g0)

        return np.array([vx, vy, ax, ay, mdot])

    def step_rk4(self, state, control, dt):
        """Single RK4 integration step."""
        k1 = self.state_derivative(state, control)
        k2 = self.state_derivative(state + 0.5 * dt * k1, control)
        k3 = self.state_derivative(state + 0.5 * dt * k2, control)
        k4 = self.state_derivative(state + dt * k3, control)

        new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Clamp: mass can't go negative, altitude can't go below ground
        new_state[4] = max(new_state[4], 0.0)
        return new_state


    def simulate(self, initial_state, control_profile, dt, n_steps):
        states = np.zeros((n_steps + 1, 5))
        states[0] = initial_state

        for i in range(n_steps):
            try:
                states[i + 1] = self.step_rk4(states[i], control_profile[i], dt)
            except ValueError:
                # Fuel exhausted mid-step — truncate here, same idea as ground contact
                states = states[: i + 1]
                break

            if states[i + 1][1] <= 0:  # hit ground
                states = states[: i + 2]
                break

        return states