import numpy as np
from config import parameters as prm

class RocketDynamics:
    def __init__(self):
        self.g = prm.G
        self.isp = prm.ISP
        self.g0 = prm.G0
        self.dry_mass = prm.DRY_MASS

    def state_derivative(self, state, control):
        x, y, vx, vy, m = state
        Tx, Ty = control

        if m <= self.dry_mass:
            # Fuel exhausted — engine cannot produce thrust with no propellant
            Tx, Ty = 0.0, 0.0
            mdot = 0.0
        else:
            thrust_mag = np.hypot(Tx, Ty)
            mdot = -thrust_mag / (self.isp * self.g0)

        # Accelerations
        ax = Tx / m
        ay = (Ty / m) - self.g

        return np.array([vx, vy, ax, ay, mdot])

    def step_rk4(self, state, control, dt):
        """Single RK4 integration step."""
        k1 = self.state_derivative(state, control)
        k2 = self.state_derivative(state + 0.5 * dt * k1, control)
        k3 = self.state_derivative(state + 0.5 * dt * k2, control)
        k4 = self.state_derivative(state + dt * k3, control)

        new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Clamp: mass can never go below dry mass (structural mass floor)
        new_state[4] = max(new_state[4], self.dry_mass)
        return new_state

    def simulate(self, initial_state, control_profile, dt, n_steps):
        states = np.zeros((n_steps + 1, 5))
        states[0] = initial_state

        for i in range(n_steps):
            states[i + 1] = self.step_rk4(states[i], control_profile[i], dt)
            if states[i + 1][1] <= 0:  # hit ground
                states = states[: i + 2]
                break

        return states
    
    def simulate_closed_loop(self, planned_states, planned_controls, controller, dt,
                          wind_force=None):
        n_steps = len(planned_controls)
        actual_states = np.zeros((n_steps + 1, 5))
        actual_controls = np.zeros((n_steps, 2))
        actual_states[0] = planned_states[0]

        for i in range(n_steps):
            u_corrected = controller.compute_control(
                planned_state=planned_states[i],
                actual_state=actual_states[i],
                planned_control=planned_controls[i],)

            if wind_force is not None:
                disturbance = wind_force(i, actual_states[i])
                u_applied = u_corrected + disturbance
            else:
                u_applied = u_corrected

            actual_controls[i] = u_corrected  # log controller's commanded thrust, pre-wind
            actual_states[i + 1] = self.step_rk4(actual_states[i], u_applied, dt)

            if actual_states[i + 1][1] <= 0:  # hit ground
                actual_states = actual_states[: i + 2]
                actual_controls = actual_controls[: i + 1]
                break

        return actual_states, actual_controls