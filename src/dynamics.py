import numpy as np
from config import parameters as prm
from src.kalman import sensor_model  # add to imports at top of file

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
    
    def simulate_closed_loop_estimated(self, planned_states, planned_controls, controller,
                                        kf, dt, wind_force=None, rng=None):

        n_steps = len(planned_controls)
        true_states = np.zeros((n_steps + 1, 5))
        estimated_states = np.zeros((n_steps + 1, 4))
        actual_controls = np.zeros((n_steps, 2))

        true_states[0] = planned_states[0]
        estimated_states[0] = kf.x_est.copy()

        if rng is None:
            rng = np.random.default_rng()

        for i in range(n_steps):
            # Controller acts on the ESTIMATE, with true mass (mass tracked
            # exactly, not estimated — see kalman.py design notes)
            estimate_with_mass = np.array([
                kf.x_est[0], kf.x_est[1], kf.x_est[2], kf.x_est[3], true_states[i][4]
            ])
            u_corrected = controller.compute_control(
                planned_state=planned_states[i],
                actual_state=estimate_with_mass,
                planned_control=planned_controls[i],
            )

            if wind_force is not None:
                disturbance = wind_force(i, true_states[i])
            else:
                disturbance = np.array([0.0, 0.0])
            u_applied = u_corrected + disturbance

            actual_controls[i] = u_corrected  # log commanded thrust, pre-wind
            true_states[i + 1] = self.step_rk4(true_states[i], u_applied, dt)

            # Sensors read the TRUE post-step state; accelerometer senses
            # total applied force (thrust + wind), not just commanded thrust
            accel_meas, alt_meas = sensor_model(
                true_states[i + 1], u_applied, mass=true_states[i][4], rng=rng
            )
            kf.step(accel_meas, alt_meas, dt)
            estimated_states[i + 1] = kf.x_est.copy()

            if true_states[i + 1][1] <= 0:  # hit ground
                true_states = true_states[: i + 2]
                estimated_states = estimated_states[: i + 2]
                actual_controls = actual_controls[: i + 1]
                break

        return true_states, estimated_states, actual_controls