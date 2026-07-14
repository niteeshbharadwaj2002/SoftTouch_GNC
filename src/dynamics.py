import numpy as np
import config.parameters as prm
from src.kalman import sensor_model  # add to imports at top of file

class RocketDynamics:
    def __init__(self):
        self.g = prm.G
        self.isp = prm.ISP
        self.g0 = prm.G0
        self.dry_mass = prm.DRY_MASS

    def state_derivative(self, state, control, external_force=(0.0, 0.0)):
        x, y, vx, vy, m = state
        Tx, Ty = control
        Fx, Fy = external_force

        if m <= self.dry_mass:
            # Fuel exhausted — engine cannot produce thrust with no propellant
            Tx, Ty = 0.0, 0.0
            mdot = 0.0
        else:
            thrust_mag = np.hypot(Tx, Ty)
            mdot = -thrust_mag / (self.isp * self.g0)

        # Accelerations: total applied force = engine thrust + external force
        ax = (Tx + Fx) / m
        ay = (Ty + Fy) / m - self.g

        return np.array([vx, vy, ax, ay, mdot])

    def step_rk4(self, state, control, dt, external_force=(0.0, 0.0)):
        k1 = self.state_derivative(state, control, external_force)
        k2 = self.state_derivative(state + 0.5 * dt * k1, control, external_force)
        k3 = self.state_derivative(state + 0.5 * dt * k2, control, external_force)
        k4 = self.state_derivative(state + dt * k3, control, external_force)

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
            else:
                disturbance = np.array([0.0, 0.0])

            actual_controls[i] = u_corrected  # log controller's commanded (engine) thrust
            # Thrust and wind are passed separately: wind affects acceleration
            # but must NOT be counted as propellant-consuming thrust.
            actual_states[i + 1] = self.step_rk4(actual_states[i], u_corrected, dt,
                                                  external_force=disturbance)

            if actual_states[i + 1][1] <= 0:  # hit ground
                actual_states = actual_states[: i + 2]
                actual_controls = actual_controls[: i + 1]
                break

        return actual_states, actual_controls
    
    def simulate_closed_loop_estimated(self, planned_states, planned_controls, controller,
                                        kf, dt, wind_force=None, rng=None,
                                        max_extra_steps=400, landing_speed_tol=0.5):

        n_steps = len(planned_controls)
        max_steps = n_steps + max_extra_steps

        true_states = [np.array(planned_states[0], dtype=float)]
        estimated_states = [kf.x_est.copy()]
        actual_controls = []

        if rng is None:
            rng = np.random.default_rng()

        def setpoint_state(i):
            return planned_states[i] if i < len(planned_states) else planned_states[-1]

        def setpoint_control(i):
            return planned_controls[i] if i < len(planned_controls) else planned_controls[-1]

        i = 0
        while i < max_steps:
            state_i = true_states[i]

            # Controller acts on the ESTIMATE, with true mass (mass tracked
            # exactly, not estimated — see kalman.py design notes)
            estimate_with_mass = np.array([
                kf.x_est[0], kf.x_est[1], kf.x_est[2], kf.x_est[3], state_i[4]
            ])
            u_corrected = controller.compute_control(
                planned_state=setpoint_state(i),
                actual_state=estimate_with_mass,
                planned_control=setpoint_control(i),
            )

            if wind_force is not None:
                disturbance = wind_force(i, state_i)
            else:
                disturbance = np.array([0.0, 0.0])

            actual_controls.append(u_corrected)  # log commanded (engine) thrust
            # Thrust and wind are passed separately: wind affects acceleration
            # but must NOT be counted as propellant-consuming thrust.
            next_state = self.step_rk4(state_i, u_corrected, dt, external_force=disturbance)
            true_states.append(next_state)

            total_force = np.asarray(u_corrected) + np.asarray(disturbance)
            accel_meas, alt_meas = sensor_model(
                next_state, total_force, mass=state_i[4], rng=rng
            )
            kf.step(accel_meas, alt_meas, dt)
            estimated_states.append(kf.x_est.copy())

            i += 1

            touched_down = next_state[1] <= 0
            speed = np.hypot(next_state[2], next_state[3])
            past_horizon_and_settled = (i >= n_steps) and (speed < landing_speed_tol)

            if touched_down or past_horizon_and_settled:
                break

        return np.array(true_states), np.array(estimated_states), np.array(actual_controls)