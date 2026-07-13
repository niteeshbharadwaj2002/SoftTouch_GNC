import numpy as np

from src.dynamics import RocketDynamics
from src.optimiser import GuidanceOptimizer
from src.controls import LQRController
from src.kalman import KalmanFilter
from src import visualisation as viz
from src import animate 
from config import parameters as prm


def main():
    # ------------------------------------------------------------------
    # 1. Guidance: solve for the fuel-optimal reference trajectory
    # ------------------------------------------------------------------
    N_GUIDANCE = 120
    DT_GUIDANCE = 0.5  # 60s fixed final time

    optimizer = GuidanceOptimizer(N=N_GUIDANCE, dt=DT_GUIDANCE)

    initial_state = prm.INITIAL_STATE
    target_state = [prm.TARGET_X, prm.TARGET_Y, prm.TARGET_VX, prm.TARGET_VY]

    result = optimizer.solve(initial_state, target_state)

    if result["status"] not in ("optimal", "optimal_inaccurate"):
        print(f"Guidance failed to converge: {result['status']}")
        print("Try increasing N_GUIDANCE/DT_GUIDANCE, or check MAX_THRUST "
              "vs initial velocity/altitude for feasibility.")
        return

    print(f"Guidance solved: status = {result['status']}")

    planned_states = optimizer.to_state_trajectory(result)
    planned_controls = optimizer.to_control_profile(result)

    planned_fuel_used = planned_states[0, 4] - planned_states[-1, 4]
    print(f"Planned fuel use: {planned_fuel_used:.1f} kg "
          f"(of {prm.FUEL_MASS:.1f} kg available)")

    # ------------------------------------------------------------------
    # 2. Closed-loop LQR + Kalman estimation, with wind gust
    # ------------------------------------------------------------------
    dynamics = RocketDynamics()
    controller = LQRController()
    kf = KalmanFilter(initial_state=initial_state[:4])  # [x, y, vx, vy]
    rng = np.random.default_rng(seed=42)

    gust_start = N_GUIDANCE // 3
    gust_end = N_GUIDANCE // 2
    gust_force = np.array([-300.0, 200])  # N, steady horizontal push during gust

    def wind_gust(step, state):
        if gust_start <= step < gust_end:
            return gust_force
        return np.array([0.0, 0.0])

    print(f"\nSimulating closed-loop descent: noisy sensors -> Kalman estimate -> LQR "
          f"(wind gust steps {gust_start}-{gust_end}, {gust_force[0]:.0f}N horizontal)...")

    true_states, estimated_states, actual_controls = dynamics.simulate_closed_loop_estimated(
        planned_states, planned_controls, controller, kf, DT_GUIDANCE,
        wind_force=wind_gust, rng=rng,
    )

    actual_fuel_used = true_states[0, 4] - true_states[-1, 4]
    print(f"Actual fuel use: {actual_fuel_used:.1f} kg")

    # ------------------------------------------------------------------
    # 3. Results
    # ------------------------------------------------------------------
    n = min(len(planned_states), len(true_states))
    tracking_error = np.hypot(
        planned_states[:n, 0] - true_states[:n, 0],
        planned_states[:n, 1] - true_states[:n, 1],
    )
    final_landing_error = np.hypot(
        true_states[-1, 0] - prm.TARGET_X,
        true_states[-1, 1] - prm.TARGET_Y,
    )

    n_est = min(len(true_states), len(estimated_states))
    est_error_x = np.abs(estimated_states[:n_est, 0] - true_states[:n_est, 0])
    est_error_y = np.abs(estimated_states[:n_est, 1] - true_states[:n_est, 1])
    std_x_final, std_y_final = kf.position_std()

    true_terminal_speed = np.hypot(true_states[-1, 2], true_states[-1, 3])
    planned_terminal_speed = np.hypot(planned_states[-1, 2], planned_states[-1, 3])

    print(f"\n--- Results ---")
    print(f"Planned landing:   x={planned_states[-1,0]:.2f}, y={planned_states[-1,1]:.2f}, "
          f"speed={planned_terminal_speed:.2f} m/s")
    print(f"True landing:      x={true_states[-1,0]:.2f}, y={true_states[-1,1]:.2f}, "
          f"speed={true_terminal_speed:.2f} m/s, steps={len(true_states)-1} "
          f"(planned horizon={N_GUIDANCE})")
    print(f"Landing accuracy (true, from target): {final_landing_error:.2f} m")
    print(f"Max tracking error during flight: {np.max(tracking_error):.2f} m")
    print(f"Fuel used vs planned: {actual_fuel_used:.1f} kg vs {planned_fuel_used:.1f} kg "
          f"({actual_fuel_used - planned_fuel_used:+.1f} kg)")
    print(f"\n--- Estimation (Kalman) ---")
    print(f"Final x estimation error: {est_error_x[-1]:.2f} m  (unobserved — expect large/drifting)")
    print(f"Final y estimation error: {est_error_y[-1]:.2f} m  (observed — expect small)")
    print(f"Max x estimation error over flight: {np.max(est_error_x):.2f} m")
    print(f"Max y estimation error over flight: {np.max(est_error_y):.2f} m")
    print(f"Kalman std at landing: std_x={std_x_final:.2f} m, std_y={std_y_final:.2f} m")

    # ------------------------------------------------------------------
    # 4. Visualization: planned vs true vs estimated
    # ------------------------------------------------------------------
    target = (prm.TARGET_X, prm.TARGET_Y)

    print("\nPlotting comparison plots (Blue=Planned, Red=True, Green=Estimated)...")
    viz.plot_trajectory(planned_states, true_states, estimated_states, target=target)
    viz.plot_altitude_vs_time(planned_states, true_states, estimated_states, dt=DT_GUIDANCE)
    viz.plot_velocity(planned_states, true_states, estimated_states, dt=DT_GUIDANCE)
    viz.plot_distance_to_target(planned_states, true_states, estimated_states,
                                 target=target, dt=DT_GUIDANCE)

    # Mass/thrust: no estimated variant — mass isn't part of the Kalman state,
    # thrust isn't a state at all
    viz.plot_mass(planned_states, true_states, dt=DT_GUIDANCE, dry_mass=prm.DRY_MASS)
    viz.plot_thrust(planned_controls, actual_controls, dt=DT_GUIDANCE,
                     max_thrust=prm.MAX_THRUST, min_thrust=prm.MIN_THRUST)
    viz.plot_thrust_angle(planned_controls, actual_controls, dt=DT_GUIDANCE)

    print("Plotting tracking error (planned vs true)...")
    viz.plot_tracking_error(planned_states, true_states, dt=DT_GUIDANCE)

    print("Plotting Kalman estimation error (x unobserved vs y observed)...")
    viz.plot_estimation_error(true_states, estimated_states, dt=DT_GUIDANCE)


    #Animation
    print("\nGenerating animations for README...")
    animate.animate_trajectory(planned_states, true_states, estimated_states,
                                target=target, dt=DT_GUIDANCE, save_path="results/animate/trajectory_1.gif")
    animate.animate_dashboard(planned_states, planned_controls, DT_GUIDANCE,
                            actual_states=true_states, actual_controls=actual_controls,
                            estimated_states=estimated_states, target=target,
                            save_path="results/animate/dashboard_1.gif")
    

if __name__ == "__main__":
    main()