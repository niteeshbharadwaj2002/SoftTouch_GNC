import numpy as np

from src.dynamics import RocketDynamics
from src.optimiser import GuidanceOptimizer
from src.controls import LQRController
from src import visualisation as viz
from config import parameters as prm


def main():
    # ------------------------------------------------------------------
    # 1. Guidance Path
    # ------------------------------------------------------------------
    N_GUIDANCE = 120
    DT_GUIDANCE = 0.5  # 30s fixed final time

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
    # 2. Closed-loop LQR
    # ------------------------------------------------------------------
    dynamics = RocketDynamics()
    controller = LQRController()

    gust_start = N_GUIDANCE // 3
    gust_end = N_GUIDANCE // 2
    gust_force = np.array([3000.0, 0.0])  # N, steady horizontal push during gust

    def wind_gust(step, state):
        if gust_start <= step < gust_end:
            return gust_force
        return np.array([0.0, 0.0])

    print(f"\nSimulating closed-loop descent with wind gust "
          f"(steps {gust_start}-{gust_end}, {gust_force[0]:.0f}N horizontal)...")

    actual_states, actual_controls = dynamics.simulate_closed_loop(
        planned_states, planned_controls, controller, DT_GUIDANCE,
        wind_force=wind_gust,
    )

    actual_fuel_used = actual_states[0, 4] - actual_states[-1, 4]
    print(f"Actual fuel use: {actual_fuel_used:.1f} kg")

    # ------------------------------------------------------------------
    # 3. Compare planned vs actual
    # ------------------------------------------------------------------
    n = min(len(planned_states), len(actual_states))
    tracking_error = np.hypot(
        planned_states[:n, 0] - actual_states[:n, 0],
        planned_states[:n, 1] - actual_states[:n, 1],
    )
    final_landing_error = np.hypot(
        actual_states[-1, 0] - prm.TARGET_X,
        actual_states[-1, 1] - prm.TARGET_Y,
    )

    print(f"\n--- Results ---")
    print(f"Planned landing:  x={planned_states[-1,0]:.2f}, y={planned_states[-1,1]:.2f}")
    print(f"Actual landing:   x={actual_states[-1,0]:.2f}, y={actual_states[-1,1]:.2f}")
    print(f"Landing accuracy (distance from target): {final_landing_error:.2f} m")
    print(f"Max tracking error during flight: {np.max(tracking_error):.2f} m")
    print(f"Fuel used vs planned: {actual_fuel_used:.1f} kg vs {planned_fuel_used:.1f} kg "
          f"({actual_fuel_used - planned_fuel_used:+.1f} kg)")

    # ------------------------------------------------------------------
    # 4. Visualization: planned vs actual, overlaid in single calls
    # ------------------------------------------------------------------
    target = (prm.TARGET_X, prm.TARGET_Y)

    print("\nPlotting comparison plots (Blue=Planned, Red=Actual)...")
    viz.plot_trajectory(planned_states, actual_states, target=target)
    viz.plot_altitude_vs_time(planned_states, actual_states, dt=DT_GUIDANCE)
    viz.plot_velocity(planned_states, actual_states, dt=DT_GUIDANCE)
    viz.plot_mass(planned_states, actual_states, dt=DT_GUIDANCE, dry_mass=prm.DRY_MASS)
    viz.plot_thrust(planned_controls, actual_controls, dt=DT_GUIDANCE,
                     max_thrust=prm.MAX_THRUST, min_thrust=prm.MIN_THRUST)
    viz.plot_thrust_angle(planned_controls, actual_controls, dt=DT_GUIDANCE)
    viz.plot_distance_to_target(planned_states, actual_states, target=target, dt=DT_GUIDANCE)

    print("Plotting tracking error...")
    viz.plot_tracking_error(planned_states, actual_states, dt=DT_GUIDANCE)


if __name__ == "__main__":
    main()