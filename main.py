import numpy as np

from src.dynamics import RocketDynamics
from src import visualisation as viz
from config import parameters as prm


"""
main.py
Entry point — wires guidance (convex optimization) -> dynamics (validation)
-> visualization together.

Guidance solves for the fuel-optimal reference trajectory from the fixed
initial conditions in config/params.py. That thrust profile is then run
back through the actual RK4 dynamics model as an open-loop check —
confirms the guidance solution is physically consistent, not just
optimal under its own (Euler-discretized) approximation. Control/LQR
tracking comes in this afternoon; this is still open-loop.
"""

import numpy as np

from src.dynamics import RocketDynamics
from src.optimiser import GuidanceOptimizer
from src import visualisation as viz
from config import parameters as prm


def main():

    N_GUIDANCE = 120
    DT_GUIDANCE = 0.5  # 30s fixed final time

    optimizer = GuidanceOptimizer(N=N_GUIDANCE, dt=DT_GUIDANCE)

    initial_state = prm.INITIAL_STATE
    target_state = [prm.TARGET_X, prm.TARGET_Y, prm.TARGET_VX, prm.TARGET_VY]

    result = optimizer.solve(initial_state, target_state)

    if result["status"] not in ("optimal", "optimal_inaccurate"):
        print(f"Guidance failed to converge: {result['status']}")
        print("Try increasing N_GUIDANCE/DT_GUIDANCE (more time) or check "
              "MAX_THRUST vs initial velocity/altitude for feasibility.")
        return

    print(f"Guidance solved: status = {result['status']}")

    planned_states = optimizer.to_state_trajectory(result)
    control_profile = optimizer.to_control_profile(result)

    fuel_used = planned_states[0, 4] - planned_states[-1, 4]
    print(f"Planned fuel use: {fuel_used:.1f} kg "
          f"(of {prm.FUEL_MASS:.1f} kg available)")
    print(f"Planned landing: x={planned_states[-1,0]:.2f}, "
          f"y={planned_states[-1,1]:.2f}, "
          f"vx={planned_states[-1,2]:.2f}, vy={planned_states[-1,3]:.2f}")

    # Guidance Validation
    dynamics = RocketDynamics()
    actual_states = dynamics.simulate(
        initial_state=np.array(initial_state),
        control_profile=control_profile,
        dt=DT_GUIDANCE,
        n_steps=N_GUIDANCE,
    )

    print(f"\nActual (RK4) landing: x={actual_states[-1,0]:.2f}, "
          f"y={actual_states[-1,1]:.2f}, "
          f"vx={actual_states[-1,2]:.2f}, vy={actual_states[-1,3]:.2f}")

    landing_error = np.hypot(
        actual_states[-1, 0] - planned_states[-1, 0],
        actual_states[-1, 1] - planned_states[-1, 1],
    )
    print(f"Guidance-vs-actual landing position discrepancy: {landing_error:.3f} m "
          f"(expected small — Euler vs RK4 discretization difference; "
          f"this is exactly the gap the controller will need to correct "
          f"once real-world disturbances are added)")

    # --- Visualization ---
    target = (prm.TARGET_X, prm.TARGET_Y)

    print("\nPlotting guidance (planned) trajectory...")
    viz.plot_trajectory(planned_states, target=target)
    viz.plot_altitude_vs_time(planned_states, dt=DT_GUIDANCE)
    viz.plot_velocity(planned_states, dt=DT_GUIDANCE)
    viz.plot_mass(planned_states, dt=DT_GUIDANCE, dry_mass=prm.DRY_MASS)
    viz.plot_distance_to_target(planned_states, target=target, dt=DT_GUIDANCE)
    viz.plot_thrust(control_profile, dt=DT_GUIDANCE,
                     max_thrust=prm.MAX_THRUST, min_thrust=prm.MIN_THRUST)
    viz.plot_thrust_angle(control_profile, dt=DT_GUIDANCE)

    print("Plotting actual (RK4-propagated) trajectory...")
    viz.plot_trajectory(actual_states, target=target)


if __name__ == "__main__":
    main()
