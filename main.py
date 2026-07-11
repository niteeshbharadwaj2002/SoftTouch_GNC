import numpy as np

from src.dynamics import RocketDynamics
from src import visualisation as viz
from config import parameters as prm


def main():
    dynamics = RocketDynamics()

    # Constant thrust straight down (+y) — controlled descent, no guidance/control yet.
    # Thrust > weight so the rocket actually decelerates instead of free-falling.
    weight = prm.WET_MASS * prm.G
    thrust_y = weight * 1.2  # 30% margin above weight for deceleration
    control_profile = np.tile([0.0, thrust_y], (prm.N_STEPS, 1))

    states = dynamics.simulate(
        initial_state=np.array(prm.INITIAL_STATE),
        control_profile=control_profile,
        dt=prm.DT,
        n_steps=prm.N_STEPS,
    )

    print(f"Simulated {len(states)} steps "
          f"({(len(states) - 1) * prm.DT:.1f}s of flight)")
    print(f"Final state: x={states[-1,0]:.1f}, y={states[-1,1]:.1f}, "
          f"vx={states[-1,2]:.1f}, vy={states[-1,3]:.1f}, m={states[-1,4]:.1f}")

    # --- Visualization ---
    target = (prm.TARGET_X, prm.TARGET_Y)

    viz.plot_trajectory(states, target=target)
    viz.plot_altitude_vs_time(states, dt=prm.DT)
    viz.plot_velocity(states, dt=prm.DT)
    viz.plot_mass(states, dt=prm.DT, dry_mass=prm.DRY_MASS)
    viz.plot_distance_to_target(states, target=target, dt=prm.DT)

    n = len(states) - 1
    viz.plot_thrust(control_profile[:n], dt=prm.DT,
                     max_thrust=prm.MAX_THRUST, min_thrust=prm.MIN_THRUST)

if __name__ == "__main__":
    main()