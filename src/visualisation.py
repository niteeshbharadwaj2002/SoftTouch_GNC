import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')  # or 'Qt5Agg'


def plot_trajectory(states, target=(0.0, 0.0), save_path=None):
    """2D trajectory: downrange (x) vs altitude (y). X-axis symmetric about 0."""
    x, y = states[:, 0], states[:, 1]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(x, y, '-b', label="Trajectory")
    ax.plot(x[0], y[0], 'go', markersize=10, label="Start")
    ax.plot(x[-1], y[-1], 'rx', markersize=10, label="Landing")
    ax.plot(target[0], target[1], 'k^', markersize=10, label="Target")

    # Symmetric x-axis: 0 is centered, regardless of sign of x values
    x_max_abs = max(np.abs(x).max(), abs(target[0]))
    margin = x_max_abs * 0.15 if x_max_abs > 0 else 10.0  # padding, avoid zero-width if all x=0
    ax.set_xlim(-x_max_abs - margin, x_max_abs + margin)
    ax.axvline(0, color='gray', linewidth=0.8, linestyle=':')  # visual center reference

    ax.set_xlabel("Downrange x (m)")
    ax.set_ylabel("Altitude y (m)")
    ax.set_title("Rocket Trajectory")
    ax.legend()
    ax.grid(True)
    ax.set_aspect('auto')
    _save_or_show(fig, save_path)

def plot_altitude_vs_time(states, dt, save_path=None):
    t = np.arange(len(states)) * dt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, states[:, 1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude y (m)")
    ax.set_title("Altitude vs Time")
    ax.grid(True)
    _save_or_show(fig, save_path)


def plot_velocity(states, dt, save_path=None):
    """Vx and Vy vs time."""
    t = np.arange(len(states)) * dt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, states[:, 2], label="Vx")
    ax.plot(t, states[:, 3], label="Vy")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Velocity vs Time")
    ax.legend()
    ax.grid(True)
    _save_or_show(fig, save_path)


def plot_mass(states, dt, dry_mass=None, save_path=None):
    """Total mass vs time, with optional dry-mass reference line."""
    t = np.arange(len(states)) * dt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, states[:, 4], label="Total mass")
    if dry_mass is not None:
        ax.axhline(dry_mass, color='r', linestyle='--', label="Dry mass")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mass (kg)")
    ax.set_title("Mass vs Time")
    ax.legend()
    ax.grid(True)
    _save_or_show(fig, save_path)


def plot_thrust(control_profile, dt, max_thrust=None, min_thrust=None, save_path=None):
    """Thrust magnitude vs time, with optional throttle bounds."""
    control_profile = np.asarray(control_profile)
    t = np.arange(len(control_profile)) * dt
    thrust_mag = np.hypot(control_profile[:, 0], control_profile[:, 1])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, thrust_mag, label="|Thrust|")
    if max_thrust is not None:
        ax.axhline(max_thrust, color='r', linestyle='--', label="Max thrust")
    if min_thrust is not None:
        ax.axhline(min_thrust, color='orange', linestyle='--', label="Min thrust")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Thrust (N)")
    ax.set_title("Thrust Magnitude vs Time")
    ax.legend()
    ax.grid(True)
    _save_or_show(fig, save_path)


def plot_thrust_angle(control_profile, dt, save_path=None):
    """Thrust direction (angle from vertical) vs time."""
    control_profile = np.asarray(control_profile)
    t = np.arange(len(control_profile)) * dt
    angle_deg = np.degrees(np.arctan2(control_profile[:, 0], control_profile[:, 1]))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, angle_deg)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Thrust angle from vertical (deg)")
    ax.set_title("Thrust Direction vs Time")
    ax.grid(True)
    _save_or_show(fig, save_path)


def plot_distance_to_target(states, target=(0.0, 0.0), dt=0.1, save_path=None):
    t = np.arange(len(states)) * dt
    dist = np.hypot(states[:, 0] - target[0], states[:, 1] - target[1])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, dist)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance to target (m)")
    ax.set_title("Distance to Target vs Time")
    ax.grid(True)
    _save_or_show(fig, save_path)


def plot_mass_flow_rate(states, dt, save_path=None):
    """mdot vs time — derivative of mass curve, sanity-check against thrust profile."""
    t = np.arange(len(states) - 1) * dt
    mdot = np.diff(states[:, 4]) / dt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, mdot)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mass flow rate (kg/s)")
    ax.set_title("Fuel Burn Rate vs Time")
    ax.grid(True)
    _save_or_show(fig, save_path)


def plot_tracking_error(planned_states, actual_states, dt, save_path=None):
    """
    Step 3 (control) stub — position tracking error between guidance plan and
    actual closed-loop trajectory. planned_states/actual_states must be same length.
    """
    n = min(len(planned_states), len(actual_states))
    t = np.arange(n) * dt
    error = np.hypot(
        planned_states[:n, 0] - actual_states[:n, 0],
        planned_states[:n, 1] - actual_states[:n, 1],
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, error)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tracking error (m)")
    ax.set_title("Planned vs Actual Position Error")
    ax.grid(True)
    _save_or_show(fig, save_path)


def plot_summary(states, control_profile, dt, target=(0.0, 0.0),
                  max_thrust=None, min_thrust=None, dry_mass=None, save_path=None):
    """One figure, multiple subplots — quick full-run overview."""
    t_states = np.arange(len(states)) * dt
    control_profile = np.asarray(control_profile)
    t_control = np.arange(len(control_profile)) * dt
    thrust_mag = np.hypot(control_profile[:, 0], control_profile[:, 1])

    fig, axs = plt.subplots(2, 3, figsize=(16, 9))

    axs[0, 0].plot(states[:, 0], states[:, 1], '-b')
    axs[0, 0].plot(target[0], target[1], 'k^', markersize=10)
    axs[0, 0].set_title("Trajectory (x vs y)")
    axs[0, 0].set_xlabel("x (m)"); axs[0, 0].set_ylabel("y (m)")
    axs[0, 0].grid(True)

    axs[0, 1].plot(t_states, states[:, 2], label="Vx")
    axs[0, 1].plot(t_states, states[:, 3], label="Vy")
    axs[0, 1].set_title("Velocity vs Time")
    axs[0, 1].legend(); axs[0, 1].grid(True)

    axs[0, 2].plot(t_states, states[:, 4])
    if dry_mass is not None:
        axs[0, 2].axhline(dry_mass, color='r', linestyle='--')
    axs[0, 2].set_title("Mass vs Time")
    axs[0, 2].grid(True)

    axs[1, 0].plot(t_control, thrust_mag)
    if max_thrust is not None:
        axs[1, 0].axhline(max_thrust, color='r', linestyle='--')
    if min_thrust is not None:
        axs[1, 0].axhline(min_thrust, color='orange', linestyle='--')
    axs[1, 0].set_title("Thrust Magnitude vs Time")
    axs[1, 0].grid(True)

    dist = np.hypot(states[:, 0] - target[0], states[:, 1] - target[1])
    axs[1, 1].plot(t_states, dist)
    axs[1, 1].set_title("Distance to Target vs Time")
    axs[1, 1].grid(True)

    axs[1, 2].plot(t_states, states[:, 1])
    axs[1, 2].set_title("Altitude vs Time")
    axs[1, 2].grid(True)

    fig.tight_layout()
    _save_or_show(fig, save_path)


def _save_or_show(fig, save_path):
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()