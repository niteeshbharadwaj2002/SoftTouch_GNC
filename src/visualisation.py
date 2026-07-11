import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

# Consistent style constants
PLANNED_STYLE = dict(color='red', linestyle=':', linewidth=2.5)
ACTUAL_STYLE = dict(color='blue', linestyle=':', linewidth=2.5)
ESTIMATED_STYLE = dict(color='green', linestyle=':', linewidth=2.5)


def plot_trajectory(planned_states, actual_states=None, estimated_states=None,
                     target=(0.0, 0.0), save_path=None):
    """2D trajectory: downrange (x) vs altitude (y). X-axis symmetric about 0."""
    xp, yp = planned_states[:, 0], planned_states[:, 1]
    has_actual = actual_states is not None
    has_estimated = estimated_states is not None

    fig, ax = plt.subplots(figsize=(7, 6))
    label = "Planned" if (has_actual or has_estimated) else "Trajectory"
    ax.plot(xp, yp, label=label, **PLANNED_STYLE)
    ax.plot(xp[0], yp[0], 'go', markersize=10, label="Start")
    ax.plot(target[0], target[1], 'k^', markersize=10, label="Target")

    x_all = np.concatenate([xp, [target[0]]])

    if has_actual:
        xa, ya = actual_states[:, 0], actual_states[:, 1]
        ax.plot(xa, ya, label="Actual", **ACTUAL_STYLE)
        ax.plot(xa[-1], ya[-1], 'bx', markersize=10, label="Actual Landing")
        x_all = np.concatenate([x_all, xa])
    elif not has_estimated:
        ax.plot(xp[-1], yp[-1], 'rx', markersize=10, label="Landing")

    if has_estimated:
        xe, ye = estimated_states[:, 0], estimated_states[:, 1]
        ax.plot(xe, ye, label="Estimated", **ESTIMATED_STYLE)
        x_all = np.concatenate([x_all, xe])

    x_max_abs = np.abs(x_all).max()
    margin = x_max_abs * 0.15 if x_max_abs > 0 else 10.0
    ax.set_xlim(-x_max_abs - margin, x_max_abs + margin)
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='-')

    title_suffix = _comparison_suffix(has_actual, has_estimated)
    ax.set_xlabel("Downrange x (m)")
    ax.set_ylabel("Altitude y (m)")
    ax.set_title("Rocket Trajectory" + title_suffix)
    ax.legend()
    ax.grid(True)
    ax.set_aspect('auto')
    _save_or_show(fig, save_path)


def plot_altitude_vs_time(planned_states, actual_states=None, estimated_states=None,
                           dt=0.1, save_path=None):
    tp = np.arange(len(planned_states)) * dt
    has_actual = actual_states is not None
    has_estimated = estimated_states is not None

    fig, ax = plt.subplots(figsize=(7, 4))
    label = "Planned" if (has_actual or has_estimated) else "Altitude"
    ax.plot(tp, planned_states[:, 1], label=label, **PLANNED_STYLE)

    if has_actual:
        ta = np.arange(len(actual_states)) * dt
        ax.plot(ta, actual_states[:, 1], label="Actual", **ACTUAL_STYLE)

    if has_estimated:
        te = np.arange(len(estimated_states)) * dt
        ax.plot(te, estimated_states[:, 1], label="Estimated", **ESTIMATED_STYLE)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude y (m)")
    ax.set_title("Altitude vs Time" + _comparison_suffix(has_actual, has_estimated))
    ax.legend()
    ax.grid(True)
    _save_or_show(fig, save_path)


def plot_velocity(planned_states, actual_states=None, estimated_states=None,
                   dt=0.1, save_path=None):
    """Vx and Vy vs time."""
    tp = np.arange(len(planned_states)) * dt
    has_actual = actual_states is not None
    has_estimated = estimated_states is not None

    if has_actual or has_estimated:
        fig, axs = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

        axs[0].plot(tp, planned_states[:, 2], label="Planned Vx", **PLANNED_STYLE)
        if has_actual:
            ta = np.arange(len(actual_states)) * dt
            axs[0].plot(ta, actual_states[:, 2], label="Actual Vx", **ACTUAL_STYLE)
        if has_estimated:
            te = np.arange(len(estimated_states)) * dt
            axs[0].plot(te, estimated_states[:, 2], label="Estimated Vx", **ESTIMATED_STYLE)
        axs[0].set_ylabel("Vx (m/s)")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(tp, planned_states[:, 3], label="Planned Vy", **PLANNED_STYLE)
        if has_actual:
            axs[1].plot(ta, actual_states[:, 3], label="Actual Vy", **ACTUAL_STYLE)
        if has_estimated:
            axs[1].plot(te, estimated_states[:, 3], label="Estimated Vy", **ESTIMATED_STYLE)
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Vy (m/s)")
        axs[1].legend()
        axs[1].grid(True)

        fig.suptitle("Velocity" + _comparison_suffix(has_actual, has_estimated))
    else:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(tp, planned_states[:, 2], linestyle=':', linewidth=2, label="Vx")
        ax.plot(tp, planned_states[:, 3], linestyle=':', linewidth=2, label="Vy")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (m/s)")
        ax.set_title("Velocity vs Time")
        ax.legend()
        ax.grid(True)

    _save_or_show(fig, save_path)


def plot_mass(planned_states, actual_states=None, dt=0.1, dry_mass=None, save_path=None):
    """Total mass vs time, with optional dry-mass reference line.
    No estimated_states param — mass is not part of the Kalman state."""
    tp = np.arange(len(planned_states)) * dt
    has_actual = actual_states is not None

    fig, ax = plt.subplots(figsize=(7, 4))
    label = "Planned mass" if has_actual else "Total mass"
    ax.plot(tp, planned_states[:, 4], label=label, **PLANNED_STYLE)

    if has_actual:
        ta = np.arange(len(actual_states)) * dt
        ax.plot(ta, actual_states[:, 4], label="Actual mass", **ACTUAL_STYLE)

    if dry_mass is not None:
        ax.axhline(dry_mass, color='k', linestyle='--', label="Dry mass")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mass (kg)")
    ax.set_title("Mass vs Time" + (": Planned vs Actual" if has_actual else ""))
    ax.legend()
    ax.grid(True)
    _save_or_show(fig, save_path)


def plot_thrust(planned_controls, actual_controls=None, dt=0.1,
                 max_thrust=None, min_thrust=None, save_path=None):
    """Thrust magnitude vs time. No estimated variant — thrust isn't estimated."""
    planned_controls = np.asarray(planned_controls)
    tp = np.arange(len(planned_controls)) * dt
    planned_mag = np.hypot(planned_controls[:, 0], planned_controls[:, 1])
    has_actual = actual_controls is not None

    fig, ax = plt.subplots(figsize=(7, 4))
    label = "Planned |Thrust|" if has_actual else "|Thrust|"
    ax.plot(tp, planned_mag, label=label, **PLANNED_STYLE)

    if has_actual:
        actual_controls = np.asarray(actual_controls)
        ta = np.arange(len(actual_controls)) * dt
        actual_mag = np.hypot(actual_controls[:, 0], actual_controls[:, 1])
        ax.plot(ta, actual_mag, label="Actual |Thrust|", **ACTUAL_STYLE)

    if max_thrust is not None:
        ax.axhline(max_thrust, color='k', linestyle='--', label="Max thrust")
    if min_thrust is not None:
        ax.axhline(min_thrust, color='gray', linestyle='--', label="Min thrust")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Thrust (N)")
    ax.set_title("Thrust Magnitude vs Time" + (": Planned vs Actual" if has_actual else ""))
    ax.legend()
    ax.grid(True)
    _save_or_show(fig, save_path)


def plot_thrust_angle(planned_controls, actual_controls=None, dt=0.1, save_path=None):
    """Thrust direction (angle from vertical) vs time."""
    planned_controls = np.asarray(planned_controls)
    tp = np.arange(len(planned_controls)) * dt
    planned_angle = np.degrees(np.arctan2(planned_controls[:, 0], planned_controls[:, 1]))
    has_actual = actual_controls is not None

    fig, ax = plt.subplots(figsize=(7, 4))
    label = "Planned" if has_actual else "Thrust angle"
    ax.plot(tp, planned_angle, label=label, **PLANNED_STYLE)

    if has_actual:
        actual_controls = np.asarray(actual_controls)
        ta = np.arange(len(actual_controls)) * dt
        actual_angle = np.degrees(np.arctan2(actual_controls[:, 0], actual_controls[:, 1]))
        ax.plot(ta, actual_angle, label="Actual", **ACTUAL_STYLE)
        ax.legend()

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Thrust angle from vertical (deg)")
    ax.set_title("Thrust Direction vs Time" + (": Planned vs Actual" if has_actual else ""))
    ax.grid(True)
    _save_or_show(fig, save_path)


def plot_distance_to_target(planned_states, actual_states=None, estimated_states=None,
                             target=(0.0, 0.0), dt=0.1, save_path=None):
    tp = np.arange(len(planned_states)) * dt
    has_actual = actual_states is not None
    has_estimated = estimated_states is not None
    dist_planned = np.hypot(planned_states[:, 0] - target[0], planned_states[:, 1] - target[1])

    fig, ax = plt.subplots(figsize=(7, 4))
    label = "Planned" if (has_actual or has_estimated) else "Distance to target"
    ax.plot(tp, dist_planned, label=label, **PLANNED_STYLE)

    if has_actual:
        ta = np.arange(len(actual_states)) * dt
        dist_actual = np.hypot(actual_states[:, 0] - target[0], actual_states[:, 1] - target[1])
        ax.plot(ta, dist_actual, label="Actual", **ACTUAL_STYLE)

    if has_estimated:
        te = np.arange(len(estimated_states)) * dt
        dist_estimated = np.hypot(estimated_states[:, 0] - target[0], estimated_states[:, 1] - target[1])
        ax.plot(te, dist_estimated, label="Estimated", **ESTIMATED_STYLE)

    if has_actual or has_estimated:
        ax.legend()

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance to target (m)")
    ax.set_title("Distance to Target vs Time" + _comparison_suffix(has_actual, has_estimated))
    ax.grid(True)
    _save_or_show(fig, save_path)


def plot_mass_flow_rate(planned_states, actual_states=None, dt=0.1, save_path=None):
    """mdot vs time — no estimated variant, mass isn't part of Kalman state."""
    tp = np.arange(len(planned_states) - 1) * dt
    mdot_planned = np.diff(planned_states[:, 4]) / dt
    has_actual = actual_states is not None

    fig, ax = plt.subplots(figsize=(7, 4))
    label = "Planned" if has_actual else "Mass flow rate"
    ax.plot(tp, mdot_planned, label=label, **PLANNED_STYLE)

    if has_actual:
        ta = np.arange(len(actual_states) - 1) * dt
        mdot_actual = np.diff(actual_states[:, 4]) / dt
        ax.plot(ta, mdot_actual, label="Actual", **ACTUAL_STYLE)
        ax.legend()

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mass flow rate (kg/s)")
    ax.set_title("Fuel Burn Rate vs Time" + (": Planned vs Actual" if has_actual else ""))
    ax.grid(True)
    _save_or_show(fig, save_path)


def plot_tracking_error(planned_states, actual_states, dt, save_path=None):
    """
    Position tracking error between guidance plan and actual (true) closed-
    loop trajectory. Requires both — no meaningful "error" with one trajectory.
    """
    n = min(len(planned_states), len(actual_states))
    t = np.arange(n) * dt
    error = np.hypot(
        planned_states[:n, 0] - actual_states[:n, 0],
        planned_states[:n, 1] - actual_states[:n, 1],
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, error, color='blue', linestyle=':', linewidth=2.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tracking error (m)")
    ax.set_title("Planned vs Actual Position Error")
    ax.grid(True)
    _save_or_show(fig, save_path)


def plot_estimation_error(true_states, estimated_states, dt, save_path=None):
    n = min(len(true_states), len(estimated_states))
    t = np.arange(n) * dt

    error_x = estimated_states[:n, 0] - true_states[:n, 0]
    error_y = estimated_states[:n, 1] - true_states[:n, 1]

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(t, error_x, color='magenta', linestyle=':', linewidth=2.5)
    axs[0].axhline(0, color='k', linewidth=0.5)
    axs[0].set_ylabel("x error (m)")
    axs[0].set_title("Kalman Estimation Error: x (unobserved) vs y (observed)")
    axs[0].grid(True)

    axs[1].plot(t, error_y, color='green', linestyle=':', linewidth=2.5)
    axs[1].axhline(0, color='k', linewidth=0.5)
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("y error (m)")
    axs[1].grid(True)

    _save_or_show(fig, save_path)


def plot_summary(planned_states, planned_controls, dt, actual_states=None, actual_controls=None,
                  estimated_states=None, target=(0.0, 0.0), max_thrust=None, min_thrust=None,
                  dry_mass=None, save_path=None):
    """One figure, multiple subplots — quick full-run overview. Works single-trajectory,
    planned-vs-actual, or full planned-vs-actual-vs-estimated."""
    tp_states = np.arange(len(planned_states)) * dt
    planned_controls = np.asarray(planned_controls)
    tp_control = np.arange(len(planned_controls)) * dt
    planned_thrust_mag = np.hypot(planned_controls[:, 0], planned_controls[:, 1])

    has_actual = actual_states is not None
    has_estimated = estimated_states is not None

    if has_actual:
        ta_states = np.arange(len(actual_states)) * dt
        actual_controls = np.asarray(actual_controls)
        ta_control = np.arange(len(actual_controls)) * dt
        actual_thrust_mag = np.hypot(actual_controls[:, 0], actual_controls[:, 1])

    if has_estimated:
        te_states = np.arange(len(estimated_states)) * dt

    fig, axs = plt.subplots(2, 3, figsize=(16, 9))

    axs[0, 0].plot(planned_states[:, 0], planned_states[:, 1],
                   label="Planned" if (has_actual or has_estimated) else None, **PLANNED_STYLE)
    if has_actual:
        axs[0, 0].plot(actual_states[:, 0], actual_states[:, 1], label="Actual", **ACTUAL_STYLE)
    if has_estimated:
        axs[0, 0].plot(estimated_states[:, 0], estimated_states[:, 1], label="Estimated", **ESTIMATED_STYLE)
    if has_actual or has_estimated:
        axs[0, 0].legend()
    axs[0, 0].plot(target[0], target[1], 'k^', markersize=10)
    axs[0, 0].set_title("Trajectory (x vs y)")
    axs[0, 0].set_xlabel("x (m)"); axs[0, 0].set_ylabel("y (m)")
    axs[0, 0].grid(True)

    axs[0, 1].plot(tp_states, planned_states[:, 2], color='red', linestyle=':', linewidth=2,
                   label="Planned Vx" if has_actual else "Vx")
    axs[0, 1].plot(tp_states, planned_states[:, 3], color='red', linestyle=':', linewidth=2.5,
                   label="Planned Vy" if has_actual else "Vy")
    if has_actual:
        axs[0, 1].plot(ta_states, actual_states[:, 2], color='blue', linestyle=':', linewidth=2, label="Actual Vx")
        axs[0, 1].plot(ta_states, actual_states[:, 3], color='blue', linestyle=':', linewidth=2.5, label="Actual Vy")
    if has_estimated:
        axs[0, 1].plot(te_states, estimated_states[:, 2], color='green', linestyle=':', linewidth=2, label="Est Vx")
        axs[0, 1].plot(te_states, estimated_states[:, 3], color='green', linestyle=':', linewidth=2.5, label="Est Vy")
    axs[0, 1].set_title("Velocity vs Time")
    axs[0, 1].legend(fontsize=7); axs[0, 1].grid(True)

    axs[0, 2].plot(tp_states, planned_states[:, 4], label="Planned" if has_actual else "Mass", **PLANNED_STYLE)
    if has_actual:
        axs[0, 2].plot(ta_states, actual_states[:, 4], label="Actual", **ACTUAL_STYLE)
    if dry_mass is not None:
        axs[0, 2].axhline(dry_mass, color='k', linestyle='--')
    axs[0, 2].set_title("Mass vs Time")
    axs[0, 2].legend(); axs[0, 2].grid(True)

    axs[1, 0].plot(tp_control, planned_thrust_mag, label="Planned" if has_actual else "|Thrust|", **PLANNED_STYLE)
    if has_actual:
        axs[1, 0].plot(ta_control, actual_thrust_mag, label="Actual", **ACTUAL_STYLE)
    if max_thrust is not None:
        axs[1, 0].axhline(max_thrust, color='k', linestyle='--')
    if min_thrust is not None:
        axs[1, 0].axhline(min_thrust, color='gray', linestyle='--')
    axs[1, 0].set_title("Thrust Magnitude vs Time")
    axs[1, 0].legend(); axs[1, 0].grid(True)

    dist_planned = np.hypot(planned_states[:, 0] - target[0], planned_states[:, 1] - target[1])
    axs[1, 1].plot(tp_states, dist_planned,
                   label="Planned" if (has_actual or has_estimated) else "Distance", **PLANNED_STYLE)
    if has_actual:
        dist_actual = np.hypot(actual_states[:, 0] - target[0], actual_states[:, 1] - target[1])
        axs[1, 1].plot(ta_states, dist_actual, label="Actual", **ACTUAL_STYLE)
    if has_estimated:
        dist_est = np.hypot(estimated_states[:, 0] - target[0], estimated_states[:, 1] - target[1])
        axs[1, 1].plot(te_states, dist_est, label="Estimated", **ESTIMATED_STYLE)
    axs[1, 1].set_title("Distance to Target vs Time")
    axs[1, 1].legend(); axs[1, 1].grid(True)

    if has_actual and has_estimated:
        n = min(len(actual_states), len(estimated_states))
        t_err = np.arange(n) * dt
        err_x = estimated_states[:n, 0] - actual_states[:n, 0]
        err_y = estimated_states[:n, 1] - actual_states[:n, 1]
        axs[1, 2].plot(t_err, err_x, color='magenta', linestyle=':', linewidth=2.5, label="x err (est)")
        axs[1, 2].plot(t_err, err_y, color='green', linestyle=':', linewidth=2.5, label="y err (est)")
        axs[1, 2].legend(fontsize=7)
        axs[1, 2].set_title("Estimation Error vs Time")
    elif has_actual:
        n = min(len(planned_states), len(actual_states))
        t_err = np.arange(n) * dt
        error = np.hypot(planned_states[:n, 0] - actual_states[:n, 0],
                          planned_states[:n, 1] - actual_states[:n, 1])
        axs[1, 2].plot(t_err, error, color='blue', linestyle=':', linewidth=2.5)
        axs[1, 2].set_title("Tracking Error vs Time")
    else:
        axs[1, 2].plot(tp_states, planned_states[:, 1], color='red', linestyle=':', linewidth=2.5)
        axs[1, 2].set_title("Altitude vs Time")
    axs[1, 2].grid(True)

    fig.tight_layout()
    _save_or_show(fig, save_path)


def _comparison_suffix(has_actual, has_estimated):
    if has_actual and has_estimated:
        return ": Planned vs Actual vs Estimated"
    if has_actual:
        return ": Planned vs Actual"
    if has_estimated:
        return ": Planned vs Estimated"
    return ""


def _save_or_show(fig, save_path):
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()