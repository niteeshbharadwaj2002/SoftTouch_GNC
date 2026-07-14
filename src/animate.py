import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving — avoids GUI popups while rendering

PLANNED_COLOR = 'red'
ACTUAL_COLOR = 'blue'
ESTIMATED_COLOR = 'green'


def _hold_last(arr, idx):
    return arr[min(idx, len(arr) - 1)]


def animate_trajectory(planned_states, actual_states=None, estimated_states=None,
                        target=(0.0, 0.0), dt=0.5, speedup=2, fps=20,
                        save_path="trajectory.gif"):

    n_frames = max(len(planned_states),
                    len(actual_states) if actual_states is not None else 0,
                    len(estimated_states) if estimated_states is not None else 0)
    frame_indices = list(range(0, n_frames, speedup))
    if frame_indices[-1] != n_frames - 1:
        frame_indices.append(n_frames - 1)

    all_x = [planned_states[:, 0]]
    if actual_states is not None:
        all_x.append(actual_states[:, 0])
    if estimated_states is not None:
        all_x.append(estimated_states[:, 0])
    x_max_abs = max(np.abs(np.concatenate(all_x)).max(), abs(target[0])) * 1.15
    y_max = max(planned_states[:, 1].max(),
                actual_states[:, 1].max() if actual_states is not None else 0,
                estimated_states[:, 1].max() if estimated_states is not None else 0) * 1.1

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlim(-x_max_abs, x_max_abs)
    ax.set_ylim(-y_max * 0.05, y_max)
    ax.axvline(0, color='gray', linewidth=0.8)
    ax.plot(target[0], target[1], 'k^', markersize=12, label="Target", zorder=5)
    ax.set_xlabel("Downrange x (m)")
    ax.set_ylabel("Altitude y (m)")
    ax.set_title("Rocket Descent")
    ax.grid(True)

    planned_line, = ax.plot([], [], color=PLANNED_COLOR, linestyle=':', linewidth=2.5, label="Planned")
    planned_dot, = ax.plot([], [], 'o', color=PLANNED_COLOR, markersize=8)

    actual_line = actual_dot = None
    if actual_states is not None:
        actual_line, = ax.plot([], [], color=ACTUAL_COLOR, linestyle=':', linewidth=2.5, label="Actual")
        actual_dot, = ax.plot([], [], 'o', color=ACTUAL_COLOR, markersize=8)

    estimated_line = estimated_dot = None
    if estimated_states is not None:
        estimated_line, = ax.plot([], [], color=ESTIMATED_COLOR, linestyle=':', linewidth=2.5, label="Estimated")
        estimated_dot, = ax.plot([], [], 'o', color=ESTIMATED_COLOR, markersize=8)

    ax.legend(loc='upper right')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10,
                         bbox=dict(facecolor='white', alpha=0.7))

    def update(frame_num):
        idx = frame_indices[frame_num]

        p_idx = min(idx, len(planned_states) - 1)
        planned_line.set_data(planned_states[:p_idx + 1, 0], planned_states[:p_idx + 1, 1])
        planned_dot.set_data([planned_states[p_idx, 0]], [planned_states[p_idx, 1]])

        artists = [planned_line, planned_dot, time_text]

        if actual_states is not None:
            a_idx = min(idx, len(actual_states) - 1)
            actual_line.set_data(actual_states[:a_idx + 1, 0], actual_states[:a_idx + 1, 1])
            actual_dot.set_data([actual_states[a_idx, 0]], [actual_states[a_idx, 1]])
            artists += [actual_line, actual_dot]

        if estimated_states is not None:
            e_idx = min(idx, len(estimated_states) - 1)
            estimated_line.set_data(estimated_states[:e_idx + 1, 0], estimated_states[:e_idx + 1, 1])
            estimated_dot.set_data([estimated_states[e_idx, 0]], [estimated_states[e_idx, 1]])
            artists += [estimated_line, estimated_dot]

        time_text.set_text(f"t = {idx * dt:.1f}s")
        return artists

    anim = animation.FuncAnimation(fig, update, frames=len(frame_indices),
                                    interval=1000 / fps, blit=True)
    anim.save(save_path, writer='pillow', fps=fps)
    plt.close(fig)
    print(f"Saved trajectory animation to {save_path}")


def animate_dashboard(planned_states, planned_controls, dt, actual_states=None,
                       actual_controls=None, estimated_states=None,
                       target=(0.0, 0.0), speedup=2, fps=20, save_path="dashboard.gif"):

    has_actual = actual_states is not None
    has_estimated = estimated_states is not None

    n_frames = max(len(planned_states),
                    len(actual_states) if has_actual else 0,
                    len(estimated_states) if has_estimated else 0)
    frame_indices = list(range(0, n_frames, speedup))
    if frame_indices[-1] != n_frames - 1:
        frame_indices.append(n_frames - 1)

    planned_controls = np.asarray(planned_controls)
    planned_thrust_mag = np.hypot(planned_controls[:, 0], planned_controls[:, 1])
    if has_actual and actual_controls is not None:
        actual_controls = np.asarray(actual_controls)
        actual_thrust_mag = np.hypot(actual_controls[:, 0], actual_controls[:, 1])

    dist_planned = np.hypot(planned_states[:, 0] - target[0], planned_states[:, 1] - target[1])
    if has_actual:
        dist_actual = np.hypot(actual_states[:, 0] - target[0], actual_states[:, 1] - target[1])

    tp_full = np.arange(len(planned_states)) * dt
    t_max = tp_full[-1]

    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Rocket Landing GNC — Live Dashboard", fontsize=14)

    # --- Panel setup (static elements) ---
    all_x = [planned_states[:, 0]]
    if has_actual:
        all_x.append(actual_states[:, 0])
    x_max_abs = max(np.abs(np.concatenate(all_x)).max(), abs(target[0])) * 1.15
    axs[0, 0].set_xlim(-x_max_abs, x_max_abs)
    axs[0, 0].set_ylim(-50, planned_states[:, 1].max() * 1.1)
    axs[0, 0].plot(target[0], target[1], 'k^', markersize=10)
    axs[0, 0].set_title("Trajectory"); axs[0, 0].set_xlabel("x (m)"); axs[0, 0].set_ylabel("y (m)")
    axs[0, 0].grid(True)

    axs[0, 1].set_xlim(0, t_max); axs[0, 1].set_ylim(0, planned_states[:, 1].max() * 1.1)
    axs[0, 1].set_title("Altitude vs Time"); axs[0, 1].grid(True)

    v_all = [planned_states[:, 2], planned_states[:, 3]]
    if has_actual:
        v_all += [actual_states[:, 2], actual_states[:, 3]]
    v_max = np.abs(np.concatenate(v_all)).max() * 1.2
    axs[0, 2].set_xlim(0, t_max); axs[0, 2].set_ylim(-v_max, v_max)
    axs[0, 2].set_title("Velocity vs Time"); axs[0, 2].grid(True)

    thrust_max = planned_thrust_mag.max() * 1.2
    axs[1, 0].set_xlim(0, t_max); axs[1, 0].set_ylim(0, thrust_max)
    axs[1, 0].set_title("Thrust Magnitude vs Time"); axs[1, 0].grid(True)

    dist_max = dist_planned.max() * 1.2
    axs[1, 1].set_xlim(0, t_max); axs[1, 1].set_ylim(0, dist_max)
    axs[1, 1].set_title("Distance to Target vs Time"); axs[1, 1].grid(True)

    axs[1, 2].set_xlim(0, t_max)
    axs[1, 2].set_title("Tracking / Estimation Error vs Time"); axs[1, 2].grid(True)

    # --- Dynamic lines ---
    def make_lines(ax, label_p, label_a=None, label_e=None):
        lines = {}
        lines['p'], = ax.plot([], [], color=PLANNED_COLOR, linestyle=':', linewidth=2.5, label=label_p)
        if label_a:
            lines['a'], = ax.plot([], [], color=ACTUAL_COLOR, linestyle=':', linewidth=2.5, label=label_a)
        if label_e:
            lines['e'], = ax.plot([], [], color=ESTIMATED_COLOR, linestyle=':', linewidth=2.5, label=label_e)
        ax.legend(fontsize=7)
        return lines

    traj_lines = make_lines(axs[0, 0], "Planned", "Actual" if has_actual else None,
                             "Estimated" if has_estimated else None)
    alt_lines = make_lines(axs[0, 1], "Planned", "Actual" if has_actual else None,
                            "Estimated" if has_estimated else None)
    vx_line, = axs[0, 2].plot([], [], color=PLANNED_COLOR, linestyle=':', linewidth=1.5, label="Planned Vx")
    vy_line, = axs[0, 2].plot([], [], color=PLANNED_COLOR, linestyle=':', linewidth=2.5, label="Planned Vy")
    axs[0, 2].legend(fontsize=6)
    thrust_lines = make_lines(axs[1, 0], "Planned", "Actual" if has_actual else None)
    dist_lines = make_lines(axs[1, 1], "Planned", "Actual" if has_actual else None,
                             "Estimated" if has_estimated else None)
    err_line, = axs[1, 2].plot([], [], color=ACTUAL_COLOR, linestyle=':', linewidth=2.5)

    time_text = fig.text(0.02, 0.02, '', fontsize=10)

    def update(frame_num):
        idx = frame_indices[frame_num]
        p_idx = min(idx, len(planned_states) - 1)
        t_p = tp_full[:p_idx + 1]

        traj_lines['p'].set_data(planned_states[:p_idx + 1, 0], planned_states[:p_idx + 1, 1])
        alt_lines['p'].set_data(t_p, planned_states[:p_idx + 1, 1])
        vx_line.set_data(t_p, planned_states[:p_idx + 1, 2])
        vy_line.set_data(t_p, planned_states[:p_idx + 1, 3])
        thrust_lines['p'].set_data(t_p[:-1] if len(t_p) > 1 else t_p,
                                    planned_thrust_mag[:max(p_idx, 1)])
        dist_lines['p'].set_data(t_p, dist_planned[:p_idx + 1])

        if has_actual:
            a_idx = min(idx, len(actual_states) - 1)
            t_a = np.arange(a_idx + 1) * dt
            traj_lines['a'].set_data(actual_states[:a_idx + 1, 0], actual_states[:a_idx + 1, 1])
            alt_lines['a'].set_data(t_a, actual_states[:a_idx + 1, 1])
            thrust_lines['a'].set_data(t_a[:-1] if len(t_a) > 1 else t_a,
                                        actual_thrust_mag[:max(a_idx, 1)])
            dist_lines['a'].set_data(t_a, dist_actual[:a_idx + 1])

            n_err = min(p_idx, a_idx) + 1
            err = np.hypot(planned_states[:n_err, 0] - actual_states[:n_err, 0],
                            planned_states[:n_err, 1] - actual_states[:n_err, 1])
            err_line.set_data(tp_full[:n_err], err)
            if len(err) > 0:
                axs[1, 2].set_ylim(0, max(err.max() * 1.2, 1))

        if has_estimated:
            e_idx = min(idx, len(estimated_states) - 1)
            t_e = np.arange(e_idx + 1) * dt
            traj_lines['e'].set_data(estimated_states[:e_idx + 1, 0], estimated_states[:e_idx + 1, 1])
            alt_lines['e'].set_data(t_e, estimated_states[:e_idx + 1, 1])
            dist_est = np.hypot(estimated_states[:e_idx + 1, 0] - target[0],
                                 estimated_states[:e_idx + 1, 1] - target[1])
            dist_lines['e'].set_data(t_e, dist_est)

        time_text.set_text(f"t = {idx * dt:.1f}s")
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(frame_indices),
                                    interval=1000 / fps, blit=False)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    anim.save(save_path, writer='pillow', fps=fps)
    plt.close(fig)
    print(f"Saved dashboard animation to {save_path}")