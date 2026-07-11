#Convex-optimization trajectory guidance for powered descent.

import numpy as np
import cvxpy as cp

from config import parameters as prm


class GuidanceOptimizer:
    def __init__(self, N, dt):
        self.N = N
        self.dt = dt
        self.g = prm.G
        self.isp = prm.ISP
        self.g0 = prm.G0
        self.dry_mass = prm.DRY_MASS
        self.max_thrust = prm.MAX_THRUST
        self.min_thrust = prm.MIN_THRUST

    def _initial_mass_guess(self, m0):

        rough_burn_rate = self.max_thrust / (self.isp * self.g0)
        m_end_guess = max(m0 - rough_burn_rate * self.N * self.dt, self.dry_mass)
        return np.linspace(m0, m_end_guess, self.N + 1)

    def _build_and_solve(self, initial_state, target_state, m_nominal):

        N, dt = self.N, self.dt

        x = cp.Variable(N + 1)
        y = cp.Variable(N + 1)
        vx = cp.Variable(N + 1)
        vy = cp.Variable(N + 1)
        m = cp.Variable(N + 1)
        Tx = cp.Variable(N)
        Ty = cp.Variable(N)
        Gamma = cp.Variable(N)  # thrust-magnitude slack (lossless convexification)

        x0, y0, vx0, vy0, m0 = initial_state
        xt, yt, vxt, vyt = target_state

        constraints = []

        # --- Initial conditions ---
        constraints += [x[0] == x0, y[0] == y0, vx[0] == vx0, vy[0] == vy0, m[0] == m0]

        # --- Terminal conditions (soft landing) ---
        constraints += [x[N] == xt, y[N] == yt, vx[N] == vxt, vy[N] == vyt]

        # --- Dynamics (zero-order hold, Euler discretization) ---
        for k in range(N):
            constraints += [
                x[k + 1] == x[k] + vx[k] * dt,
                y[k + 1] == y[k] + vy[k] * dt,
                vx[k + 1] == vx[k] + (Tx[k] / m_nominal[k]) * dt,
                vy[k + 1] == vy[k] + (Ty[k] / m_nominal[k] - self.g) * dt,
                m[k + 1] == m[k] - (Gamma[k] / (self.isp * self.g0)) * dt,
            ]

        # --- Ground constraint: never below pad ---
        constraints += [y >= 0]

        # --- Fuel floor: mass never below dry mass ---
        constraints += [m >= self.dry_mass]

        # --- Thrust bounds via lossless convexification ---
        for k in range(N):
            constraints += [
                cp.norm(cp.vstack([Tx[k], Ty[k]])) <= Gamma[k],  # SOC constraint
                Gamma[k] >= self.min_thrust,
                Gamma[k] <= self.max_thrust,
            ]

        # --- Objective: minimize fuel used == maximize final mass ---
        objective = cp.Maximize(m[N])

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL,
                      max_iter=50000,
                      tol_gap_abs=1e-6,
                      tol_gap_rel=1e-6,
                      tol_feas=1e-6,)

        return {
            "status": problem.status,
            "x": x.value, "y": y.value,
            "vx": vx.value, "vy": vy.value,
            "m": m.value,
            "Tx": Tx.value, "Ty": Ty.value,
            "Gamma": Gamma.value,
        }

    def solve(self, initial_state, target_state, max_iters=3, tol=1e-2):

        m0 = initial_state[4]
        m_nominal = self._initial_mass_guess(m0)

        result = None
        for iteration in range(max_iters):
            result = self._build_and_solve(initial_state, target_state, m_nominal)

            if result["status"] != "optimal":
                print(f"[guidance] iteration {iteration}: solver status = {result['status']}")
                break

            new_m_nominal = result["m"]
            delta = np.max(np.abs(new_m_nominal - m_nominal))
            m_nominal = new_m_nominal

            print(f"[guidance] iteration {iteration}: max mass update = {delta:.4f} kg")
            if delta < tol:
                break

        return result

    def to_control_profile(self, result):
        
        #The Control Profile for the optimised Path
        return np.column_stack([result["Tx"], result["Ty"]])

    def to_state_trajectory(self, result):
        
        #The Optimised Trajectory 
        return np.column_stack([result["x"], result["y"], result["vx"], result["vy"], result["m"]])