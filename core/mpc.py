"""Model Predictive Control of the motor speed (same formulation as Lab 7).

Prediction model x = [ω, iₐ], decision variables Δu = u_k − u_{k−1}, augmented state z = [ω, iₐ, u_{k−1}]:
    Ŷ = Φ z_k + Γ ΔU,     Û = 1·u_{k−1} + S ΔU
    min  q‖Ŷ − ω*·1‖² + r‖ΔU‖²
    s.t. −V_max ≤ Û ≤ V_max,   ω_min ≤ Ŷ ≤ ω_max  (soft, slack ε with a large weight)
Only the first move is applied (receding horizon). The state z_k fed to the MPC is either the true state,
the noisy measurement, or the Kalman-filter estimate.
"""

import numpy as np
import osqp
from scipy import sparse


def mpc_params(cfg):
    return dict(Ts=float(cfg["mpc_Ts"]), N=int(cfg["mpc_N"]), q=float(cfg["mpc_q"]), r=float(cfg["mpc_r"]),
                Vmax=float(cfg["mpc_Vmax"]), wmin=float(cfg["mpc_wmin"]), wmax=float(cfg["mpc_wmax"]))


def prediction_matrices(Ad, Bd, N):
    At = np.zeros((3, 3))
    At[:2, :2], At[:2, 2], At[2, 2] = Ad, Bd, 1.0
    Bt = np.array([Bd[0], Bd[1], 1.0])
    Ct = np.array([1.0, 0.0, 0.0])
    Phi = np.zeros((N, 3))
    h = np.zeros(N)
    Ap = np.eye(3)
    for j in range(N):
        h[j] = Ct @ Ap @ Bt
        Ap = Ap @ At
        Phi[j] = Ct @ Ap
    Gam = np.zeros((N, N))
    for j in range(N):
        Gam[j, :j + 1] = h[j::-1]
    return At, Bt, Phi, Gam, np.tril(np.ones((N, N)))


class MPC:
    def __init__(self, Ad, Bd, p):
        self.p = p
        N, q, r = p["N"], p["q"], p["r"]
        self.At, self.Bt, self.Phi, self.Gam, self.S = prediction_matrices(Ad, Bd, N)
        M = np.linalg.solve(q * self.Gam.T @ self.Gam + r * np.eye(N), q * self.Gam.T)
        self.K, self.Kr = M[0] @ self.Phi, M[0] @ np.ones(N)          # unconstrained law Δu = −Kz + K_r ω*
        self.poles = np.linalg.eigvals(self.At - np.outer(self.Bt, self.K))
        self.rho = 1e3 * max(q, 1e-6)
        H = 2 * (q * self.Gam.T @ self.Gam + r * np.eye(N))
        P = np.zeros((N + 1, N + 1))
        P[:N, :N], P[N, N] = H, 2 * self.rho
        A = np.zeros((3 * N + 1, N + 1))
        A[:N, :N] = self.S
        A[N:2 * N, :N], A[N:2 * N, N] = self.Gam, -1
        A[2 * N:3 * N, :N], A[2 * N:3 * N, N] = self.Gam, 1
        A[3 * N, N] = 1
        self.solver = osqp.OSQP()
        self.solver.setup(P=sparse.triu(sparse.csc_matrix(P)).tocsc(), q=np.zeros(N + 1), A=sparse.csc_matrix(A),
                          l=-np.inf * np.ones(3 * N + 1), u=np.inf * np.ones(3 * N + 1), verbose=False,
                          eps_abs=1e-6, eps_rel=1e-6, polishing=True, max_iter=20000, check_termination=10,
                          warm_starting=True)

    def solve(self, x, u_prev, ref):
        p, N = self.p, self.p["N"]
        free = self.Phi @ np.array([x[0], x[1], u_prev])
        f = np.zeros(N + 1)
        f[:N] = 2 * p["q"] * self.Gam.T @ (free - ref)
        f[N] = 10 * self.rho
        inf = np.inf * np.ones(N)
        lo = np.concatenate([-p["Vmax"] - u_prev * np.ones(N), -inf, p["wmin"] - free, [0.0]])
        hi = np.concatenate([p["Vmax"] - u_prev * np.ones(N), p["wmax"] - free, inf, [np.inf]])
        self.solver.update(q=f, l=lo, u=hi)
        sol = self.solver.solve().x
        if sol is None or not np.all(np.isfinite(sol)):
            return 0.0
        return float(sol[0])
