"""Closed-loop simulation: noisy motor + sensors, (optional) Kalman filter, MPC.

Every sample k (period Ts):
    1. sensors:      y_k = x_k + v_k                         (noisy speed and current)
    2. estimate:     true case   -> x̂ = x_k  (ideal sensors, reference case)
                     direct case -> x̂ = y_k  (noisy measurement used as it is)
                     kalman case -> x̂ = x̂_{k|k} from the Kalman filter
    3. MPC:          u_k = u_{k−1} + Δu*(x̂, u_{k−1}, ω*),   |u_k| ≤ V_max
    4. motor:        x_{k+1} = A_d x_k + B_d u_k + w_k        (process noise)
    5. filter:       prediction x̂_{k+1|k} with u_k
The three cases use exactly the same noise sequences (same seed), so they can be compared fairly.
"""

import math
import time

import numpy as np

from .config import CASES
from .kalman import KalmanFilter
from .model import C, discrete, noise_covariances
from .mpc import MPC, mpc_params


def noise_sequences(cfg, n):
    rng = np.random.default_rng(int(cfg["seed"]))
    W = rng.standard_normal((n + 1, 2)) * [float(cfg["proc_w"]), float(cfg["proc_i"])]
    V = rng.standard_normal((n + 1, 2)) * [float(cfg["meas_w"]), float(cfg["meas_i"])]
    return W, V


def simulate(cfg, case):
    p = mpc_params(cfg)
    Ts, T, ref = p["Ts"], float(cfg["t_end"]), float(cfg["ref"])
    n = int(math.ceil(T / Ts))
    Ad, Bd = discrete(cfg, Ts)
    mpc = MPC(Ad, Bd, p)
    Q, R = noise_covariances(cfg)
    kf = KalmanFilter(Ad, Bd, C, float(cfg["kf_q_scale"]) * Q, float(cfg["kf_r_scale"]) * R,
                      x0=[float(cfg["kf_x0_w"]), 0.0], P0=float(cfg["kf_p0"]) * np.eye(2))
    W, V = noise_sequences(cfg, n)

    t = np.arange(n + 1) * Ts
    X = np.zeros((n + 1, 2))          # true states
    Y = np.zeros((n + 1, 2))          # measurements
    Xh = np.full((n + 1, 2), np.nan)  # state given to the MPC (estimate)
    Pd = np.full((n + 1, 2), np.nan)  # diagonal of P_{k|k}
    Kg = np.full((n + 1, 4), np.nan)  # Kalman gain entries
    U = np.zeros(n + 1)
    x = np.zeros(2)
    u_prev = 0.0
    t0 = time.perf_counter()
    for k in range(n + 1):
        y = C @ x + V[k]
        if case == "kalman":
            if k > 0:
                kf.predict(u_prev)
            xh = kf.update(y).copy()
            Pd[k], Kg[k] = np.diag(kf.P), kf.K.ravel()
        elif case == "direct":
            xh = y.copy()
        else:
            xh = x.copy()
        u = float(np.clip(u_prev + mpc.solve(xh, u_prev, ref), -p["Vmax"], p["Vmax"]))
        X[k], Y[k], Xh[k], U[k] = x, y, xh, u
        x = Ad @ x + Bd * u + W[k]
        u_prev = u
    return dict(case=case, t=t, x=X, y=Y, xh=Xh, P=Pd, K=Kg, u=U, ref=ref, Ts=Ts,
                solve_ms=1000 * (time.perf_counter() - t0) / (n + 1))


def simulate_all(cfg):
    return {c: simulate(cfg, c) for c in CASES}
