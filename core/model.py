"""DC motor state-space model (speed control, two states).

    L·diₐ/dt = Vₐ − R·iₐ − K_b·ω            (armature circuit)
    J·dω/dt  = Kₐ·iₐ − B·ω                  (rotor)

    x = [ω, iₐ]ᵀ,  u = Vₐ
    ẋ = A x + B u,   A = [[−B/J, Kₐ/J], [−K_b/L, −R/L]],   B = [0, 1/L]ᵀ

Discrete time with sample time Ts (zero-order hold, exact) and additive noise:
    x_{k+1} = A_d x_k + B_d u_k + w_k,     w_k ~ N(0, Q)    (process noise)
    y_k     = C x_k + v_k,                 v_k ~ N(0, R)    (measurement noise)
Both states are measured (speed sensor and current sensor), so C = I.
"""

import numpy as np
from scipy.linalg import expm


def motor(cfg):
    return {k: float(cfg[k]) for k in ("R", "L", "J", "B", "Ka", "Kb")}


def continuous(cfg):
    p = motor(cfg)
    A = np.array([[-p["B"] / p["J"], p["Ka"] / p["J"]],
                  [-p["Kb"] / p["L"], -p["R"] / p["L"]]])
    B = np.array([0.0, 1.0 / p["L"]])
    return A, B


def discrete(cfg, Ts):
    """Exact zero-order-hold discretisation: A_d = e^{A Ts}, B_d = ∫ e^{Aτ} dτ B."""
    A, B = continuous(cfg)
    M = np.zeros((3, 3))
    M[:2, :2], M[:2, 2] = A, B
    Md = expm(M * Ts)
    return Md[:2, :2], Md[:2, 2]


C = np.eye(2)                       # both ω and iₐ are measured


def noise_covariances(cfg):
    """True process and measurement noise covariances (diagonal) from the standard deviations."""
    Q = np.diag([float(cfg["proc_w"]) ** 2, float(cfg["proc_i"]) ** 2])
    R = np.diag([float(cfg["meas_w"]) ** 2, float(cfg["meas_i"]) ** 2])
    return Q, R


def motor_summary(cfg):
    p = motor(cfg)
    A, B = continuous(cfg)
    a1 = p["R"] * p["B"] + p["Ka"] * p["Kb"]
    return dict(A=A, B=B, poles=np.linalg.eigvals(A), dc_gain=p["Ka"] / a1,
                tau_e=p["L"] / p["R"], tau_m=p["R"] * p["J"] / a1,
                hold_v=a1 / p["Ka"], hold_i=p["B"] / p["Ka"])
