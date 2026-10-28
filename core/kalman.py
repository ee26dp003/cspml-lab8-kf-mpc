"""Discrete-time Kalman filter for the motor states x = [ω, iₐ].

Prediction (uses the model and the input that was applied):
    x̂_{k|k−1} = A_d x̂_{k−1|k−1} + B_d u_{k−1}
    P_{k|k−1} = A_d P_{k−1|k−1} A_dᵀ + Q

Update / correction (uses the new noisy measurement y_k):
    K_k       = P_{k|k−1} Cᵀ (C P_{k|k−1} Cᵀ + R)⁻¹
    x̂_{k|k}  = x̂_{k|k−1} + K_k (y_k − C x̂_{k|k−1})        (y_k − C x̂_{k|k−1} = innovation)
    P_{k|k}   = (I − K_k C) P_{k|k−1}

P is the covariance of the estimation error e_k = x_k − x̂_{k|k}.
"""

import numpy as np
from scipy.linalg import solve_discrete_are


class KalmanFilter:
    def __init__(self, Ad, Bd, C, Q, R, x0, P0):
        self.Ad, self.Bd, self.C = Ad, Bd, C
        self.Q = Q
        self.R = R + 1e-12 * np.eye(R.shape[0])     # keeps (C P Cᵀ + R) invertible if a sensor is noise-free
        self.x = np.array(x0, float)                # x̂_{k|k}
        self.P = np.array(P0, float)                # P_{k|k}
        self.K = np.zeros((Ad.shape[0], C.shape[0]))
        self.innovation = np.zeros(C.shape[0])

    def predict(self, u):
        self.x = self.Ad @ self.x + self.Bd * u
        self.P = self.Ad @ self.P @ self.Ad.T + self.Q
        return self.x

    def update(self, y):
        S = self.C @ self.P @ self.C.T + self.R
        self.K = self.P @ self.C.T @ np.linalg.inv(S)
        self.innovation = y - self.C @ self.x
        self.x = self.x + self.K @ self.innovation
        I = np.eye(self.P.shape[0])
        self.P = (I - self.K @ self.C) @ self.P
        return self.x


def steady_state(Ad, C, Q, R):
    """Steady-state prior covariance and gain (the values the filter settles to)."""
    Rr = R + 1e-12 * np.eye(R.shape[0])
    try:
        Pp = solve_discrete_are(Ad.T, C.T, Q + 1e-14 * np.eye(Q.shape[0]), Rr)
    except (np.linalg.LinAlgError, ValueError):
        return None, None, None
    K = Pp @ C.T @ np.linalg.inv(C @ Pp @ C.T + Rr)
    Pu = (np.eye(Ad.shape[0]) - K @ C) @ Pp
    return Pp, K, Pu
