# DC Motor Speed Control with MPC and a Kalman Filter

A simulation study of DC motor speed control when the sensors are noisy. A Kalman filter estimates the motor's states (speed ω and armature current iₐ) from noisy measurements, and a Model Predictive Controller (MPC) uses the estimates to regulate the speed. Three cases are simulated with exactly the same noise and compared:

| Case | What the MPC receives |
| --- | --- |
| True states | the exact states (ideal sensors), the reference result |
| Direct measurement | the noisy sensor readings as they are |
| Kalman filter | the filter's estimates x̂ₖ|ₖ |

All calculations are in `core/`; users only see the web front end.

## Run it

**Online:** upload this folder to a public GitHub repository and deploy it at [share.streamlit.io](https://share.streamlit.io) (Create app → repository, branch `main`, file `app.py` → Deploy).

**On your computer:**

```bash
cd kfmpc_lab
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py               # opens http://localhost:8501
```

## Model

    x = [ω, iₐ]ᵀ, u = Vₐ
    ẋ = A x + B u,  A = [[−B/J, Kₐ/J], [−K_b/L, −R/L]],  B = [0, 1/L]ᵀ
    x_{k+1} = A_d x_k + B_d u_k + w_k,   w_k ~ N(0, Q)   (process noise)
    y_k     = x_k + v_k,                 v_k ~ N(0, R)   (measurement noise, both states measured)

## Kalman filter

    Prediction:  x̂_{k|k−1} = A_d x̂_{k−1|k−1} + B_d u_{k−1},    P_{k|k−1} = A_d P_{k−1|k−1} A_dᵀ + Q
    Update:      K_k = P_{k|k−1} Cᵀ (C P_{k|k−1} Cᵀ + R)⁻¹
                 x̂_{k|k} = x̂_{k|k−1} + K_k (y_k − C x̂_{k|k−1}),    P_{k|k} = (I − K_k C) P_{k|k−1}

## MPC

Same formulation as Lab 7: minimise Σ q(ŷ − ω*)² + Σ r Δû² over N steps subject to the prediction model, −V_max ≤ û ≤ V_max and ω_min ≤ ŷ ≤ ω_max, solved as a quadratic program (OSQP). The initial state of each prediction is the true state, the measurement, or the Kalman estimate.

## The app

| Tab | Contents |
| --- | --- |
| 📘 Overview | What, why and how; block diagram; the three cases; terms |
| ⚙️ Model | Continuous and discrete state-space model with the current numbers; noise model Q and R |
| 🧮 Kalman filter | Filter equations; current Q, R, x̂₀, P₀; steady-state gain; estimation error with the filter's ±2σ band; gain and √P over time; consistency check |
| 🔮 MPC | MPC formulation, current settings, stability |
| 📈 Estimation & control | Actual, measured and estimated speed and current; reference; applied voltage; the three cases side by side |
| 📊 Comparison | Estimation accuracy (raw sensor vs Kalman); control performance of the three cases; observations; conclusion |
| 🔬 Noise study | Sweeps of measurement noise, process noise and filter tuning (R scale) |

Sidebar: motor parameters, measurement and process noise (and random seed), Kalman filter tuning (Q and R scales, initial estimate, P₀), MPC settings (N, Tₛ, q, r, V_max, ω_min, ω_max), target speed and simulation time.

## Results with the default settings

| | Speed error RMS (rad/s) | Current error RMS (A) | Speed ripple RMS (rad/s) | Voltage jitter std (V) | Settling ±2 % (s) |
| --- | --- | --- | --- | --- | --- |
| Raw sensor / direct measurement | 0.0519 | 0.541 | 0.0137 | 2.50 | never settles |
| Kalman filter | 0.0091 | 0.119 | 0.0068 | 0.35 | 0.41 |
| True states (ideal) | 0 | 0 | 0.0045 | 0.23 | 0.49 |

## Files

```
app.py              web front end
requirements.txt    Python packages
assets/             block diagram
core/config.py      every setting with default, range and help text
core/model.py       state-space model, exact discretisation, noise covariances
core/kalman.py      Kalman filter (prediction and update) and its steady state
core/mpc.py         MPC quadratic program (OSQP)
core/simulate.py    closed loop: noisy motor, sensors, filter, MPC; three cases
core/metrics.py     estimation accuracy and control performance
core/study.py       noise study
core/plots.py       charts
```
