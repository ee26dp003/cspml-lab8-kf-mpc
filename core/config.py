"""Every tunable value: key, label, default, min, max, step, unit, help text.
The sidebar is built from these lists."""

MOTOR = [
    ("R", "Armature resistance R", 1.0, 0.01, 50.0, 0.1, "Ω", "Resistance of the armature winding."),
    ("L", "Armature inductance L", 0.5, 0.0001, 10.0, 0.01, "H", "Sets how fast the current can change (L/R)."),
    ("J", "Rotor inertia J", 0.01, 0.00001, 10.0, 0.001, "kg·m²", "Resists changes in speed."),
    ("B", "Viscous friction B", 0.1, 0.0, 10.0, 0.01, "N·m·s/rad", "Friction torque per unit speed."),
    ("Ka", "Torque constant Kₐ", 0.01, 0.0001, 10.0, 0.001, "N·m/A", "Torque per ampere: T = Kₐ·iₐ."),
    ("Kb", "Back-EMF constant K_b", 0.01, 0.0001, 10.0, 0.001, "V·s/rad", "Back-EMF per unit speed: e_b = K_b·ω."),
]

NOISE = [
    ("meas_w", "Speed-sensor noise σ_v,ω", 0.05, 0.0, 10.0, 0.01, "rad/s",
     "Standard deviation of the noise on each speed measurement (encoder / tachometer)."),
    ("meas_i", "Current-sensor noise σ_v,i", 0.5, 0.0, 100.0, 0.1, "A",
     "Standard deviation of the noise on each current measurement."),
    ("proc_w", "Process noise on speed σ_w,ω", 0.002, 0.0, 10.0, 0.001, "rad/s",
     "Random change of the true speed per sample (load-torque fluctuations, unmodelled friction)."),
    ("proc_i", "Process noise on current σ_w,i", 0.02, 0.0, 100.0, 0.01, "A",
     "Random change of the true current per sample (supply ripple, unmodelled electrical effects)."),
    ("seed", "Random seed", 1, 0, 100000, 1, "", "Same seed = same noise, so runs can be compared fairly."),
]

KALMAN = [
    ("kf_q_scale", "Kalman Q scale", 1.0, 1e-4, 1e4, 0.1, "×",
     "The filter uses Q = scale × (true process-noise covariance). 1 = filter tuned to the true noise."),
    ("kf_r_scale", "Kalman R scale", 1.0, 1e-4, 1e4, 0.1, "×",
     "The filter uses R = scale × (true measurement-noise covariance). 1 = filter tuned to the true noise."),
    ("kf_x0_w", "Initial speed estimate ω̂₀", 0.0, -100.0, 100.0, 0.1, "rad/s",
     "Starting guess of the filter (the true motor starts at rest). Try a wrong guess to see the filter converge."),
    ("kf_p0", "Initial uncertainty P₀", 1.0, 1e-6, 1e4, 0.1, "",
     "Initial error covariance P₀ = value × I. Large = 'I don't trust my initial guess'."),
]

MPC = [
    ("mpc_N", "Prediction horizon N", 30, 1, 200, 1, "steps", "Number of future samples predicted and optimised."),
    ("mpc_Ts", "Sample time Ts", 0.01, 0.001, 0.2, 0.001, "s",
     "How often the sensors are read, the filter runs and the MPC re-plans."),
    ("mpc_q", "Tracking weight q", 1.0, 0.0001, 1e6, 0.1, "", "Penalty on (predicted speed − reference)²."),
    ("mpc_r", "Input-change weight r", 0.0005, 1e-8, 1e3, 0.0001, "", "Penalty on (change of voltage)²."),
    ("mpc_Vmax", "Voltage limit Vmax", 24.0, 0.1, 1000.0, 1.0, "V", "The MPC keeps −Vmax ≤ Vₐ ≤ Vmax."),
    ("mpc_wmin", "Minimum speed ω_min", 0.0, -1000.0, 1000.0, 0.1, "rad/s", "Lower limit on the predicted speed."),
    ("mpc_wmax", "Maximum speed ω_max", 1.1, -1000.0, 1000.0, 0.01, "rad/s",
     "Upper limit on the predicted speed (set a little above the target speed)."),
]

TEST = [
    ("ref", "Target speed ω*", 1.0, 0.1, 1000.0, 0.1, "rad/s", "Reference speed (a step at t = 0)."),
    ("t_end", "Simulation time", 2.0, 0.2, 20.0, 0.1, "s", "Length of each simulated test."),
]

GROUPS = [("Motor parameters", MOTOR), ("Noise", NOISE), ("Kalman filter", KALMAN),
          ("MPC controller", MPC), ("Target", TEST)]

CASES = ["true", "direct", "kalman"]
CASE_NAMES = {"true": "True states (ideal sensors)", "direct": "Direct noisy measurements",
              "kalman": "Kalman filter estimates"}
CASE_SHORT = {"true": "True states", "direct": "Direct measurement", "kalman": "Kalman filter"}


def default_config():
    cfg = {}
    for _, group in GROUPS:
        cfg.update({k: d for k, _, d, *_ in group})
    return cfg
