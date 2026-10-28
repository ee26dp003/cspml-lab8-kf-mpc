"""
DC Motor Speed Control with MPC and a Kalman Filter — web front end.     Run with:   streamlit run app.py

All calculations live in core/. This file only collects settings and shows results.
"""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from core import plots  # noqa: E402
from core.config import CASE_NAMES, CASE_SHORT, CASES, GROUPS, default_config  # noqa: E402
from core.kalman import steady_state  # noqa: E402
from core.metrics import control_metrics, estimation_metrics  # noqa: E402
from core.model import C, discrete, motor_summary, noise_covariances  # noqa: E402
from core.mpc import MPC, mpc_params  # noqa: E402
from core.simulate import simulate_all  # noqa: E402
from core.study import run_study  # noqa: E402

st.set_page_config(page_title="DC Motor: MPC + Kalman Filter", page_icon="⚙️", layout="wide")

# ------------------------------------------------------------------ settings
if "cfg_R" not in st.session_state or st.session_state.pop("reset", False):
    for k, v in default_config().items():
        st.session_state["cfg_" + k] = v
    st.session_state["study"] = None


def num_input(key, label, lo, hi, step, unit, help_):
    lab = f"{label} [{unit}]" if unit else label
    if isinstance(step, int) and isinstance(lo, int):
        st.number_input(lab, min_value=lo, max_value=hi, step=step, key="cfg_" + key, help=help_)
    else:
        st.number_input(lab, min_value=float(lo), max_value=float(hi), step=float(step), key="cfg_" + key,
                        help=help_, format="%g")


with st.sidebar:
    st.title("⚙️ MPC + Kalman Filter")
    st.caption("Change any value; every result updates at once. Hover ⓘ for what a value means.")
    for name, group in GROUPS:
        with st.expander(name, expanded=name in ("Noise", "Target")):
            for k, label, _, lo, hi, step, unit, h in group:
                num_input(k, label, lo, hi, step, unit, h)
    if st.button("Reset to default values", width="stretch"):
        st.session_state["reset"] = True
        st.rerun()

cfg = {k: st.session_state["cfg_" + k] for k in default_config()}


@st.cache_data(show_spinner="Simulating the three cases…", max_entries=40)
def run_all(items):
    c = dict(items)
    sims = simulate_all(c)
    return sims, {k: estimation_metrics(s) for k, s in sims.items()}, {k: control_metrics(s) for k, s in sims.items()}


sims, est, ctl = run_all(tuple(sorted(cfg.items())))
p = mpc_params(cfg)
Ad, Bd = discrete(cfg, p["Ts"])
Qt, Rt = noise_covariances(cfg)
Qkf, Rkf = cfg["kf_q_scale"] * Qt, cfg["kf_r_scale"] * Rt
Pp_ss, K_ss, Pu_ss = steady_state(Ad, C, Qkf, Rkf)
MS = motor_summary(cfg)


def bm(X, fmt="{:.4g}"):
    X = np.atleast_2d(X)
    return r"\begin{bmatrix}" + r"\\".join("&".join(fmt.format(v) for v in row) for row in X) + r"\end{bmatrix}"


def f(v, d=3):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


# ------------------------------------------------------------------ header
st.title("DC motor speed control with MPC and a Kalman filter")
st.caption("The motor's speed and current sensors are noisy. A Kalman filter estimates the true states, and the "
           "MPC uses those estimates to control the speed.")
def _metric(col, label, value, sub):
    try:
        col.metric(label, value, sub, delta_color="off", delta_arrow="off")
    except TypeError:
        col.metric(label, value, sub, delta_color="off")


c1, c2, c3, c4 = st.columns(4)
_metric(c1, "Speed error: raw sensor → Kalman", f"{est['kalman']['meas_rmse_w']:.4f} → {est['kalman']['rmse_w']:.4f}",
          f"{est['kalman']['reduction_w']:.0f} % smaller (RMS, rad/s)" if np.isfinite(est['kalman']['reduction_w'])
          else "no measurement noise")
_metric(c2, "Current error: raw sensor → Kalman", f"{est['kalman']['meas_rmse_i']:.3f} → {est['kalman']['rmse_i']:.3f}",
          f"{est['kalman']['reduction_i']:.0f} % smaller (RMS, A)" if np.isfinite(est['kalman']['reduction_i'])
          else "no measurement noise")
_metric(c3, "Speed ripple, direct → Kalman", f"{ctl['direct']['track_rms']:.4f} → {ctl['kalman']['track_rms']:.4f}",
          "RMS of ω − ω* in steady state (rad/s)")
_metric(c4, "Voltage jitter, direct → Kalman", f"{ctl['direct']['u_std_ss']:.2f} → {ctl['kalman']['u_std_ss']:.2f} V",
          "std of Vₐ in steady state")

tabs = st.tabs(["📘 Overview", "⚙️ Model", "🧮 Kalman filter", "🔮 MPC", "📈 Estimation & control", "📊 Comparison",
                "🔬 Noise study"])

# ================================================================== overview
with tabs[0]:
    st.subheader("What this lab does")
    st.markdown("""
A DC motor must run at a target speed ω\\*. The **MPC** (Model Predictive Control) needs the motor's states, the
speed ω and the armature current iₐ, to plan its voltages. Real sensors are **noisy**, and the motor itself is
disturbed by small random effects (**process noise**). Feeding noisy measurements straight into the controller
makes the voltage jitter and the speed ripple.

A **Kalman filter** combines the motor model with the noisy measurements to produce a much better estimate of the
true states. This lab simulates three cases with exactly the same noise and compares them:
""")
    st.markdown("""
| Case | What the MPC receives | Why it is included |
|---|---|---|
| **True states** | the exact ω and iₐ (ideal, noise-free sensors) | best possible result, the reference |
| **Direct measurement** | the noisy sensor readings as they are | what happens without a filter |
| **Kalman filter** | the filter's estimates x̂ₖ|ₖ | the method studied in this lab |
""")
    st.image(os.path.join(HERE, "assets", "loop_kf_mpc.png"), width="stretch",
             caption="The control loop. The motor is disturbed by process noise, the sensors add measurement noise, "
                     "the Kalman filter estimates the states, and the MPC uses the estimates (or, for comparison, the "
                     "raw measurements).")
    st.markdown("""
**How to use this app**
1. In the sidebar, set the **noise** levels, the **Kalman filter** tuning, the **MPC** settings, the **motor**
   parameters and the **target speed**.
2. **📈 Estimation & control** shows the actual, measured and estimated speed and current, the reference and the
   applied voltage.
3. **📊 Comparison** gives the estimation accuracy and the control performance of the three cases.
4. **🔬 Noise study** repeats the test for many noise levels and filter tunings.

| Term | Meaning |
|---|---|
| Process noise wₖ | Random disturbance of the motor itself (load-torque or supply fluctuations). Covariance Q. |
| Measurement noise vₖ | Random error added by the sensors. Covariance R. |
| Estimate x̂ₖ|ₖ | The filter's best guess of the true state at sample k, using all measurements up to k. |
| P | Covariance of the estimation error: how uncertain the filter thinks it is. |
| Innovation | yₖ − C x̂ₖ|ₖ₋₁: the difference between the new measurement and what the model predicted. |
| RMS error | Root-mean-square difference from the true value. |
""")

# ================================================================== model
with tabs[1]:
    st.subheader("State-space model of the DC motor")
    st.markdown("From the armature circuit (Kirchhoff) and the rotor (Newton), with states x = [ω, iₐ]ᵀ and input "
                "u = Vₐ:")
    st.latex(r"J\dot\omega=K_ai_a-B\omega,\qquad L\dot i_a=V_a-Ri_a-K_b\omega")
    st.latex(r"\dot x=\underbrace{\begin{bmatrix}-\frac{B}{J}&\frac{K_a}{J}\\-\frac{K_b}{L}&-\frac{R}{L}\end{bmatrix}}_{A}x"
             r"+\underbrace{\begin{bmatrix}0\\\frac{1}{L}\end{bmatrix}}_{B}u,\qquad y=\underbrace{\begin{bmatrix}1&0\\0&1"
             r"\end{bmatrix}}_{C}x\quad(\text{both }\omega\text{ and }i_a\text{ are measured})")
    st.latex(r"A=" + bm(MS["A"]) + r",\qquad B=" + bm(MS["B"].reshape(-1, 1)))
    st.subheader("Discrete-time model with noise")
    st.markdown(f"Discretised exactly (zero-order hold) with the sample time Tₛ = {p['Ts']:g} s, and with the two noise "
                "sources added:")
    st.latex(r"x_{k+1}=A_dx_k+B_du_k+w_k,\quad w_k\sim\mathcal N(0,Q);\qquad y_k=Cx_k+v_k,\quad v_k\sim\mathcal N(0,R)")
    st.latex(r"A_d=e^{AT_s}=" + bm(Ad, "{:.5g}") + r",\qquad B_d=\int_0^{T_s}e^{A\tau}d\tau\,B=" + bm(Bd.reshape(-1, 1), "{:.5g}"))
    st.latex(r"Q=" + bm(Qt, "{:.3g}") + r"\ (\text{process}),\qquad R=" + bm(Rt, "{:.3g}") + r"\ (\text{measurement})")
    st.caption("Q and R are diagonal: their entries are the squares of the noise standard deviations set in the sidebar.")
    st.dataframe(pd.DataFrame([
        ("Motor poles", ", ".join(f"{z.real:.3f}" for z in MS["poles"]), "both negative: the motor alone is stable"),
        ("DC gain", f"{MS['dc_gain']:.4g} rad/s per V", "steady speed for 1 V"),
        ("Electrical time constant L/R", f"{MS['tau_e']:.3g} s", ""),
        ("Mechanical time constant", f"{MS['tau_m']:.3g} s", ""),
        ("Voltage / current to hold 1 rad/s", f"{MS['hold_v']:.3g} V / {MS['hold_i']:.3g} A", ""),
    ], columns=["Quantity", "Value", "Note"]), hide_index=True, width="stretch")

# ================================================================== kalman
with tabs[2]:
    st.subheader("Kalman filter formulation")
    st.markdown("The filter keeps an estimate x̂ and its error covariance P = E[eₖeₖᵀ], with eₖ = xₖ − x̂ₖ|ₖ. Each "
                "sample has two steps.")
    k1, k2 = st.columns(2)
    with k1:
        st.markdown("**Prediction** (model only)")
        st.latex(r"\hat x_{k|k-1}=A_d\,\hat x_{k-1|k-1}+B_d\,u_{k-1}")
        st.latex(r"P_{k|k-1}=A_d\,P_{k-1|k-1}\,A_d^T+Q")
    with k2:
        st.markdown("**Update** (correct with the new measurement)")
        st.latex(r"K_k=P_{k|k-1}C^T\left(CP_{k|k-1}C^T+R\right)^{-1}")
        st.latex(r"\hat x_{k|k}=\hat x_{k|k-1}+K_k\big(\underbrace{y_k-C\hat x_{k|k-1}}_{\text{innovation}}\big)")
        st.latex(r"P_{k|k}=(I-K_kC)\,P_{k|k-1}")
    st.markdown("The gain Kₖ balances the two sources of information: if the sensors are noisy (large R) the filter "
                "trusts the model more; if the motor is strongly disturbed (large Q) it trusts the measurements more. "
                "With noise-free sensors (no-noise case) the states can be used directly; with noise, the estimated "
                "states x̂ₖ|ₖ are used to compute the control.")
    st.subheader("Current filter settings")
    st.latex(r"Q_{KF}=" + bm(Qkf, "{:.3g}") + r",\quad R_{KF}=" + bm(Rkf, "{:.3g}") + r",\quad \hat x_{0}="
             + bm(np.array([[cfg["kf_x0_w"]], [0.0]]), "{:g}") + r",\quad P_0=" + f"{cfg['kf_p0']:g}" + r"\,I")
    if cfg["kf_q_scale"] != 1 or cfg["kf_r_scale"] != 1:
        st.info(f"The filter is deliberately mistuned: it assumes Q × {cfg['kf_q_scale']:g} and R × "
                f"{cfg['kf_r_scale']:g} compared with the true noise.")
    if K_ss is not None:
        st.latex(r"\text{Steady state:}\quad K_\infty=" + bm(K_ss) + r",\qquad \sqrt{\operatorname{diag}P_{k|k}}="
                 + bm(np.sqrt(np.diag(Pu_ss)).reshape(1, -1)))
    e = est["kalman"]
    if "pred_std_w" in e and np.isfinite(e["pred_std_w"]):
        st.dataframe(pd.DataFrame([
            ("Speed ω [rad/s]", e["meas_rmse_w"], e["act_std_w"], e["pred_std_w"]),
            ("Current iₐ [A]", e["meas_rmse_i"], e["act_std_i"], e["pred_std_i"]),
        ], columns=["State", "Raw sensor error (RMS)", "Kalman error (RMS, 2nd half)", "Filter's own estimate √P"]),
            hide_index=True, width="stretch",
            column_config={c: st.column_config.NumberColumn(format="%.4f") for c in
                           ("Raw sensor error (RMS)", "Kalman error (RMS, 2nd half)", "Filter's own estimate √P")})
        st.caption("When the filter is tuned to the true noise, its own uncertainty √P matches the error it actually "
                   "makes: the filter is consistent. A mistuned Q or R breaks this agreement.")
    st.plotly_chart(plots.estimation_error(sims["kalman"]), width="stretch")
    st.caption("Grey dots: error of the raw sensor. Purple line: error of the Kalman estimate, which is much smaller. "
               "Shaded: the ±2σ band the filter predicts from P; about 95 % of the purple line should lie inside it.")
    st.plotly_chart(plots.kalman_gain(sims["kalman"]), width="stretch")
    st.caption("The gain and the uncertainty start large (the initial guess is not trusted) and settle within a few "
               "samples to the steady-state values above.")

# ================================================================== mpc
with tabs[3]:
    st.subheader("MPC formulation (same as Lab 7)")
    st.markdown("At every sample k the MPC takes the current state, the Kalman estimate in the main case, predicts "
                "the speed over N samples, and solves:")
    st.latex(r"\min_{\Delta\hat u_{k|k},\dots,\Delta\hat u_{k+N-1|k}}\;\sum_{j=1}^{N}q\big(\hat y_{k+j|k}-\omega^*\big)^2"
             r"+\sum_{j=0}^{N-1}r\big(\Delta\hat u_{k+j|k}\big)^2")
    st.latex(r"\text{s.t.}\;\hat x_{k|k}=\text{(state estimate)},\;\hat x_{k+j+1|k}=A_d\hat x_{k+j|k}+B_d\hat u_{k+j|k},"
             r"\;-V_{max}\le\hat u_{k+j|k}\le V_{max},\;\omega_{min}\le\hat y_{k+j+1|k}\le\omega_{max}")
    st.markdown("In matrix form, with z = [ω, iₐ, uₖ₋₁]ᵀ: Ŷ = Φz + ΓΔU and Û = 1·uₖ₋₁ + SΔU, giving a quadratic "
                "program min ½ΔUᵀHΔU + fᵀΔU with H = 2(qΓᵀΓ + rI), f = 2qΓᵀ(Φz − ω\\*1), solved with OSQP. Only "
                "the first move is applied: uₖ = uₖ₋₁ + Δû\\*ₖ|ₖ.")
    mpc = MPC(Ad, Bd, p)
    st.dataframe(pd.DataFrame([
        ("Prediction horizon N", f"{p['N']} steps ({p['N'] * p['Ts']:.3g} s)"),
        ("Sample time Tₛ", f"{p['Ts']:g} s"),
        ("Weights q / r", f"{p['q']:g} / {p['r']:g}"),
        ("Voltage limit", f"±{p['Vmax']:g} V"),
        ("Speed limits", f"{p['wmin']:g} … {p['wmax']:g} rad/s"),
        ("Stable without constraints", "✅ yes" if np.all(np.abs(mpc.poles) < 1) else "❌ no"),
        ("Average time per sample (filter + MPC)", f"{sims['kalman']['solve_ms']:.2f} ms"),
    ], columns=["Setting", "Value"]), hide_index=True, width="stretch")
    st.markdown("**Why the state estimate matters for MPC.** Every prediction starts from the current state. A noisy "
                "starting state makes every plan, and therefore every applied voltage, jump from sample to sample. A "
                "good estimate gives smooth, consistent plans.")
    if p["wmax"] < cfg["ref"]:
        st.warning("ω_max is below the target speed, so the MPC cannot reach it. Set ω_max a little above ω*.")

# ================================================================== responses
with tabs[4]:
    case = st.radio("State information used by the MPC", CASES, index=2, horizontal=True,
                    format_func=lambda c: CASE_NAMES[c])
    st.plotly_chart(plots.states_estimation(sims[case], "Kalman estimate" if case == "kalman" else "used by the MPC"),
                    width="stretch")
    st.caption({"kalman": "Grey dots are the noisy sensor readings. The dark line is the actual state and the dashed "
                          "purple line the Kalman estimate, which follows the actual state closely.",
                "direct": "Here the MPC uses the noisy readings themselves (dashed orange = grey dots), so it reacts "
                          "to the noise.",
                "true": "Reference case: the MPC knows the exact states. Grey dots show what the sensors would read."}[case])
    st.plotly_chart(plots.voltage(sims, p["Vmax"], [case]), width="stretch")
    st.subheader("The three cases side by side")
    st.plotly_chart(plots.compare_speed(sims, CASES), width="stretch")
    st.plotly_chart(plots.voltage(sims, p["Vmax"], CASES), width="stretch")
    st.caption("With direct measurements the voltage jumps every sample because the MPC re-plans from a noisy state; "
               "the speed ripples as a result. With the Kalman filter the voltage and speed are almost as smooth as "
               "with perfect sensors.")

# ================================================================== comparison
with tabs[5]:
    st.subheader("State estimation accuracy")
    rows = []
    for s, name, unit in (("w", "Speed ω", "rad/s"), ("i", "Current iₐ", "A")):
        e = est["kalman"]
        rows.append({"State": f"{name} [{unit}]", "Raw sensor RMS error": e[f"meas_rmse_{s}"],
                     "Kalman RMS error": e[f"rmse_{s}"], "Reduction (%)": e[f"reduction_{s}"]})
    edf = pd.DataFrame(rows)
    st.dataframe(edf, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.4f") for c in edf.columns if c != "State"})
    st.subheader("Control performance (measured on the actual speed)")
    rows = []
    for c in CASES:
        m = ctl[c]
        rows.append({"Case": CASE_NAMES[c], "Rise time (s)": m["tr"], "Settling time ±2 % (s)": m["ts"],
                     "Overshoot (%)": m["Mp"], "Steady error (%)": m["ess_pct"],
                     "Speed ripple RMS (rad/s)": m["track_rms"], "Voltage jitter std (V)": m["u_std_ss"],
                     "IAE": m["iae"]})
    cdf = pd.DataFrame(rows)
    st.dataframe(cdf, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.4f") for c in cdf.columns if c != "Case"})
    st.caption("Settling time is blank when the speed never stays within ±2 % of ω* (noise keeps pushing it out). "
               "Ripple and jitter are measured over the last 40 % of the test, after the step.")
    st.download_button("⬇ Download both tables (CSV)", pd.concat([edf, cdf]).to_csv(index=False),
                       file_name="kf_mpc_comparison.csv")

    st.subheader("Observations")
    e, mt, md, mk = est["kalman"], ctl["true"], ctl["direct"], ctl["kalman"]
    obs = []
    if np.isfinite(e["reduction_w"]):
        obs.append(f"**Estimation.** The Kalman filter reduces the speed error from {e['meas_rmse_w']:.4f} to "
                   f"{e['rmse_w']:.4f} rad/s ({e['reduction_w']:.0f} % smaller) and the current error from "
                   f"{e['meas_rmse_i']:.3f} to {e['rmse_i']:.3f} A ({e['reduction_i']:.0f} % smaller).")
    obs.append(f"**Direct measurements.** The MPC reacts to every noisy reading: the voltage jitters with a standard "
               f"deviation of {md['u_std_ss']:.2f} V and the speed ripples by {md['track_rms']:.4f} rad/s RMS"
               + (", so it never settles within ±2 %." if not np.isfinite(md["ts"]) else
                  f", settling in {md['ts']:.3f} s."))
    obs.append(f"**Kalman filter.** Voltage jitter {mk['u_std_ss']:.2f} V and speed ripple {mk['track_rms']:.4f} rad/s, "
               f"close to the ideal case with perfect sensors ({mt['u_std_ss']:.2f} V, {mt['track_rms']:.4f} rad/s)"
               + (f"; settling time {mk['ts']:.3f} s." if np.isfinite(mk["ts"]) else "."))
    obs.append("**What remains.** Even perfect state knowledge cannot remove the process noise: the 'true states' "
               "case still ripples a little, because the motor itself is disturbed.")
    for o in obs:
        st.markdown("- " + o)
    st.subheader("Conclusion")
    better = mk["track_rms"] < md["track_rms"] and mk["u_std_ss"] < md["u_std_ss"]
    (st.success if better else st.warning)(
        "**Using Kalman-filter estimates in the MPC gives better control than using the noisy measurements "
        "directly**: smoother voltage, less speed ripple, and performance close to having perfect sensors."
        if better else "With the current settings the Kalman filter does not improve on direct measurements. Check "
                       "the filter tuning (Q and R scales) and the noise levels.")

# ================================================================== noise study
with tabs[6]:
    st.subheader("Effect of noise level and filter tuning")
    st.markdown("The test is repeated for **measurement noise** from 0 to 4 × the sidebar value, **process noise** "
                "from 0 to 8 × the sidebar value, and for a filter that assumes the measurement noise is 0.01 to 100 "
                "× its true value. All other settings come from the sidebar.")
    if st.button("▶ Run the noise study (about 15 s)", type="primary"):
        bar = st.progress(0.0, "Running…")
        st.session_state["study"] = run_study(cfg, progress=lambda x: bar.progress(x))
        bar.empty()
    sd = st.session_state.get("study")
    if sd:
        meas, proc, tune = sd["meas"], sd["proc"], sd["tune"]
        st.markdown("**1 · Measurement noise**")
        a, b = st.columns(2)
        a.plotly_chart(plots.sweep_plot(meas, "sigma_w", "speed-sensor noise σ [rad/s]  (current sensor scaled alike)", [
            ("meas_rmse_w", "raw sensor error", "#8a8f98", "dot"),
            ("kalman_rmse_w", "Kalman estimate error", "#6250d6", "solid")], "speed estimation RMS error [rad/s]"),
            width="stretch")
        b.plotly_chart(plots.sweep_plot(meas, "sigma_w", "speed-sensor noise σ [rad/s]", [
            ("direct_ustd", "direct measurement", "#eb6834", "solid"),
            ("kalman_ustd", "Kalman filter", "#6250d6", "solid")], "voltage jitter std [V]"), width="stretch")
        st.caption("The raw error grows in proportion to the sensor noise; the Kalman error grows much more slowly. "
                   "Without a filter the voltage jitter grows with the noise until it hits the voltage limit; with the "
                   "filter it stays small.")
        st.markdown("**2 · Process noise**")
        a, b = st.columns(2)
        a.plotly_chart(plots.sweep_plot(proc, "sigma_w", "process noise on speed σ [rad/s per sample]", [
            ("kalman_rmse_w", "Kalman estimate error", "#6250d6", "solid"),
            ("meas_rmse_w", "raw sensor error", "#8a8f98", "dot")], "speed estimation RMS error [rad/s]"),
            width="stretch")
        b.plotly_chart(plots.sweep_plot(proc, "sigma_w", "process noise on speed σ [rad/s per sample]", [
            ("true_track", "true states", "#2a78d6", "solid"), ("direct_track", "direct measurement", "#eb6834", "solid"),
            ("kalman_track", "Kalman filter", "#6250d6", "solid")], "speed ripple RMS [rad/s]"), width="stretch")
        st.caption("More process noise makes the model less reliable, so the filter trusts the sensors more and its "
                   "error rises toward the raw sensor error. The speed ripple of every case grows, because the motor "
                   "itself is disturbed more. At very large process noise the filter's advantage shrinks: the "
                   "model no longer predicts the motor well, so the estimate is little better than the measurement.")
        st.markdown("**3 · Filter tuning (assumed measurement noise R)**")
        a, b = st.columns(2)
        a.plotly_chart(plots.sweep_plot(tune, "level", "R scale used by the filter (1 = true value)", [
            ("kalman_rmse_w", "speed estimate error [rad/s]", "#6250d6", "solid")], "RMS error", log_x=True),
            width="stretch")
        b.plotly_chart(plots.sweep_plot(tune, "level", "R scale used by the filter (1 = true value)", [
            ("kalman_ustd", "voltage jitter std [V]", "#6250d6", "solid")], "voltage jitter std [V]", log_x=True),
            width="stretch")
        st.caption("Too small an R makes the filter trust the noisy sensors too much (noisy estimate). Too large an R "
                   "makes it ignore them and follow the model (slow, lagging estimate). The error is smallest near the "
                   "true value, R scale = 1. The voltage keeps getting smoother as R grows, but only because the filter "
                   "stops listening to the sensors and its estimate lags behind the real motor.")
        out = pd.concat([pd.DataFrame(meas).assign(study="measurement noise"),
                         pd.DataFrame(proc).assign(study="process noise"),
                         pd.DataFrame(tune).assign(study="R scale")])
        st.download_button("⬇ Download noise-study results (CSV)", out.to_csv(index=False), file_name="noise_study.csv")
    else:
        st.info("Press the button to run the study with the current sidebar settings.")
