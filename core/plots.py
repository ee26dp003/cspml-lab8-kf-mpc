"""Plotly charts. Fixed colours: one per case, and one role per signal."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import CASE_NAMES, CASE_SHORT

CASE_COL = {"true": "#2a78d6", "direct": "#eb6834", "kalman": "#6250d6"}
TRUE_COL = "#1f3556"        # actual (true) state
EST_COL = "#6250d6"         # Kalman estimate
MEAS_COL = "rgba(128,128,128,0.45)"
REF_COL = "rgba(128,128,128,0.9)"
GRID = "rgba(128,128,128,0.18)"


def _style(fig, height, title=None, hover="x unified"):
    fig.update_layout(height=height, title=dict(text=title, font=dict(size=15)) if title else None,
                      margin=dict(l=60, r=20, t=60 if title else 30, b=50), hovermode=hover,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def states_estimation(sim, est_label="Kalman estimate"):
    """Speed and current: measured (dots), actual (solid), estimate given to the MPC (dashed), reference."""
    t, x, y, xh = sim["t"], sim["x"], sim["y"], sim["xh"]
    fig = make_subplots(2, 1, shared_xaxes=True, vertical_spacing=0.09,
                        subplot_titles=["Motor speed ω [rad/s]", "Armature current iₐ [A]"])
    fig.add_trace(go.Scatter(x=t, y=np.full_like(t, sim["ref"]), name="reference ω*",
                             line=dict(color=REF_COL, dash="dash", width=1.5), hovertemplate="%{y:.3f}"), 1, 1)
    for row, j, unit in ((1, 0, "rad/s"), (2, 1, "A")):
        sl = row == 1
        fig.add_trace(go.Scatter(x=t, y=y[:, j], name="measured (noisy)", mode="markers", legendgroup="m",
                                 showlegend=sl, marker=dict(size=3.5, color=MEAS_COL),
                                 hovertemplate=f"%{{y:.4g}} {unit}"), row, 1)
        fig.add_trace(go.Scatter(x=t, y=x[:, j], name="actual", legendgroup="a", showlegend=sl,
                                 line=dict(color=TRUE_COL, width=2.4), hovertemplate=f"%{{y:.4g}} {unit}"), row, 1)
        if sim["case"] != "true":
            fig.add_trace(go.Scatter(x=t, y=xh[:, j], name=est_label, legendgroup="e", showlegend=sl,
                                     line=dict(color=EST_COL if sim["case"] == "kalman" else CASE_COL["direct"],
                                               width=1.8, dash="dash"),
                                     hovertemplate=f"%{{y:.4g}} {unit}"), row, 1)
    fig.update_xaxes(title_text="time [s]", row=2, col=1)
    _style(fig, 620, None)
    fig.update_layout(margin=dict(t=70), legend=dict(y=1.07))
    return fig


def voltage(sims, vmax, cases):
    fig = go.Figure()
    for c in cases:
        s = sims[c]
        fig.add_trace(go.Scatter(x=s["t"], y=s["u"], name=CASE_SHORT[c],
                                 line=dict(color=CASE_COL[c], width=1.8, shape="hv"), hovertemplate="%{y:.3g} V"))
    fig.add_hline(y=vmax, line=dict(color="#e34948", dash="dot", width=1), annotation_text=f"±{vmax:g} V limit",
                  annotation_position="top right")
    fig.add_hline(y=-vmax, line=dict(color="#e34948", dash="dot", width=1))
    fig.update_xaxes(title_text="time [s]")
    lo = min(0.0, min(float(np.min(sims[c]["u"])) for c in cases))
    fig.update_yaxes(title_text="armature voltage Vₐ [V]", range=[lo - 0.06 * vmax, 1.12 * vmax])
    return _style(fig, 360, "Applied armature voltage Vₐ")


def compare_speed(sims, cases):
    fig = go.Figure()
    s0 = sims[cases[0]]
    fig.add_hline(y=s0["ref"], line=dict(color=REF_COL, dash="dash"), annotation_text="reference ω*",
                  annotation_position="bottom right")
    for c in cases:
        s = sims[c]
        fig.add_trace(go.Scatter(x=s["t"], y=s["x"][:, 0], name=CASE_NAMES[c], line=dict(color=CASE_COL[c], width=2.2),
                                 hovertemplate="%{y:.4f} rad/s"))
    fig.update_xaxes(title_text="time [s]")
    fig.update_yaxes(title_text="actual speed ω [rad/s]")
    return _style(fig, 400, "Actual motor speed when the MPC uses each kind of state information")


def estimation_error(sim, sim_direct=None):
    """Kalman estimation error with the filter's own ±2σ band, and the raw measurement error for comparison."""
    t, x, xh, y, P = sim["t"], sim["x"], sim["xh"], sim["y"], sim["P"]
    fig = make_subplots(2, 1, shared_xaxes=True, vertical_spacing=0.09,
                        subplot_titles=["Speed error [rad/s]", "Current error [A]"])
    for row, j in ((1, 0), (2, 1)):
        sl = row == 1
        sd = 2 * np.sqrt(P[:, j])
        fig.add_trace(go.Scatter(x=np.concatenate([t, t[::-1]]), y=np.concatenate([sd, -sd[::-1]]), fill="toself",
                                 fillcolor="rgba(98,80,214,0.15)", line=dict(width=0), name="filter's ±2σ (from P)",
                                 legendgroup="b", showlegend=sl, hoverinfo="skip"), row, 1)
        fig.add_trace(go.Scatter(x=t, y=y[:, j] - x[:, j], mode="markers", name="measurement error y − x",
                                 marker=dict(size=3, color=MEAS_COL), legendgroup="m", showlegend=sl,
                                 hovertemplate="%{y:.4g}"), row, 1)
        fig.add_trace(go.Scatter(x=t, y=xh[:, j] - x[:, j], name="estimation error x̂ − x",
                                 line=dict(color=EST_COL, width=1.8), legendgroup="e", showlegend=sl,
                                 hovertemplate="%{y:.4g}"), row, 1)
        m = np.nanmax(np.abs(y[:, j] - x[:, j]))
        if np.isfinite(m) and m > 0:
            fig.update_yaxes(range=[-1.1 * m, 1.1 * m], row=row, col=1)
    fig.update_xaxes(title_text="time [s]", row=2, col=1)
    _style(fig, 560, None)
    fig.update_layout(margin=dict(t=70), legend=dict(y=1.08))
    return fig


def kalman_gain(sim):
    t, K, P = sim["t"], sim["K"], sim["P"]
    fig = make_subplots(1, 2, horizontal_spacing=0.1,
                        subplot_titles=["Kalman gain entries K_k", "Filter's error std √P (diagonal)"])
    names = ["K₁₁ (ω from ω)", "K₁₂ (ω from iₐ)", "K₂₁ (iₐ from ω)", "K₂₂ (iₐ from iₐ)"]
    cols = ["#2a78d6", "#1baf7a", "#eb6834", "#6250d6"]
    for j in range(4):
        fig.add_trace(go.Scatter(x=t, y=K[:, j], name=names[j], line=dict(color=cols[j], width=1.8)), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=np.sqrt(P[:, 0]), name="√P₁₁ speed [rad/s]", line=dict(color=TRUE_COL, width=2)), 1, 2)
    fig.add_trace(go.Scatter(x=t, y=np.sqrt(P[:, 1]), name="√P₂₂ current [A]", line=dict(color=TRUE_COL, width=2, dash="dash")),
                  1, 2)
    fig.update_yaxes(type="log", row=1, col=2)
    fig.update_xaxes(title_text="time [s]")
    _style(fig, 420, None)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.2, x=0), margin=dict(t=40, b=100))
    return fig


def sweep_plot(rows, xkey, xlabel, series, ylabel, log_x=False):
    """series: list of (key, label, colour, dash)."""
    fig = go.Figure()
    xs = [r[xkey] for r in rows]
    for key, label, col, dash in series:
        fig.add_trace(go.Scatter(x=xs, y=[r[key] for r in rows], name=label, mode="lines+markers",
                                 line=dict(color=col, width=2.2, dash=dash), marker=dict(size=8),
                                 hovertemplate="%{x:g} → %{y:.4g}"))
    fig.update_xaxes(title_text=xlabel, type="log" if log_x else "linear")
    fig.update_yaxes(title_text=ylabel, rangemode="tozero")
    return _style(fig, 360, None, hover="closest")
