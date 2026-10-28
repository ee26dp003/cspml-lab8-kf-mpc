"""Estimation-accuracy and control-performance metrics."""

import numpy as np


def _rms(a):
    a = np.asarray(a, float)
    return float(np.sqrt(np.nanmean(a ** 2)))


def estimation_metrics(sim):
    """RMS error of the state used by the controller, against the true state (whole run)."""
    x, xh, y = sim["x"], sim["xh"], sim["y"]
    out = dict(rmse_w=_rms(xh[:, 0] - x[:, 0]), rmse_i=_rms(xh[:, 1] - x[:, 1]),
               meas_rmse_w=_rms(y[:, 0] - x[:, 0]), meas_rmse_i=_rms(y[:, 1] - x[:, 1]))
    for s in ("w", "i"):
        m = out[f"meas_rmse_{s}"]
        out[f"reduction_{s}"] = 100 * (1 - out[f"rmse_{s}"] / m) if m > 0 else np.nan
    if np.isfinite(sim["P"]).any():                       # filter's own error prediction (2nd half of the run)
        h = len(sim["t"]) // 2
        out["pred_std_w"] = float(np.sqrt(np.nanmean(sim["P"][h:, 0])))
        out["pred_std_i"] = float(np.sqrt(np.nanmean(sim["P"][h:, 1])))
        out["act_std_w"] = _rms(xh[h:, 0] - x[h:, 0])
        out["act_std_i"] = _rms(xh[h:, 1] - x[h:, 1])
    return out


def control_metrics(sim, band=2.0):
    """Measured on the TRUE speed. Settling: last time |ω − ω*| > band % of ω*."""
    t, w, u, ref = sim["t"], sim["x"][:, 0], sim["u"], sim["ref"]
    i10 = np.flatnonzero(w >= 0.1 * ref)
    i90 = np.flatnonzero(w >= 0.9 * ref)
    tr = t[i90[0]] - t[i10[0]] if len(i10) and len(i90) else np.nan
    Mp = max(0.0, (np.max(w) - ref) / ref * 100)
    out_band = np.flatnonzero(np.abs(w - ref) > band / 100 * abs(ref))
    ts = 0.0 if not len(out_band) else (t[out_band[-1] + 1] if out_band[-1] + 1 < len(t) else np.nan)
    tail = t >= 0.6 * t[-1]
    return dict(tr=float(tr), ts=float(ts), Mp=float(Mp),
                ess=float(ref - np.mean(w[tail])), ess_pct=float(100 * (ref - np.mean(w[tail])) / ref),
                track_rms=_rms(w[tail] - ref), iae=float(np.sum(np.abs(w - ref)) * sim["Ts"]),
                du_rms=_rms(np.diff(u)), u_std_ss=float(np.std(u[tail])), peak_v=float(np.max(np.abs(u))))
