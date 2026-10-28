"""Noise study: how estimation accuracy and control quality change with the noise levels and the filter tuning."""

from .metrics import control_metrics, estimation_metrics
from .simulate import simulate

MEAS_LEVELS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
PROC_LEVELS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
R_SCALES = [0.01, 0.1, 1.0, 10.0, 100.0]


def _row(c, cases):
    row = {}
    for case in cases:
        s = simulate(c, case)
        e, m = estimation_metrics(s), control_metrics(s)
        row.update({f"{case}_rmse_w": e["rmse_w"], f"{case}_rmse_i": e["rmse_i"],
                    f"{case}_track": m["track_rms"], f"{case}_ustd": m["u_std_ss"], f"{case}_ts": m["ts"]})
        row["meas_rmse_w"], row["meas_rmse_i"] = e["meas_rmse_w"], e["meas_rmse_i"]
    return row


def run_study(cfg, progress=None):
    total = len(MEAS_LEVELS) + len(PROC_LEVELS) + len(R_SCALES)
    done = [0]

    def tick():
        done[0] += 1
        if progress:
            progress(done[0] / total)

    meas, proc, tune = [], [], []
    for m in MEAS_LEVELS:
        c = dict(cfg, meas_w=cfg["meas_w"] * m, meas_i=cfg["meas_i"] * m)
        meas.append(dict(level=m, sigma_w=c["meas_w"], **_row(c, ["direct", "kalman"])))
        tick()
    for m in PROC_LEVELS:
        c = dict(cfg, proc_w=cfg["proc_w"] * m, proc_i=cfg["proc_i"] * m)
        proc.append(dict(level=m, sigma_w=c["proc_w"], **_row(c, ["true", "direct", "kalman"])))
        tick()
    for rs in R_SCALES:
        c = dict(cfg, kf_r_scale=rs, kf_q_scale=1.0)
        tune.append(dict(level=rs, **_row(c, ["kalman"])))
        tick()
    return dict(meas=meas, proc=proc, tune=tune)
