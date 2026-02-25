import sys
import numpy as np
import pandas as pd
from pathlib import Path
from data_loader import QuarterlyDataLoader
from models import SeasonalRW, HoltWinters, ThetaModel
from stats_utils import StatTests, metrics
from visualizer import make_plot

def run(filepath: str):
    SEP = "═" * 65
    print(SEP)
    print("  Quarterly Health Insurance — Final Forecasting Pipeline")
    print(SEP)

    print("\n[1] Loading data...")
    full = QuarterlyDataLoader(filepath).load()
    print(f"    Full series: {full.index[0]} → {full.index[-1]}  ({len(full)} obs)")

    print("\n[2] Regime analysis...")
    print("    2008-2017: old regime (irrelevant, 0.02–0.76B)")
    print("    2018-2024: stable seasonal regime — Q1-Q3 high, Q4 low")
    print("    2025:      level shift — all quarters ~1.04B (flat)")
    print("    → Training on 2018Q1-2023Q4 (recent stable regime)")

    recent = full["2018Q1":]
    train  = recent["2018Q1":"2023Q4"]
    val    = recent["2024Q1":"2024Q4"] 
    y_tr, y_val = train.values, val.values

    print(f"\n[3] Data split:")
    print(f"    Train:      {train.index[0]} → {train.index[-1]}  ({len(train)} obs)")
    print(f"    Validation: {val.index[0]}  → {val.index[-1]}   ({len(val)} obs)")

    print("\n[4] ADF stationarity (log-train)...")
    adf_res = StatTests.adf(np.log(y_tr))
    print(f"    tau={adf_res['tau']}  p={adf_res['p']}  → {adf_res['verdict']}")

    print("\n[5] Fitting models & validating on 2024...")
    srw   = SeasonalRW(period=4).fit(y_tr)
    hw    = HoltWinters(period=4).fit(y_tr)
    theta = ThetaModel(period=4).fit(y_tr)

    srw_val_fc,   _, _ = srw.forecast(4)
    hw_val_fc,    _, _ = hw.forecast(4)
    theta_val_fc, _, _ = theta.forecast(4)

    m_srw   = metrics(y_val, srw_val_fc)
    m_hw    = metrics(y_val, hw_val_fc)
    m_theta = metrics(y_val, theta_val_fc)

    print(f"    Seasonal RW  — MAPE={m_srw['MAPE']:>6.2f}%  "
          f"MAE={m_srw['MAE']/1e6:.1f}M  RMSE={m_srw['RMSE']/1e6:.1f}M")
    print(f"    Holt-Winters — MAPE={m_hw['MAPE']:>6.2f}%  "
          f"MAE={m_hw['MAE']/1e6:.1f}M  RMSE={m_hw['RMSE']/1e6:.1f}M")
    print(f"    Theta        — MAPE={m_theta['MAPE']:>6.2f}%  "
          f"MAE={m_theta['MAE']/1e6:.1f}M  RMSE={m_theta['RMSE']/1e6:.1f}M")

    best = min({"SRW": m_srw["MAPE"], "HW": m_hw["MAPE"],
                "Theta": m_theta["MAPE"]}, key=lambda k: {
                    "SRW": m_srw["MAPE"], "HW": m_hw["MAPE"],
                    "Theta": m_theta["MAPE"]}[k])
    print(f"\n    ★ Best on validation: {best}")

    best_model = {"SRW": srw, "HW": hw, "Theta": theta}[best]
    best_resid = best_model.residuals
    print(f"\n[6] Statistical diagnostics ({best})...")
    lb_res  = StatTests.ljung_box(best_resid)
    sw_res  = StatTests.shapiro(best_resid)
    cusum_r = StatTests.cusum(best_resid)
    mapes_all = {"SRW": m_srw["MAPE"], "HW": m_hw["MAPE"], "Theta": m_theta["MAPE"]}
    sorted_models = sorted(mapes_all, key=mapes_all.get)
    second = sorted_models[1]
    preds_map = {"SRW": srw_val_fc, "HW": hw_val_fc, "Theta": theta_val_fc}
    dm_res = StatTests.diebold_mariano(y_val, preds_map[best], preds_map[second])

    print(f"    Ljung-Box:     Q={lb_res['Q']}  p={lb_res['p']}  {lb_res['verdict']}")
    print(f"    Shapiro-Wilk:  W={sw_res['W']}  p={sw_res['p']}  {sw_res['verdict']}")
    print(f"    DM ({best} vs {second}): DM={dm_res['DM']}  p={dm_res['p']}  {dm_res['verdict']}")
    print(f"    CUSUM: max={cusum_r['max_dev']}  {cusum_r['verdict']}")

    print("\n[7] Refitting on 2018-2025 (full recent regime)...")
    y_recent = recent.values
    srw_f   = SeasonalRW(period=4).fit(y_recent)
    hw_f    = HoltWinters(period=4).fit(y_recent)
    theta_f = ThetaModel(period=4).fit(y_recent)
    print(f"    HW params: α={hw_f._params['alpha']:.4f}  "
          f"β={hw_f._params['beta']:.5f}  γ={hw_f._params['gamma']:.4f}  "
          f"φ={hw_f._params['phi']:.4f}")
    print(f"    Theta: α_SES={theta_f._alpha:.4f}  drift={theta_f._drift:.6f}")

    print("\n[8] Building ensemble (inverse-MAPE weights)...")
    raw_w = {n: 1 / max(mapes_all[n], 0.01) for n in ["SRW", "HW", "Theta"]}
    tot   = sum(raw_w.values())
    w     = {n: raw_w[n] / tot for n in raw_w}
    for n in ["SRW", "HW", "Theta"]:
        print(f"    {n:<12}  MAPE={mapes_all[n]:.2f}%  weight={w[n]:.3f}")

    srw_fc,   srw_lo,   srw_hi   = srw_f.forecast(4,   alpha_ci=0.10)
    hw_fc,    hw_lo,    hw_hi    = hw_f.forecast(4,    alpha_ci=0.10)
    theta_fc, theta_lo, theta_hi = theta_f.forecast(4, alpha_ci=0.10)

    ens_fc = w["SRW"]*srw_fc + w["HW"]*hw_fc + w["Theta"]*theta_fc
    ens_lo = w["SRW"]*srw_lo + w["HW"]*hw_lo + w["Theta"]*theta_lo
    ens_hi = w["SRW"]*srw_hi + w["HW"]*hw_hi + w["Theta"]*theta_hi

    future_idx = pd.period_range(start=full.index[-1] + 1, periods=4, freq="Q")

    # 9. Print
    print(f"\n[9] FINAL FORECAST — 2026  (90% Prediction Interval)")
    print(f"\n    {'Quarter':<10} {'SRW':>10} {'HW':>10} {'Theta':>10} "
          f"{'Ensemble':>10}   90% PI")
    for i, prd in enumerate(future_idx):
        print(f"    {str(prd):<10} "
              f"{srw_fc[i]/1e9:>9.3f}B "
              f"{hw_fc[i]/1e9:>9.3f}B "
              f"{theta_fc[i]/1e9:>9.3f}B "
              f"{ens_fc[i]/1e9:>9.3f}B   "
              f"[{ens_lo[i]/1e9:.3f}B – {ens_hi[i]/1e9:.3f}B]")

    print(f"\n    Validation MAPE summary:")
    for n in sorted_models:
        marker = " ★" if n == best else ""
        print(f"    {n:<14}: {mapes_all[n]:.2f}%{marker}")

    diag = dict(
        adf=adf_res, lb=lb_res, sw=sw_res, dm=dm_res, cusum=cusum_r,
        val_metrics={"SRW": m_srw, "HW": m_hw, "Theta": m_theta},
        best_model=best, weights=w,
        params={
            "HW":    hw_f._params,
            "Theta": {"alpha": theta_f._alpha, "drift": theta_f._drift},
            "SRW":   {},
        },
    )
    out = str(Path(filepath).parent / "forecast_final.png")
    print(f"\n[10] Generating plot...")
    make_plot(
        full, train, val, ["2024Q1","2024Q2","2024Q3","2024Q4"],
        srw.fitted_orig if hasattr(srw, "fitted_orig") else None,
        hw.fitted_orig,
        theta.fitted_orig,
        srw_val_fc, hw_val_fc, theta_val_fc,
        ens_fc, ens_lo, ens_hi,
        srw_fc, hw_fc, theta_fc,
        future_idx, diag, out)

    print(f"\n{SEP}\n  Done!\n{SEP}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "./data.xlsx"
    run(filepath=path)