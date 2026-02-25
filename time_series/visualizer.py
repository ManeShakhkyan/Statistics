import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib

matplotlib.use("Agg")

C = dict(
    hist="#1E3A8A", srw="#16A34A", hw="#D97706", theta="#7C3AED",
    ens="#DC2626", test="#475569", ci="#DDD6FE",
)

def make_plot(full_series, train, val, test_labels,
              srw_fit, hw_fit, theta_fit,
              srw_val, hw_val, theta_val,
              ens_fc, ens_lo, ens_hi,
              srw_fc, hw_fc, theta_fc,
              future_idx, diag: dict, save_path: str):

    sc    = 1e9
    fig   = plt.figure(figsize=(18, 16))
    fig.patch.set_facecolor("#F0F4F8")
    
    gs    = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.25)

    recent_full = full_series["2018Q1":]
    n_rf  = len(recent_full)
    n_tr  = len(train)
    n_val = len(val)
    x_rf  = np.arange(n_rf)
    x_tr  = np.arange(n_tr)
    x_val = np.arange(n_tr, n_tr + n_val)
    
    # ── 1. Մոդելների համեմատություն (Validation) ────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(x_rf, recent_full.values / sc, color=C["hist"],
             lw=1.8, label="Actual (2018–2025)", alpha=0.9)
    ax1.plot(x_val, val.values / sc, color=C["test"],
             lw=2.5, ls="--", label="Validation — 2024", alpha=0.85)
    
    m_srw   = diag["val_metrics"]["SRW"]["MAPE"]
    m_hw    = diag["val_metrics"]["HW"]["MAPE"]
    m_theta = diag["val_metrics"]["Theta"]["MAPE"]
    
    ax1.plot(x_val, srw_val / sc,   color=C["srw"],   lw=2, marker="o", ms=5,
             label=f"Seasonal RW (MAPE={m_srw:.1f}%)")
    ax1.plot(x_val, hw_val / sc,    color=C["hw"],    lw=2, marker="s", ms=5,
             label=f"Holt-Winters (MAPE={m_hw:.1f}%)")
    ax1.plot(x_val, theta_val / sc, color=C["theta"], lw=2, marker="^", ms=5,
             label=f"Theta (MAPE={m_theta:.1f}%)")
    
    ax1.axvline(n_tr - 0.5, color="gray", ls=":", lw=1.2)
    ax1.set_title("Model Comparison — Validation Set (2024Q1–2024Q4)", fontsize=13, pad=10)
    ax1.legend(fontsize=10, ncol=2)
    _style(ax1)
    _xticks(ax1, recent_full, step=4)

    # ── 2. Լավագույն մոդելի ֆիթը & Կանխատեսում ──────────
    best = diag["best_model"]
    best_color = {"SRW": C["srw"], "HW": C["hw"], "Theta": C["theta"]}[best]
    best_fit   = {"SRW": srw_fit, "HW": hw_fit, "Theta": theta_fit}[best]
    best_fc    = {"SRW": srw_fc, "HW": hw_fc, "Theta": theta_fc}[best]

    last_tr_val = train.values[-1]
    best_fc_plot = np.concatenate(([last_tr_val], best_fc))
    x_fc_plot = np.arange(n_tr - 1, n_tr + 4)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(x_tr, train.values / sc, color=C["hist"],
             lw=1.2, alpha=0.4, label="Train (2018–2023)")
    
    if best_fit is not None:
        ax2.plot(x_tr, best_fit[:n_tr] / sc,
                 color=best_color, lw=1.8, ls="--", label=f"Fitted ({best})")
    
    ax2.plot(x_fc_plot, best_fc_plot / sc, color=best_color,
             lw=2.2, ls="--", marker="D", ms=6, label=f"Forecast {best} (2026)")
    
    ax2.axvline(n_tr - 0.5, color="gray", ls=":", lw=1)
    ax2.set_title(f"Best Individual Model: {best}\n{_param_str(diag, best)}", fontsize=11, pad=8)
    ax2.legend(fontsize=9)
    _style(ax2)
    _xticks(ax2, recent_full, step=4)

    # ── 3. MAPE սյունակային գրաֆիկ ───────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    names  = ["SRW", "HW", "Theta"]
    colors = [C["srw"], C["hw"], C["theta"]]
    val_mapes = [diag["val_metrics"][n]["MAPE"] for n in names]
    bars = ax3.bar(range(3), val_mapes, color=colors, alpha=0.85, width=0.55)
    best_idx = names.index(best)
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(3)
    ax3.set_xticks(range(3))
    ax3.set_xticklabels(["Seasonal RW", "Holt-Winters", "Theta"], fontsize=10)
    ax3.set_ylabel("MAPE (%)", fontsize=10)
    ax3.set_title("Validation Accuracy (lower is better)", fontsize=11, pad=8)
    for bar, val_m in zip(bars, val_mapes):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.2,
                 f"{val_m:.1f}%", ha="center", fontsize=10, fontweight="bold")
    _style(ax3)

    # ── 4. Final Ensemble Forecast 2026 ───────────────
    ax4 = fig.add_subplot(gs[2, :])
    tail   = 12
    recent = full_series.iloc[-tail:]
    
    last_hist_val = recent.values[-1]
    ens_fc_plot = np.concatenate(([last_hist_val], ens_fc))
    
    xh = np.arange(tail)
    xf_plot = np.arange(tail - 1, tail + 4) 
    xf_only = np.arange(tail, tail + 4)     

    ax4.plot(xh, recent.values / sc, color=C["hist"], lw=2.5, label="Historical (last 3 years)")
    
    ax4.plot(xf_plot, ens_fc_plot / sc, color=C["ens"], lw=3, 
             ls="--", marker="D", ms=9, label="Ensemble Forecast 2026")
    
    ax4.fill_between(xf_only, ens_lo / sc, ens_hi / sc, alpha=0.15, color=C["ens"], label="90% Prediction Interval")
    
    for xi, (prd, fv, lv, hv) in enumerate(zip(future_idx, ens_fc, ens_lo, ens_hi)):
        ax4.annotate(
            f"{fv/sc:.3f}B\n[{lv/sc:.2f}–{hv/sc:.2f}]",
            xy=(xf_only[xi], fv / sc), xytext=(0, 30), textcoords="offset points",
            ha="center", fontsize=10, fontweight="bold", color=C["ens"],
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["ens"], alpha=0.9))
            
    ax4.axvline(tail - 0.5, color="gray", ls=":", lw=1.5)
    
    lbls = [str(p) for p in recent.index] + [str(p) for p in future_idx]
    ax4.set_xticks(range(len(lbls)))
    ax4.set_xticklabels(lbls, rotation=45, ha="right", fontsize=9)
    
    w = diag["weights"]
    w_str = f"Weights: SRW={w['SRW']:.2f}, HW={w['HW']:.2f}, Theta={w['Theta']:.2f}"
    ax4.set_title(f"Final Combined Forecast 2026 (AMD Billion)\n{w_str}", fontsize=13, pad=10)
    ax4.legend(fontsize=10, loc="upper left")
    _style(ax4)

    fig.suptitle("Health Insurance Forecast Report — 2026", fontsize=18, fontweight="bold", y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓  Report updated and saved: {save_path}")

def _style(ax):
    ax.grid(axis="y", ls="--", alpha=0.3)
    ax.set_facecolor("#FAFAFA")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

def _xticks(ax, series, step=4):
    x = np.arange(len(series))
    ax.set_xticks(x[::step])
    ax.set_xticklabels([str(p) for p in series.index][::step],
                       rotation=45, ha="right", fontsize=8)

def _param_str(diag, best):
    p = diag.get("params", {}).get(best, {})
    if best == "HW":
        return f"(α={p.get('alpha',0):.2f}, β={p.get('beta',0):.2f}, γ={p.get('gamma',0):.2f}, φ={p.get('phi',0):.2f})"
    if best == "Theta":
        return f"(α_SES={p.get('alpha',0):.2f}, drift={p.get('drift',0):.4f})"
    return "Seasonal Random Walk"