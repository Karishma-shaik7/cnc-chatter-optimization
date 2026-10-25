"""
Industrial Visualization Module for CNC Chatter Diagnostics.
Generates publication-quality Matplotlib figures for Streamlit display.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Configure dark/industrial styling palette
BG_COLOR = "#0E1117"
PANEL_COLOR = "#161B22"
TEXT_COLOR = "#E6EDF3"
MUTED_TEXT = "#8B949E"
GRID_COLOR = "#30363D"

COLOR_STABLE = "#2EA043"    # Green
COLOR_WARNING = "#D29922"   # Yellow / Gold
COLOR_HIGH_RISK = "#F85149" # Red
COLOR_ACCENT = "#58A6FF"    # Blue
COLOR_RECOMMENDED = "#39D353"


def apply_industrial_style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Apply consistent high-contrast engineering theme to an axis."""
    ax.set_facecolor(PANEL_COLOR)
    ax.figure.set_facecolor(BG_COLOR)
    ax.set_title(title, color=TEXT_COLOR, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=TEXT_COLOR, fontsize=10, labelpad=8)
    ax.set_ylabel(ylabel, color=TEXT_COLOR, fontsize=10, labelpad=8)
    ax.tick_params(colors=MUTED_TEXT, labelsize=9)
    ax.grid(True, linestyle="--", alpha=0.35, color=GRID_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def plot_stability_lobes(
    rpm_grid: np.ndarray,
    ap_curve: np.ndarray,
    sweet_rpms: List[float],
    baseline_rpm: float,
    baseline_depth: float,
    recommended_rpm: Optional[float] = None,
    recommended_depth: Optional[float] = None,
    feasible_rpms: Optional[List[float]] = None,
    feasible_depths: Optional[List[float]] = None,
) -> plt.Figure:
    """
    Plot the analytical Stability Lobe Diagram (SLD) with baseline and candidate points.
    """
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=120)
    apply_industrial_style(
        ax,
        title="Stability Lobe Diagram (SLD) & Operating Regime",
        xlabel="Spindle Speed (RPM)",
        ylabel="Axial Depth of Cut ap (mm)",
    )

    # Fill zones
    ax.fill_between(rpm_grid, 0, ap_curve, color=COLOR_STABLE, alpha=0.15, label="Stable Cutting Region")
    ax.fill_between(
        rpm_grid,
        ap_curve,
        max(ap_curve.max() * 1.3, 7.0),
        color=COLOR_HIGH_RISK,
        alpha=0.12,
        label="Regenerative Chatter Region",
    )

    # Plot analytical lobe boundary
    ax.plot(rpm_grid, ap_curve, color=COLOR_ACCENT, linewidth=2.2, label="Stability Boundary (ap,lim)")

    # Mark sweet spot pockets
    for s_rpm in sweet_rpms:
        ax.axvline(s_rpm, color=COLOR_WARNING, linestyle=":", alpha=0.45, linewidth=1.2)

    # Plot other feasible alternative candidates
    if feasible_rpms and feasible_depths and len(feasible_rpms) > 0:
        ax.scatter(
            feasible_rpms,
            feasible_depths,
            color="#A371F7",
            s=40,
            alpha=0.6,
            edgecolors="none",
            label="Feasible Candidates",
            zorder=4,
        )

    # Plot baseline point
    ax.scatter(
        [baseline_rpm],
        [baseline_depth],
        color=COLOR_HIGH_RISK,
        s=140,
        marker="X",
        edgecolors="#FFFFFF",
        linewidths=1.5,
        label=f"Baseline ({baseline_rpm:.0f} RPM, {baseline_depth:.2f} mm)",
        zorder=5,
    )

    # Plot recommended point
    if recommended_rpm is not None and recommended_depth is not None:
        ax.scatter(
            [recommended_rpm],
            [recommended_depth],
            color=COLOR_RECOMMENDED,
            s=220,
            marker="*",
            edgecolors="#FFFFFF",
            linewidths=1.5,
            label=f"Recommended ({recommended_rpm:.0f} RPM, {recommended_depth:.2f} mm)",
            zorder=6,
        )

    ax.set_ylim(0, max(ap_curve.max() * 1.25, baseline_depth * 1.3, 6.5))
    ax.set_xlim(rpm_grid.min(), rpm_grid.max())
    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor=PANEL_COLOR,
        edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR,
        fontsize=8.5,
    )
    plt.tight_layout()
    return fig


def plot_vibration_spectrum(
    baseline_ft: float,
    baseline_dom_freq: float,
    baseline_rms: float,
    natural_freq_hz: float = 850.0,
    rec_ft: Optional[float] = None,
    rec_rms: Optional[float] = None,
) -> plt.Figure:
    """
    Generate synthetic frequency response spectrum showing baseline vs recommended vibration modes.
    """
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=120)
    apply_industrial_style(
        ax,
        title="Simulated Vibration Frequency Spectrum (Baseline vs. Recommended)",
        xlabel="Frequency (Hz)",
        ylabel="Vibration Spectral Amplitude (g)",
    )

    freqs = np.linspace(20.0, 3000.0, 1000)

    # Baseline synthetic spectrum
    # Baseline noise floor
    base_floor = 0.05 * baseline_rms
    base_spec = np.full_like(freqs, base_floor) + np.random.normal(0, 0.01 * baseline_rms, len(freqs))

    # Baseline tooth passing harmonics
    for harmonic in [1, 2, 3]:
        h_freq = harmonic * baseline_ft
        base_spec += (0.4 * baseline_rms / harmonic) * np.exp(-0.5 * ((freqs - h_freq) / 12.0) ** 2)

    # Chatter resonance peak if unstable / warning
    if baseline_rms > 0.8:
        chatter_amp = baseline_rms * 1.8
        base_spec += chatter_amp * np.exp(-0.5 * ((freqs - natural_freq_hz) / 18.0) ** 2)

    ax.plot(freqs, np.maximum(base_spec, 0), color=COLOR_HIGH_RISK, linewidth=1.8, label="Baseline Spectrum", alpha=0.9)

    # Recommended spectrum
    if rec_ft is not None and rec_rms is not None:
        rec_floor = 0.04 * rec_rms
        rec_spec = np.full_like(freqs, rec_floor) + np.random.normal(0, 0.008 * rec_rms, len(freqs))
        for harmonic in [1, 2, 3]:
            h_freq = harmonic * rec_ft
            rec_spec += (0.5 * rec_rms / harmonic) * np.exp(-0.5 * ((freqs - h_freq) / 12.0) ** 2)

        ax.plot(freqs, np.maximum(rec_spec, 0), color=COLOR_RECOMMENDED, linewidth=1.8, label="Recommended Spectrum (Suppressed)", alpha=0.9)

    # Highlight structural natural frequency
    ax.axvline(natural_freq_hz, color="#FF7B72", linestyle="--", linewidth=1.2, alpha=0.7, label=f"Natural Mode fn ({natural_freq_hz:.0f} Hz)")
    ax.axvline(baseline_ft, color="#79C0FF", linestyle=":", linewidth=1.2, alpha=0.7, label=f"Baseline Tooth Pass ft ({baseline_ft:.0f} Hz)")

    ax.set_xlim(20.0, 2500.0)
    ax.set_ylim(0, max(base_spec.max() * 1.25, 1.0))
    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor=PANEL_COLOR,
        edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR,
        fontsize=8.5,
    )
    plt.tight_layout()
    return fig


def plot_tradeoff_pareto(
    feasible_candidates: List[Any],
    baseline_mrr: float,
    baseline_risk: float,
    recommended_candidate: Optional[Any] = None,
) -> plt.Figure:
    """
    Scatter plot illustrating the engineering trade-off between Material Removal Rate (MRR)
    and Chatter Risk Score.
    """
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=120)
    apply_industrial_style(
        ax,
        title="Multi-Objective Trade-Off: Productivity (MRR) vs. Chatter Risk",
        xlabel="Chatter Risk Score (0 = Completely Stable, 100 = Severe Chatter)",
        ylabel="Material Removal Rate (cm3/min)",
    )

    if feasible_candidates:
        risks = [c.risk_score for c in feasible_candidates]
        mrrs = [c.mrr_cm3_min for c in feasible_candidates]
        scores = [c.composite_score for c in feasible_candidates]

        scatter = ax.scatter(
            risks,
            mrrs,
            c=scores,
            cmap="viridis",
            s=70,
            alpha=0.85,
            edgecolors=GRID_COLOR,
            label="Feasible Candidates",
            zorder=3,
        )
        cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label("Composite Ranking Score", color=TEXT_COLOR, fontsize=9)
        cbar.ax.yaxis.set_tick_params(color=MUTED_TEXT, labelsize=8)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=MUTED_TEXT)

    # Plot baseline
    ax.scatter(
        [baseline_risk],
        [baseline_mrr],
        color=COLOR_HIGH_RISK,
        s=160,
        marker="X",
        edgecolors="#FFFFFF",
        linewidths=1.5,
        label=f"Baseline (Risk {baseline_risk:.1f}, MRR {baseline_mrr:.1f})",
        zorder=5,
    )

    # Plot recommended
    if recommended_candidate is not None:
        ax.scatter(
            [recommended_candidate.risk_score],
            [recommended_candidate.mrr_cm3_min],
            color=COLOR_RECOMMENDED,
            s=240,
            marker="*",
            edgecolors="#FFFFFF",
            linewidths=1.5,
            label=f"Recommended (Risk {recommended_candidate.risk_score:.1f}, MRR {recommended_candidate.mrr_cm3_min:.1f})",
            zorder=6,
        )

    ax.axvspan(0, 30, color=COLOR_STABLE, alpha=0.08, label="Low Risk Zone")
    ax.axvspan(30, 65, color=COLOR_WARNING, alpha=0.08, label="Warning Zone")
    ax.axvspan(65, 100, color=COLOR_HIGH_RISK, alpha=0.08, label="High Risk Zone")

    ax.set_xlim(-2, 102)
    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor=PANEL_COLOR,
        edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR,
        fontsize=8.0,
    )
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, classes: List[str]) -> plt.Figure:
    """Plot Confusion Matrix Heatmap."""
    fig, ax = plt.subplots(figsize=(5.5, 4.4), dpi=120)
    apply_industrial_style(
        ax,
        title="Model Hold-Out Confusion Matrix",
        xlabel="Predicted Label",
        ylabel="True Label",
    )

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color=MUTED_TEXT, labelsize=8)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=MUTED_TEXT)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, color=TEXT_COLOR, fontsize=9)
    ax.set_yticklabels(classes, color=TEXT_COLOR, fontsize=9)

    # Text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            text_color = "white" if val > thresh else "#24292F"
            ax.text(
                j, i, f"{val}",
                ha="center", va="center",
                color=text_color, fontweight="bold", fontsize=11,
            )

    plt.tight_layout()
    return fig


def plot_feature_importances(feature_importances: Dict[str, float]) -> plt.Figure:
    """Plot horizontal bar chart of feature importances."""
    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=120)
    apply_industrial_style(
        ax,
        title="Random Forest Feature Importance",
        xlabel="Gini Relative Importance",
        ylabel="Feature Name",
    )

    # Sort descending
    sorted_items = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    names = [k.replace("_", " ").title() for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, align="center", color=COLOR_ACCENT, edgecolor=GRID_COLOR, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, color=TEXT_COLOR, fontsize=9)
    ax.invert_yaxis()

    for i, v in enumerate(values):
        ax.text(v + 0.005, i, f"{v * 100:.1f}%", va="center", color=TEXT_COLOR, fontsize=8.5, fontweight="bold")

    ax.set_xlim(0, max(values) * 1.25)
    plt.tight_layout()
    return fig
