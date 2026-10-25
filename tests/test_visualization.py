import pytest
import numpy as np
import matplotlib.pyplot as plt
from visualization.plots import (
    plot_stability_lobes,
    plot_vibration_spectrum,
    plot_tradeoff_pareto,
    plot_confusion_matrix,
    plot_feature_importances,
)


def test_plot_stability_lobes():
    rpm_grid = np.linspace(2000, 12000, 50)
    ap_curve = np.full(50, 2.5)
    fig = plot_stability_lobes(
        rpm_grid=rpm_grid,
        ap_curve=ap_curve,
        sweet_rpms=[4000.0, 8000.0],
        baseline_rpm=5000.0,
        baseline_depth=3.0,
        recommended_rpm=4000.0,
        recommended_depth=2.2,
    )
    assert fig is not None
    plt.close(fig)


def test_plot_vibration_spectrum():
    fig = plot_vibration_spectrum(
        baseline_ft=350.0,
        baseline_dom_freq=850.0,
        baseline_rms=2.2,
        natural_freq_hz=850.0,
        rec_ft=400.0,
        rec_rms=0.45,
    )
    assert fig is not None
    plt.close(fig)


def test_plot_tradeoff_pareto():
    class DummyCandidate:
        risk_score = 12.0
        mrr_cm3_min = 28.5
        composite_score = 88.0

    fig = plot_tradeoff_pareto(
        feasible_candidates=[DummyCandidate()],
        baseline_mrr=20.0,
        baseline_risk=75.0,
        recommended_candidate=DummyCandidate(),
    )
    assert fig is not None
    plt.close(fig)


def test_plot_confusion_matrix():
    cm = np.array([[20, 2, 0], [1, 18, 3], [0, 2, 22]])
    fig = plot_confusion_matrix(cm, classes=["Stable", "Warning", "High Risk"])
    assert fig is not None
    plt.close(fig)


def test_plot_feature_importances():
    importances = {
        "vibration_rms_g": 0.42,
        "vibration_peak_g": 0.28,
        "dominant_frequency_hz": 0.15,
        "depth_of_cut_mm": 0.08,
        "spindle_speed_rpm": 0.05,
        "feed_rate_mm_min": 0.02,
    }
    fig = plot_feature_importances(importances)
    assert fig is not None
    plt.close(fig)
