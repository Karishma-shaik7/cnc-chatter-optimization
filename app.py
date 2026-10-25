"""
YUKTHI 2026: CNC Chatter Suppression and Tool-Path Optimization
Industrial Diagnostic Dashboard & Optimization Prototype

Notice:
- Algorithmic and Physical Simulation Prototype.
- Uses domain-physics-grounded synthetic data and illustrative trade-off models.
- Not connected to, nor controlling, a physical CNC machine.
"""

from typing import Dict, Any, List, Optional
from dataclasses import asdict
import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from core.physics import (
    calculate_chip_load,
    calculate_mrr,
    calculate_cutting_speed,
    calculate_tooth_passing_frequency,
    estimate_cutting_force_and_power,
    compute_stability_lobe_curve,
    get_stability_limit_at_rpm,
)
from core.bounds import (
    MachineToolingBounds,
    check_candidate_feasibility,
)
from core.synthetic_data import (
    generate_synthetic_dataset,
    LABEL_STABLE,
    LABEL_WARNING,
    LABEL_HIGH_RISK,
    VALID_LABELS,
)
from core.model import (
    ChatterClassifier,
    get_trained_default_model,
    FEATURE_COLUMNS,
)
from core.optimizer import (
    ChatterOptimizationEngine,
    MachiningCandidate,
)
from visualization.plots import (
    plot_stability_lobes,
    plot_vibration_spectrum,
    plot_tradeoff_pareto,
    plot_confusion_matrix,
    plot_feature_importances,
)

# Page configuration
st.set_page_config(
    page_title="CNC Chatter Suppression & Optimization | YUKTHI 2026",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Industrial Styling
st.markdown(
    """
    <style>
    /* Global background and typography */
    .stApp {
        background-color: #0E1117;
        color: #E6EDF3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header banner styling */
    .banner-title {
        background: linear-gradient(90deg, #161B22 0%, #1F242C 100%);
        border: 1px solid #30363D;
        border-left: 5px solid #58A6FF;
        border-radius: 8px;
        padding: 16px 22px;
        margin-bottom: 18px;
    }
    .banner-title h1 {
        color: #FFFFFF;
        font-size: 1.7rem;
        font-weight: 700;
        margin: 0;
        padding: 0;
    }
    .banner-title p {
        color: #8B949E;
        font-size: 0.88rem;
        margin: 6px 0 0 0;
    }

    /* Disclaimer box */
    .disclaimer-card {
        background-color: #161B22;
        border: 1px solid #D29922;
        border-left: 5px solid #D29922;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 0.82rem;
        color: #E3B341;
        margin-bottom: 20px;
    }

    /* KPI metric cards */
    .metric-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    }
    .metric-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        color: #8B949E;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #F0F6FC;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 0.76rem;
        color: #8B949E;
        margin-top: 2px;
    }

    /* State banners */
    .state-banner-stable {
        background-color: rgba(46, 160, 67, 0.15);
        border: 1px solid #2EA043;
        border-radius: 8px;
        padding: 14px 18px;
        color: #3FB950;
        font-weight: 700;
        font-size: 1.25rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .state-banner-warning {
        background-color: rgba(210, 153, 34, 0.15);
        border: 1px solid #D29922;
        border-radius: 8px;
        padding: 14px 18px;
        color: #E3B341;
        font-weight: 700;
        font-size: 1.25rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .state-banner-risk {
        background-color: rgba(248, 81, 73, 0.15);
        border: 1px solid #F85149;
        border-radius: 8px;
        padding: 14px 18px;
        color: #FF7B72;
        font-weight: 700;
        font-size: 1.25rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    /* Delta badges */
    .badge-good {
        background-color: rgba(46, 160, 67, 0.2);
        color: #3FB950;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-bad {
        background-color: rgba(248, 81, 73, 0.2);
        color: #FF7B72;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Cached Synthetic Data Generation & Model Training
@st.cache_resource(show_spinner=False)
def load_trained_model_and_data(n_samples: int, seed: int):
    """
    Cached loader that produces a reproducible synthetic dataset and fits
    the Scikit-Learn Random Forest Classifier.
    """
    df = generate_synthetic_dataset(n_samples=n_samples, random_state=seed)
    clf = ChatterClassifier(n_estimators=100, max_depth=12, random_state=seed)
    clf.fit(df)
    return clf, df


# Initialize session state for preset parameters
if "preset_loaded" not in st.session_state:
    st.session_state["preset_loaded"] = False
    st.session_state["spindle_speed"] = 4200.0
    st.session_state["feed_rate"] = 1450.0
    st.session_state["depth_of_cut"] = 4.20
    st.session_state["vibration_rms"] = 3.25
    st.session_state["vibration_peak"] = 13.10
    st.session_state["dominant_freq"] = 850.0


def apply_preset(rpm: float, feed: float, depth: float, rms: float, peak: float, freq: float):
    st.session_state["spindle_speed"] = float(rpm)
    st.session_state["feed_rate"] = float(feed)
    st.session_state["depth_of_cut"] = float(depth)
    st.session_state["vibration_rms"] = float(rms)
    st.session_state["vibration_peak"] = float(peak)
    st.session_state["dominant_freq"] = float(freq)


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ CNC System & Tooling")

    # Preset selection
    st.markdown("**Load Preset Scenario:**")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        if st.button("🔴 Chatter", use_container_width=True, help="Load severe chatter condition"):
            apply_preset(4200.0, 1450.0, 4.20, 3.25, 13.10, 850.0)
            st.rerun()
    with col_p2:
        if st.button("🟡 Warning", use_container_width=True, help="Load marginal onset condition"):
            apply_preset(5800.0, 1200.0, 2.80, 1.35, 4.20, 840.0)
            st.rerun()
    with col_p3:
        if st.button("🟢 Stable", use_container_width=True, help="Load stable cutting condition"):
            apply_preset(6375.0, 1800.0, 2.20, 0.38, 0.95, 425.0)
            st.rerun()

    st.markdown("---")
    st.markdown("#### 📐 Tooling & Physics Parameters")
    tool_diameter = st.number_input("Tool Diameter D (mm)", min_value=3.0, max_value=50.0, value=12.0, step=1.0)
    num_flutes = st.number_input("Number of Flutes (N)", min_value=1, max_value=8, value=4, step=1)
    radial_depth = st.number_input("Radial Depth of Cut ae (mm)", min_value=0.5, max_value=tool_diameter, value=6.0, step=0.5)
    natural_freq = st.number_input("Dominant Natural Freq fn (Hz)", min_value=200.0, max_value=3500.0, value=850.0, step=25.0)

    st.markdown("---")
    st.markdown("#### 🛑 Machine Configured Limits (Bound Checks)")
    min_rpm = st.number_input("Min Spindle Speed (RPM)", value=1200.0, step=100.0)
    max_rpm = st.number_input("Max Spindle Speed (RPM)", value=15000.0, step=500.0)
    min_feed = st.number_input("Min Feed Rate (mm/min)", value=100.0, step=50.0)
    max_feed = st.number_input("Max Feed Rate (mm/min)", value=3800.0, step=100.0)
    max_depth = st.number_input("Max Flute Cut Depth ap (mm)", value=6.0, step=0.5)
    max_power = st.number_input("Spindle Continuous Power (kW)", value=7.5, step=0.5)
    min_chip_load = st.number_input("Min Chip Load fz (mm/tooth)", value=0.015, step=0.005, format="%.3f")
    max_chip_load = st.number_input("Max Chip Load fz (mm/tooth)", value=0.160, step=0.010, format="%.3f")

    st.markdown("---")
    st.markdown("#### 🎯 Optimization Objectives")
    w_stability = st.slider("Stability / Chatter Suppression Weight", 0.0, 1.0, 0.50, 0.05)
    w_productivity = st.slider("Productivity (MRR) Weight", 0.0, 1.0, 0.35, 0.05)
    w_tool_life = st.slider("Tool Life / Wear Factor Weight", 0.0, 1.0, 0.15, 0.05)

    st.markdown("---")
    st.markdown("#### 🧪 Synthetic Data & ML Controls")
    n_samples = st.selectbox("Synthetic Dataset Size", [800, 1500, 3000], index=1)
    dataset_seed = st.number_input("Random Seed (Reproducibility)", value=42, step=1)

    if st.button("🔄 Retrain ML Model", use_container_width=True):
        st.cache_resource.clear()
        st.success("Cache cleared! Model retraining on synthetic dataset.")
        st.rerun()

# Build Bounds Configuration
bounds = MachineToolingBounds(
    min_spindle_speed_rpm=min_rpm,
    max_spindle_speed_rpm=max_rpm,
    max_spindle_power_kw=max_power,
    min_feed_rate_mm_min=min_feed,
    max_feed_rate_mm_min=max_feed,
    min_depth_of_cut_mm=0.30,
    max_depth_of_cut_mm=max_depth,
    tool_diameter_mm=tool_diameter,
    num_flutes=int(num_flutes),
    radial_depth_mm=radial_depth,
    min_chip_load_mm_tooth=min_chip_load,
    max_chip_load_mm_tooth=max_chip_load,
)

# Load / Train Random Forest Classifier
with st.spinner("Loading synthetic dataset and training Random Forest model..."):
    clf, synthetic_df = load_trained_model_and_data(n_samples=n_samples, seed=int(dataset_seed))

optimizer_engine = ChatterOptimizationEngine(
    bounds=bounds,
    model=clf,
    natural_freq_hz=natural_freq,
)

# ==========================================
# MAIN DASHBOARD CONTENT
# ==========================================

# Top Banner
st.markdown(
    """
    <div class="banner-title">
        <h1>⚙️ CNC Chatter Suppression & Tool-Path Optimization</h1>
        <p>YUKTHI 2026 Hackathon Prototype | Real-Time Milling Dynamics, Machine Learning Diagnosis & Multi-Objective Parameter Tuning</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Mandatory Prototype Disclaimer Card
st.markdown(
    """
    <div class="disclaimer-card">
        <strong>⚠️ SIMULATION & ALGORITHMIC PROTOTYPE DISCLAIMER:</strong>
        This software is an engineering simulation prototype. All datasets, sensor signals, and frequency spectra are 
        <strong>synthetically generated</strong> using Altintas-Budak regenerative chatter theory approximations and forced vibration models. 
        Calculated metrics (MRR, cycle time delta, tool wear) are illustrative trade-off estimates for algorithmic demonstration. 
        <strong>This application is NOT connected to and does NOT control any physical CNC machinery.</strong> No real-world machining performance claims are made.
    </div>
    """,
    unsafe_allow_html=True,
)

# Section 1: Real-Time Machining Inputs & Sensor Features
st.markdown("### 🎛️ Baseline Machining Settings & Sensor Signals")
with st.container():
    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        inp_rpm = st.number_input(
            "Spindle Speed (RPM)",
            min_value=500.0,
            max_value=25000.0,
            value=float(st.session_state["spindle_speed"]),
            step=100.0,
            help="Spindle rotational frequency",
        )
        inp_rms = st.number_input(
            "Vibration RMS (g)",
            min_value=0.01,
            max_value=15.0,
            value=float(st.session_state["vibration_rms"]),
            step=0.1,
            format="%.2f",
            help="Root mean square vibration acceleration",
        )

    with col_in2:
        inp_feed = st.number_input(
            "Feed Rate (mm/min)",
            min_value=50.0,
            max_value=10000.0,
            value=float(st.session_state["feed_rate"]),
            step=50.0,
            help="Workpiece table traverse speed",
        )
        inp_peak = st.number_input(
            "Vibration Peak (g)",
            min_value=0.05,
            max_value=40.0,
            value=float(st.session_state["vibration_peak"]),
            step=0.2,
            format="%.2f",
            help="Peak vibration acceleration",
        )

    with col_in3:
        inp_depth = st.number_input(
            "Axial Depth of Cut ap (mm)",
            min_value=0.1,
            max_value=15.0,
            value=float(st.session_state["depth_of_cut"]),
            step=0.1,
            format="%.2f",
            help="Axial engagement along tool axis",
        )
        inp_dom_freq = st.number_input(
            "Dominant Frequency (Hz)",
            min_value=10.0,
            max_value=5000.0,
            value=float(st.session_state["dominant_freq"]),
            step=10.0,
            format="%.1f",
            help="Dominant peak in the vibration FFT spectrum",
        )

# Calculate live physical indicators for the baseline
live_fz = calculate_chip_load(inp_feed, inp_rpm, bounds.num_flutes)
live_vc = calculate_cutting_speed(inp_rpm, bounds.tool_diameter_mm)
live_ft = calculate_tooth_passing_frequency(inp_rpm, bounds.num_flutes)
live_mrr = calculate_mrr(inp_depth, bounds.radial_depth_mm, inp_feed)
live_fc, live_power = estimate_cutting_force_and_power(
    axial_depth_mm=inp_depth,
    radial_depth_mm=bounds.radial_depth_mm,
    feed_rate_mm_min=inp_feed,
    spindle_speed_rpm=inp_rpm,
    num_flutes=bounds.num_flutes,
    tool_diameter_mm=bounds.tool_diameter_mm,
)
baseline_bounds_check = check_candidate_feasibility(inp_rpm, inp_feed, inp_depth, bounds)

# KPI Metrics Bar for Baseline Physical State
kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5 = st.columns(5)
with kpi_c1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Chip Load (fz)</div>
            <div class="metric-value">{live_fz:.4f} <span style="font-size:0.9rem; font-weight:normal;">mm/t</span></div>
            <div class="metric-sub">Allowable: [{bounds.min_chip_load_mm_tooth:.3f} - {bounds.max_chip_load_mm_tooth:.3f}]</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi_c2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Tooth Pass Freq (ft)</div>
            <div class="metric-value">{live_ft:.1f} <span style="font-size:0.9rem; font-weight:normal;">Hz</span></div>
            <div class="metric-sub">Natural Mode: {natural_freq:.0f} Hz</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi_c3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Cutting Speed (Vc)</div>
            <div class="metric-value">{live_vc:.1f} <span style="font-size:0.9rem; font-weight:normal;">m/min</span></div>
            <div class="metric-sub">D = {bounds.tool_diameter_mm:.0f} mm</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi_c4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Est. Cutting Power</div>
            <div class="metric-value">{live_power:.2f} <span style="font-size:0.9rem; font-weight:normal;">kW</span></div>
            <div class="metric-sub">Spindle Limit: {bounds.max_spindle_power_kw:.1f} kW</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi_c5:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Removal Rate (MRR)</div>
            <div class="metric-value">{live_mrr:.1f} <span style="font-size:0.9rem; font-weight:normal;">cm³/min</span></div>
            <div class="metric-sub">ae = {bounds.radial_depth_mm:.1f} mm</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# Run Inference with Random Forest Model
baseline_input_dict = {
    "spindle_speed_rpm": inp_rpm,
    "feed_rate_mm_min": inp_feed,
    "depth_of_cut_mm": inp_depth,
    "vibration_rms_g": inp_rms,
    "vibration_peak_g": inp_peak,
    "dominant_frequency_hz": inp_dom_freq,
}
baseline_pred = clf.predict(baseline_input_dict)
pred_state = baseline_pred["predicted_state"]
pred_probs = baseline_pred["probabilities"]
risk_score = baseline_pred["risk_score"]

# Section 2: Real-Time Machine Learning Diagnostic Card
st.markdown("### 🔍 Real-Time ML Chatter Diagnosis")
col_diag1, col_diag2 = st.columns([1.2, 1.0])

with col_diag1:
    if pred_state == LABEL_STABLE:
        st.markdown(
            f"""
            <div class="state-banner-stable">
                <span>🟢</span>
                <div>
                    <div>STATE: STABLE MACHINING</div>
                    <div style="font-size:0.85rem; font-weight:normal; color:#8B949E;">
                        Vibration within nominal threshold. Tool tooth passing harmonics dominate the spectrum.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif pred_state == LABEL_WARNING:
        st.markdown(
            f"""
            <div class="state-banner-warning">
                <span>🟡</span>
                <div>
                    <div>STATE: WARNING - INCIPIENT CHATTER</div>
                    <div style="font-size:0.85rem; font-weight:normal; color:#8B949E;">
                        Operating near stability boundary. Transient vibration spikes and surface finish degradation detected.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="state-banner-risk">
                <span>🔴</span>
                <div>
                    <div>STATE: HIGH RISK - SEVERE REGENERATIVE CHATTER</div>
                    <div style="font-size:0.85rem; font-weight:normal; color:#8B949E;">
                        Vibration locked onto tool structural resonance ({natural_freq:.0f} Hz). High probability of tool chipping, spindle overload, or scrap part.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Bound feasibility warning
    if not baseline_bounds_check.is_feasible:
        st.warning(f"⚠️ Baseline violates configured bounds: {'; '.join(baseline_bounds_check.rejection_reasons)}")
    else:
        st.success("✅ Baseline parameters are within machine, tooling, and power constraints.")

with col_diag2:
    st.markdown("**Random Forest Class Probabilities:**")
    p_stable = pred_probs.get(LABEL_STABLE, 0.0)
    p_warn = pred_probs.get(LABEL_WARNING, 0.0)
    p_risk = pred_probs.get(LABEL_HIGH_RISK, 0.0)

    st.progress(p_stable, text=f"Stable: {p_stable * 100:.1f}%")
    st.progress(p_warn, text=f"Warning: {p_warn * 100:.1f}%")
    st.progress(p_risk, text=f"High Risk: {p_risk * 100:.1f}%")
    st.caption(f"Composite Chatter Risk Score: **{risk_score:.1f} / 100** | Classifier Confidence: **{baseline_pred['confidence']:.1f}%**")

st.markdown("---")

# Section 3: Optimization Engine & Candidate Generation
optimization_results = optimizer_engine.generate_and_rank_candidates(
    baseline_rpm=inp_rpm,
    baseline_feed=inp_feed,
    baseline_depth=inp_depth,
    baseline_rms=inp_rms,
    baseline_peak=inp_peak,
    baseline_dom_freq=inp_dom_freq,
    weight_stability=w_stability,
    weight_mrr=w_productivity,
    weight_tool_life=w_tool_life,
)

recommended = optimization_results["recommended"]
feasible_cands = optimization_results["feasible_candidates"]
rejected_cands = optimization_results["rejected_candidates"]

st.markdown("### 🚀 Optimization Engine & Recommended Alternative")

if recommended is not None:
    # Side-by-side comparison table
    st.markdown(
        f"""
        <div style="background-color:#161B22; border:1px solid #30363D; border-radius:8px; padding:16px; margin-bottom:18px;">
            <div style="font-size:1.15rem; font-weight:700; color:#58A6FF; margin-bottom:12px;">
                🏆 Best Recommended Alternative: <span style="color:#39D353;">{recommended.strategy_name}</span> (Candidate {recommended.candidate_id})
            </div>
            <div style="font-size:0.86rem; color:#8B949E; margin-bottom:14px;">
                Identified via stability lobe harmonic pocket analysis and multi-objective Pareto trade-off optimization.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_comp1, col_comp2, col_comp3 = st.columns([1.1, 1.1, 1.2])

    with col_comp1:
        st.markdown("**Current Baseline Setting:**")
        st.write(f"- Spindle Speed: **{inp_rpm:.0f} RPM**")
        st.write(f"- Feed Rate: **{inp_feed:.1f} mm/min**")
        st.write(f"- Depth of Cut ap: **{inp_depth:.2f} mm**")
        st.write(f"- Chip Load fz: **{live_fz:.4f} mm/tooth**")
        st.write(f"- Material Removal Rate: **{live_mrr:.2f} cm³/min**")
        st.write(f"- Predicted State: **{pred_state}**")
        st.write(f"- Chatter Risk Score: **{risk_score:.1f} / 100**")
        st.write(f"- Est. Cutting Power: **{live_power:.2f} kW**")

    with col_comp2:
        st.markdown("**Recommended Setting:**")
        st.write(f"- Spindle Speed: **{recommended.spindle_speed_rpm:.0f} RPM**")
        st.write(f"- Feed Rate: **{recommended.feed_rate_mm_min:.1f} mm/min**")
        st.write(f"- Depth of Cut ap: **{recommended.depth_of_cut_mm:.2f} mm**")
        st.write(f"- Chip Load fz: **{recommended.chip_load_mm_tooth:.4f} mm/tooth**")
        st.write(f"- Material Removal Rate: **{recommended.mrr_cm3_min:.2f} cm³/min**")
        st.write(f"- Predicted State: **{recommended.predicted_state}**")
        st.write(f"- Chatter Risk Score: **{recommended.risk_score:.1f} / 100**")
        st.write(f"- Est. Cutting Power: **{recommended.cutting_power_kw:.2f} kW**")

    with col_comp3:
        st.markdown("**Illustrative Improvement Deltas:**")
        # Chatter Risk Delta
        d_risk = recommended.risk_delta_pts
        risk_badge = f"<span class='badge-good'>{d_risk:+.1f} pts (Chatter Suppressed)</span>" if d_risk < 0 else f"<span class='badge-bad'>{d_risk:+.1f} pts</span>"
        st.markdown(f"- Chatter Risk Change: {risk_badge}", unsafe_allow_html=True)

        # MRR Delta
        d_mrr = recommended.mrr_delta_pct
        mrr_badge = f"<span class='badge-good'>{d_mrr:+.1f}%</span>" if d_mrr >= 0 else f"<span class='badge-bad'>{d_mrr:+.1f}%</span>"
        st.markdown(f"- Productivity (MRR) Change: {mrr_badge}", unsafe_allow_html=True)

        # Cycle Time Delta
        d_cycle = recommended.cycle_time_delta_pct
        cycle_badge = f"<span class='badge-good'>{d_cycle:+.1f}% (Faster)</span>" if d_cycle <= 0 else f"<span class='badge-bad'>{d_cycle:+.1f}% (Slower)</span>"
        st.markdown(f"- Estimated Cycle Time Change: {cycle_badge}", unsafe_allow_html=True)

        # Composite Score
        st.markdown(f"- Multi-Objective Composite Score: **{recommended.composite_score:.1f} / 100**")
        st.markdown(f"- Tool Wear Index: **{recommended.wear_factor:.2f}** (baseline wear ~ 1.00)")

        # Engineering explanation
        rpm_shift = recommended.spindle_speed_rpm - inp_rpm
        st.info(
            f"💡 **Optimization Strategy:** Shifted spindle by {rpm_shift:+.0f} RPM to leverage stability pocket, "
            f"maintaining chip load ({recommended.chip_load_mm_tooth:.3f} mm/tooth) inside safe shearing regime."
        )

else:
    st.error("No feasible candidates found within the configured machine limits. Please widen the bounds.")

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# Section 4: Multi-Tab Interactive Diagnostics and Visualizations
st.markdown("### 📊 Interactive Diagnostics & Visualizations")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Stability Lobe Diagram (SLD)",
    "🔊 Vibration Frequency Spectrum",
    "⚖️ Multi-Objective Trade-Off Frontier",
    "📋 Candidate Audit & Bound Checks",
    "🤖 ML Model Performance & Explainability",
])

with tab1:
    st.markdown("#### Analytical Stability Lobe Diagram (Altintas Regenerative Model)")
    st.caption("Visualizes the boundary between stable cutting and regenerative chatter pockets across the spindle speed range.")

    rpm_grid = np.linspace(bounds.min_spindle_speed_rpm, bounds.max_spindle_speed_rpm, 160)
    _, ap_curve, sweet_rpms = compute_stability_lobe_curve(
        rpm_range=rpm_grid,
        natural_freq_hz=natural_freq,
        num_flutes=bounds.num_flutes,
    )

    feasible_rpms = [c.spindle_speed_rpm for c in feasible_cands]
    feasible_depths = [c.depth_of_cut_mm for c in feasible_cands]

    fig_sld = plot_stability_lobes(
        rpm_grid=rpm_grid,
        ap_curve=ap_curve,
        sweet_rpms=sweet_rpms,
        baseline_rpm=inp_rpm,
        baseline_depth=inp_depth,
        recommended_rpm=recommended.spindle_speed_rpm if recommended else None,
        recommended_depth=recommended.depth_of_cut_mm if recommended else None,
        feasible_rpms=feasible_rpms,
        feasible_depths=feasible_depths,
    )
    st.pyplot(fig_sld, clear_figure=True)
    plt.close(fig_sld)

with tab2:
    st.markdown("#### Simulated Vibration Frequency Spectrum (FFT)")
    st.caption("Shows frequency peaks: baseline tooth passing harmonics vs. chatter resonance locked to natural mode.")

    fig_fft = plot_vibration_spectrum(
        baseline_ft=live_ft,
        baseline_dom_freq=inp_dom_freq,
        baseline_rms=inp_rms,
        natural_freq_hz=natural_freq,
        rec_ft=recommended.tooth_pass_freq_hz if recommended else None,
        rec_rms=recommended.est_vibration_rms_g if recommended else None,
    )
    st.pyplot(fig_fft, clear_figure=True)
    plt.close(fig_fft)

with tab3:
    st.markdown("#### Trade-Off Analysis: Material Removal Rate vs. Chatter Risk")
    st.caption("Illustrates the Pareto trade-off between maximizing production throughput (MRR) and minimizing chatter probability.")

    fig_pareto = plot_tradeoff_pareto(
        feasible_candidates=feasible_cands,
        baseline_mrr=live_mrr,
        baseline_risk=risk_score,
        recommended_candidate=recommended,
    )
    st.pyplot(fig_pareto, clear_figure=True)
    plt.close(fig_pareto)

with tab4:
    st.markdown("#### Optimization Candidate Audit & Bounds Filtering")
    st.caption(
        f"Generated {optimization_results['total_candidates_generated']} total candidates: "
        f"**{len(feasible_cands)} Feasible**, **{len(rejected_cands)} Rejected by configured bounds**."
    )

    st.markdown("**Feasible Machining Candidates (Ranked by Multi-Objective Score):**")
    if feasible_cands:
        feasible_records = [
            {
                "Candidate ID": c.candidate_id,
                "Strategy": c.strategy_name,
                "Spindle (RPM)": c.spindle_speed_rpm,
                "Feed (mm/min)": c.feed_rate_mm_min,
                "Depth ap (mm)": c.depth_of_cut_mm,
                "Chip Load (mm/t)": c.chip_load_mm_tooth,
                "MRR (cm³/min)": c.mrr_cm3_min,
                "State": c.predicted_state,
                "Risk Score": c.risk_score,
                "MRR Δ (%)": f"{c.mrr_delta_pct:+.1f}%",
                "Cycle Time Δ (%)": f"{c.cycle_time_delta_pct:+.1f}%",
                "Composite Score": c.composite_score,
            }
            for c in feasible_cands
        ]
        st.dataframe(pd.DataFrame(feasible_records), use_container_width=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("**Rejected Infeasible Candidates (Bound Violations Log):**")
    if rejected_cands:
        rejected_records = [
            {
                "Candidate ID": c.candidate_id,
                "Proposed Strategy": c.strategy_name,
                "Spindle (RPM)": c.spindle_speed_rpm,
                "Feed (mm/min)": c.feed_rate_mm_min,
                "Depth ap (mm)": c.depth_of_cut_mm,
                "Chip Load (mm/t)": c.chip_load_mm_tooth,
                "Est. Power (kW)": c.cutting_power_kw,
                "Rejection Reasons": " | ".join(c.rejection_reasons),
            }
            for c in rejected_cands
        ]
        st.dataframe(pd.DataFrame(rejected_records), use_container_width=True)

with tab5:
    st.markdown("#### Random Forest Classifier Evaluation Metrics")
    st.caption(f"Trained on {clf.metrics_.get('train_samples', 0)} synthetic samples; evaluated on {clf.metrics_.get('test_samples', 0)} hold-out samples.")

    col_m1, col_m2 = st.columns([1.0, 1.3])
    with col_m1:
        acc = clf.metrics_.get("accuracy", 0.0)
        st.metric("Hold-Out Overall Accuracy", f"{acc * 100:.2f}%")

        # Class breakdown table
        class_metrics = clf.metrics_.get("class_metrics", {})
        if class_metrics:
            cm_df = pd.DataFrame(class_metrics).T[["precision", "recall", "f1_score", "support"]]
            cm_df.columns = ["Precision", "Recall", "F1-Score", "Support"]
            st.dataframe(cm_df.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1-Score": "{:.3f}"}), use_container_width=True)

        # Confusion Matrix
        cm_data = np.array(clf.metrics_.get("confusion_matrix", [[0]]))
        fig_cm = plot_confusion_matrix(cm_data, clf.classes_)
        st.pyplot(fig_cm, clear_figure=True)
        plt.close(fig_cm)

    with col_m2:
        st.markdown("**Feature Importance Analysis (Gini Impurity):**")
        fig_imp = plot_feature_importances(clf.feature_importances_)
        st.pyplot(fig_imp, clear_figure=True)
        plt.close(fig_imp)

st.markdown("---")

# Section 5: Data Exports & Downloadable Results
st.markdown("### 📥 Downloadable Results & Datasets")

col_dl1, col_dl2, col_dl3 = st.columns(3)

with col_dl1:
    # Download evaluated candidates CSV
    all_evaluated = [asdict(c) for c in (feasible_cands + rejected_cands)]
    evaluated_df = pd.DataFrame(all_evaluated)
    csv_candidates = evaluated_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Evaluated Candidates (CSV)",
        data=csv_candidates,
        file_name="cnc_chatter_optimization_candidates.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col_dl2:
    # Download synthetic training dataset CSV
    csv_synthetic = synthetic_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Synthetic Dataset (CSV)",
        data=csv_synthetic,
        file_name="cnc_chatter_synthetic_training_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col_dl3:
    # Download executive summary report
    report_text = f"""=======================================================
CNC CHATTER SUPPRESSION & TOOL-PATH OPTIMIZATION REPORT
YUKTHI 2026 Hackathon Prototype
=======================================================
Disclaimer: Simulation prototype with synthetic data.
Estimated metrics are illustrative trade-off models.
Not connected to physical machine hardware.

BASELINE MACHINING CONDITION:
- Spindle Speed: {inp_rpm:.0f} RPM
- Feed Rate: {inp_feed:.1f} mm/min
- Axial Depth of Cut: {inp_depth:.2f} mm
- Vibration RMS: {inp_rms:.2f} g
- Vibration Peak: {inp_peak:.2f} g
- Dominant Frequency: {inp_dom_freq:.1f} Hz
- Predicted State: {pred_state}
- Chatter Risk Score: {risk_score:.1f} / 100
- Baseline MRR: {live_mrr:.2f} cm3/min
- Baseline Power: {live_power:.2f} kW

RECOMMENDED ALTERNATIVE:
- Strategy: {recommended.strategy_name if recommended else 'N/A'}
- Candidate ID: {recommended.candidate_id if recommended else 'N/A'}
- Recommended Spindle Speed: {recommended.spindle_speed_rpm if recommended else 0:.0f} RPM
- Recommended Feed Rate: {recommended.feed_rate_mm_min if recommended else 0:.1f} mm/min
- Recommended Depth of Cut: {recommended.depth_of_cut_mm if recommended else 0:.2f} mm
- Recommended Chip Load: {recommended.chip_load_mm_tooth if recommended else 0:.4f} mm/tooth
- Recommended MRR: {recommended.mrr_cm3_min if recommended else 0:.2f} cm3/min
- Predicted State: {recommended.predicted_state if recommended else 'N/A'}
- Chatter Risk Score: {recommended.risk_score if recommended else 0:.1f} / 100
- Chatter Risk Delta: {recommended.risk_delta_pts if recommended else 0:+.1f} pts
- MRR Delta: {recommended.mrr_delta_pct if recommended else 0:+.1f} %
- Cycle Time Delta: {recommended.cycle_time_delta_pct if recommended else 0:+.1f} %
- Multi-Objective Composite Score: {recommended.composite_score if recommended else 0:.1f} / 100

FEASIBILITY SUMMARY:
- Total Candidates Generated: {optimization_results['total_candidates_generated']}
- Feasible Candidates: {len(feasible_cands)}
- Rejected Infeasible Candidates: {len(rejected_cands)}
=======================================================
"""
    st.download_button(
        label="📥 Download Optimization Summary (TXT)",
        data=report_text.encode("utf-8"),
        file_name="cnc_optimization_summary_report.txt",
        mime="text/plain",
        use_container_width=True,
    )
