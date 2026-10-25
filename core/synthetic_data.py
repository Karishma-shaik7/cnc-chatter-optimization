"""
Reproducible Synthetic Machining and Vibration Dataset Generator.
Simulates milling dynamics, regenerative chatter onset, and sensor vibration features.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from .physics import (
    calculate_chip_load,
    calculate_mrr,
    calculate_tooth_passing_frequency,
    calculate_cutting_speed,
    get_stability_limit_at_rpm,
    estimate_cutting_force_and_power,
)

# Label constants
LABEL_STABLE = "Stable"
LABEL_WARNING = "Warning"
LABEL_HIGH_RISK = "High Risk"
VALID_LABELS = [LABEL_STABLE, LABEL_WARNING, LABEL_HIGH_RISK]


def generate_synthetic_dataset(
    n_samples: int = 1200,
    random_state: int = 42,
    natural_freq_hz: float = 850.0,
    tool_diameter_mm: float = 12.0,
    num_flutes: int = 4,
    radial_depth_mm: float = 6.0,
) -> pd.DataFrame:
    """
    Generate a physically grounded, reproducible synthetic dataset for CNC chatter diagnosis.
    
    Features generated:
    - spindle_speed_rpm: Spindle rotational speed (RPM)
    - feed_rate_mm_min: Table feed speed (mm/min)
    - depth_of_cut_mm: Axial depth of cut ap (mm)
    - vibration_rms_g: RMS vibration amplitude (g)
    - vibration_peak_g: Peak vibration amplitude (g)
    - dominant_frequency_hz: Dominant spectral peak frequency (Hz)
    
    Target label:
    - chatter_state: "Stable" | "Warning" | "High Risk"
    
    Physics metadata:
    - chip_load_mm_tooth, mrr_cm3_min, tooth_pass_freq_hz, stability_limit_ap_mm, cutting_power_kw
    """
    rng = np.random.default_rng(random_state)

    # 1. Sample primary controllable machining parameters
    # Mix uniform random exploration with focused operating bands
    spindle_speeds = rng.uniform(2200.0, 13500.0, size=n_samples)
    feed_rates = rng.uniform(300.0, 2600.0, size=n_samples)
    depths_of_cut = rng.uniform(0.5, 5.5, size=n_samples)

    records = []

    for i in range(n_samples):
        rpm = float(spindle_speeds[i])
        feed = float(feed_rates[i])
        ap = float(depths_of_cut[i])

        # Domain physics calculations
        ft = calculate_tooth_passing_frequency(rpm, num_flutes)
        fz = calculate_chip_load(feed, rpm, num_flutes)
        mrr = calculate_mrr(ap, radial_depth_mm, feed)
        fc, power_kw = estimate_cutting_force_and_power(
            axial_depth_mm=ap,
            radial_depth_mm=radial_depth_mm,
            feed_rate_mm_min=feed,
            spindle_speed_rpm=rpm,
            num_flutes=num_flutes,
            tool_diameter_mm=tool_diameter_mm,
        )

        # Altintas stability boundary limit for this RPM
        ap_lim = get_stability_limit_at_rpm(
            spindle_speed_rpm=rpm,
            natural_freq_hz=natural_freq_hz,
            num_flutes=num_flutes,
        )

        depth_ratio = ap / max(ap_lim, 0.1)

        # Baseline forced vibration from tool engagement forces
        base_rms = 0.20 + 0.15 * (ap / 5.0) + 0.20 * (fz / 0.10) + rng.normal(0.0, 0.03)
        base_rms = max(0.12, base_rms)

        # Determine stability state and synthesize vibration signatures
        if depth_ratio < 0.88:
            # Stable machining regime
            state = LABEL_STABLE
            rms = base_rms + rng.uniform(0.02, 0.15)
            crest_factor = rng.uniform(2.1, 2.7)
            peak = rms * crest_factor + rng.normal(0.0, 0.04)

            # Dominant frequency tracks tooth passing frequency or 2nd harmonic
            harmonic_choice = rng.choice([1.0, 2.0], p=[0.75, 0.25])
            dom_freq = ft * harmonic_choice + rng.normal(0.0, 5.0)

        elif depth_ratio <= 1.18:
            # Warning regime: near stability lobe threshold, transient chatter bursts
            state = LABEL_WARNING
            chatter_intensity = (depth_ratio - 0.88) / (1.18 - 0.88)
            rms = 0.80 + 0.85 * chatter_intensity + rng.uniform(-0.10, 0.20)
            crest_factor = rng.uniform(2.8, 3.8)
            peak = rms * crest_factor + rng.normal(0.0, 0.15)

            # Dominant frequency transitions between tooth passing and structural resonance
            if rng.random() > 0.45:
                dom_freq = natural_freq_hz + rng.normal(0.0, 18.0)
            else:
                dom_freq = ft + rng.normal(0.0, 12.0)

        else:
            # High Risk regime: fully developed regenerative chatter
            state = LABEL_HIGH_RISK
            overshoot = min(depth_ratio - 1.18, 3.0)
            rms = 1.85 + 1.20 * overshoot + rng.uniform(0.1, 0.7)
            crest_factor = rng.uniform(3.5, 5.2)
            peak = rms * crest_factor + rng.normal(0.0, 0.35)

            # Dominant frequency firmly locks onto structural resonance (chatter frequency fc ~ fn)
            dom_freq = natural_freq_hz + rng.normal(0.0, 12.0)

        # Sanity clipping to realistic sensor bounds
        rms = float(np.clip(rms, 0.08, 8.5))
        peak = float(np.clip(peak, rms * 1.5, 25.0))
        dom_freq = float(np.clip(dom_freq, 40.0, 4500.0))

        records.append({
            "spindle_speed_rpm": round(rpm, 1),
            "feed_rate_mm_min": round(feed, 1),
            "depth_of_cut_mm": round(ap, 2),
            "vibration_rms_g": round(rms, 3),
            "vibration_peak_g": round(peak, 3),
            "dominant_frequency_hz": round(dom_freq, 1),
            "chatter_state": state,
            # Synthetic domain context
            "chip_load_mm_tooth": round(fz, 4),
            "mrr_cm3_min": round(mrr, 2),
            "tooth_pass_freq_hz": round(ft, 1),
            "stability_limit_ap_mm": round(ap_lim, 2),
            "cutting_force_n": round(fc, 1),
            "cutting_power_kw": round(power_kw, 2),
            "depth_ratio_to_limit": round(depth_ratio, 3),
        })

    df = pd.DataFrame(records)
    # Tag dataframe with metadata attribute
    df.attrs["dataset_type"] = "Physically-grounded Synthetic Simulation Dataset"
    df.attrs["random_state"] = random_state
    return df
