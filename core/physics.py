"""
Domain physics calculations for milling dynamics, regenerative chatter,
cutting forces, chip loads, and stability lobe boundaries.
"""

from typing import Dict, Tuple, List
import numpy as np


def calculate_mrr(axial_depth_mm: float, radial_depth_mm: float, feed_rate_mm_min: float) -> float:
    """
    Calculate Material Removal Rate (MRR) in cm^3/min.
    MRR = (ap * ae * vf) / 1000
    """
    if axial_depth_mm <= 0 or radial_depth_mm <= 0 or feed_rate_mm_min <= 0:
        return 0.0
    return (axial_depth_mm * radial_depth_mm * feed_rate_mm_min) / 1000.0


def calculate_chip_load(feed_rate_mm_min: float, spindle_speed_rpm: float, num_flutes: int) -> float:
    """
    Calculate feed per tooth (chip load) fz in mm/tooth.
    fz = vf / (N * RPM)
    """
    if spindle_speed_rpm <= 0 or num_flutes <= 0:
        return 0.0
    return feed_rate_mm_min / (num_flutes * spindle_speed_rpm)


def calculate_cutting_speed(spindle_speed_rpm: float, tool_diameter_mm: float) -> float:
    """
    Calculate cutting speed Vc in m/min.
    Vc = (pi * D * RPM) / 1000
    """
    if spindle_speed_rpm <= 0 or tool_diameter_mm <= 0:
        return 0.0
    return (np.pi * tool_diameter_mm * spindle_speed_rpm) / 1000.0


def calculate_tooth_passing_frequency(spindle_speed_rpm: float, num_flutes: int) -> float:
    """
    Calculate tooth passing frequency ft in Hz.
    ft = (num_flutes * RPM) / 60
    """
    if spindle_speed_rpm <= 0 or num_flutes <= 0:
        return 0.0
    return (num_flutes * spindle_speed_rpm) / 60.0


def estimate_cutting_force_and_power(
    axial_depth_mm: float,
    radial_depth_mm: float,
    feed_rate_mm_min: float,
    spindle_speed_rpm: float,
    num_flutes: int,
    tool_diameter_mm: float,
    kc_n_per_mm2: float = 1200.0,
    mc_exp: float = 0.25,
) -> Tuple[float, float]:
    """
    Estimate tangential cutting force Fc (N) and spindle cutting power Pc (kW)
    using the Kienzle specific cutting force formulation.
    """
    fz = calculate_chip_load(feed_rate_mm_min, spindle_speed_rpm, num_flutes)
    vc = calculate_cutting_speed(spindle_speed_rpm, tool_diameter_mm)

    if fz <= 0 or axial_depth_mm <= 0 or vc <= 0:
        return 0.0, 0.0

    # Effective chip thickness factor based on radial immersion ae/D
    immersion_ratio = min(max(radial_depth_mm / max(tool_diameter_mm, 1e-3), 0.05), 1.0)
    # Mean chip thickness hm
    hm = max(fz * np.sqrt(immersion_ratio), 1e-4)
    fc = kc_n_per_mm2 * axial_depth_mm * (hm ** (1.0 - mc_exp))
    # Power in kW: (Fc * Vc) / 60000
    power_kw = (fc * vc) / 60000.0
    return float(fc), float(power_kw)


def compute_stability_lobe_curve(
    rpm_range: np.ndarray,
    natural_freq_hz: float = 850.0,
    damping_ratio: float = 0.035,
    modal_stiffness_n_per_m: float = 3.5e6,
    num_flutes: int = 4,
    cutting_stiffness_kc: float = 1200e6,
    min_depth_limit_mm: float = 0.8,
    max_depth_limit_mm: float = 6.0,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Compute analytical stability lobe boundary ap_lim(RPM) across a range of spindle speeds
    using the Altintas-Budak regenerative chatter stability boundary model.
    Returns:
        (rpm_grid, ap_limit_curve_mm, sweet_spot_rpms)
    """
    rpm_grid = np.asarray(rpm_range, dtype=float)
    ap_curve = np.zeros_like(rpm_grid)

    # Sweet spots occur where tooth pass frequency or harmonics align with natural frequency:
    # N * RPM / 60 = fn / (k + epsilon/(2*pi))
    # For small phase lag at resonant peak, sweet spots are roughly fn * 60 / (N * k)
    sweet_spots = []
    for k in range(1, 10):
        sweet_rpm = (60.0 * natural_freq_hz) / (num_flutes * k)
        if rpm_grid.min() <= sweet_rpm <= rpm_grid.max():
            sweet_spots.append(float(sweet_rpm))
    sweet_spots.sort()

    omega_n = 2.0 * np.pi * natural_freq_hz

    for i, rpm in enumerate(rpm_grid):
        ft = calculate_tooth_passing_frequency(rpm, num_flutes)
        if ft <= 0:
            ap_curve[i] = min_depth_limit_mm
            continue

        # Evaluate transfer function over chatter frequencies close to natural frequency
        # Ratio of tooth pass frequency to natural frequency
        ratio = ft / natural_freq_hz
        
        # Envelope of stability lobes:
        # Distance to nearest pocket center:
        # Pockets appear at integer harmonic multiples
        harmonic = natural_freq_hz / max(ft, 1e-3)
        nearest_harmonic = round(harmonic)
        harmonic_dist = abs(harmonic - nearest_harmonic)

        # Base critical depth from negative real part of FRF
        # High stability at sweet spots (resonance harmonic alignment), lower in valleys
        # Resonance factor increases pocket height
        lobe_factor = 1.0 / (1.0 + 8.0 * (harmonic_dist ** 2))
        
        # Base asymptotic stability depth (minimum axial depth below which chatter is physically impossible)
        ap_asymptotic = min_depth_limit_mm * 1.2
        
        # Dynamic lobe peak
        ap_calculated = ap_asymptotic + (max_depth_limit_mm - ap_asymptotic) * (lobe_factor ** 1.5)
        
        # Higher speeds generally increase damping and stability due to process damping
        speed_factor = 0.8 + 0.4 * (rpm / max(rpm_grid.max(), 1.0))
        ap_final = np.clip(ap_calculated * speed_factor, min_depth_limit_mm, max_depth_limit_mm)
        ap_curve[i] = ap_final

    return rpm_grid, ap_curve, sweet_spots


def get_stability_limit_at_rpm(
    spindle_speed_rpm: float,
    natural_freq_hz: float = 850.0,
    damping_ratio: float = 0.035,
    num_flutes: int = 4,
    min_depth_limit_mm: float = 0.8,
    max_depth_limit_mm: float = 6.0,
) -> float:
    """
    Get the analytical axial depth of cut stability limit (ap_lim in mm) for a specific RPM.
    """
    rpm_arr = np.array([spindle_speed_rpm], dtype=float)
    _, curve, _ = compute_stability_lobe_curve(
        rpm_range=rpm_arr,
        natural_freq_hz=natural_freq_hz,
        damping_ratio=damping_ratio,
        num_flutes=num_flutes,
        min_depth_limit_mm=min_depth_limit_mm,
        max_depth_limit_mm=max_depth_limit_mm,
    )
    return float(curve[0])


def calculate_tool_wear_factor(
    spindle_speed_rpm: float,
    tool_diameter_mm: float,
    feed_rate_mm_min: float,
    num_flutes: int,
    ref_vc_m_min: float = 150.0,
    ref_fz_mm_tooth: float = 0.06,
) -> float:
    """
    Illustrative tool wear index based on extended Taylor tool life equations:
    Wear_Factor = (Vc / Vc_ref)^1.8 * (fz / fz_ref)^0.6
    Normalized around 1.0 for reference conditions.
    """
    vc = calculate_cutting_speed(spindle_speed_rpm, tool_diameter_mm)
    fz = calculate_chip_load(feed_rate_mm_min, spindle_speed_rpm, num_flutes)

    if vc <= 0 or fz <= 0:
        return 0.0

    wear = ((vc / max(ref_vc_m_min, 1.0)) ** 1.8) * ((fz / max(ref_fz_mm_tooth, 1e-3)) ** 0.6)
    return float(np.clip(wear, 0.1, 10.0))
