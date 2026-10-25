"""
Machine, Tooling, and Process Boundary Constraints Validation.
Rejects infeasible machining candidates with explicit diagnostic logging.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple
import numpy as np

from .physics import (
    calculate_chip_load,
    calculate_cutting_speed,
    estimate_cutting_force_and_power,
    calculate_mrr,
)


@dataclass
class MachineToolingBounds:
    """
    Configurable engineering boundaries for CNC machine, cutting tool, and workpiece.
    """
    # Spindle limits
    min_spindle_speed_rpm: float = 1200.0
    max_spindle_speed_rpm: float = 15000.0
    max_spindle_power_kw: float = 7.5

    # Feed limits
    min_feed_rate_mm_min: float = 100.0
    max_feed_rate_mm_min: float = 3800.0

    # Depth of cut limits
    min_depth_of_cut_mm: float = 0.30
    max_depth_of_cut_mm: float = 6.00

    # Tooling geometry & chip load limits
    tool_diameter_mm: float = 12.0
    num_flutes: int = 4
    radial_depth_mm: float = 6.0
    min_chip_load_mm_tooth: float = 0.015  # Minimum to prevent rubbing/burnishing
    max_chip_load_mm_tooth: float = 0.160  # Maximum to prevent tooth fracture

    # Force & speed limits
    max_cutting_force_n: float = 1400.0
    max_cutting_speed_m_min: float = 550.0  # Thermal limit for tool coating
    kc_n_per_mm2: float = 1200.0


@dataclass
class BoundsCheckResult:
    """
    Outcome of bounds check for a candidate machining parameter set.
    """
    is_feasible: bool
    rejection_reasons: List[str] = field(default_factory=list)
    chip_load_mm_tooth: float = 0.0
    cutting_speed_m_min: float = 0.0
    cutting_force_n: float = 0.0
    cutting_power_kw: float = 0.0
    mrr_cm3_min: float = 0.0


def check_candidate_feasibility(
    spindle_speed_rpm: float,
    feed_rate_mm_min: float,
    depth_of_cut_mm: float,
    bounds: MachineToolingBounds,
) -> BoundsCheckResult:
    """
    Perform rigorous engineering bound checks on a proposed machining candidate.
    Identifies all violations and returns feasibility flag with descriptive reasons.
    """
    reasons = []

    # 1. Spindle speed range check
    if spindle_speed_rpm < bounds.min_spindle_speed_rpm:
        reasons.append(
            f"Spindle speed ({spindle_speed_rpm:.0f} RPM) below minimum machine capability ({bounds.min_spindle_speed_rpm:.0f} RPM)"
        )
    elif spindle_speed_rpm > bounds.max_spindle_speed_rpm:
        reasons.append(
            f"Spindle speed ({spindle_speed_rpm:.0f} RPM) exceeds maximum spindle rating ({bounds.max_spindle_speed_rpm:.0f} RPM)"
        )

    # 2. Feed rate range check
    if feed_rate_mm_min < bounds.min_feed_rate_mm_min:
        reasons.append(
            f"Feed rate ({feed_rate_mm_min:.1f} mm/min) below axis drive minimum ({bounds.min_feed_rate_mm_min:.1f} mm/min)"
        )
    elif feed_rate_mm_min > bounds.max_feed_rate_mm_min:
        reasons.append(
            f"Feed rate ({feed_rate_mm_min:.1f} mm/min) exceeds axis drive maximum ({bounds.max_feed_rate_mm_min:.1f} mm/min)"
        )

    # 3. Axial depth of cut check
    if depth_of_cut_mm < bounds.min_depth_of_cut_mm:
        reasons.append(
            f"Depth of cut ({depth_of_cut_mm:.2f} mm) below practical minimum engagement ({bounds.min_depth_of_cut_mm:.2f} mm)"
        )
    elif depth_of_cut_mm > bounds.max_depth_of_cut_mm:
        reasons.append(
            f"Depth of cut ({depth_of_cut_mm:.2f} mm) exceeds tool flute length ({bounds.max_depth_of_cut_mm:.2f} mm)"
        )

    # 4. Chip load calculation & rubbing/breakage check
    fz = calculate_chip_load(feed_rate_mm_min, spindle_speed_rpm, bounds.num_flutes)
    if fz < bounds.min_chip_load_mm_tooth:
        reasons.append(
            f"Chip load ({fz:.4f} mm/tooth) < min threshold ({bounds.min_chip_load_mm_tooth:.4f} mm/tooth): severe rubbing & work-hardening risk"
        )
    elif fz > bounds.max_chip_load_mm_tooth:
        reasons.append(
            f"Chip load ({fz:.4f} mm/tooth) > max threshold ({bounds.max_chip_load_mm_tooth:.4f} mm/tooth): tool tooth breakage risk"
        )

    # 5. Cutting speed check
    vc = calculate_cutting_speed(spindle_speed_rpm, bounds.tool_diameter_mm)
    if vc > bounds.max_cutting_speed_m_min:
        reasons.append(
            f"Cutting speed ({vc:.1f} m/min) exceeds tool thermal limit ({bounds.max_cutting_speed_m_min:.1f} m/min)"
        )

    # 6. Cutting force and spindle motor power check
    fc, power_kw = estimate_cutting_force_and_power(
        axial_depth_mm=depth_of_cut_mm,
        radial_depth_mm=bounds.radial_depth_mm,
        feed_rate_mm_min=feed_rate_mm_min,
        spindle_speed_rpm=spindle_speed_rpm,
        num_flutes=bounds.num_flutes,
        tool_diameter_mm=bounds.tool_diameter_mm,
        kc_n_per_mm2=bounds.kc_n_per_mm2,
    )

    if power_kw > bounds.max_spindle_power_kw:
        reasons.append(
            f"Estimated power ({power_kw:.2f} kW) exceeds spindle motor continuous power ({bounds.max_spindle_power_kw:.2f} kW)"
        )

    if fc > bounds.max_cutting_force_n:
        reasons.append(
            f"Estimated tangential force ({fc:.1f} N) exceeds maximum tool shank limit ({bounds.max_cutting_force_n:.1f} N)"
        )

    mrr = calculate_mrr(depth_of_cut_mm, bounds.radial_depth_mm, feed_rate_mm_min)

    is_feasible = (len(reasons) == 0)

    return BoundsCheckResult(
        is_feasible=is_feasible,
        rejection_reasons=reasons,
        chip_load_mm_tooth=float(fz),
        cutting_speed_m_min=float(vc),
        cutting_force_n=float(fc),
        cutting_power_kw=float(power_kw),
        mrr_cm3_min=float(mrr),
    )
