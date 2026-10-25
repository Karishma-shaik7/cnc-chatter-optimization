import pytest
import numpy as np
from core.physics import (
    calculate_mrr,
    calculate_chip_load,
    calculate_cutting_speed,
    calculate_tooth_passing_frequency,
    estimate_cutting_force_and_power,
    compute_stability_lobe_curve,
    get_stability_limit_at_rpm,
    calculate_tool_wear_factor,
)


def test_calculate_mrr():
    # ap = 2.0 mm, ae = 6.0 mm, vf = 1000.0 mm/min
    # MRR = 2.0 * 6.0 * 1000 / 1000 = 12.0 cm^3/min
    mrr = calculate_mrr(2.0, 6.0, 1000.0)
    assert pytest.approx(mrr, rel=1e-3) == 12.0

    # Non-positive input returns 0.0
    assert calculate_mrr(0, 6.0, 1000.0) == 0.0
    assert calculate_mrr(2.0, -1.0, 1000.0) == 0.0


def test_calculate_chip_load():
    # vf = 1200 mm/min, RPM = 6000, flutes = 4
    # fz = 1200 / (4 * 6000) = 0.05 mm/tooth
    fz = calculate_chip_load(1200.0, 6000.0, 4)
    assert pytest.approx(fz, rel=1e-3) == 0.05

    # Safe division
    assert calculate_chip_load(1200.0, 0, 4) == 0.0


def test_calculate_cutting_speed():
    # D = 10 mm, RPM = 6000
    # Vc = pi * 10 * 6000 / 1000 = 60 * pi = 188.495 m/min
    vc = calculate_cutting_speed(6000.0, 10.0)
    assert pytest.approx(vc, rel=1e-3) == 60.0 * np.pi


def test_calculate_tooth_passing_frequency():
    # RPM = 6000, flutes = 4
    # ft = 4 * 6000 / 60 = 400 Hz
    ft = calculate_tooth_passing_frequency(6000.0, 4)
    assert ft == 400.0


def test_estimate_cutting_force_and_power():
    fc, power = estimate_cutting_force_and_power(
        axial_depth_mm=2.5,
        radial_depth_mm=6.0,
        feed_rate_mm_min=1200.0,
        spindle_speed_rpm=5000.0,
        num_flutes=4,
        tool_diameter_mm=12.0,
    )
    assert fc > 0.0
    assert power > 0.0
    assert power < 20.0  # realistic cutting power


def test_compute_stability_lobe_curve():
    rpm_range = np.linspace(2000, 12000, 50)
    grid, curve, sweet_spots = compute_stability_lobe_curve(
        rpm_range=rpm_range,
        natural_freq_hz=850.0,
        num_flutes=4,
    )
    assert len(grid) == 50
    assert len(curve) == 50
    assert np.all(curve > 0.0)
    assert len(sweet_spots) > 0
    # Sweet spots must be within rpm range
    for s in sweet_spots:
        assert rpm_range.min() <= s <= rpm_range.max()


def test_get_stability_limit_at_rpm():
    lim = get_stability_limit_at_rpm(6000.0, natural_freq_hz=850.0, num_flutes=4)
    assert lim > 0.5
    assert lim < 8.0


def test_calculate_tool_wear_factor():
    wear = calculate_tool_wear_factor(
        spindle_speed_rpm=6000.0,
        tool_diameter_mm=12.0,
        feed_rate_mm_min=1200.0,
        num_flutes=4,
    )
    assert wear > 0.1
