import pytest
from core.bounds import (
    MachineToolingBounds,
    check_candidate_feasibility,
)


@pytest.fixture
def default_bounds():
    return MachineToolingBounds(
        min_spindle_speed_rpm=1200.0,
        max_spindle_speed_rpm=15000.0,
        min_feed_rate_mm_min=100.0,
        max_feed_rate_mm_min=3800.0,
        min_depth_of_cut_mm=0.30,
        max_depth_of_cut_mm=6.0,
        min_chip_load_mm_tooth=0.015,
        max_chip_load_mm_tooth=0.160,
        max_spindle_power_kw=7.5,
    )


def test_feasible_nominal_parameters(default_bounds):
    res = check_candidate_feasibility(
        spindle_speed_rpm=6000.0,
        feed_rate_mm_min=1200.0,
        depth_of_cut_mm=2.0,
        bounds=default_bounds,
    )
    assert res.is_feasible is True
    assert len(res.rejection_reasons) == 0
    assert pytest.approx(res.chip_load_mm_tooth, rel=1e-3) == 0.05
    assert res.cutting_power_kw < default_bounds.max_spindle_power_kw


def test_over_spindle_speed_rejected(default_bounds):
    res = check_candidate_feasibility(
        spindle_speed_rpm=18000.0,  # > 15000 max
        feed_rate_mm_min=1200.0,
        depth_of_cut_mm=2.0,
        bounds=default_bounds,
    )
    assert res.is_feasible is False
    assert any("exceeds maximum spindle rating" in r for r in res.rejection_reasons)


def test_under_chip_load_rubbing_rejected(default_bounds):
    # fz = 100 / (4 * 10000) = 0.0025 mm/tooth < 0.015 min
    res = check_candidate_feasibility(
        spindle_speed_rpm=10000.0,
        feed_rate_mm_min=100.0,
        depth_of_cut_mm=1.0,
        bounds=default_bounds,
    )
    assert res.is_feasible is False
    assert any("rubbing" in r.lower() for r in res.rejection_reasons)


def test_over_chip_load_breakage_rejected(default_bounds):
    # fz = 3000 / (4 * 3000) = 0.25 mm/tooth > 0.16 max
    res = check_candidate_feasibility(
        spindle_speed_rpm=3000.0,
        feed_rate_mm_min=3000.0,
        depth_of_cut_mm=2.0,
        bounds=default_bounds,
    )
    assert res.is_feasible is False
    assert any("breakage" in r.lower() for r in res.rejection_reasons)


def test_spindle_power_overload_rejected(default_bounds):
    # Set continuous rated power limit to 5.0 kW (e.g. compact machining center)
    default_bounds.max_spindle_power_kw = 5.0
    res = check_candidate_feasibility(
        spindle_speed_rpm=14000.0,
        feed_rate_mm_min=3600.0,
        depth_of_cut_mm=5.9,
        bounds=default_bounds,
    )
    assert res.is_feasible is False
    assert any("power" in r.lower() for r in res.rejection_reasons)
