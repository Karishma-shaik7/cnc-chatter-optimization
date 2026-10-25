import pytest
import pandas as pd
import numpy as np
from core.synthetic_data import (
    generate_synthetic_dataset,
    LABEL_STABLE,
    LABEL_WARNING,
    LABEL_HIGH_RISK,
    VALID_LABELS,
)


def test_generate_synthetic_dataset_reproducibility():
    df1 = generate_synthetic_dataset(n_samples=100, random_state=42)
    df2 = generate_synthetic_dataset(n_samples=100, random_state=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_synthetic_dataset_different_seeds():
    df1 = generate_synthetic_dataset(n_samples=50, random_state=42)
    df2 = generate_synthetic_dataset(n_samples=50, random_state=99)
    assert not df1.equals(df2)


def test_generate_synthetic_dataset_schema():
    df = generate_synthetic_dataset(n_samples=150, random_state=123)
    assert len(df) == 150
    expected_cols = [
        "spindle_speed_rpm",
        "feed_rate_mm_min",
        "depth_of_cut_mm",
        "vibration_rms_g",
        "vibration_peak_g",
        "dominant_frequency_hz",
        "chatter_state",
        "chip_load_mm_tooth",
        "mrr_cm3_min",
        "tooth_pass_freq_hz",
        "stability_limit_ap_mm",
    ]
    for col in expected_cols:
        assert col in df.columns


def test_generate_synthetic_dataset_classes_represented():
    df = generate_synthetic_dataset(n_samples=500, random_state=42)
    unique_classes = set(df["chatter_state"].unique())
    assert set(VALID_LABELS).issubset(unique_classes)


def test_sensor_features_physical_consistency():
    df = generate_synthetic_dataset(n_samples=200, random_state=42)
    # Peak vibration should strictly exceed or equal RMS
    assert np.all(df["vibration_peak_g"] >= df["vibration_rms_g"])
    # RMS vibration strictly positive
    assert np.all(df["vibration_rms_g"] > 0)
    # Dominant frequency positive
    assert np.all(df["dominant_frequency_hz"] > 0)
    # High Risk samples should have higher mean RMS than Stable samples
    stable_rms = df[df["chatter_state"] == LABEL_STABLE]["vibration_rms_g"].mean()
    chatter_rms = df[df["chatter_state"] == LABEL_HIGH_RISK]["vibration_rms_g"].mean()
    assert chatter_rms > stable_rms
