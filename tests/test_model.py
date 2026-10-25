import pytest
import numpy as np
import pandas as pd
from core.synthetic_data import generate_synthetic_dataset, LABEL_STABLE, LABEL_WARNING, LABEL_HIGH_RISK
from core.model import ChatterClassifier, FEATURE_COLUMNS


@pytest.fixture
def trained_classifier():
    df = generate_synthetic_dataset(n_samples=400, random_state=42)
    clf = ChatterClassifier(n_estimators=30, random_state=42)
    clf.fit(df)
    return clf


def test_model_fit(trained_classifier):
    assert trained_classifier.is_trained
    assert "accuracy" in trained_classifier.metrics_
    assert trained_classifier.metrics_["accuracy"] > 0.80
    assert len(trained_classifier.feature_importances_) == len(FEATURE_COLUMNS)
    assert sum(trained_classifier.feature_importances_.values()) == pytest.approx(1.0, rel=1e-2)


def test_model_predict_single_sample(trained_classifier):
    sample = {
        "spindle_speed_rpm": 6000.0,
        "feed_rate_mm_min": 1200.0,
        "depth_of_cut_mm": 1.5,
        "vibration_rms_g": 0.35,
        "vibration_peak_g": 0.85,
        "dominant_frequency_hz": 400.0,
    }
    pred = trained_classifier.predict(sample)
    assert "predicted_state" in pred
    assert pred["predicted_state"] in [LABEL_STABLE, LABEL_WARNING, LABEL_HIGH_RISK]
    assert "probabilities" in pred
    assert "risk_score" in pred
    assert 0.0 <= pred["risk_score"] <= 100.0

    prob_sum = sum(pred["probabilities"].values())
    assert prob_sum == pytest.approx(1.0, rel=1e-2)


def test_model_predict_high_chatter_sample(trained_classifier):
    sample = {
        "spindle_speed_rpm": 4500.0,
        "feed_rate_mm_min": 1500.0,
        "depth_of_cut_mm": 4.5,
        "vibration_rms_g": 3.8,
        "vibration_peak_g": 14.5,
        "dominant_frequency_hz": 850.0,
    }
    pred = trained_classifier.predict(sample)
    # Severe vibration at resonant mode should flag High Risk or Warning with high risk score
    assert pred["predicted_state"] in [LABEL_HIGH_RISK, LABEL_WARNING]
    assert pred["risk_score"] > 50.0


def test_model_predict_unfitted_raises():
    clf = ChatterClassifier()
    with pytest.raises(RuntimeError):
        clf.predict({"spindle_speed_rpm": 5000})
