import pytest
from dataclasses import asdict
import pandas as pd
from core.bounds import MachineToolingBounds, check_candidate_feasibility
from core.model import get_trained_default_model
from core.optimizer import ChatterOptimizationEngine


def test_full_pipeline_integration():
    bounds = MachineToolingBounds()
    model, synthetic_df = get_trained_default_model(n_samples=300, random_state=42)

    assert len(synthetic_df) == 300
    assert model.is_trained

    engine = ChatterOptimizationEngine(bounds=bounds, model=model, natural_freq_hz=850.0)

    # Test running optimization across 3 distinct baseline conditions
    scenarios = [
        # Severe chatter
        {"rpm": 4200.0, "feed": 1450.0, "depth": 4.2, "rms": 3.25, "peak": 13.1, "freq": 850.0},
        # Warning
        {"rpm": 5800.0, "feed": 1200.0, "depth": 2.8, "rms": 1.35, "peak": 4.2, "freq": 840.0},
        # Stable
        {"rpm": 6375.0, "feed": 1800.0, "depth": 2.2, "rms": 0.38, "peak": 0.95, "freq": 425.0},
    ]

    for sc in scenarios:
        res = engine.generate_and_rank_candidates(
            baseline_rpm=sc["rpm"],
            baseline_feed=sc["feed"],
            baseline_depth=sc["depth"],
            baseline_rms=sc["rms"],
            baseline_peak=sc["peak"],
            baseline_dom_freq=sc["freq"],
        )
        assert res["recommended"] is not None
        assert res["recommended"].is_feasible is True
        assert len(res["feasible_candidates"]) > 0

        # Check export serializability
        all_cands = [asdict(c) for c in (res["feasible_candidates"] + res["rejected_candidates"])]
        df_export = pd.DataFrame(all_cands)
        assert len(df_export) > 0
        csv_bytes = df_export.to_csv(index=False).encode("utf-8")
        assert len(csv_bytes) > 0
