import pytest
from core.bounds import MachineToolingBounds
from core.model import get_trained_default_model
from core.optimizer import ChatterOptimizationEngine


@pytest.fixture(scope="module")
def default_engine():
    bounds = MachineToolingBounds()
    model, _ = get_trained_default_model(n_samples=400, random_state=42)
    return ChatterOptimizationEngine(bounds=bounds, model=model, natural_freq_hz=850.0)


def test_optimizer_generate_and_rank(default_engine):
    # Severe chatter baseline
    results = default_engine.generate_and_rank_candidates(
        baseline_rpm=4200.0,
        baseline_feed=1400.0,
        baseline_depth=4.2,
        baseline_rms=3.2,
        baseline_peak=12.5,
        baseline_dom_freq=850.0,
    )

    assert "baseline" in results
    assert "recommended" in results
    assert "feasible_candidates" in results
    assert "rejected_candidates" in results

    assert results["total_candidates_generated"] > 5
    assert len(results["feasible_candidates"]) > 0
    assert len(results["rejected_candidates"]) > 0

    # Verify feasible ranking order: sorted descending by composite_score
    scores = [c.composite_score for c in results["feasible_candidates"]]
    assert scores == sorted(scores, reverse=True)

    # Recommended candidate should be rank 1
    rec = results["recommended"]
    assert rec is not None
    assert rec.is_feasible is True
    assert rec.composite_score == scores[0]


def test_rejected_candidates_contain_reasons(default_engine):
    results = default_engine.generate_and_rank_candidates(
        baseline_rpm=5000.0,
        baseline_feed=1000.0,
        baseline_depth=2.0,
        baseline_rms=0.4,
        baseline_peak=1.0,
        baseline_dom_freq=333.0,
    )
    rejected = results["rejected_candidates"]
    assert len(rejected) > 0
    for cand in rejected:
        assert cand.is_feasible is False
        assert len(cand.rejection_reasons) > 0
        assert isinstance(cand.rejection_reasons[0], str)
