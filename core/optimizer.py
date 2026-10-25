"""
Optimization Engine for CNC Chatter Suppression and Tool-Path Parameter Selection.
Generates candidate machining parameters, filters infeasible states via bound checks,
evaluates trade-offs, and ranks optimal operational points.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd

from .physics import (
    calculate_chip_load,
    calculate_mrr,
    calculate_tooth_passing_frequency,
    calculate_cutting_speed,
    get_stability_limit_at_rpm,
    compute_stability_lobe_curve,
    calculate_tool_wear_factor,
    estimate_cutting_force_and_power,
)
from .bounds import (
    MachineToolingBounds,
    check_candidate_feasibility,
    BoundsCheckResult,
)
from .model import ChatterClassifier, LABEL_STABLE, LABEL_WARNING, LABEL_HIGH_RISK


@dataclass
class MachiningCandidate:
    """
    Representation of a single candidate machining configuration.
    """
    candidate_id: str
    strategy_name: str
    spindle_speed_rpm: float
    feed_rate_mm_min: float
    depth_of_cut_mm: float

    # Feasibility
    is_feasible: bool
    rejection_reasons: List[str]

    # Engineering metrics
    chip_load_mm_tooth: float
    cutting_speed_m_min: float
    cutting_power_kw: float
    cutting_force_n: float
    mrr_cm3_min: float
    tooth_pass_freq_hz: float
    stability_limit_ap_mm: float
    wear_factor: float

    # Estimated sensor responses
    est_vibration_rms_g: float
    est_vibration_peak_g: float
    est_dominant_freq_hz: float

    # Model predictions
    predicted_state: str
    prob_stable: float
    prob_warning: float
    prob_high_risk: float
    risk_score: float  # 0 to 100 (lower is safer)

    # Multi-objective trade-off metrics
    cycle_time_delta_pct: float  # vs baseline (negative = faster)
    mrr_delta_pct: float         # vs baseline (positive = higher productivity)
    risk_delta_pts: float        # vs baseline (negative = chatter suppression)
    composite_score: float = 0.0 # 0 to 100 (higher is better)


def estimate_sensor_vibration_for_params(
    spindle_speed_rpm: float,
    depth_of_cut_mm: float,
    feed_rate_mm_min: float,
    bounds: MachineToolingBounds,
    natural_freq_hz: float = 850.0,
    seed: int = 123,
) -> Tuple[float, float, float]:
    """
    Estimate the expected vibration features (RMS, Peak, Dominant Frequency)
    if the machine were to operate at the candidate settings.
    Physical regenerative chatter model grounds the expected response.
    """
    rng = np.random.default_rng(int(spindle_speed_rpm + feed_rate_mm_min + depth_of_cut_mm * 100) + seed)

    ap_lim = get_stability_limit_at_rpm(
        spindle_speed_rpm=spindle_speed_rpm,
        natural_freq_hz=natural_freq_hz,
        num_flutes=bounds.num_flutes,
    )
    ft = calculate_tooth_passing_frequency(spindle_speed_rpm, bounds.num_flutes)
    fz = calculate_chip_load(feed_rate_mm_min, spindle_speed_rpm, bounds.num_flutes)

    ratio = depth_of_cut_mm / max(ap_lim, 0.1)

    # Forced cutting vibration base
    base_rms = 0.18 + 0.12 * (depth_of_cut_mm / 5.0) + 0.15 * (fz / 0.10)

    if ratio < 0.88:
        # Stable
        est_rms = base_rms + rng.uniform(0.04, 0.12)
        est_peak = est_rms * rng.uniform(2.2, 2.6)
        est_dom_freq = ft + rng.normal(0.0, 3.0)
    elif ratio <= 1.18:
        # Warning
        intensity = (ratio - 0.88) / 0.30
        est_rms = 0.85 + 0.70 * intensity + rng.uniform(-0.05, 0.10)
        est_peak = est_rms * rng.uniform(2.8, 3.5)
        est_dom_freq = natural_freq_hz if rng.random() > 0.5 else ft
    else:
        # Severe chatter
        overshoot = min(ratio - 1.18, 3.0)
        est_rms = 1.90 + 1.10 * overshoot + rng.uniform(0.1, 0.4)
        est_peak = est_rms * rng.uniform(3.6, 4.8)
        est_dom_freq = natural_freq_hz + rng.normal(0.0, 8.0)

    return float(np.clip(est_rms, 0.12, 7.5)), float(np.clip(est_peak, 0.3, 22.0)), float(np.clip(est_dom_freq, 40.0, 4000.0))


class ChatterOptimizationEngine:
    """
    Engine to generate alternative machining settings, filter by configured machine bounds,
    predict stability, evaluate trade-offs, and rank Pareto-optimal solutions.
    """

    def __init__(
        self,
        bounds: MachineToolingBounds,
        model: ChatterClassifier,
        natural_freq_hz: float = 850.0,
    ):
        self.bounds = bounds
        self.model = model
        self.natural_freq_hz = natural_freq_hz

    def evaluate_candidate(
        self,
        candidate_id: str,
        strategy_name: str,
        spindle_speed_rpm: float,
        feed_rate_mm_min: float,
        depth_of_cut_mm: float,
        baseline_mrr: float,
        baseline_risk: float,
        baseline_feed: float,
    ) -> MachiningCandidate:
        """
        Evaluate a single candidate parameter set:
        1. Run bound checks.
        2. Compute physics metrics.
        3. Predict chatter probability & risk.
        4. Calculate trade-offs against baseline.
        """
        # Step 1: Rigorous configured bound checks
        bounds_res = check_candidate_feasibility(
            spindle_speed_rpm=spindle_speed_rpm,
            feed_rate_mm_min=feed_rate_mm_min,
            depth_of_cut_mm=depth_of_cut_mm,
            bounds=self.bounds,
        )

        ft = calculate_tooth_passing_frequency(spindle_speed_rpm, self.bounds.num_flutes)
        ap_lim = get_stability_limit_at_rpm(
            spindle_speed_rpm=spindle_speed_rpm,
            natural_freq_hz=self.natural_freq_hz,
            num_flutes=self.bounds.num_flutes,
        )
        wear = calculate_tool_wear_factor(
            spindle_speed_rpm=spindle_speed_rpm,
            tool_diameter_mm=self.bounds.tool_diameter_mm,
            feed_rate_mm_min=feed_rate_mm_min,
            num_flutes=self.bounds.num_flutes,
        )

        # Step 2: Sensor response estimation
        est_rms, est_peak, est_dom = estimate_sensor_vibration_for_params(
            spindle_speed_rpm=spindle_speed_rpm,
            depth_of_cut_mm=depth_of_cut_mm,
            feed_rate_mm_min=feed_rate_mm_min,
            bounds=self.bounds,
            natural_freq_hz=self.natural_freq_hz,
        )

        # Step 3: Run Random Forest classifier prediction
        pred = self.model.predict({
            "spindle_speed_rpm": spindle_speed_rpm,
            "feed_rate_mm_min": feed_rate_mm_min,
            "depth_of_cut_mm": depth_of_cut_mm,
            "vibration_rms_g": est_rms,
            "vibration_peak_g": est_peak,
            "dominant_frequency_hz": est_dom,
        })

        pred_state = pred["predicted_state"]
        probs = pred["probabilities"]
        risk_score = pred["risk_score"]

        # Step 4: Deltas against baseline
        mrr = bounds_res.mrr_cm3_min
        mrr_delta_pct = ((mrr - baseline_mrr) / max(baseline_mrr, 1e-3)) * 100.0
        risk_delta_pts = risk_score - baseline_risk
        # Cycle time is inversely proportional to feed rate: (T_new - T_base) / T_base = (v_base - v_new) / v_new
        cycle_time_delta_pct = ((baseline_feed - feed_rate_mm_min) / max(feed_rate_mm_min, 1e-3)) * 100.0

        return MachiningCandidate(
            candidate_id=candidate_id,
            strategy_name=strategy_name,
            spindle_speed_rpm=round(spindle_speed_rpm, 1),
            feed_rate_mm_min=round(feed_rate_mm_min, 1),
            depth_of_cut_mm=round(depth_of_cut_mm, 2),
            is_feasible=bounds_res.is_feasible,
            rejection_reasons=bounds_res.rejection_reasons,
            chip_load_mm_tooth=round(bounds_res.chip_load_mm_tooth, 4),
            cutting_speed_m_min=round(bounds_res.cutting_speed_m_min, 1),
            cutting_power_kw=round(bounds_res.cutting_power_kw, 2),
            cutting_force_n=round(bounds_res.cutting_force_n, 1),
            mrr_cm3_min=round(mrr, 2),
            tooth_pass_freq_hz=round(ft, 1),
            stability_limit_ap_mm=round(ap_lim, 2),
            wear_factor=round(wear, 2),
            est_vibration_rms_g=round(est_rms, 3),
            est_vibration_peak_g=round(est_peak, 3),
            est_dominant_freq_hz=round(est_dom, 1),
            predicted_state=pred_state,
            prob_stable=round(probs.get(LABEL_STABLE, 0.0), 3),
            prob_warning=round(probs.get(LABEL_WARNING, 0.0), 3),
            prob_high_risk=round(probs.get(LABEL_HIGH_RISK, 0.0), 3),
            risk_score=round(risk_score, 1),
            cycle_time_delta_pct=round(cycle_time_delta_pct, 1),
            mrr_delta_pct=round(mrr_delta_pct, 1),
            risk_delta_pts=round(risk_delta_pts, 1),
            composite_score=0.0,
        )

    def generate_and_rank_candidates(
        self,
        baseline_rpm: float,
        baseline_feed: float,
        baseline_depth: float,
        baseline_rms: float,
        baseline_peak: float,
        baseline_dom_freq: float,
        weight_stability: float = 0.50,
        weight_mrr: float = 0.35,
        weight_tool_life: float = 0.15,
    ) -> Dict[str, Any]:
        """
        Generate candidate alternative parameters across multiple physical strategies:
        - Stability Lobe Sweet Spot hopping (targeting resonant harmonic nodes)
        - Depth reduction with feed compensation (retaining MRR)
        - Chip thinning compensation
        - High-productivity mode
        - Conservative / maximum stability mode
        - Intentionally bound-violating candidates (for bounds validation audit)
        
        Filters by configured bounds, scores feasible candidates, and ranks them.
        """
        # Baseline diagnostics
        baseline_pred = self.model.predict({
            "spindle_speed_rpm": baseline_rpm,
            "feed_rate_mm_min": baseline_feed,
            "depth_of_cut_mm": baseline_depth,
            "vibration_rms_g": baseline_rms,
            "vibration_peak_g": baseline_peak,
            "dominant_frequency_hz": baseline_dom_freq,
        })
        baseline_mrr = calculate_mrr(baseline_depth, self.bounds.radial_depth_mm, baseline_feed)
        baseline_risk = baseline_pred["risk_score"]
        baseline_bounds_res = check_candidate_feasibility(
            spindle_speed_rpm=baseline_rpm,
            feed_rate_mm_min=baseline_feed,
            depth_of_cut_mm=baseline_depth,
            bounds=self.bounds,
        )

        # Identify stability lobe pockets across the machine's speed range
        rpm_grid = np.linspace(
            self.bounds.min_spindle_speed_rpm,
            self.bounds.max_spindle_speed_rpm,
            120,
        )
        _, ap_curve, sweet_rpms = compute_stability_lobe_curve(
            rpm_range=rpm_grid,
            natural_freq_hz=self.natural_freq_hz,
            num_flutes=self.bounds.num_flutes,
        )

        candidates_pool = []
        cand_idx = 1

        # Strategy A: Stability Lobe Pocket Tuning (Pocket Sweet Spots)
        for pocket_rpm in sweet_rpms:
            # Set target depth safely inside the pocket peak
            pocket_limit = get_stability_limit_at_rpm(
                spindle_speed_rpm=pocket_rpm,
                natural_freq_hz=self.natural_freq_hz,
                num_flutes=self.bounds.num_flutes,
            )
            target_depth = min(baseline_depth, pocket_limit * 0.85)

            # Preserve or scale feed to maintain chip load
            target_feed = baseline_feed * (pocket_rpm / max(baseline_rpm, 1.0))
            candidates_pool.append({
                "id": f"C-{cand_idx:02d}",
                "strategy": f"Stability Pocket @ {pocket_rpm:.0f} RPM",
                "rpm": pocket_rpm,
                "feed": target_feed,
                "depth": target_depth,
            })
            cand_idx += 1

            # High-productivity variant in pocket: increase feed up to 1.3x if depth is reduced
            boosted_feed = target_feed * 1.25
            candidates_pool.append({
                "id": f"C-{cand_idx:02d}",
                "strategy": f"High MRR Pocket @ {pocket_rpm:.0f} RPM",
                "rpm": pocket_rpm,
                "feed": boosted_feed,
                "depth": target_depth,
            })
            cand_idx += 1

        # Strategy B: Axial Depth Reduction with Feed Compensation (same RPM)
        local_ap_lim = get_stability_limit_at_rpm(
            spindle_speed_rpm=baseline_rpm,
            natural_freq_hz=self.natural_freq_hz,
            num_flutes=self.bounds.num_flutes,
        )
        safe_depth = max(self.bounds.min_depth_of_cut_mm, local_ap_lim * 0.80)
        # Compensate feed so that ap * vf ~ baseline_ap * baseline_vf
        depth_ratio = baseline_depth / max(safe_depth, 0.1)
        compensated_feed = baseline_feed * min(depth_ratio, 1.6)

        candidates_pool.append({
            "id": f"C-{cand_idx:02d}",
            "strategy": "Depth Reduction + Feed Compensation",
            "rpm": baseline_rpm,
            "feed": compensated_feed,
            "depth": safe_depth,
        })
        cand_idx += 1

        # Strategy C: Conservative Max Stability Mode
        # Drop depth to 65% of local limit, moderate feed for ultra-smooth cut
        fz_nominal = 0.045
        conservative_feed = fz_nominal * self.bounds.num_flutes * baseline_rpm
        candidates_pool.append({
            "id": f"C-{cand_idx:02d}",
            "strategy": "Conservative / Minimum Vibration",
            "rpm": baseline_rpm,
            "feed": conservative_feed,
            "depth": max(self.bounds.min_depth_of_cut_mm, local_ap_lim * 0.65),
        })
        cand_idx += 1

        # Strategy D: Small Speed Shift (detuning tooth pass frequency by +/- 10-15%)
        for shift_pct in [-15.0, -8.0, 8.0, 15.0]:
            shifted_rpm = baseline_rpm * (1.0 + shift_pct / 100.0)
            if self.bounds.min_spindle_speed_rpm <= shifted_rpm <= self.bounds.max_spindle_speed_rpm:
                candidates_pool.append({
                    "id": f"C-{cand_idx:02d}",
                    "strategy": f"Speed Detuning ({shift_pct:+.0f}%)",
                    "rpm": shifted_rpm,
                    "feed": baseline_feed * (shifted_rpm / baseline_rpm),
                    "depth": min(baseline_depth, local_ap_lim * 0.90),
                })
                cand_idx += 1

        # Strategy E: Deliberate Boundary-Violating Candidates (for configured-bound rejection testing)
        # Over-speed violation
        candidates_pool.append({
            "id": f"C-{cand_idx:02d}",
            "strategy": "Excessive RPM (Infeasible Probe)",
            "rpm": self.bounds.max_spindle_speed_rpm + 2500.0,
            "feed": baseline_feed,
            "depth": baseline_depth,
        })
        cand_idx += 1

        # Excessive chip load violation (feed too high for RPM)
        breakage_feed = 0.25 * self.bounds.num_flutes * 3000.0  # fz = 0.25 > 0.16 max
        candidates_pool.append({
            "id": f"C-{cand_idx:02d}",
            "strategy": "High Chip Load Fracture Risk (Infeasible Probe)",
            "rpm": 3000.0,
            "feed": breakage_feed,
            "depth": baseline_depth,
        })
        cand_idx += 1

        # Spindle motor power overload violation
        candidates_pool.append({
            "id": f"C-{cand_idx:02d}",
            "strategy": "Spindle Power Overload (Infeasible Probe)",
            "rpm": 12000.0,
            "feed": 3500.0,
            "depth": 5.8,
        })
        cand_idx += 1

        # Evaluate all candidates
        evaluated_candidates: List[MachiningCandidate] = []
        for cand in candidates_pool:
            eval_cand = self.evaluate_candidate(
                candidate_id=cand["id"],
                strategy_name=cand["strategy"],
                spindle_speed_rpm=cand["rpm"],
                feed_rate_mm_min=cand["feed"],
                depth_of_cut_mm=cand["depth"],
                baseline_mrr=baseline_mrr,
                baseline_risk=baseline_risk,
                baseline_feed=baseline_feed,
            )
            evaluated_candidates.append(eval_cand)

        # Separate feasible and rejected candidates
        feasible_candidates = [c for c in evaluated_candidates if c.is_feasible]
        rejected_candidates = [c for c in evaluated_candidates if not c.is_feasible]

        # Calculate composite ranking score for feasible candidates
        if feasible_candidates:
            max_mrr = max(c.mrr_cm3_min for c in feasible_candidates)
            max_mrr = max(max_mrr, 1.0)
            max_wear = max(c.wear_factor for c in feasible_candidates)
            max_wear = max(max_wear, 0.5)

            total_weight = weight_stability + weight_mrr + weight_tool_life
            w_stab = weight_stability / total_weight
            w_mrr = weight_mrr / total_weight
            w_wear = weight_tool_life / total_weight

            for c in feasible_candidates:
                # Stability component (0 to 1): lower risk is better
                stab_score = (100.0 - c.risk_score) / 100.0
                # MRR component (0 to 1): higher MRR is better
                mrr_score = c.mrr_cm3_min / max_mrr
                # Tool life component (0 to 1): lower wear is better
                wear_score = 1.0 - (c.wear_factor / (max_wear * 1.5))
                wear_score = max(0.0, min(1.0, wear_score))

                # Composite score out of 100
                comp = (w_stab * stab_score + w_mrr * mrr_score + w_wear * wear_score) * 100.0
                c.composite_score = round(comp, 1)

            # Sort feasible candidates by composite score descending
            feasible_candidates.sort(key=lambda x: x.composite_score, reverse=True)

        recommended_candidate = feasible_candidates[0] if feasible_candidates else None

        # Build comparison summary
        baseline_summary = {
            "spindle_speed_rpm": baseline_rpm,
            "feed_rate_mm_min": baseline_feed,
            "depth_of_cut_mm": baseline_depth,
            "vibration_rms_g": baseline_rms,
            "vibration_peak_g": baseline_peak,
            "dominant_frequency_hz": baseline_dom_freq,
            "predicted_state": baseline_pred["predicted_state"],
            "probabilities": baseline_pred["probabilities"],
            "risk_score": round(baseline_risk, 1),
            "mrr_cm3_min": round(baseline_mrr, 2),
            "chip_load_mm_tooth": round(baseline_bounds_res.chip_load_mm_tooth, 4),
            "cutting_speed_m_min": round(baseline_bounds_res.cutting_speed_m_min, 1),
            "cutting_power_kw": round(baseline_bounds_res.cutting_power_kw, 2),
            "cutting_force_n": round(baseline_bounds_res.cutting_force_n, 1),
            "is_feasible": baseline_bounds_res.is_feasible,
            "rejection_reasons": baseline_bounds_res.rejection_reasons,
        }

        return {
            "baseline": baseline_summary,
            "recommended": recommended_candidate,
            "feasible_candidates": feasible_candidates,
            "rejected_candidates": rejected_candidates,
            "total_candidates_generated": len(evaluated_candidates),
            "num_feasible": len(feasible_candidates),
            "num_rejected": len(rejected_candidates),
            "sweet_spot_rpms": sweet_rpms,
        }
