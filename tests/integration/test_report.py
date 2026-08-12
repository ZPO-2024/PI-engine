"""Integration coverage for transparent, theory-neutral prediction reports."""

import pytest

from pi_engine.analysis.closure import ClosureThresholds, analyze_closure
from pi_engine.analysis.convergence import analyze_trajectory_convergence
from pi_engine.analysis.divergence import analyze_trajectory_spread
from pi_engine.analysis.residuals import analyze_residual
from pi_engine.evaluation.holdout import prepare_holdout, reveal_holdout
from pi_engine.evaluation.scoring import score_revealed_evaluation
from pi_engine.registry.applicability import ApplicabilityResult
from pi_engine.schemas.common import Confidence, Provenance
from pi_engine.schemas.residual import (
    Residual,
    ResidualCategory,
    ResidualClassification,
)
from pi_engine.simulation.runner import simulate_deterministic
from pi_engine.synthetic.systems import deterministic_divergence, linear_convergence


def _residual_for_linear_forecast(trajectory: object) -> object:
    fixture = linear_convergence()
    outcome = fixture.outcomes[0]
    return analyze_residual(
        Residual(
            residual_id="linear-residual-1",
            trajectory_id=getattr(trajectory, "trajectory_id"),
            prediction_time=fixture.case.prediction_cutoff,
            model_id=fixture.model.model_id,
            model_version=fixture.model.version,
            case_id=fixture.case.case_id,
            variable="x",
            unit="a.u.",
            predicted_value=1.0,
            observed_outcome=outcome,
            error=0.0,
            error_convention="predicted_minus_observed",
            classification=ResidualClassification(
                category=ResidualCategory.STRUCTURED_UNKNOWN,
                basis="awaiting inspectable residual evidence",
            ),
            provenance=Provenance(
                source="report integration fixture",
                observed_at=fixture.case.prediction_cutoff,
                reference="report:linear-residual-1",
            ),
        )
    )


def _linear_report_inputs() -> dict[str, object]:
    fixture = linear_convergence()
    trajectory = simulate_deterministic(
        fixture.case, fixture.model, horizon=3
    )
    spread = analyze_trajectory_spread(
        trajectory, normalization_scales={"x": 1.0}
    )
    alternate = deterministic_divergence().model
    return {
        "case": fixture.case,
        "state_confidence": Confidence(
            score=0.8, basis="three direct measurements at the cutoff"
        ),
        "models": (fixture.model, alternate),
        "applicability": (
            ApplicabilityResult(
                model_id=fixture.model.model_id,
                model_version=fixture.model.version,
                applicable=True,
                structural_score=0,
                structural_score_max=0,
                rank=1,
                reasons=("required variable present: x",),
                rejection_causes=(),
            ),
            ApplicabilityResult(
                model_id=alternate.model_id,
                model_version=alternate.version,
                applicable=False,
                structural_score=0,
                structural_score_max=0,
                rank=None,
                reasons=(),
                rejection_causes=("synthetic competing-model rejection",),
            ),
        ),
        "trajectories": (trajectory,),
        "convergence": (
            analyze_trajectory_convergence(
                trajectory, normalization_scales={"x": 1.0}
            ),
        ),
        "spread": (spread,),
        "closure": (
            analyze_closure(
                spread,
                thresholds=ClosureThresholds(
                    minimum_relative_contraction=0.25,
                    abrupt_relative_contraction=0.75,
                ),
            ),
        ),
        "residuals": (_residual_for_linear_forecast(trajectory),),
        "observability_warnings": (
            "latent state is not directly observed",
        ),
    }


def test_report_retains_separate_models_and_analysis_artifacts() -> None:
    """Collapsing decisions or analyses into a selected model must fail here."""
    from pi_engine.reporting.report import (
        build_prediction_report,
        render_prediction_report,
    )

    inputs = _linear_report_inputs()
    report = build_prediction_report(**inputs)
    structured = report.model_dump(mode="json")
    text = render_prediction_report(report)

    assert structured["case"]["case_id"] == "synthetic-linear-convergence"
    assert structured["state_confidence"] == {
        "score": 0.8,
        "basis": "three direct measurements at the cutoff",
    }
    assert [item["applicable"] for item in structured["applicability"]] == [
        True,
        False,
    ]
    assert structured["applicability"][1]["rejection_causes"] == [
        "synthetic competing-model rejection"
    ]
    trajectory = inputs["trajectories"][0]
    assert structured["trajectories"][0]["trajectory_id"] == getattr(
        trajectory, "trajectory_id"
    )
    assert structured["convergence"][0]["patterns"][0]["observed_pattern"] == (
        "strictly_contracting_step_distance"
    )
    assert structured["spread"][0]["patterns"][0]["observed_pattern"] == (
        "deterministic_singleton_no_spread"
    )
    assert structured["closure"][0]["events"] == []
    assert structured["residuals"][0]["classification"]["category"] == (
        "structured_unknown"
    )
    assert structured["held_out_scores"] is None
    assert "Model applicability:" in text
    assert "synthetic-linear-affine-convergence@1.0.0: applicable" in text
    assert "synthetic competing-model rejection" in text
    assert "Model-conditioned trajectories:" in text
    assert "Model uncertainty:" in text
    assert "Observability warnings:" in text
    assert "Closure:" in text
    assert "Residual status:" in text
    assert "Held-out scores: NOT REVEALED" in text


def test_report_marks_a_case_unidentifiable_when_no_model_is_applicable() -> None:
    """Replacing evidence absence with an implied forecast must fail here."""
    from pi_engine.reporting.report import (
        UNIDENTIFIABLE_FROM_AVAILABLE_EVIDENCE,
        build_prediction_report,
        render_prediction_report,
    )

    fixture = linear_convergence()
    report = build_prediction_report(
        case=fixture.case,
        state_confidence=Confidence(
            score=0.0, basis="no applicable model supports a forecast"
        ),
        models=(fixture.model,),
        applicability=(
            ApplicabilityResult(
                model_id=fixture.model.model_id,
                model_version=fixture.model.version,
                applicable=False,
                structural_score=0,
                structural_score_max=0,
                rank=None,
                reasons=(),
                rejection_causes=("required variable unavailable",),
            ),
        ),
        observability_warnings=("x is unavailable at the cutoff",),
    )

    assert report.identifiability == UNIDENTIFIABLE_FROM_AVAILABLE_EVIDENCE
    assert UNIDENTIFIABLE_FROM_AVAILABLE_EVIDENCE in render_prediction_report(report)


def test_report_includes_artifact_bound_scores_only_after_reveal() -> None:
    """Dropping the revealed score artifact or inventing a total score must fail."""
    from pi_engine.reporting.report import (
        build_prediction_report,
        render_prediction_report,
    )

    inputs = _linear_report_inputs()
    fixture = linear_convergence()
    trajectory = inputs["trajectories"][0]
    prepared = prepare_holdout(fixture.case, fixture.outcomes)
    revealed = reveal_holdout(prepared, trajectory)
    scores = score_revealed_evaluation(revealed, (trajectory,))

    report = build_prediction_report(**inputs, held_out_scores=scores)
    structured = report.model_dump(mode="json")
    text = render_prediction_report(report)

    assert structured["held_out_scores"]["case_id"] == fixture.case.case_id
    assert structured["held_out_scores"]["artifacts"][0]["continuous_metrics"][0][
        "mean_absolute_error"
    ] == 0.0
    assert "Held-out scores: REVEALED" in text
    assert "mean_absolute_error=0.0" in text
    assert "master score" not in text.lower()


def test_report_rejects_scores_for_a_same_id_but_changed_trajectory() -> None:
    """Replacing a report artifact after reveal must not preserve its score link."""
    from pi_engine.reporting.report import build_prediction_report

    inputs = _linear_report_inputs()
    fixture = linear_convergence()
    trajectory = inputs["trajectories"][0]
    prepared = prepare_holdout(fixture.case, fixture.outcomes)
    scores = score_revealed_evaluation(
        reveal_holdout(prepared, trajectory), (trajectory,)
    )
    changed_point = trajectory.points[0].model_copy(
        update={"values": {"x": 9.0}}
    )
    inputs["trajectories"] = (
        trajectory.model_copy(
            update={"points": (changed_point, *trajectory.points[1:])}
        ),
    )

    with pytest.raises(ValueError, match="exact report trajectory artifacts"):
        build_prediction_report(**inputs, held_out_scores=scores)
