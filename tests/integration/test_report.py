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
from pi_engine.simulation.stochastic import simulate_stochastic
from pi_engine.synthetic.systems import (
    deterministic_divergence,
    hierarchical_nested_dynamics,
    linear_convergence,
    stochastic_branching,
)


def _report_provenance(reference: str) -> Provenance:
    return Provenance(
        source="report integration fixture",
        observed_at=linear_convergence().case.prediction_cutoff,
        reference=reference,
    )


def _state_confidence_assessment(score: float = 0.8) -> dict[str, object]:
    return {
        "confidence": Confidence(
            score=score, basis="three direct measurements at the cutoff"
        ),
        "provenance": _report_provenance("report:state-confidence"),
    }


def _observability_warning(message: str) -> dict[str, object]:
    return {
        "message": message,
        "provenance": _report_provenance("report:observability-warning"),
    }


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
        "provenance": _report_provenance("report:linear"),
        "state_confidence": _state_confidence_assessment(),
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
            _observability_warning("latent state is not directly observed"),
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
    assert structured["provenance"]["reference"] == "report:linear"
    assert structured["state_confidence"] == {
        "confidence": {
            "score": 0.8,
            "basis": "three direct measurements at the cutoff",
        },
        "provenance": {
            "source": "report integration fixture",
            "observed_at": "2026-08-08T12:00:00Z",
            "reference": "report:state-confidence",
        },
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
    assert structured["observability_warnings"][0]["provenance"]["reference"] == (
        "report:observability-warning"
    )
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
        provenance=_report_provenance("report:unidentifiable"),
        state_confidence={
            "confidence": Confidence(
                score=0.0, basis="no applicable model supports a forecast"
            ),
            "provenance": _report_provenance("report:unidentifiable-confidence"),
        },
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
        observability_warnings=(
            _observability_warning("x is unavailable at the cutoff"),
        ),
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


def test_report_renders_canonical_state_and_graph_relationship_views() -> None:
    """Removing canonical variable or graph relations must make this view fail."""
    from pi_engine.reporting.report import (
        build_prediction_report,
        render_prediction_report,
    )

    fixture = hierarchical_nested_dynamics()
    report = build_prediction_report(
        case=fixture.case,
        provenance=_report_provenance("report:hierarchy"),
        state_confidence=_state_confidence_assessment(),
        models=(fixture.model,),
        applicability=(
            ApplicabilityResult(
                model_id=fixture.model.model_id,
                model_version=fixture.model.version,
                applicable=True,
                structural_score=0,
                structural_score_max=0,
                rank=1,
                reasons=("hierarchical fixture applies",),
                rejection_causes=(),
            ),
        ),
        observability_warnings=(),
    )

    text = render_prediction_report(report)

    assert "Canonical state-space:" in text
    assert "levels [a.u.]: observed=(1.0, 0.0, 0.0)" in text
    assert "Graph nodes:" in text
    assert "root: variables=levels" in text
    assert "Graph relationships:" in text
    assert "root -> middle: coupling_type=parent_to_child; strength=0.5" in text


def test_report_renders_all_revealed_score_metric_classes() -> None:
    """Dropping any retained score class from text must fail these real reports."""
    from pi_engine.reporting.report import (
        build_prediction_report,
        render_prediction_report,
    )

    linear = linear_convergence()
    binary_prediction = simulate_deterministic(
        linear.case, linear.model, horizon=3
    ).model_copy(
        update={
            "points": tuple(
                point.model_copy(update={"values": {"x": value}})
                for point, value in zip(
                    simulate_deterministic(linear.case, linear.model, horizon=3).points,
                    (0.2, 0.8, 0.6),
                    strict=True,
                )
            )
        }
    )
    binary_outcomes = tuple(
        outcome.model_copy(update={"value": value})
        for outcome, value in zip(linear.outcomes, (0.0, 1.0, 1.0), strict=True)
    )
    binary_scores = score_revealed_evaluation(
        reveal_holdout(prepare_holdout(linear.case, binary_outcomes), binary_prediction),
        (binary_prediction,),
        binary_probability_variables=("x",),
        include_log_score=True,
    )
    binary_inputs = _linear_report_inputs()
    binary_inputs["trajectories"] = (binary_prediction,)
    binary_text = render_prediction_report(
        build_prediction_report(**binary_inputs, held_out_scores=binary_scores)
    )

    assert "brier_score=" in binary_text
    assert "log_score=" in binary_text
    assert "mean_brier_score=" in binary_text
    assert "mean_log_score=" in binary_text

    branching = stochastic_branching(seed=7)
    ensemble = simulate_stochastic(
        branching.case, branching.model, horizon=4, samples=4, seed=7
    )
    interval_scores = score_revealed_evaluation(
        reveal_holdout(prepare_holdout(branching.case, branching.outcomes), ensemble),
        (ensemble,),
        interval_levels=(0.5,),
    )
    interval_text = render_prediction_report(
        build_prediction_report(
            case=branching.case,
            provenance=_report_provenance("report:interval-scores"),
            state_confidence=_state_confidence_assessment(),
            models=(branching.model,),
            applicability=(
                ApplicabilityResult(
                    model_id=branching.model.model_id,
                    model_version=branching.model.version,
                    applicable=True,
                    structural_score=0,
                    structural_score_max=0,
                    rank=1,
                    reasons=("branching fixture applies",),
                    rejection_causes=(),
                ),
            ),
            trajectories=(ensemble,),
            held_out_scores=interval_scores,
        )
    )

    assert "mean_error=" in interval_text
    assert "mean_absolute_error=" in interval_text
    assert "mean_squared_error=" in interval_text
    assert "root_mean_squared_error=" in interval_text
    assert "nominal_coverage=0.5" in interval_text
    assert "covered=" in interval_text


def test_report_requires_every_model_to_have_one_applicability_decision() -> None:
    """Omitting a competing model's decision must not leave it unexplained."""
    from pi_engine.reporting.report import build_prediction_report

    inputs = _linear_report_inputs()
    inputs["applicability"] = inputs["applicability"][:1]

    with pytest.raises(ValueError, match="exactly one applicability decision"):
        build_prediction_report(**inputs)


def test_report_rejects_trajectory_for_a_rejected_model() -> None:
    """A rejected model cannot silently retain a prediction artifact."""
    from pi_engine.reporting.report import build_prediction_report

    inputs = _linear_report_inputs()
    accepted = inputs["applicability"][0]
    inputs["applicability"] = (
        accepted.model_copy(update={"applicable": False, "rank": None}),
        *inputs["applicability"][1:],
    )

    with pytest.raises(ValueError, match="applicable decisions"):
        build_prediction_report(**inputs)
