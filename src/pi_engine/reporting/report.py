"""Transparent, model-conditioned prediction reporting."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pi_engine.analysis.closure import ClosureAnalysis
from pi_engine.analysis.convergence import ConvergenceAnalysis
from pi_engine.analysis.divergence import SpreadAnalysis
from pi_engine.analysis.residuals import ResidualAnalysis
from pi_engine.evaluation.scoring import ForecastScoreReport
from pi_engine.registry.applicability import ApplicabilityResult
from pi_engine.schemas.case import Case
from pi_engine.schemas.common import Confidence
from pi_engine.schemas.model import ExplicitModel
from pi_engine.schemas.trajectory import Trajectory, TrajectoryEnsemble


UNIDENTIFIABLE_FROM_AVAILABLE_EVIDENCE = "UNIDENTIFIABLE FROM AVAILABLE EVIDENCE"
IDENTIFIED_MODEL_CONDITIONED = "MODEL-CONDITIONED PREDICTIONS AVAILABLE"
PredictionArtifact = Trajectory | TrajectoryEnsemble
Identifiability = Literal[
    "MODEL-CONDITIONED PREDICTIONS AVAILABLE",
    "UNIDENTIFIABLE FROM AVAILABLE EVIDENCE",
]


class PredictionReport(BaseModel):
    """A structured, provenance-retaining view of available prediction evidence."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )

    case: Case
    state_confidence: Confidence
    models: tuple[ExplicitModel, ...]
    applicability: tuple[ApplicabilityResult, ...]
    trajectories: tuple[PredictionArtifact, ...] = ()
    convergence: tuple[ConvergenceAnalysis, ...] = ()
    spread: tuple[SpreadAnalysis, ...] = ()
    closure: tuple[ClosureAnalysis, ...] = ()
    residuals: tuple[ResidualAnalysis, ...] = ()
    observability_warnings: tuple[str, ...] = ()
    held_out_scores: ForecastScoreReport | None = None
    identifiability: Identifiability

    @field_validator("observability_warnings")
    @classmethod
    def warnings_must_be_nonempty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item for item in value):
            raise ValueError("observability warnings must be nonempty")
        return value

    @model_validator(mode="after")
    def validate_identity_alignment(self) -> "PredictionReport":
        model_keys = _model_keys(self.models)
        if len(model_keys) != len(set(model_keys)):
            raise ValueError("report model identities must be unique")

        applicability_keys = tuple(
            (item.model_id, item.model_version) for item in self.applicability
        )
        if len(applicability_keys) != len(set(applicability_keys)):
            raise ValueError("applicability identities must be unique")
        if set(applicability_keys) - set(model_keys):
            raise ValueError("applicability results must reference retained models")

        expected_identifiability: Identifiability = (
            IDENTIFIED_MODEL_CONDITIONED
            if any(item.applicable for item in self.applicability)
            else UNIDENTIFIABLE_FROM_AVAILABLE_EVIDENCE
        )
        if self.identifiability != expected_identifiability:
            raise ValueError("identifiability must derive from applicability evidence")

        _validate_artifacts(
            self.case.case_id,
            set(model_keys),
            self.trajectories,
            role="trajectory",
        )
        _validate_analyses(self.case.case_id, set(model_keys), self)
        _validate_residuals(self.case.case_id, set(model_keys), self.residuals)
        _validate_scores(self.case.case_id, self.trajectories, self.held_out_scores)
        return self


def build_prediction_report(
    *,
    case: Case,
    state_confidence: Confidence,
    models: Sequence[ExplicitModel] = (),
    applicability: Sequence[ApplicabilityResult] = (),
    trajectories: Sequence[PredictionArtifact] = (),
    convergence: Sequence[ConvergenceAnalysis] = (),
    spread: Sequence[SpreadAnalysis] = (),
    closure: Sequence[ClosureAnalysis] = (),
    residuals: Sequence[ResidualAnalysis] = (),
    observability_warnings: Sequence[str] = (),
    held_out_scores: ForecastScoreReport | None = None,
) -> PredictionReport:
    """Build a report without selecting a model or combining its evidence."""
    model_records = tuple(models)
    applicability_records = tuple(applicability)
    identifiability: Identifiability = (
        IDENTIFIED_MODEL_CONDITIONED
        if any(item.applicable for item in applicability_records)
        else UNIDENTIFIABLE_FROM_AVAILABLE_EVIDENCE
    )
    return PredictionReport(
        case=case,
        state_confidence=state_confidence,
        models=model_records,
        applicability=applicability_records,
        trajectories=tuple(trajectories),
        convergence=tuple(convergence),
        spread=tuple(spread),
        closure=tuple(closure),
        residuals=tuple(residuals),
        observability_warnings=tuple(observability_warnings),
        held_out_scores=held_out_scores,
        identifiability=identifiability,
    )


def render_prediction_report(report: PredictionReport) -> str:
    """Render retained evidence in a concise, human-readable form."""
    validated = PredictionReport.model_validate(report.model_dump(warnings=False))
    lines = [
        "PI-ENGINE PREDICTION REPORT",
        f"Case: {validated.case.case_id} — {validated.case.title}",
        f"Domain: {validated.case.domain}",
        f"Prediction cutoff: {validated.case.prediction_cutoff.isoformat()}",
        f"Identifiability: {validated.identifiability}",
        _render_confidence(validated.state_confidence),
        _render_state_uncertainty(validated.case),
        "Model uncertainty:",
        *_render_model_uncertainty(validated.models),
        "Observability warnings:",
        *_bullets(validated.observability_warnings, "none declared"),
        "Model applicability:",
        *_render_applicability(validated.applicability),
        "Model-conditioned trajectories:",
        *_render_trajectories(validated.trajectories),
        "Convergence:",
        *_render_convergence(validated.convergence),
        "Divergence/spread:",
        *_render_spread(validated.spread),
        "Closure:",
        *_render_closure(validated.closure),
        "Residual status:",
        *_render_residuals(validated.residuals),
        *_render_scores(validated.held_out_scores),
    ]
    return "\n".join(lines)


def _model_keys(models: Sequence[ExplicitModel]) -> tuple[tuple[str, str], ...]:
    return tuple((model.model_id, model.version) for model in models)


def _artifact_key(artifact: PredictionArtifact) -> tuple[str, str]:
    return artifact.model_id, artifact.model_version


def _validate_artifacts(
    case_id: str,
    model_keys: set[tuple[str, str]],
    artifacts: Sequence[PredictionArtifact],
    *,
    role: str,
) -> None:
    for artifact in artifacts:
        if artifact.case_id != case_id:
            raise ValueError(f"{role} case_id must match report case")
        if _artifact_key(artifact) not in model_keys:
            raise ValueError(f"{role} must reference a retained model")


def _validate_analyses(
    case_id: str,
    model_keys: set[tuple[str, str]],
    report: PredictionReport,
) -> None:
    for analysis in report.convergence:
        _validate_artifacts(case_id, model_keys, (analysis.source,), role="convergence")
    for analysis in report.spread:
        _validate_artifacts(case_id, model_keys, (analysis.source,), role="spread")
    for analysis in report.closure:
        _validate_artifacts(
            case_id, model_keys, (analysis.source.source,), role="closure"
        )


def _validate_residuals(
    case_id: str,
    model_keys: set[tuple[str, str]],
    residuals: Sequence[ResidualAnalysis],
) -> None:
    for analysis in residuals:
        residual = analysis.residual
        if residual.case_id != case_id:
            raise ValueError("residual case_id must match report case")
        if (residual.model_id, residual.model_version) not in model_keys:
            raise ValueError("residual must reference a retained model")


def _validate_scores(
    case_id: str,
    report_artifacts: Sequence[PredictionArtifact],
    scores: ForecastScoreReport | None,
) -> None:
    if scores is None:
        return
    if scores.case_id != case_id:
        raise ValueError("held-out scores case_id must match report case")
    report_by_key = {
        _prediction_artifact_identity(item): item for item in report_artifacts
    }
    for artifact in scores.prediction_artifacts:
        key = _prediction_artifact_identity(artifact)
        retained = report_by_key.get(key)
        if retained is None or retained.model_dump(mode="json") != artifact.model_dump(
            mode="json"
        ):
            raise ValueError(
                "held-out scores must retain exact report trajectory artifacts"
            )


def _prediction_artifact_identity(artifact: PredictionArtifact) -> tuple[str, str]:
    if isinstance(artifact, TrajectoryEnsemble):
        return "trajectory_ensemble", artifact.ensemble_id
    return "trajectory", artifact.trajectory_id


def _render_confidence(confidence: Confidence) -> str:
    basis = f" (basis: {confidence.basis})" if confidence.basis else ""
    return f"State confidence: {confidence.score}{basis}"


def _render_state_uncertainty(case: Case) -> str:
    if not case.state.uncertainty:
        return "State uncertainty: none retained"
    values = ", ".join(
        f"{variable}={value!r}"
        for variable, value in sorted(case.state.uncertainty.items())
    )
    return f"State uncertainty: {values}"


def _render_model_uncertainty(
    models: Sequence[ExplicitModel],
) -> tuple[str, ...]:
    lines = []
    for model in models:
        uncertainty = model.uncertainty
        fields = (
            ("parameter", uncertainty.parameter_uncertainty),
            ("process_disturbance", uncertainty.process_disturbance),
            ("model_discrepancy", uncertainty.model_discrepancy),
            ("structured_unknowns", uncertainty.structured_unknowns),
        )
        descriptions = "; ".join(
            f"{name}={', '.join(values) if values else 'none declared'}"
            for name, values in fields
        )
        lines.append(f"- {model.model_id}@{model.version}: {descriptions}")
    return tuple(lines) or ("- no model uncertainty records retained",)


def _bullets(values: Sequence[str], empty: str) -> tuple[str, ...]:
    return tuple(f"- {item}" for item in values) or (f"- {empty}",)


def _render_applicability(
    results: Sequence[ApplicabilityResult],
) -> tuple[str, ...]:
    lines: list[str] = []
    for result in results:
        identity = f"{result.model_id}@{result.model_version}"
        if result.applicable:
            rank = f", rank {result.rank}" if result.rank is not None else ""
            lines.append(
                f"- {identity}: applicable{rank}; structural score "
                f"{result.structural_score}/{result.structural_score_max}"
            )
        else:
            causes = "; ".join(result.rejection_causes) or "no cause retained"
            lines.append(f"- {identity}: rejected; causes: {causes}")
    return tuple(lines) or ("- no model applicability results retained",)


def _render_trajectories(
    trajectories: Sequence[PredictionArtifact],
) -> tuple[str, ...]:
    lines: list[str] = []
    for artifact in trajectories:
        if isinstance(artifact, TrajectoryEnsemble):
            lines.append(
                f"- ensemble {artifact.ensemble_id}: "
                f"{artifact.model_id}@{artifact.model_version}, "
                f"{len(artifact.trajectories)} retained trajectories"
            )
        else:
            lines.append(
                f"- trajectory {artifact.trajectory_id}: "
                f"{artifact.model_id}@{artifact.model_version}, "
                f"{len(artifact.points)} retained points"
            )
    return tuple(lines) or ("- no model-conditioned trajectories retained",)


def _render_convergence(
    analyses: Sequence[ConvergenceAnalysis],
) -> tuple[str, ...]:
    lines = [
        "- "
        f"{analysis.source.trajectory_id}: "
        + "; ".join(
            f"{pattern.variable}={pattern.observed_pattern}"
            for pattern in analysis.patterns
        )
        for analysis in analyses
    ]
    return tuple(lines) or ("- no convergence analysis retained",)


def _render_spread(analyses: Sequence[SpreadAnalysis]) -> tuple[str, ...]:
    lines = [
        "- "
        f"{analysis.source_kind}, {analysis.member_count} member(s): "
        + "; ".join(
            f"{pattern.variable}={pattern.observed_pattern}"
            for pattern in analysis.patterns
        )
        for analysis in analyses
    ]
    return tuple(lines) or ("- no divergence/spread analysis retained",)


def _render_closure(analyses: Sequence[ClosureAnalysis]) -> tuple[str, ...]:
    lines: list[str] = []
    for analysis in analyses:
        if not analysis.events:
            lines.append("- no qualifying closure events; no closure conclusion is implied")
            continue
        lines.extend(
            f"- {event.event_id}: {event.variable}, "
            f"domain={event.domain}, firmness={event.firmness}, "
            f"basis={event.classification_basis}"
            for event in analysis.events
        )
    return tuple(lines) or ("- no closure analysis retained",)


def _render_residuals(
    residuals: Sequence[ResidualAnalysis],
) -> tuple[str, ...]:
    lines = [
        f"- {analysis.residual.residual_id}: "
        f"{analysis.classification.category.value}; "
        f"basis: {analysis.classification.basis}"
        for analysis in residuals
    ]
    return tuple(lines) or ("- no residual analysis retained",)


def _render_scores(scores: ForecastScoreReport | None) -> tuple[str, ...]:
    if scores is None:
        return ("Held-out scores: NOT REVEALED",)
    lines = ["Held-out scores: REVEALED"]
    for artifact in scores.artifacts:
        identity = f"{artifact.model_id}@{artifact.model_version}"
        metrics = "; ".join(
            f"{metric.variable}: mean_absolute_error={metric.mean_absolute_error}"
            for metric in artifact.continuous_metrics
        ) or "no continuous metrics retained"
        lines.append(f"- {identity}: {metrics}")
    return tuple(lines)


__all__ = [
    "IDENTIFIED_MODEL_CONDITIONED",
    "PredictionReport",
    "UNIDENTIFIABLE_FROM_AVAILABLE_EVIDENCE",
    "build_prediction_report",
    "render_prediction_report",
]
