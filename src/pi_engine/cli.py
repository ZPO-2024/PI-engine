"""Minimal, auditable command line harness for PI-engine experiments."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from pi_engine.analysis.closure import ClosureThresholds, analyze_closure
from pi_engine.analysis.convergence import analyze_trajectory_convergence
from pi_engine.analysis.divergence import analyze_trajectory_spread
from pi_engine.analysis.residuals import ResidualAnalysis, analyze_residual
from pi_engine.evaluation.holdout import prepare_holdout, reveal_holdout
from pi_engine.evaluation.scoring import score_revealed_evaluation
from pi_engine.registry import ModelRegistry
from pi_engine.reporting.report import (
    ObservabilityWarning,
    StateConfidenceAssessment,
    build_prediction_report,
    render_prediction_report,
)
from pi_engine.schemas.common import Confidence, NumericValue, Provenance
from pi_engine.schemas.model import ExplicitModel
from pi_engine.schemas.residual import Residual, ResidualClassification
from pi_engine.schemas.trajectory import Trajectory, TrajectoryEnsemble
from pi_engine.simulation.runner import simulate_deterministic
from pi_engine.simulation.stochastic import simulate_stochastic
from pi_engine.synthetic.controls import CONTROL_CATALOG
from pi_engine.synthetic.systems import SYSTEM_CATALOG, SyntheticSystem


PredictionArtifact = Trajectory | TrajectoryEnsemble
_CATALOG = {**SYSTEM_CATALOG, **CONTROL_CATALOG}


def _fixture(case_name: str, seed: int) -> SyntheticSystem:
    try:
        factory = _CATALOG[case_name]
    except KeyError as exc:
        names = ", ".join(sorted(_CATALOG))
        raise ValueError(f"unknown synthetic case {case_name!r}; choose one of: {names}") from exc
    return factory(seed)


def _registry() -> tuple[ModelRegistry, tuple[ExplicitModel, ...]]:
    registry = ModelRegistry()
    models = tuple(factory(0).model for factory in SYSTEM_CATALOG.values())
    for model in models:
        registry.register(model)
    retained = tuple(registry.get(model.model_id, model.version) for model in models)
    return registry, retained


def _normalization_scales(fixture: SyntheticSystem) -> dict[str, object]:
    return {
        variable.name: 1.0
        for variable in fixture.case.canonical_variables
    }


def _simulate(
    fixture: SyntheticSystem, model: ExplicitModel, horizon: int, seed: int
) -> PredictionArtifact:
    if model.dynamics.classification == "deterministic":
        return simulate_deterministic(fixture.case, model, horizon)
    return simulate_stochastic(
        fixture.case,
        model,
        horizon=horizon,
        samples=8,
        seed=seed,
    )


def _analysis_trajectory(artifact: PredictionArtifact) -> Trajectory:
    if isinstance(artifact, TrajectoryEnsemble):
        return artifact.trajectories[0]
    return artifact


def _difference(predicted: NumericValue, observed: NumericValue) -> NumericValue:
    if isinstance(predicted, tuple):
        if not isinstance(observed, tuple):
            raise ValueError("prediction and outcome shapes must match")
        return tuple(left - right for left, right in zip(predicted, observed, strict=True))
    if isinstance(observed, tuple):
        raise ValueError("prediction and outcome shapes must match")
    return predicted - observed


def _residuals(
    fixture: SyntheticSystem, artifacts: Sequence[PredictionArtifact]
) -> tuple[ResidualAnalysis, ...]:
    analyses = []
    for artifact in artifacts:
        if not isinstance(artifact, Trajectory):
            continue
        by_time = {point.at: point for point in artifact.points}
        for outcome in fixture.outcomes:
            point = by_time[outcome.event_time]
            predicted = point.values[outcome.variable]
            analyses.append(
                analyze_residual(
                    Residual(
                        residual_id=(
                            f"{artifact.trajectory_id}:{outcome.outcome_id}"
                        ),
                        trajectory_id=artifact.trajectory_id,
                        prediction_time=fixture.case.prediction_cutoff,
                        model_id=artifact.model_id,
                        model_version=artifact.model_version,
                        case_id=fixture.case.case_id,
                        variable=outcome.variable,
                        unit=outcome.unit,
                        predicted_value=predicted,
                        observed_outcome=outcome,
                        error=_difference(predicted, outcome.value),
                        error_convention="predicted_minus_observed",
                        classification=ResidualClassification(
                            category="structured_unknown",
                            basis="classification is derived from retained evidence",
                        ),
                        provenance=Provenance(
                            source="PI-engine experiment CLI",
                            observed_at=outcome.available_at,
                            reference=f"cli:residual:{outcome.outcome_id}",
                        ),
                    )
                )
            )
    return tuple(analyses)


def _render_experiment(
    fixture: SyntheticSystem, *, horizon: int, reveal: bool
) -> str:
    registry, models = _registry()
    applicability = registry.rank_applicable_models(fixture.case)
    artifacts = tuple(
        _simulate(
            fixture,
            registry.get(result.model_id, result.model_version),
            horizon,
            fixture.seed,
        )
        for result in applicability
        if result.applicable
    )
    scales = _normalization_scales(fixture)
    provenance = Provenance(
        source="PI-engine experiment CLI",
        observed_at=fixture.case.prediction_cutoff,
        reference=f"cli:{fixture.name}:prediction-report",
    )
    held_out_scores = None
    residuals: tuple[ResidualAnalysis, ...] = ()
    if reveal and not artifacts:
        raise ValueError("held-out outcomes cannot be revealed without a prediction artifact")
    if reveal:
        prepared = prepare_holdout(fixture.case, fixture.outcomes)
        revealed = tuple(reveal_holdout(prepared, artifact) for artifact in artifacts)
        combined_reveal = revealed[0].model_copy(
            update={
                "prediction_references": tuple(
                    reference
                    for item in revealed
                    for reference in item.prediction_references
                )
            }
        )
        held_out_scores = score_revealed_evaluation(combined_reveal, artifacts)
        residuals = _residuals(fixture, artifacts)
    convergence = tuple(
        analyze_trajectory_convergence(
            _analysis_trajectory(artifact), normalization_scales=scales
        )
        for artifact in artifacts
    )
    spread = tuple(
        analyze_trajectory_spread(artifact, normalization_scales=scales)
        for artifact in artifacts
    )
    closure = tuple(
        analyze_closure(
            analysis,
            thresholds=ClosureThresholds(
                minimum_relative_contraction=0.25,
                abrupt_relative_contraction=0.75,
            ),
        )
        for analysis in spread
    )
    report = build_prediction_report(
        case=fixture.case,
        provenance=provenance,
        state_confidence=StateConfidenceAssessment(
            confidence=Confidence(
                score=1.0,
                basis="synthetic state is explicit at the prediction cutoff",
            ),
            provenance=provenance,
        ),
        models=models,
        applicability=applicability,
        trajectories=artifacts,
        convergence=convergence,
        spread=spread,
        closure=closure,
        residuals=residuals,
        observability_warnings=(
            ObservabilityWarning(
                message=(
                    "held-out outcomes remain sealed until the explicit reveal command"
                    if artifacts
                    else "no registered synthetic-system model applies to this negative control"
                ),
                provenance=provenance,
            ),
        ),
        held_out_scores=held_out_scores,
    )
    calibration = (
        "held-out forecast scores are retained by model artifact; no aggregate "
        "calibration verdict or model selection is inferred."
        if reveal
        else "held-out outcomes have not been revealed, so no calibration score is available."
    )
    return "\n".join(
        (
            "Model disagreement: retained separately; registry applicability and "
            "model-conditioned artifacts do not imply a selected winner.",
            f"Calibration information: {calibration}",
            render_prediction_report(report),
        )
    )


def _list_cases() -> str:
    rows = ["PI-ENGINE SYNTHETIC CASES"]
    for name in sorted(SYSTEM_CATALOG):
        fixture = SYSTEM_CATALOG[name](0)
        rows.append(f"- {name}: regime={fixture.regime}; synthetic_system")
    for name in sorted(CONTROL_CATALOG):
        fixture = CONTROL_CATALOG[name](0)
        rows.append(f"- {name}: regime={fixture.regime}; negative_control")
    return "\n".join(rows)


def _add_experiment_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("case", choices=sorted(_CATALOG))
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pi-engine",
        description="Run explicit, cutoff-safe PI-engine synthetic experiments.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list synthetic systems and negative controls")
    _add_experiment_arguments(
        commands.add_parser("run", help="run a cutoff-safe prediction report")
    )
    _add_experiment_arguments(
        commands.add_parser("report", help="render a cutoff-safe prediction report")
    )
    _add_experiment_arguments(
        commands.add_parser("reveal", help="reveal and score held-out outcomes")
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parser = _parser()
    namespace = parser.parse_args(arguments)
    if namespace.command == "list":
        print(_list_cases())
        return 0
    fixture = _fixture(namespace.case, namespace.seed)
    horizon = namespace.horizon or int(fixture.ground_truth["horizon_steps"])
    if horizon <= 0:
        parser.error("--horizon must be a positive integer")
    if namespace.command == "reveal" and horizon < fixture.ground_truth["horizon_steps"]:
        parser.error("--horizon must cover every held-out outcome before reveal")
    try:
        print(_render_experiment(fixture, horizon=horizon, reveal=namespace.command == "reveal"))
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
