"""Minimal, auditable command line harness for PI-engine experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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

# Run-artifact envelope: a signed, replayable record of a `run` invocation.
# `report`/`reveal` consume this file rather than re-simulating, so a report
# always reflects exactly what was run, never a fresh (and potentially
# divergent) recomputation. `_ARTIFACT_FORMAT`/`_ARTIFACT_VERSION` are
# forward-compatibility markers, not currently branched on.
_ARTIFACT_FORMAT = "pi-engine-run-artifact"
_ARTIFACT_VERSION = "1.0"


class RunArtifactError(ValueError):
    """Raised when a saved run artifact is missing, malformed, or tampered."""


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


def _simulate_one(
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


def _simulate_artifacts(
    fixture: SyntheticSystem, *, horizon: int, seed: int
) -> tuple[PredictionArtifact, ...]:
    """Run every applicable registered model once. The only place simulation happens."""
    registry, _models = _registry()
    applicability = registry.rank_applicable_models(fixture.case)
    return tuple(
        _simulate_one(
            fixture,
            registry.get(result.model_id, result.model_version),
            horizon,
            seed,
        )
        for result in applicability
        if result.applicable
    )


def _artifact_kind(artifact: PredictionArtifact) -> str:
    return "trajectory_ensemble" if isinstance(artifact, TrajectoryEnsemble) else "trajectory"


def _artifact_from_kind(kind: str, data: dict[str, Any]) -> PredictionArtifact:
    if kind == "trajectory_ensemble":
        return TrajectoryEnsemble.model_validate(data)
    if kind == "trajectory":
        return Trajectory.model_validate(data)
    raise RunArtifactError(f"run artifact integrity check failed: unknown artifact kind {kind!r}")


def _payload_checksum(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _save_run_artifact(
    path: str,
    *,
    case_name: str,
    horizon: int,
    seed: int,
    artifacts: Sequence[PredictionArtifact],
) -> None:
    payload = {
        "case_name": case_name,
        "horizon": horizon,
        "seed": seed,
        "artifacts": [
            {"kind": _artifact_kind(artifact), "data": artifact.model_dump(mode="json")}
            for artifact in artifacts
        ],
    }
    envelope = {
        "format": _ARTIFACT_FORMAT,
        "version": _ARTIFACT_VERSION,
        "payload": payload,
        "checksum": _payload_checksum(payload),
    }
    Path(path).write_text(json.dumps(envelope, indent=2), encoding="utf-8")


def _load_run_artifact(
    path: str,
) -> tuple[SyntheticSystem, int, tuple[PredictionArtifact, ...]]:
    try:
        raw = Path(path).read_text(encoding="utf-8")
        envelope = json.loads(raw)
    except OSError as exc:
        raise RunArtifactError(f"run artifact not found at {path!r}") from exc
    except json.JSONDecodeError as exc:
        raise RunArtifactError(f"run artifact integrity check failed: not valid JSON ({exc})") from exc

    try:
        payload = envelope["payload"]
        stored_checksum = envelope["checksum"]
    except (KeyError, TypeError) as exc:
        raise RunArtifactError("run artifact integrity check failed: malformed envelope") from exc

    if _payload_checksum(payload) != stored_checksum:
        raise RunArtifactError("run artifact integrity check failed: content does not match checksum")

    try:
        case_name = payload["case_name"]
        horizon = int(payload["horizon"])
        seed = int(payload["seed"])
        artifacts = tuple(
            _artifact_from_kind(entry["kind"], entry["data"]) for entry in payload["artifacts"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RunArtifactError("run artifact integrity check failed: malformed payload") from exc

    fixture = _fixture(case_name, seed)
    return fixture, horizon, artifacts


def _difference(predicted: NumericValue, observed: NumericValue) -> NumericValue:
    if isinstance(predicted, tuple):
        if not isinstance(observed, tuple):
            raise ValueError("prediction and outcome shapes must match")
        return tuple(left - right for left, right in zip(predicted, observed, strict=True))
    if isinstance(observed, tuple):
        raise ValueError("prediction and outcome shapes must match")
    return predicted - observed


def _residual_trajectories(
    artifacts: Sequence[PredictionArtifact],
) -> tuple[Trajectory, ...]:
    """Flatten ensembles to their members so every stochastic sample gets its
    own residual analysis rather than being dropped at the ensemble boundary."""
    flattened: list[Trajectory] = []
    for artifact in artifacts:
        if isinstance(artifact, TrajectoryEnsemble):
            flattened.extend(artifact.trajectories)
        else:
            flattened.append(artifact)
    return tuple(flattened)


def _residuals(
    fixture: SyntheticSystem, artifacts: Sequence[PredictionArtifact]
) -> tuple[ResidualAnalysis, ...]:
    analyses = []
    for artifact in _residual_trajectories(artifacts):
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


def _analysis_trajectory(artifact: PredictionArtifact) -> Trajectory:
    if isinstance(artifact, TrajectoryEnsemble):
        return artifact.trajectories[0]
    return artifact


def _render_from_artifacts(
    fixture: SyntheticSystem,
    artifacts: tuple[PredictionArtifact, ...],
    *,
    reveal: bool,
) -> str:
    """Build and render a prediction report from already-produced artifacts.

    Never simulates. `run` is the only place that calls `_simulate_artifacts`;
    `report`/`reveal` always pass artifacts loaded from a saved run so the
    rendered report reflects exactly what was run, not a fresh recomputation.
    """
    registry, models = _registry()
    applicability = registry.rank_applicable_models(fixture.case)
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
            f"Synthetic fixture regime: {fixture.regime}",
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pi-engine",
        description="Run explicit, cutoff-safe PI-engine synthetic experiments.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list synthetic systems and negative controls")

    run_parser = commands.add_parser(
        "run", help="simulate a case to cutoff and save a signed run artifact"
    )
    run_parser.add_argument("case", choices=sorted(_CATALOG))
    run_parser.add_argument("--horizon", type=int, default=None)
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument(
        "--artifact", required=True, help="path to write the signed run artifact to"
    )

    report_parser = commands.add_parser(
        "report", help="render a cutoff-safe prediction report from a saved run artifact"
    )
    report_parser.add_argument(
        "--artifact", required=True, help="path to a run artifact produced by `run`"
    )

    reveal_parser = commands.add_parser(
        "reveal", help="reveal and score held-out outcomes for a saved run artifact"
    )
    reveal_parser.add_argument(
        "--artifact", required=True, help="path to a run artifact produced by `run`"
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parser = _parser()
    namespace = parser.parse_args(arguments)

    if namespace.command == "list":
        print(_list_cases())
        return 0

    if namespace.command == "run":
        fixture = _fixture(namespace.case, namespace.seed)
        horizon = namespace.horizon or int(fixture.ground_truth["horizon_steps"])
        if horizon <= 0:
            parser.error("--horizon must be a positive integer")
        try:
            artifacts = _simulate_artifacts(fixture, horizon=horizon, seed=namespace.seed)
            _save_run_artifact(
                namespace.artifact,
                case_name=namespace.case,
                horizon=horizon,
                seed=namespace.seed,
                artifacts=artifacts,
            )
        except (TypeError, ValueError) as exc:
            parser.error(str(exc))
        print(f"Run artifact saved: {namespace.artifact}")
        return 0

    # report / reveal both replay a saved run artifact; neither simulates.
    try:
        fixture, horizon, artifacts = _load_run_artifact(namespace.artifact)
    except RunArtifactError as exc:
        parser.error(str(exc))
    if namespace.command == "reveal" and horizon < fixture.ground_truth["horizon_steps"]:
        parser.error(
            "the saved run artifact's horizon does not cover every held-out outcome; "
            "re-run with a longer --horizon before reveal"
        )
    try:
        print(_render_from_artifacts(fixture, artifacts, reveal=namespace.command == "reveal"))
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
