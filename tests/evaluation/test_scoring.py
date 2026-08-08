"""Forecast scoring stays bound to revealed, immutable prediction artifacts."""

from datetime import UTC, datetime
import math
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from pi_engine.evaluation.calibration import (
    CalibrationSummary,
    CalibrationError,
    summarize_calibration,
)
from pi_engine.evaluation.holdout import prepare_holdout, reveal_holdout
from pi_engine.evaluation.scoring import (
    ForecastScoreReport,
    ScoringError,
    score_revealed_evaluation,
)
from pi_engine.simulation.runner import simulate_deterministic
from pi_engine.simulation.stochastic import simulate_stochastic
from pi_engine.synthetic.systems import (
    linear_convergence,
    stochastic_branching,
)


def _revealed_trajectory(
    forecasts: tuple[object, object, object],
    *,
    outcomes: tuple[object, object, object] | None = None,
    event_times: tuple[datetime, datetime, datetime] | None = None,
):
    fixture = linear_convergence()
    prediction = simulate_deterministic(
        fixture.case, fixture.model, horizon=3
    )
    points = tuple(
        point.model_copy(update={"values": {"x": forecast}})
        for point, forecast in zip(prediction.points, forecasts, strict=True)
    )
    prediction = prediction.model_copy(update={"points": points})
    outcome_values = outcomes or tuple(item.value for item in fixture.outcomes)
    outcome_times = event_times or tuple(
        item.event_time for item in fixture.outcomes
    )
    withheld = tuple(
        outcome.model_copy(
            update={"value": value, "event_time": event_time}
        )
        for outcome, value, event_time in zip(
            fixture.outcomes,
            outcome_values,
            outcome_times,
            strict=True,
        )
    )
    prepared = prepare_holdout(fixture.case, withheld)
    return prediction, reveal_holdout(prepared, prediction)


def _revealed_ensemble():
    fixture = stochastic_branching(seed=7)
    prediction = simulate_stochastic(
        fixture.case,
        fixture.model,
        horizon=4,
        samples=4,
        seed=7,
    )
    prepared = prepare_holdout(fixture.case, fixture.outcomes)
    return prediction, reveal_holdout(prepared, prediction)


def test_known_continuous_errors_are_reported_per_model_and_variable() -> None:
    """A sign flip or opaque aggregation would hide inspectable error behavior."""
    prediction, revealed = _revealed_trajectory((0.0, 2.0, 1.25))

    report = score_revealed_evaluation(revealed, (prediction,))

    assert report.case_id == revealed.case_id
    assert report.case_sha256 == revealed.case_sha256
    assert len(report.artifacts) == 1
    artifact = report.artifacts[0]
    assert (
        artifact.artifact_id,
        artifact.model_id,
        artifact.model_version,
    ) == (
        prediction.trajectory_id,
        prediction.model_id,
        prediction.model_version,
    )
    assert artifact.point_estimate_method == "trajectory_value"
    assert [item.variable for item in artifact.continuous_points] == [
        "x",
        "x",
        "x",
    ]
    assert [item.forecast for item in artifact.continuous_points] == [
        0.0,
        2.0,
        1.25,
    ]
    assert [item.observed for item in artifact.continuous_points] == [
        1.0,
        0.5,
        0.25,
    ]
    assert [item.error for item in artifact.continuous_points] == [
        -1.0,
        1.5,
        1.0,
    ]
    assert [item.absolute_error for item in artifact.continuous_points] == [
        1.0,
        1.5,
        1.0,
    ]
    assert [item.squared_error for item in artifact.continuous_points] == [
        1.0,
        2.25,
        1.0,
    ]
    assert len(artifact.continuous_metrics) == 1
    metrics = artifact.continuous_metrics[0]
    assert (metrics.variable, metrics.count) == ("x", 3)
    assert metrics.mean_error == pytest.approx(0.5)
    assert metrics.mean_absolute_error == pytest.approx(7.0 / 6.0)
    assert metrics.mean_squared_error == pytest.approx(17.0 / 12.0)
    assert metrics.root_mean_squared_error == pytest.approx(
        math.sqrt(17.0 / 12.0)
    )
    assert artifact.probability_points == ()
    assert artifact.probability_metrics == ()
    assert artifact.intervals == ()
    assert "master_score" not in report.model_dump()
    with pytest.raises(ValidationError, match="frozen"):
        metrics.count = 4  # type: ignore[misc]


def test_binary_brier_and_optional_log_scores_use_explicit_labels() -> None:
    """Treating every bounded continuous value as a probability is invalid."""
    prediction, revealed = _revealed_trajectory(
        (0.25, 0.5, 0.8), outcomes=(0.0, 1.0, 1.0)
    )

    report = score_revealed_evaluation(
        revealed,
        (prediction,),
        binary_probability_variables=("x",),
        include_log_score=True,
    )

    artifact = report.artifacts[0]
    assert artifact.continuous_points == ()
    assert artifact.continuous_metrics == ()
    assert [item.predicted_probability for item in artifact.probability_points] == [
        0.25,
        0.5,
        0.8,
    ]
    assert [item.label for item in artifact.probability_points] == [0, 1, 1]
    assert [item.brier_score for item in artifact.probability_points] == pytest.approx(
        [0.0625, 0.25, 0.04]
    )
    assert [item.log_score for item in artifact.probability_points] == pytest.approx(
        [-math.log(0.75), -math.log(0.5), -math.log(0.8)]
    )
    assert len(artifact.probability_metrics) == 1
    metrics = artifact.probability_metrics[0]
    assert (metrics.variable, metrics.count) == ("x", 3)
    assert metrics.mean_brier_score == pytest.approx(0.1175)
    assert metrics.mean_log_score == pytest.approx(
        (-math.log(0.75) - math.log(0.5) - math.log(0.8)) / 3.0
    )


def test_log_score_rejects_zero_probability_for_observed_event_without_epsilon() -> None:
    """Clipping an impossible event with an arbitrary epsilon invents evidence."""
    prediction, revealed = _revealed_trajectory(
        (0.0, 0.0, 0.0), outcomes=(1.0, 0.0, 0.0)
    )

    brier_only = score_revealed_evaluation(
        revealed,
        (prediction,),
        binary_probability_variables=("x",),
    )
    assert brier_only.artifacts[0].probability_points[0].brier_score == 1.0
    assert brier_only.artifacts[0].probability_points[0].log_score is None

    with pytest.raises(
        ScoringError, match="zero probability.*observed label"
    ):
        score_revealed_evaluation(
            revealed,
            (prediction,),
            binary_probability_variables=("x",),
            include_log_score=True,
        )


def test_empirical_intervals_and_calibration_use_raw_ensemble_samples() -> None:
    """A fitted distribution would replace retained sample evidence."""
    prediction, revealed = _revealed_ensemble()

    report = score_revealed_evaluation(
        revealed,
        (prediction,),
        interval_levels=(0.5, 0.75),
    )

    artifact = report.artifacts[0]
    assert artifact.point_estimate_method == "equal_weight_raw_samples"
    assert [item.forecast for item in artifact.continuous_points] == [
        0.5,
        0.5,
        0.5,
        0.5,
    ]
    half = [item for item in artifact.intervals if item.nominal_coverage == 0.5]
    assert [(item.lower, item.upper, item.covered) for item in half] == [
        (-1.0, 1.0, True),
        (-2.0, 2.0, True),
        (-1.0, 1.0, False),
        (0.0, 0.0, False),
    ]
    assert all(
        item.interval_method == "empirical_equal_tail_inverse_cdf"
        for item in artifact.intervals
    )

    calibration = summarize_calibration(report)
    assert calibration.probability == ()
    assert len(calibration.intervals) == 2
    fifty, seventy_five = calibration.intervals
    assert (
        fifty.artifact_id,
        fifty.model_id,
        fifty.variable,
        fifty.nominal_coverage,
        fifty.count,
        fifty.covered_count,
    ) == (
        prediction.ensemble_id,
        prediction.model_id,
        "x",
        0.5,
        4,
        2,
    )
    assert fifty.observed_coverage == 0.5
    assert fifty.calibration_error == 0.0
    assert fifty.mean_interval_width == 2.0
    assert seventy_five.observed_coverage == 0.5
    assert seventy_five.calibration_error == -0.25
    assert seventy_five.mean_interval_width == 3.0


def test_probability_calibration_requires_explicit_bins_and_known_boundaries() -> None:
    """Hidden default bins or ambiguous edge membership distort calibration."""
    prediction, revealed = _revealed_trajectory(
        (0.25, 0.5, 0.8), outcomes=(0.0, 1.0, 1.0)
    )
    report = score_revealed_evaluation(
        revealed,
        (prediction,),
        binary_probability_variables=("x",),
    )

    calibration = summarize_calibration(
        report, probability_bin_edges=(0.0, 0.5, 1.0)
    )

    assert len(calibration.probability) == 1
    summary = calibration.probability[0]
    assert (
        summary.artifact_id,
        summary.model_id,
        summary.variable,
        summary.count,
    ) == (
        prediction.trajectory_id,
        prediction.model_id,
        "x",
        3,
    )
    assert [item.count for item in summary.bins] == [1, 2]
    assert [item.lower_inclusive for item in summary.bins] == [0.0, 0.5]
    assert [item.upper for item in summary.bins] == [0.5, 1.0]
    assert [item.upper_inclusive for item in summary.bins] == [False, True]
    assert [item.mean_predicted_probability for item in summary.bins] == pytest.approx(
        [0.25, 0.65]
    )
    assert [item.observed_frequency for item in summary.bins] == [0.0, 1.0]
    assert [item.calibration_error for item in summary.bins] == pytest.approx(
        [-0.25, 0.35]
    )


def test_scoring_rejects_unrevealed_outcomes_and_prepared_holdout() -> None:
    """Raw or merely committed outcomes must not bypass the reveal boundary."""
    fixture = linear_convergence()
    prediction = simulate_deterministic(
        fixture.case, fixture.model, horizon=3
    )
    prepared = prepare_holdout(fixture.case, fixture.outcomes)

    for unrevealed in (fixture.outcomes, prepared):
        with pytest.raises(
            TypeError, match="revealed must be a RevealedEvaluation"
        ):
            score_revealed_evaluation(unrevealed, (prediction,))


def test_scoring_revalidates_and_binds_exact_artifact_digest_before_values() -> None:
    """Same-ID tampering must not be scored against an authorized reference."""
    prediction, revealed = _revealed_trajectory((0.0, 2.0, 1.25))
    first = prediction.points[0].model_copy(update={"values": {"x": 0.75}})
    tampered = prediction.model_copy(
        update={"points": (first, *prediction.points[1:])}
    )

    with pytest.raises(ScoringError, match="artifact SHA-256"):
        score_revealed_evaluation(revealed, (tampered,))

    invalid_point = prediction.points[0].model_copy(
        update={"values": {"x": True}}
    )
    invalid = prediction.model_copy(
        update={"points": (invalid_point, *prediction.points[1:])}
    )
    with pytest.raises(ScoringError, match="valid Trajectory"):
        score_revealed_evaluation(revealed, (invalid,))


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"model_id": "other-model"}, "model identity"),
        ({"case_id": "other-case"}, "case identity"),
        ({"trajectory_id": "other-trajectory"}, "artifact identity"),
    ],
    ids=("model", "case", "artifact-and-trajectory-id"),
)
def test_scoring_rejects_reference_identity_mismatch_before_values(
    update: dict[str, str], message: str
) -> None:
    """Approximate identity matching could authorize an unrelated artifact."""
    prediction, revealed = _revealed_trajectory((0.0, 2.0, 1.25))
    unrelated = prediction.model_copy(update=update)

    with pytest.raises(ScoringError, match=message):
        score_revealed_evaluation(revealed, (unrelated,))


def test_scoring_matches_points_by_exact_utc_instant() -> None:
    """Timezone spelling must not replace absolute point identity."""
    prediction, original = _revealed_trajectory((0.0, 2.0, 1.25))
    eastern = ZoneInfo("America/New_York")
    event_times = tuple(
        outcome.event_time.astimezone(eastern) for outcome in original.outcomes
    )
    prediction, revealed = _revealed_trajectory(
        (0.0, 2.0, 1.25), event_times=event_times
    )

    report = score_revealed_evaluation(revealed, (prediction,))

    assert [item.event_time.astimezone(UTC) for item in report.artifacts[0].continuous_points] == [
        outcome.event_time.astimezone(UTC) for outcome in revealed.outcomes
    ]


def test_scoring_rejects_vector_shapes_without_component_semantics() -> None:
    """Flattening vectors would invent canonical component labels."""
    prediction, revealed = _revealed_trajectory(
        ((1.0, 2.0), (0.5, 1.5), (0.25, 1.25)),
        outcomes=((1.0, 2.0), (0.5, 1.5), (0.25, 1.25)),
    )

    with pytest.raises(ScoringError, match="scalar numeric"):
        score_revealed_evaluation(revealed, (prediction,))


@pytest.mark.parametrize(
    ("forecasts", "outcomes", "message"),
    [
        ((0.25, 0.5, 0.8), (True, 1.0, 1.0), "label.*bool"),
        ((1.2, 0.5, 0.8), (0.0, 1.0, 1.0), r"probability.*\[0, 1\]"),
        ((0.25, 0.5, 0.8), (2.0, 1.0, 1.0), "label.*0 or 1"),
    ],
    ids=("bool-label", "probability-out-of-range", "invalid-label"),
)
def test_probability_scoring_rejects_invalid_probabilities_and_labels(
    forecasts: tuple[object, object, object],
    outcomes: tuple[object, object, object],
    message: str,
) -> None:
    """Coercion would turn invalid probabilistic evidence into a valid score."""
    prediction, revealed = _revealed_trajectory(forecasts, outcomes=outcomes)

    with pytest.raises(ScoringError, match=message):
        score_revealed_evaluation(
            revealed,
            (prediction,),
            binary_probability_variables=("x",),
        )


def test_scoring_rejects_nonfinite_derived_errors() -> None:
    """Finite inputs can still overflow subtraction or squared error."""
    prediction, revealed = _revealed_trajectory(
        (1e308, 0.5, 0.25), outcomes=(-1e308, 0.5, 0.25)
    )

    with pytest.raises(ScoringError, match="nonfinite continuous error"):
        score_revealed_evaluation(revealed, (prediction,))


@pytest.mark.parametrize(
    "levels",
    [
        (True,),
        (0.0,),
        (1.0,),
        (math.nan,),
        (0.75, 0.5),
        (0.5, 0.5),
    ],
    ids=("bool", "zero", "one", "nan", "unordered", "duplicate"),
)
def test_interval_levels_must_be_finite_unique_and_strictly_ordered(
    levels: tuple[object, ...],
) -> None:
    """Coercion or hidden sorting would alter a caller's requested intervals."""
    prediction, revealed = _revealed_ensemble()

    with pytest.raises(
        ScoringError,
        match="interval levels must be finite floats strictly between 0 and 1|strictly increasing",
    ):
        score_revealed_evaluation(
            revealed, (prediction,), interval_levels=levels
        )


@pytest.mark.parametrize(
    "edges",
    [
        (0.1, 1.0),
        (0.0, 0.9),
        (0.0, True, 1.0),
        (0.0, math.inf, 1.0),
        (0.0, 0.5, 0.5, 1.0),
    ],
    ids=("missing-zero", "missing-one", "bool", "infinite", "duplicate"),
)
def test_probability_calibration_rejects_ambiguous_bin_edges(
    edges: tuple[object, ...],
) -> None:
    """Calibration bins need an exact, complete, non-overlapping partition."""
    prediction, revealed = _revealed_trajectory(
        (0.25, 0.5, 0.8), outcomes=(0.0, 1.0, 1.0)
    )
    report = score_revealed_evaluation(
        revealed,
        (prediction,),
        binary_probability_variables=("x",),
    )

    with pytest.raises(CalibrationError, match="probability bin edges"):
        summarize_calibration(report, probability_bin_edges=edges)


def test_score_report_revalidation_rejects_tampered_derived_arithmetic() -> None:
    """A frozen record copied through JSON must not legitimize false metrics."""
    prediction, revealed = _revealed_trajectory((0.0, 2.0, 1.25))
    report = score_revealed_evaluation(revealed, (prediction,))

    wrong_point = report.model_dump()
    wrong_point["artifacts"][0]["continuous_points"][0][
        "squared_error"
    ] = 0.0
    with pytest.raises(ValidationError, match="continuous point arithmetic"):
        ForecastScoreReport.model_validate(wrong_point)

    wrong_metrics = report.model_dump()
    wrong_metrics["artifacts"][0]["continuous_metrics"][0][
        "mean_error"
    ] = 999.0
    with pytest.raises(ValidationError, match="continuous metrics"):
        ForecastScoreReport.model_validate(wrong_metrics)


def test_interval_records_reject_inversion_and_false_coverage() -> None:
    """Coverage cannot remain true after serialized interval-bound tampering."""
    prediction, revealed = _revealed_ensemble()
    report = score_revealed_evaluation(
        revealed, (prediction,), interval_levels=(0.5,)
    )

    inverted = report.model_dump()
    inverted["artifacts"][0]["intervals"][0]["lower"] = 2.0
    inverted["artifacts"][0]["intervals"][0]["upper"] = -2.0
    with pytest.raises(ValidationError, match="interval lower"):
        ForecastScoreReport.model_validate(inverted)

    false_coverage = report.model_dump()
    false_coverage["artifacts"][0]["intervals"][0]["covered"] = False
    with pytest.raises(ValidationError, match="interval coverage flag"):
        ForecastScoreReport.model_validate(false_coverage)


def test_calibration_revalidation_rejects_impossible_covered_count() -> None:
    """A summary cannot report more covered outcomes than evaluated outcomes."""
    prediction, revealed = _revealed_ensemble()
    report = score_revealed_evaluation(
        revealed, (prediction,), interval_levels=(0.5,)
    )
    calibration = summarize_calibration(report)
    payload = calibration.model_dump()
    payload["intervals"][0]["covered_count"] = 5

    with pytest.raises(ValidationError, match="covered_count"):
        CalibrationSummary.model_validate(payload)
