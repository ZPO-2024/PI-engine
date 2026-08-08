"""Pure, deterministic structural applicability evaluation."""

from collections.abc import Iterable, Mapping

from pydantic import BaseModel, ConfigDict

from pi_engine.schemas.case import Case
from pi_engine.schemas.common import NumericValue
from pi_engine.schemas.model import ExplicitModel, NumericRange


class ApplicabilityResult(BaseModel):
    """Immutable audit record for one registered model's routing result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    model_version: str
    applicable: bool
    structural_score: int
    structural_score_max: int
    rank: int | None
    reasons: tuple[str, ...]
    rejection_causes: tuple[str, ...]


def rank_applicable_models(
    case: Case, models: Iterable[ExplicitModel]
) -> tuple[ApplicabilityResult, ...]:
    """Evaluate every model and return a deterministic, tie-preserving ranking."""
    evaluated = [_evaluate_model(case, model) for model in models]
    evaluated.sort(
        key=lambda item: (
            not item.applicable,
            -item.structural_score,
            item.model_id,
            item.model_version,
        )
    )

    ranked: list[ApplicabilityResult] = []
    prior_score: int | None = None
    prior_rank = 0
    applicable_position = 0
    for result in evaluated:
        if not result.applicable:
            ranked.append(result)
            continue

        applicable_position += 1
        if result.structural_score != prior_score:
            prior_rank = applicable_position
            prior_score = result.structural_score
        ranked.append(result.model_copy(update={"rank": prior_rank}))
    return tuple(ranked)


def _evaluate_model(case: Case, model: ExplicitModel) -> ApplicabilityResult:
    canonical_variables = {
        definition.name for definition in case.canonical_variables
    }
    constraints = set(case.constraints)
    topology_tokens = set(case.graph.topology_metadata)
    spec = model.applicability
    reasons: list[str] = []
    rejection_causes: list[str] = []

    for variable in spec.required_variables:
        if variable in canonical_variables:
            reasons.append(f"required variable present: {variable}")
        else:
            rejection_causes.append(f"missing required variable: {variable}")

    current_values = _current_state_values(case)
    for variable, valid_range in spec.valid_ranges.items():
        values = current_values.get(variable, ())
        if not values:
            rejection_causes.append(
                f"current-state value unavailable for range check: {variable}"
            )
            continue

        range_failures = _range_failures(variable, values, valid_range)
        if range_failures:
            rejection_causes.extend(range_failures)
        else:
            reasons.append(f"range satisfied: {variable}")

    for token in spec.topology_requirements:
        if token in topology_tokens:
            reasons.append(f"topology requirement satisfied: {token}")
        else:
            rejection_causes.append(f"missing topology requirement: {token}")

    for token in spec.boundary_conditions:
        if token in constraints:
            reasons.append(f"boundary condition satisfied: {token}")
        else:
            rejection_causes.append(f"missing boundary condition: {token}")

    for token in spec.exclusion_rules:
        if token in constraints:
            rejection_causes.append(f"exclusion rule present: {token}")
        else:
            reasons.append(f"exclusion rule absent: {token}")

    structural_score = 0
    for variable in spec.optional_variables:
        if variable in canonical_variables:
            structural_score += 1
            reasons.append(f"optional variable present: {variable}")
        else:
            reasons.append(f"optional variable absent: {variable}")

    return ApplicabilityResult(
        model_id=model.model_id,
        model_version=model.version,
        applicable=not rejection_causes,
        structural_score=structural_score,
        structural_score_max=len(spec.optional_variables),
        rank=None,
        reasons=tuple(reasons),
        rejection_causes=tuple(rejection_causes),
    )


def _current_state_values(
    case: Case,
) -> dict[str, tuple[NumericValue, ...]]:
    values: dict[str, list[NumericValue]] = {}
    components: tuple[Mapping[str, NumericValue], ...] = (
        case.state.observed,
        case.state.latent,
        case.state.boundary,
    )
    for component in components:
        for variable, value in component.items():
            values.setdefault(variable, []).append(value)
    return {variable: tuple(items) for variable, items in values.items()}


def _range_failures(
    variable: str,
    current_values: tuple[NumericValue, ...],
    valid_range: NumericRange,
) -> tuple[str, ...]:
    lower, upper = valid_range
    failures: list[str] = []
    multiple_current_values = len(current_values) > 1
    for current_index, current_value in enumerate(current_values):
        elements = (
            current_value
            if isinstance(current_value, tuple)
            else (current_value,)
        )
        for element_index, element in enumerate(elements):
            label = variable
            if multiple_current_values:
                label += f"#{current_index}"
            if isinstance(current_value, tuple):
                label += f"[{element_index}]"
            if (lower is not None and element < lower) or (
                upper is not None and element > upper
            ):
                failures.append(
                    f"current-state value outside valid range for {label}: "
                    f"{element} not in [{lower}, {upper}]"
                )
    return tuple(failures)
