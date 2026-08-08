"""Immutable and inspectable explicit model definitions and performance records."""

from datetime import datetime
from types import MappingProxyType
from typing import Annotated, Literal, Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from pi_engine.schemas.common import (
    Confidence,
    FiniteFloat,
    JsonScalar,
    ModelStatus,
    Provenance,
)


NonEmptyString = Annotated[str, Field(min_length=1)]
NumericRange = tuple[FiniteFloat | None, FiniteFloat | None]


class _ImmutableSchema(BaseModel):
    """Common validation policy for versioned model records."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )


class ApplicabilitySpec(_ImmutableSchema):
    """Explicit conditions used by deterministic applicability routing."""

    required_variables: tuple[NonEmptyString, ...]
    optional_variables: tuple[NonEmptyString, ...]
    valid_ranges: Mapping[NonEmptyString, NumericRange]
    topology_requirements: tuple[NonEmptyString, ...]
    boundary_conditions: tuple[NonEmptyString, ...]
    exclusion_rules: tuple[NonEmptyString, ...]

    @field_validator("valid_ranges")
    @classmethod
    def freeze_valid_ranges(
        cls, value: Mapping[str, NumericRange]
    ) -> Mapping[str, NumericRange]:
        return MappingProxyType(dict(value))

    @field_serializer("valid_ranges")
    def serialize_valid_ranges(
        self, value: Mapping[str, NumericRange]
    ) -> dict[str, NumericRange]:
        return dict(value)

    @model_validator(mode="after")
    def validate_variable_roles_and_ranges(self) -> "ApplicabilitySpec":
        overlap = set(self.required_variables) & set(self.optional_variables)
        if overlap:
            overlap_text = ", ".join(sorted(overlap))
            raise ValueError(
                f"variables cannot be both required and optional: {overlap_text}"
            )

        for variable, (lower, upper) in self.valid_ranges.items():
            if lower is not None and upper is not None and lower > upper:
                raise ValueError(f"valid range is inverted for {variable!r}")
        return self


class DynamicsSpec(_ImmutableSchema):
    """Auditable dynamics metadata resolved only through a trusted registry."""

    executor_ref: str = Field(
        min_length=1,
        pattern=r"^[a-z][a-z0-9_-]*$",
        description="Opaque registry identifier, never a module or filesystem path",
    )
    executor_version: NonEmptyString
    code_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    equations: tuple[NonEmptyString, ...]
    rules: tuple[NonEmptyString, ...]
    transition_metadata: Mapping[NonEmptyString, JsonScalar]
    time_behavior: NonEmptyString
    classification: Literal["deterministic", "stochastic"]

    @field_validator("transition_metadata")
    @classmethod
    def freeze_transition_metadata(
        cls, value: Mapping[str, JsonScalar]
    ) -> Mapping[str, JsonScalar]:
        return MappingProxyType(dict(value))

    @field_serializer("transition_metadata")
    def serialize_transition_metadata(
        self, value: Mapping[str, JsonScalar]
    ) -> dict[str, JsonScalar]:
        return dict(value)

    @model_validator(mode="after")
    def require_inspectable_dynamics(self) -> "DynamicsSpec":
        if not self.equations and not self.rules:
            raise ValueError("dynamics must declare at least one equation or rule")
        return self


class UncertaintySpec(_ImmutableSchema):
    """Distinct uncertainty sources that must not be collapsed into one score."""

    parameter_uncertainty: tuple[NonEmptyString, ...]
    process_disturbance: tuple[NonEmptyString, ...]
    model_discrepancy: tuple[NonEmptyString, ...]
    structured_unknowns: tuple[NonEmptyString, ...]


class FalsificationSpec(_ImmutableSchema):
    """Evidence conditions under which a model conflicts with observations."""

    falsifiers: Annotated[tuple[NonEmptyString, ...], Field(min_length=1)]
    contradictory_evidence_conditions: tuple[NonEmptyString, ...]
    failure_conditions: tuple[NonEmptyString, ...]


class ExplicitModel(_ImmutableSchema):
    """An immutable, versioned model definition with no executable callables."""

    model_id: NonEmptyString
    version: NonEmptyString
    name: NonEmptyString
    domain: NonEmptyString
    model_family: NonEmptyString
    provenance: Provenance
    initial_confidence: Confidence
    applicability: ApplicabilitySpec
    dynamics: DynamicsSpec
    uncertainty: UncertaintySpec
    assumptions: tuple[NonEmptyString, ...]
    relationships_used: tuple[NonEmptyString, ...]
    phase_dependencies: tuple[NonEmptyString, ...]
    effective_proximity_dependencies: tuple[NonEmptyString, ...]
    information_dependencies: tuple[NonEmptyString, ...]
    predicted_outputs: Annotated[tuple[NonEmptyString, ...], Field(min_length=1)]
    prediction_horizon: NonEmptyString
    expected_regimes: tuple[NonEmptyString, ...]
    falsification: FalsificationSpec

    @field_validator("provenance", mode="before")
    @classmethod
    def revalidate_provenance_instance(cls, value: object) -> object:
        if isinstance(value, Provenance):
            return Provenance.model_validate(value.model_dump())
        return value

    @field_validator("initial_confidence", mode="before")
    @classmethod
    def revalidate_confidence_instance(cls, value: object) -> object:
        if isinstance(value, Confidence):
            return Confidence.model_validate(value.model_dump())
        return value


class ModelPerformance(_ImmutableSchema):
    """Immutable evaluation evidence for one model ID/version as of a time."""

    model_id: NonEmptyString
    model_version: NonEmptyString
    as_of: datetime
    status: ModelStatus
    cases_tested: tuple[NonEmptyString, ...]
    calibration_metrics: Mapping[NonEmptyString, FiniteFloat]
    prediction_errors: Mapping[NonEmptyString, FiniteFloat]
    known_failure_regimes: tuple[NonEmptyString, ...]
    evidence: tuple[NonEmptyString, ...]
    provenance: Provenance

    @field_validator("provenance", mode="before")
    @classmethod
    def revalidate_provenance_instance(cls, value: object) -> object:
        if isinstance(value, Provenance):
            return Provenance.model_validate(value.model_dump())
        return value

    @field_validator("as_of")
    @classmethod
    def as_of_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("as_of must include a timezone")
        return value

    @field_validator("calibration_metrics", "prediction_errors")
    @classmethod
    def freeze_metric_mapping(
        cls, value: Mapping[str, float]
    ) -> Mapping[str, float]:
        return MappingProxyType(dict(value))

    @field_serializer("calibration_metrics", "prediction_errors")
    def serialize_metric_mapping(
        self, value: Mapping[str, float]
    ) -> dict[str, float]:
        return dict(value)
