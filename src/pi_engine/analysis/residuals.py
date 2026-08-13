"""Explicit, provisional residual analysis without opaque causal attribution."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pi_engine.schemas.common import Provenance
from pi_engine.schemas.residual import (
    Residual,
    ResidualCategory,
    ResidualClassification,
)


NonEmptyString = Annotated[str, Field(min_length=1)]
ResidualEvidenceKind = Literal[
    "known_process_noise",
    "parameter_mismatch",
    "model_mismatch",
    "phase_error",
    "topology_error",
]

_CATEGORY_BY_EVIDENCE_KIND: dict[ResidualEvidenceKind, ResidualCategory] = {
    "known_process_noise": ResidualCategory.PROCESS_NOISE,
    "parameter_mismatch": ResidualCategory.PARAMETER_UNCERTAINTY,
    "model_mismatch": ResidualCategory.MODEL_DISCREPANCY,
    "phase_error": ResidualCategory.PHASE_TIMING,
    "topology_error": ResidualCategory.TOPOLOGY_COUPLING,
}


class _ImmutableResidualAnalysisSchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )


class ResidualEvidence(_ImmutableResidualAnalysisSchema):
    """One sourced, inspectable signal for a provisional residual cause."""

    evidence_id: NonEmptyString
    kind: ResidualEvidenceKind
    basis: NonEmptyString
    provenance: Provenance

    @field_validator("provenance", mode="before")
    @classmethod
    def revalidate_provenance(cls, value: object) -> object:
        if isinstance(value, Provenance):
            return Provenance.model_validate(value.model_dump(warnings=False))
        return value


class ResidualAnalysis(_ImmutableResidualAnalysisSchema):
    """A source-bound classified residual retaining every supplied signal."""

    residual: Residual
    evidence: tuple[ResidualEvidence, ...]
    classification: ResidualClassification

    @field_validator("residual", mode="before")
    @classmethod
    def revalidate_residual(cls, value: object) -> object:
        if isinstance(value, Residual):
            return Residual.model_validate(value.model_dump(warnings=False))
        return value

    @field_validator("evidence", mode="before")
    @classmethod
    def revalidate_evidence(cls, value: object) -> object:
        if isinstance(value, (tuple, list)):
            return tuple(
                ResidualEvidence.model_validate(item.model_dump(warnings=False))
                if isinstance(item, ResidualEvidence)
                else item
                for item in value
            )
        return value

    @field_validator("classification", mode="before")
    @classmethod
    def revalidate_classification(cls, value: object) -> object:
        if isinstance(value, ResidualClassification):
            return ResidualClassification.model_validate(
                value.model_dump(warnings=False)
            )
        return value

    @model_validator(mode="after")
    def validate_derived_classification(self) -> "ResidualAnalysis":
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("residual evidence identities must be unique")

        expected = _classify_evidence(self.evidence)
        if self.classification != expected:
            raise ValueError("residual classification must recompute from evidence")
        if self.residual.classification != expected:
            raise ValueError("residual record classification must match analysis")
        return self


def _classify_evidence(
    evidence: Sequence[ResidualEvidence],
) -> ResidualClassification:
    categories = {_CATEGORY_BY_EVIDENCE_KIND[item.kind] for item in evidence}
    evidence_ids = ", ".join(item.evidence_id for item in evidence)
    if not categories:
        return ResidualClassification(
            category=ResidualCategory.STRUCTURED_UNKNOWN,
            basis="no inspectable residual evidence was supplied",
        )
    if len(categories) > 1:
        return ResidualClassification(
            category=ResidualCategory.STRUCTURED_UNKNOWN,
            basis=(
                "competing inspectable residual evidence prevents a unique "
                f"provisional classification: {evidence_ids}"
            ),
        )

    category = next(iter(categories))
    return ResidualClassification(
        category=category,
        basis=(
            f"{category.value} classification supported by inspectable evidence: "
            f"{evidence_ids}"
        ),
    )


def analyze_residual(
    residual: Residual,
    *,
    evidence: Sequence[ResidualEvidence] = (),
) -> ResidualAnalysis:
    """Classify a residual only from explicit, provenance-bearing evidence."""
    validated_evidence = tuple(
        ResidualEvidence.model_validate(item.model_dump(warnings=False))
        if isinstance(item, ResidualEvidence)
        else ResidualEvidence.model_validate(item)
        for item in evidence
    )
    classification = _classify_evidence(validated_evidence)
    classified_residual = Residual.model_validate(
        residual.model_dump(warnings=False)
        if isinstance(residual, Residual)
        else residual
    ).model_copy(update={"classification": classification})
    return ResidualAnalysis(
        residual=classified_residual,
        evidence=validated_evidence,
        classification=classification,
    )
