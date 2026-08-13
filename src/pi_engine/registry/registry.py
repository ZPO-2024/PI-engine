"""Exact, immutable registration of explicit model definitions."""

from pi_engine.registry.applicability import (
    ApplicabilityResult,
    rank_applicable_models,
)
from pi_engine.schemas.case import Case
from pi_engine.schemas.model import ExplicitModel


class ModelRegistry:
    """Store explicit models by their exact identity and version."""

    def __init__(self) -> None:
        self._models: dict[tuple[str, str], ExplicitModel] = {}

    def register(self, model: ExplicitModel) -> None:
        """Register an immutable snapshot without replacing an existing key."""
        if not isinstance(model, ExplicitModel):
            raise TypeError("model must be an ExplicitModel")

        stored_model = ExplicitModel.model_validate(model.model_dump())
        key = (stored_model.model_id, stored_model.version)
        if key in self._models:
            raise ValueError(
                f"model {stored_model.model_id!r} version "
                f"{stored_model.version!r} is already registered"
            )
        self._models[key] = stored_model

    def get(self, model_id: str, version: str) -> ExplicitModel:
        """Return only the model registered under the exact requested key."""
        return self._models[(model_id, version)]

    def rank_applicable_models(
        self, case: Case
    ) -> tuple[ApplicabilityResult, ...]:
        """Route a case without mutating the registry or its models."""
        return rank_applicable_models(case, self._models.values())
