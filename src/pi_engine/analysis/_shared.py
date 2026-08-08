"""Internal canonical-content primitives shared by trajectory analyses."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel


def canonical_model_content(value: BaseModel) -> object:
    """Return JSON-mode content so timezone offsets remain evidence identity."""
    return value.model_dump(mode="json", warnings=False)


def canonical_record_content(values: Sequence[BaseModel]) -> tuple[object, ...]:
    """Return ordered canonical content for derived-record comparison."""
    return tuple(canonical_model_content(value) for value in values)
