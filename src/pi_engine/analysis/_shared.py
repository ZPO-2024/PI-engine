"""Internal numeric and validation primitives shared by trajectory analyses."""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from datetime import datetime


def utc_instant_key(
    value: datetime, *, role: str, error_type: type[ValueError]
) -> int:
    """Return absolute microseconds without constructing a UTC datetime."""
    try:
        offset = value.utcoffset()
        if offset is None:
            raise ValueError(f"{role} must include a timezone")
        local_microseconds = (
            (
                ((value.toordinal() * 24 + value.hour) * 60 + value.minute)
                * 60
                + value.second
            )
            * 1_000_000
            + value.microsecond
        )
        offset_microseconds = (
            (offset.days * 86_400 + offset.seconds) * 1_000_000
            + offset.microseconds
        )
        return local_microseconds - offset_microseconds
    except (OverflowError, TypeError, ValueError) as exc:
        if isinstance(exc, error_type):
            raise
        raise error_type(f"{role} is not a valid absolute instant") from exc


def finite_difference_or_none(left: float, right: float) -> float | None:
    """Return ``right - left`` exactly when finite, scaling only on overflow."""
    direct = right - left
    if math.isfinite(direct):
        return float(direct)
    scale = max(abs(left), abs(right))
    if scale == 0.0:
        return 0.0
    factor = (right / scale) - (left / scale)
    if factor != 0.0 and scale > sys.float_info.max / abs(factor):
        return None
    result = scale * factor
    return float(result) if math.isfinite(result) else None


def finite_difference(
    left: float,
    right: float,
    *,
    role: str,
    error_type: type[ValueError],
) -> float:
    result = finite_difference_or_none(left, right)
    if result is None:
        raise error_type(f"{role} is not representable as finite")
    return result


def normalized_absolute_difference(
    left: float,
    right: float,
    normalization_scale: float,
    *,
    role: str,
    error_type: type[ValueError],
) -> float:
    """Normalize a finite difference without replacing exact subtraction."""
    direct = right - left
    if math.isfinite(direct):
        numerator = abs(direct)
        if numerator == 0.0:
            return 0.0
        if numerator > sys.float_info.max * normalization_scale:
            raise error_type(f"{role} is not representable as finite")
        result = numerator / normalization_scale
    else:
        computation_scale = max(abs(left), abs(right))
        factor = abs(
            (right / computation_scale) - (left / computation_scale)
        )
        ratio = computation_scale / normalization_scale
        if factor != 0.0 and ratio > sys.float_info.max / factor:
            raise error_type(f"{role} is not representable as finite")
        result = ratio * factor
    if not math.isfinite(result) or (left != right and result == 0.0):
        raise error_type(f"{role} is not representable as finite")
    return float(result)


def normalize_nonnegative_weights(
    raw_weights: Sequence[float],
    *,
    error_type: type[ValueError],
) -> tuple[float, ...]:
    """Normalize finite nonnegative weights after max scaling."""
    if not raw_weights:
        raise error_type("spread weights require at least one member")
    if any(not math.isfinite(item) or item < 0.0 for item in raw_weights):
        raise error_type("spread weights must be finite and nonnegative")
    scale = max(raw_weights)
    if scale <= 0.0:
        raise error_type("spread weights require a positive total")
    scaled = tuple(item / scale for item in raw_weights)
    if any(raw > 0.0 and item == 0.0 for raw, item in zip(raw_weights, scaled)):
        raise error_type("positive spread weight is not representable after scaling")
    total = math.fsum(scaled)
    normalized = tuple(item / total for item in scaled)
    if any(
        raw > 0.0 and item == 0.0
        for raw, item in zip(raw_weights, normalized)
    ):
        raise error_type(
            "positive spread weight is not representable after normalization"
        )
    if any(not math.isfinite(item) or item < 0.0 for item in normalized):
        raise error_type("spread weights cannot be normalized finitely")
    return normalized


def translated_population_mean_std(
    values: tuple[float, ...],
    weights: tuple[float, ...],
    *,
    error_type: type[ValueError],
) -> tuple[float, float, float, float]:
    """Return weighted mean/std using an anchored translated two-pass method."""
    if len(values) != len(weights) or not values:
        raise error_type("spread values and weights must align exactly")
    normalized_weights = normalize_nonnegative_weights(
        weights, error_type=error_type
    )
    supported = tuple(
        (value, weight)
        for value, weight in zip(values, normalized_weights, strict=True)
        if weight > 0.0
    )
    supported_values = tuple(value for value, _ in supported)
    supported_weights = tuple(weight for _, weight in supported)
    minimum = min(supported_values)
    maximum = max(supported_values)
    anchor = (minimum / 2.0) + (maximum / 2.0)
    translated = tuple(
        finite_difference(
            anchor,
            value,
            role="translated spread value",
            error_type=error_type,
        )
        for value in supported_values
    )
    translated_scale = max(abs(item) for item in translated)
    if translated_scale == 0.0:
        translated_mean = 0.0
    else:
        normalized_translated_mean = math.fsum(
            (item / translated_scale) * weight
            for item, weight in zip(
                translated, supported_weights, strict=True
            )
        )
        translated_mean = translated_scale * normalized_translated_mean
        if normalized_translated_mean != 0.0 and translated_mean == 0.0:
            raise error_type("spread mean is not representable as finite")
    mean = anchor + translated_mean
    if not math.isfinite(mean):
        raise error_type("spread mean is not representable as finite")

    deviation_scale = max(
        abs(translated_mean), *(abs(item) for item in translated)
    )
    if deviation_scale == 0.0:
        return float(mean), 0.0, float(minimum), float(maximum)
    normalized_deviations: list[float] = []
    for item in translated:
        direct = item - translated_mean
        if math.isfinite(direct):
            normalized_deviations.append(direct / deviation_scale)
        else:
            normalized_deviations.append(
                (item / deviation_scale)
                - (translated_mean / deviation_scale)
            )
    normalized_variance = math.fsum(
        weight * deviation * deviation
        for weight, deviation in zip(
            supported_weights, normalized_deviations, strict=True
        )
    )
    std = deviation_scale * math.sqrt(normalized_variance)
    if not math.isfinite(std):
        raise error_type(
            "population standard deviation is not representable as finite"
        )
    if normalized_variance != 0.0 and std == 0.0:
        raise error_type(
            "population standard deviation is not representable as finite"
        )
    return float(mean), float(std), float(minimum), float(maximum)
