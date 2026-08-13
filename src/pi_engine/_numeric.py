"""Dependency-neutral finite numeric and absolute-time primitives."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
import math
import sys


def utc_instant_key(
    value: datetime,
    *,
    role: str = "datetime",
    error_type: type[ValueError] = ValueError,
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
        raise error_type("population weights require at least one member")
    if any(not math.isfinite(item) or item < 0.0 for item in raw_weights):
        raise error_type("population weights must be finite and nonnegative")
    scale = max(raw_weights)
    if scale <= 0.0:
        raise error_type("population weights require a positive total")
    scaled = tuple(item / scale for item in raw_weights)
    if any(raw > 0.0 and item == 0.0 for raw, item in zip(raw_weights, scaled)):
        raise error_type(
            "positive population weight is not representable after scaling"
        )
    total = math.fsum(scaled)
    normalized = tuple(item / total for item in scaled)
    if any(
        raw > 0.0 and item == 0.0
        for raw, item in zip(raw_weights, normalized)
    ):
        raise error_type(
            "positive population weight is not representable after normalization"
        )
    if any(not math.isfinite(item) or item < 0.0 for item in normalized):
        raise error_type("population weights cannot be normalized finitely")
    return normalized


def stable_population_mean_std(
    values: tuple[float, ...],
    weights: tuple[float, ...],
    *,
    error_type: type[ValueError],
) -> tuple[float, float, float, float]:
    """Return weighted mean/std in coordinates anchored at supported data."""
    if len(values) != len(weights) or not values:
        raise error_type("population values and weights must align exactly")
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
    maximum_weight = max(supported_weights)
    anchor = min(
        value
        for value, weight in supported
        if weight == maximum_weight
    )
    coordinate_scale = max(abs(anchor), *(abs(item) for item in supported_values))
    if coordinate_scale == 0.0:
        translated_coordinates = tuple(0.0 for _ in supported_values)
    else:
        translated_coordinates_list: list[float] = []
        for value in supported_values:
            direct = value - anchor
            if math.isfinite(direct):
                coordinate = direct / coordinate_scale
            else:
                coordinate = (
                    (value / coordinate_scale)
                    - (anchor / coordinate_scale)
                )
            if value != anchor and coordinate == 0.0:
                raise error_type(
                    "translated population value is not representable"
                )
            translated_coordinates_list.append(float(coordinate))
        translated_coordinates = tuple(translated_coordinates_list)

    normalized_translated_mean = math.fsum(
        item * weight
        for item, weight in zip(
            translated_coordinates, supported_weights, strict=True
        )
    )
    translated_mean = coordinate_scale * normalized_translated_mean
    if normalized_translated_mean != 0.0 and translated_mean == 0.0:
        raise error_type("population mean is not representable as finite")
    mean = anchor + translated_mean
    if not math.isfinite(mean):
        normalized_mean = (
            (anchor / coordinate_scale) + normalized_translated_mean
            if coordinate_scale != 0.0
            else 0.0
        )
        mean = coordinate_scale * normalized_mean
    if not math.isfinite(mean):
        raise error_type("population mean is not representable as finite")

    normalized_deviations = tuple(
        item - normalized_translated_mean
        for item in translated_coordinates
    )
    deviation_scale = max(abs(item) for item in normalized_deviations)
    if deviation_scale == 0.0:
        return float(mean), 0.0, float(minimum), float(maximum)
    scaled_variance = math.fsum(
        weight * (deviation / deviation_scale) ** 2
        for weight, deviation in zip(
            supported_weights, normalized_deviations, strict=True
        )
    )
    standard_deviation_factor = deviation_scale * math.sqrt(scaled_variance)
    if (
        standard_deviation_factor != 0.0
        and coordinate_scale
        > sys.float_info.max / standard_deviation_factor
    ):
        raise error_type(
            "population standard deviation is not representable as finite"
        )
    std = coordinate_scale * standard_deviation_factor
    if not math.isfinite(std):
        raise error_type(
            "population standard deviation is not representable as finite"
        )
    if scaled_variance != 0.0 and std == 0.0:
        raise error_type(
            "population standard deviation is not representable as finite"
        )
    return float(mean), float(std), float(minimum), float(maximum)
