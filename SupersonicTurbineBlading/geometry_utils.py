"""Geometry operations shared by the rotor and stator designers."""

from __future__ import annotations

import numpy as np

from .common_models import SurfaceCoordinates


def densify_surface(surface: SurfaceCoordinates, minimum_number_of_stations: int) -> SurfaceCoordinates:
    """Insert BL stations while retaining every source geometry station.

    A normal resampling operation can join points on opposite sides of an MOC
    vertex and thereby shorten the piecewise-linear surface. This routine
    instead subdivides each existing segment. Extra intervals are distributed
    approximately in proportion to segment length, while every original
    station and the exact polyline arc length are preserved.

    :param SurfaceCoordinates surface: Original MOC surface whose stations must be retained.
    :param int minimum_number_of_stations: Minimum number of points required for the BL integration.
    :return: Piecewise-linear surface with additional stations inserted inside original segments.
    :rtype: SurfaceCoordinates
    :raises ValueError: If fewer than three stations are requested or the source contains duplicate points.
    """

    if not isinstance(minimum_number_of_stations, int) or minimum_number_of_stations < 3:
        raise ValueError("minimum_number_of_stations must be an integer >= 3")
    source_count = len(surface.x)
    if minimum_number_of_stations <= source_count:
        # Return independent arrays so later BL processing cannot accidentally modify the stored MOC geometry.
        return SurfaceCoordinates(
            x=np.asarray(surface.x, dtype=float).copy(),
            y=np.asarray(surface.y, dtype=float).copy(),
            mach=np.asarray(surface.mach, dtype=float).copy(),
            tangent_angle_rad=np.asarray(surface.tangent_angle_rad, dtype=float).copy(),
        )

    segment_length = np.hypot(np.diff(surface.x), np.diff(surface.y))
    if np.any(segment_length <= 0.0):
        raise ValueError("surface contains duplicate stations")

    # Allocate additional intervals in proportion to segment length. Flooring the allocation first guarantees that the
    # final remainder can be distributed deterministically to the segments with the largest fractional shares.
    extra_intervals = minimum_number_of_stations - source_count
    exact_allocation = extra_intervals * segment_length / segment_length.sum()
    added_intervals = np.floor(exact_allocation).astype(int)
    remainder = int(extra_intervals - added_intervals.sum())
    if remainder:
        fractional_order = np.argsort(-(exact_allocation - added_intervals))
        added_intervals[fractional_order[:remainder]] += 1
    intervals_per_segment = added_intervals + 1

    # Interpolate within each original straight segment. The segment endpoint is omitted here because it becomes the
    # first point of the next segment; the final endpoint is appended once after the loop.
    x_values: list[float] = []
    y_values: list[float] = []
    mach_values: list[float] = []
    for index, interval_count in enumerate(intervals_per_segment):
        fractions = np.arange(interval_count, dtype=float) / interval_count
        x_values.extend(surface.x[index] + fractions * (surface.x[index + 1] - surface.x[index]))
        y_values.extend(surface.y[index] + fractions * (surface.y[index + 1] - surface.y[index]))
        mach_values.extend(surface.mach[index] + fractions * (surface.mach[index + 1] - surface.mach[index]))
    x_values.append(float(surface.x[-1]))
    y_values.append(float(surface.y[-1]))
    mach_values.append(float(surface.mach[-1]))

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    mach = np.asarray(mach_values, dtype=float)
    # Recalculate tangent direction from the denser coordinates. Interpolating angles directly can fail at angle wraps.
    return SurfaceCoordinates(
        x=x, y=y, mach=mach, tangent_angle_rad=np.asarray(np.arctan2(np.gradient(y), np.gradient(x)), dtype=float)
    )


def resample_surface(surface: SurfaceCoordinates, number_of_stations: int) -> SurfaceCoordinates:
    """Interpolate a surface onto equally spaced arc-length stations.

    Endpoints are preserved exactly. Mach number is interpolated along the
    same arc-length coordinate, and tangent direction is recalculated from
    the resampled geometry instead of interpolating wrapped angles.

    :param SurfaceCoordinates surface: Surface to be interpolated.
    :param int number_of_stations: Exact number of equally spaced output stations.
    :return: Surface sampled uniformly in cumulative arc length.
    :rtype: SurfaceCoordinates
    :raises ValueError: If fewer than three stations are requested or the source contains duplicate points.
    """

    if not isinstance(number_of_stations, int) or number_of_stations < 3:
        raise ValueError("number_of_stations must be an integer >= 3")
    segment_length = np.hypot(np.diff(surface.x), np.diff(surface.y))
    if np.any(segment_length <= 0.0):
        raise ValueError("surface contains duplicate stations")
    # Cumulative arc length is monotonic and is therefore a convenient interpolation coordinate even for curved or
    # locally vertical surfaces where either x or y alone would not be monotonic.
    source_arc = np.concatenate(([0.0], np.cumsum(segment_length)))
    target_arc = np.linspace(0.0, float(source_arc[-1]), number_of_stations, dtype=float)
    x = np.interp(target_arc, source_arc, surface.x)
    y = np.interp(target_arc, source_arc, surface.y)
    mach = np.interp(target_arc, source_arc, surface.mach)
    tangent = np.arctan2(np.gradient(y), np.gradient(x))
    return SurfaceCoordinates(
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        mach=np.asarray(mach, dtype=float),
        tangent_angle_rad=np.asarray(tangent, dtype=float),
    )
