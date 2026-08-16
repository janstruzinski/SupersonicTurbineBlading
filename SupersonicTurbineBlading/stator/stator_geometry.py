"""Sharp-throat MOC and straight-wall conical stator geometry.

The characteristic net is a direct, zero-based transcription of the FORTRAN
``NOZZL`` routine printed in NASA TM X-1502 and repeated in NASA TM X-2343.
Only the final wall coordinates are retained; the thousands of interior
characteristic intersections were work arrays in the original program and are
not useful properties of a finished nozzle object. The alternative conical
contour is an axisymmetric de Laval nozzle sized by the ideal-gas area--Mach
relation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np

from ..common_results import SurfaceCoordinates
from ..gas_dynamics import (
    isentropic_area_ratio,
    mach_from_prandtl_meyer,
    prandtl_meyer_angle,
    supersonic_mach_from_area_ratio,
)
from .stator_results import NozzleShape

ContourMethod = Literal["moc", "conical"]


class StatorGeometryError(ValueError):
    """Raised when a requested stator contour cannot be constructed."""


@dataclass(frozen=True)
class IdealNozzleConstruction:
    """Store an ideal nozzle and diagnostics from its contour discretization.

    :ivar NozzleShape shape: Completed ideal nozzle passage.
    :ivar int contour_point_count: Number of retained points on the divergent characteristic contour.
    :ivar actual_flow_turning_increment: Actual MOC turning increment, deg, or ``None`` for a conical nozzle.
    :ivar int pressure_point_count: Number of stations belonging to the pressure-side divergent contour.
    """

    shape: NozzleShape
    contour_point_count: int
    actual_flow_turning_increment: float | None
    pressure_point_count: int


@dataclass(frozen=True)
class _MocContour:
    """Store the upper MOC half-contour before the straight section is added.

    :ivar x: Nozzle-axis coordinates divided by throat half-width, -.
    :ivar y: Transverse coordinates divided by throat half-width, -.
    :ivar absolute_flow_mach: Ordinary Mach number at each retained wall station, -.
    :ivar float actual_flow_turning_increment: Derived characteristic turning increment, deg.
    """

    x: np.ndarray
    y: np.ndarray
    absolute_flow_mach: np.ndarray
    actual_flow_turning_increment: float


def _surface(x: np.ndarray, y: np.ndarray, absolute_flow_mach: np.ndarray) -> SurfaceCoordinates:
    """Make one surface and calculate its signed local tangent direction.

    :param numpy.ndarray x: Nozzle-axis coordinates in the active length scale.
    :param numpy.ndarray y: Transverse coordinates in the active length scale.
    :param numpy.ndarray absolute_flow_mach: Ordinary Mach number at each coordinate station, -.
    :return: Validated surface with recalculated tangent angles.
    :rtype: SurfaceCoordinates
    :raises StatorGeometryError: If the arrays differ in length, contain fewer than two points, or include duplicates.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    absolute_flow_mach = np.asarray(absolute_flow_mach, dtype=float)
    if not (len(x) == len(y) == len(absolute_flow_mach)) or len(x) < 2:
        raise StatorGeometryError("a nozzle surface needs at least two points")
    if np.any(np.hypot(np.diff(x), np.diff(y)) <= 1.0e-12):
        raise StatorGeometryError("nozzle contour contains duplicate points")

    # arctan2(dy, dx) stays well behaved at the sharp throat and preserves the
    # sign difference between the pressure and suction passage walls.
    tangent = np.arctan2(np.gradient(y), np.gradient(x))
    return SurfaceCoordinates(x=x, y=y, absolute_flow_mach=absolute_flow_mach,
                              metal_angle=np.asarray(np.degrees(tangent), dtype=float))


@lru_cache(maxsize=128)
def _characteristic_contour(ideal_outlet_absolute_flow_mach: float, gamma: float, number_of_nodes: int) -> _MocContour:
    """Return the diverging half-contour normalized by throat half-width.

    NASA TM X-1502 divides half the exit Prandtl--Meyer angle into equal
    characteristic increments. ``number_of_nodes`` fixes the resulting wall
    stations, including the throat and exit, while the increment is retained
    as a diagnostic.

    :param float ideal_outlet_absolute_flow_mach: Supersonic design exit Mach number, -.
    :param float gamma: Frozen specific-heat ratio, -.
    :param int number_of_nodes: Nodes on the divergent MOC wall, including the throat and exit.
    :return: Diverging upper half-contour normalized by throat half-width.
    :rtype: _MocContour
    :raises StatorGeometryError: If fewer than two characteristic regions are requested.
    """

    # The cache prevents the same MOC net from being rebuilt during nozzle-angle
    # iterations because rotation changes metal direction, not the unrotated
    # characteristic contour.
    exit_nu = float(prandtl_meyer_angle(ideal_outlet_absolute_flow_mach, gamma))
    k_max_original = number_of_nodes
    if k_max_original < 2:
        raise StatorGeometryError("number_of_nodes must provide at least two characteristic regions")
    increment = exit_nu / (2.0 * (k_max_original - 1))
    absolute_flow_angle = np.arange(k_max_original, dtype=float) * increment

    # FORTRAN index I covers Prandtl--Meyer angles from zero to the exact exit
    # value.  Mach angles are cached in this local array because every
    # characteristic intersection uses the average of two adjacent regions.
    prandtl_meyer_angles = np.arange(2 * k_max_original - 1, dtype=float) * increment
    absolute_flow_mach = np.asarray([mach_from_prandtl_meyer(angle, gamma) for angle in prandtl_meyer_angles],
                                    dtype=float)
    mach_angle = np.arcsin(1.0 / absolute_flow_mach)

    # Only two characteristic columns are needed at once.  This is the same
    # storage reuse performed by statements 204--205 in the FORTRAN.
    x = np.full((k_max_original, 2), np.nan, dtype=float)
    y = np.full((k_max_original, 2), np.nan, dtype=float)
    contour_x: list[float] = []
    contour_y: list[float] = []
    contour_absolute_flow_mach: list[float] = []

    # Region adjacent to the throat (FORTRAN N=1).  The throat itself is
    # (0, 1); it is prepended later because these points begin downstream.
    characteristic_count = k_max_original
    characteristic_column = 0
    for k_zero in range(characteristic_count):
        i_zero = k_zero
        if k_zero == 0:
            slope_1 = -math.tan(0.5 * (mach_angle[i_zero] + mach_angle[i_zero + 1])
                - 0.5 * (absolute_flow_angle[k_zero] + absolute_flow_angle[k_zero + 1]))
            x[k_zero, 0] = -1.0 / slope_1
            y[k_zero, 0] = 0.0
        else:
            slope_2 = math.tan(0.5 * (mach_angle[i_zero] + mach_angle[i_zero + 1])
                + 0.5 * (absolute_flow_angle[k_zero] + absolute_flow_angle[k_zero - 1]))
            slope_1 = (math.tan(absolute_flow_angle[k_zero])
                if k_zero == characteristic_count - 1
                else -math.tan(0.5 * (mach_angle[i_zero] + mach_angle[i_zero + 1])
                    - 0.5 * (absolute_flow_angle[k_zero] + absolute_flow_angle[k_zero + 1])))
            x[k_zero, 0] = (1.0 - (y[k_zero - 1, 0] - slope_2 * x[k_zero - 1, 0])) / (slope_2 - slope_1)
            y[k_zero, 0] = y[k_zero - 1, 0] + slope_2 * (x[k_zero, 0] - x[k_zero - 1, 0])

    contour_x.append(float(x[characteristic_count - 1, 0]))
    contour_y.append(float(y[characteristic_count - 1, 0]))
    contour_absolute_flow_mach.append(float(absolute_flow_mach[characteristic_count - 1]))

    # Downstream region.  Each new column contains one fewer characteristic
    # point until only the centreline point and one wall point remain.
    while characteristic_count > 2:
        new_count = characteristic_count - 1
        characteristic_column += 1
        for k_zero in range(new_count):
            i_zero = k_zero + 2 * characteristic_column
            if k_zero == 0:
                slope_1 = -math.tan(0.5 * (mach_angle[i_zero] + mach_angle[i_zero + 1])
                    - 0.5 * (absolute_flow_angle[k_zero] + absolute_flow_angle[k_zero + 1]))
                x[k_zero, 1] = -(y[k_zero + 1, 0] - slope_1 * x[k_zero + 1, 0]) / slope_1
                y[k_zero, 1] = 0.0
                continue

            slope_2 = math.tan(0.5 * (mach_angle[i_zero] + mach_angle[i_zero + 1])
                + 0.5 * (absolute_flow_angle[k_zero] + absolute_flow_angle[k_zero - 1]))
            slope_1 = (math.tan(absolute_flow_angle[k_zero])
                if k_zero == new_count - 1
                else -math.tan(0.5 * (mach_angle[i_zero] + mach_angle[i_zero + 1])
                    - 0.5 * (absolute_flow_angle[k_zero] + absolute_flow_angle[k_zero + 1])))
            old_intercept = y[k_zero + 1, 0] - slope_1 * x[k_zero + 1, 0]
            new_intercept = y[k_zero - 1, 1] - slope_2 * x[k_zero - 1, 1]
            x[k_zero, 1] = (old_intercept - new_intercept) / (slope_2 - slope_1)
            y[k_zero, 1] = y[k_zero - 1, 1] + slope_2 * (x[k_zero, 1] - x[k_zero - 1, 1])

        wall_index = new_count - 1
        contour_x.append(float(x[wall_index, 1]))
        contour_y.append(float(y[wall_index, 1]))
        contour_absolute_flow_mach.append(float(absolute_flow_mach[wall_index + 2 * characteristic_column]))
        x[:new_count, 0] = x[:new_count, 1]
        y[:new_count, 0] = y[:new_count, 1]
        characteristic_count = new_count

    return _MocContour(x=np.asarray(contour_x, dtype=float), y=np.asarray(contour_y, dtype=float),
                       absolute_flow_mach=np.asarray(contour_absolute_flow_mach, dtype=float),
                       actual_flow_turning_increment=math.degrees(increment))


def design_ideal_stator_nozzle(*, ideal_outlet_absolute_flow_mach: float, outlet_metal_angle: float,
                               number_of_nodes: int, gamma: float) -> IdealNozzleConstruction:
    """Design the uncorrected supersonic passage and straight suction section.

    ``outlet_metal_angle`` is measured from the machine axial direction.  The
    old input ``ALP1`` was measured from the tangential direction, so the
    NASA TM X-1502's straight-section equation becomes

    ``L / y_throat = 2 * y_exit * tan(outlet_metal_angle)``.

    :param float ideal_outlet_absolute_flow_mach: Supersonic ideal nozzle exit Mach number, -.
    :param float outlet_metal_angle: Nozzle-axis angle measured from the machine axis, deg.
    :param int number_of_nodes: Nodes on both the MOC contour and the following straight-wall segment.
    :param float gamma: Frozen specific-heat ratio, -.
    :return: Ideal planar MOC nozzle and discretization diagnostics.
    :rtype: IdealNozzleConstruction
    :raises StatorGeometryError: If an input is outside its physical or numerical range.
    """

    if not math.isfinite(ideal_outlet_absolute_flow_mach) or ideal_outlet_absolute_flow_mach <= 1.0:
        raise StatorGeometryError("ideal_outlet_absolute_flow_mach must be supersonic (> 1)")
    if not math.isfinite(gamma) or gamma <= 1.0:
        raise StatorGeometryError("gamma must be greater than one")
    if not 0.0 < outlet_metal_angle < 90.0:
        raise StatorGeometryError("outlet_metal_angle must be between 0 and 90")
    if not isinstance(number_of_nodes, int) or isinstance(number_of_nodes, bool) or number_of_nodes < 3:
        raise StatorGeometryError("number_of_nodes must be an integer >= 3")

    # First create the universal sharp-throat contour, then attach the
    # constant-area straight required by AFMIX. Rotation occurs later in
    # ``SupersonicStatorNozzle``, keeping MOC independent of installation angle.
    contour = _characteristic_contour(float(ideal_outlet_absolute_flow_mach), float(gamma), number_of_nodes)
    outlet_metal_angle_rad = math.radians(outlet_metal_angle)
    exit_x = float(contour.x[-1])
    exit_y = float(contour.y[-1])
    straight_length = 2.0 * exit_y * math.tan(outlet_metal_angle_rad)

    stored_contour = _surface(np.concatenate(([0.0], contour.x.copy())),
                              np.concatenate(([1.0], contour.y.copy())),
                              np.concatenate(([1.0], contour.absolute_flow_mach.copy())))
    pressure_point_count = number_of_nodes
    straight_x = np.linspace(exit_x, exit_x + straight_length, number_of_nodes, dtype=float)[1:]
    straight_intervals = number_of_nodes - 1
    suction_x = np.concatenate((stored_contour.x, straight_x))
    suction_y = np.concatenate((stored_contour.y, np.full(straight_intervals, exit_y)))
    # NOZZL copies the pressure ratio at the final characteristic point onto
    # the straight section.  It does not replace it with the exact input exit
    # Mach; AFMIX alone uses that exact free-stream value.  This distinction
    # is small but preserves the FORTRAN discretization faithfully.
    suction_absolute_flow_mach = np.concatenate((
        stored_contour.absolute_flow_mach, np.full(straight_intervals, contour.absolute_flow_mach[-1])))

    pressure_x = stored_contour.x.copy()
    pressure_y = -stored_contour.y.copy()
    pressure_absolute_flow_mach = stored_contour.absolute_flow_mach.copy()
    spacing = 2.0 * exit_y / math.cos(outlet_metal_angle_rad)

    shape = NozzleShape(pressure_surface=_surface(pressure_x, pressure_y, pressure_absolute_flow_mach),
                        suction_surface=_surface(suction_x, suction_y, suction_absolute_flow_mach),
                        chord=float(suction_x[-1]), throat_width=2.0, exit_opening=2.0 * exit_y,
                        spacing=spacing, coordinate_scale="throat half-width")
    return IdealNozzleConstruction(shape=shape, contour_point_count=len(stored_contour.x),
                                   actual_flow_turning_increment=contour.actual_flow_turning_increment,
                                   pressure_point_count=pressure_point_count)


def design_conical_stator_nozzle(*, ideal_outlet_absolute_flow_mach: float, outlet_metal_angle: float,
                                 half_cone_metal_angle: float, number_of_nodes: int,
                                 gamma: float) -> IdealNozzleConstruction:
    """Design an axisymmetric straight-wall (conical) de Laval nozzle.

    The nozzle is axisymmetric. Coordinates are normalized by throat diameter:
    the throat walls start at ``y = +/-0.5`` and the exit walls end at
    ``y = +/-(0.5*sqrt(A_e/A*))``. The divergent walls make the requested
    half-cone angle with the nozzle axis.

    As in the MOC construction, the suction wall continues through a horizontal
    segment after the divergent part. Both segments contain
    ``number_of_nodes`` including their shared junction. After the complete
    shape is rotated by ``outlet_metal_angle``, that straight is parallel to
    the stator metal angle.

    :param float ideal_outlet_absolute_flow_mach: Supersonic ideal nozzle exit Mach number, -.
    :param float outlet_metal_angle: Nozzle-axis angle measured from the machine axis, deg.
    :param float half_cone_metal_angle: Angle between the divergent wall and nozzle axis, deg.
    :param int number_of_nodes: Nodes on both the divergent and straight-wall segments.
    :param float gamma: Frozen specific-heat ratio, -.
    :return: Ideal axisymmetric conical nozzle and geometry diagnostics.
    :rtype: IdealNozzleConstruction
    :raises StatorGeometryError: If Mach, gamma, or either angle is outside its physical range.
    """

    if not math.isfinite(ideal_outlet_absolute_flow_mach) or ideal_outlet_absolute_flow_mach <= 1.0:
        raise StatorGeometryError("ideal_outlet_absolute_flow_mach must be supersonic (> 1)")
    if not math.isfinite(gamma) or gamma <= 1.0:
        raise StatorGeometryError("gamma must be greater than one")
    if not (math.isfinite(half_cone_metal_angle) and 0.0 < half_cone_metal_angle < 90.0):
        raise StatorGeometryError("half_cone_metal_angle must be between 0 and 90")
    if not (math.isfinite(outlet_metal_angle) and 0.0 < outlet_metal_angle < 90.0):
        raise StatorGeometryError("outlet_metal_angle must be between 0 and 90")
    if not isinstance(number_of_nodes, int) or isinstance(number_of_nodes, bool) or number_of_nodes < 3:
        raise StatorGeometryError("number_of_nodes must be an integer >= 3")

    exit_area_ratio = float(isentropic_area_ratio(ideal_outlet_absolute_flow_mach, gamma))
    half_cone_metal_angle_rad = math.radians(half_cone_metal_angle)
    outlet_metal_angle_rad = math.radians(outlet_metal_angle)

    # Circular area scales with radius squared. With all coordinates divided
    # by throat diameter, r*/D* = 0.5 and r_e/D* =
    # 0.5*sqrt(A_e/A*).
    exit_radius_over_throat_diameter = 0.5 * math.sqrt(exit_area_ratio)
    divergent_length = (exit_radius_over_throat_diameter - 0.5) / math.tan(half_cone_metal_angle_rad)
    straight_length = 2.0 * exit_radius_over_throat_diameter * math.tan(outlet_metal_angle_rad)
    # A conical wall makes dM/dx singular at the exact sonic throat under the
    # quasi-one-dimensional area--Mach relation. A fixed fourth-root spacing
    # retains both endpoints while avoiding an artificial cluster of stations
    # in that singular neighborhood. The distribution is unchanged between
    # iterative trials, so only the physical coordinates move.
    divergent_fraction = np.linspace(0.0, 1.0, number_of_nodes, dtype=float) ** 0.25
    pressure_x = divergent_length * divergent_fraction
    pressure_radius = 0.5 + (exit_radius_over_throat_diameter - 0.5) * divergent_fraction
    pressure_y = -pressure_radius
    local_area_ratio = np.maximum((2.0 * pressure_radius) ** 2, 1.0)
    pressure_absolute_flow_mach = np.asarray([
        supersonic_mach_from_area_ratio(float(area_ratio), gamma) for area_ratio in local_area_ratio], dtype=float)

    straight_x = np.linspace(divergent_length, divergent_length + straight_length, number_of_nodes, dtype=float)[1:]
    suction_x = np.concatenate((pressure_x, straight_x))
    suction_y = np.concatenate((pressure_radius,
                                np.full(number_of_nodes - 1, exit_radius_over_throat_diameter, dtype=float)))
    suction_absolute_flow_mach = np.concatenate((
        pressure_absolute_flow_mach, np.full(number_of_nodes - 1, ideal_outlet_absolute_flow_mach, dtype=float)))

    shape = NozzleShape(pressure_surface=_surface(pressure_x, pressure_y, pressure_absolute_flow_mach),
                        suction_surface=_surface(suction_x, suction_y, suction_absolute_flow_mach),
                        chord=float(suction_x[-1]), throat_width=1.0,
                        exit_opening=2.0 * exit_radius_over_throat_diameter,
                        spacing=2.0 * exit_radius_over_throat_diameter / math.cos(outlet_metal_angle_rad),
                        coordinate_scale="throat diameter")
    return IdealNozzleConstruction(shape=shape, contour_point_count=number_of_nodes,
                                   actual_flow_turning_increment=None, pressure_point_count=number_of_nodes)
