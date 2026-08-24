"""Result containers used only by the supersonic stator designer."""

from __future__ import annotations

from dataclasses import dataclass

from ..common_results import SurfaceCoordinates


@dataclass(frozen=True)
class NozzleShape:
    """One two-dimensional stator-nozzle passage in one length scale.

    The two surfaces are the walls of a symmetric sharp-throat MOC or conical
    passage. They are named ``pressure_surface`` and ``suction_surface`` after
    rotation into the turbine blade row, even though each unrotated
    construction is a symmetric nozzle.

    ``throat_width`` is the full opening between the two plotted walls. The
    nondimensional MOC geometry uses throat half-width and therefore has
    ``throat_width == 2``. The axisymmetric conical meridional contour uses
    throat diameter and therefore has ``throat_width == 1``.

    :ivar pressure_surface: Pressure-side wall after rotation into the turbine frame.
    :ivar suction_surface: Suction-side wall after rotation into the turbine frame.
    :ivar float chord: Axial nozzle chord in the active length scale.
    :ivar float throat_width: Full ideal throat opening in the active length scale.
    :ivar float nozzle_exit_width: Full passage width where the divergent wall meets the straight section.
    :ivar float nozzle_passage_pitch: Open circumferential nozzle pitch in the active length scale.
    :ivar float total_pitch: Open nozzle pitch plus trailing-edge metal thickness in the active length scale.
    :ivar float trailing_edge_thickness: Remaining trailing-edge metal in the active length scale.
    :ivar str coordinate_scale: Human-readable description of the active length scale.
    """

    pressure_surface: SurfaceCoordinates
    suction_surface: SurfaceCoordinates
    chord: float
    throat_width: float
    nozzle_exit_width: float
    nozzle_passage_pitch: float
    total_pitch: float
    trailing_edge_thickness: float
    coordinate_scale: str

    def scaled(self, factor: float, scale_name: str) -> NozzleShape:
        """Return a copy with every length multiplied by ``factor``.

        :param float factor: Length multiplier, m per active coordinate unit.
        :param str scale_name: Description assigned to the returned coordinate scale.
        :return: New nozzle shape with all lengths scaled and flow variables unchanged.
        :rtype: NozzleShape
        """

        return NozzleShape(pressure_surface=self.pressure_surface.scaled(factor),
                           suction_surface=self.suction_surface.scaled(factor),
                           chord=self.chord * factor,
                           throat_width=self.throat_width * factor,
                           nozzle_exit_width=self.nozzle_exit_width * factor,
                           nozzle_passage_pitch=self.nozzle_passage_pitch * factor,
                           total_pitch=self.total_pitch * factor,
                           trailing_edge_thickness=self.trailing_edge_thickness * factor,
                           coordinate_scale=scale_name)


@dataclass(frozen=True)
class NozzleShapes:
    """Store uncorrected and BL-corrected stator geometries in one coordinate scale.

    :ivar NozzleShape uncorrected: Ideal nozzle geometry in the active coordinate scale.
    :ivar NozzleShape corrected: Boundary-layer-corrected nozzle geometry in the active coordinate scale.
    """

    uncorrected: NozzleShape
    corrected: NozzleShape


@dataclass(frozen=True)
class DimensionalNozzleShapes:
    """Store dimensional stator passages and the pitch-derived machine scale.

    :ivar float mean_radius: Stator mean radius, m.
    :ivar float partial_admission_fraction: Fraction of the turbine perimeter occupied by nozzles.
    :ivar int nozzle_count: Number of equal nozzles in the admitted arc.
    :ivar float dimensional_scale_factor: Metres represented by one nondimensional coordinate unit.
    :ivar NozzleShape uncorrected: Ideal nozzle coordinates in metres.
    :ivar NozzleShape corrected: Boundary-layer-corrected nozzle coordinates in metres.
    """

    mean_radius: float
    partial_admission_fraction: float
    nozzle_count: int
    dimensional_scale_factor: float
    uncorrected: NozzleShape
    corrected: NozzleShape
