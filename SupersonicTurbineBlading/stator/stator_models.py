"""Result containers used only by the supersonic stator designer."""

from __future__ import annotations

from dataclasses import dataclass

from ..common_models import SurfaceCoordinates


@dataclass(frozen=True)
class NozzleShape:
    """One two-dimensional stator-nozzle passage in one length scale.

    The two surfaces are the walls of a symmetric sharp-throat MOC or conical
    passage. They are named ``pressure`` and ``suction`` after rotation into
    the turbine blade row, even though each unrotated construction is a
    symmetric nozzle.

    ``throat_width`` is the full opening between the two plotted walls. The
    nondimensional MOC geometry uses throat half-width and therefore has
    ``throat_width == 2``. The axisymmetric conical meridional contour uses
    throat diameter and therefore has ``throat_width == 1``.

    :ivar pressure: Pressure-side wall after rotation into the turbine frame.
    :ivar suction: Suction-side wall after rotation into the turbine frame.
    :ivar float chord: Axial nozzle chord in the active length scale.
    :ivar float throat_width: Full ideal throat opening in the active length scale.
    :ivar float exit_opening: Full exit opening in the active length scale.
    :ivar float spacing: Periodic stator pitch in the active length scale.
    :ivar str coordinate_scale: Human-readable description of the active length scale.
    """

    pressure: SurfaceCoordinates
    suction: SurfaceCoordinates
    chord: float
    throat_width: float
    exit_opening: float
    spacing: float
    coordinate_scale: str

    def scaled(self, factor: float, scale_name: str) -> NozzleShape:
        """Return a copy with every length multiplied by ``factor``.

        :param float factor: Length multiplier, m per active coordinate unit.
        :param str scale_name: Description assigned to the returned coordinate scale.
        :return: New nozzle shape with all lengths scaled and flow variables unchanged.
        :rtype: NozzleShape
        """

        return NozzleShape(
            pressure=self.pressure.scaled(factor),
            suction=self.suction.scaled(factor),
            chord=self.chord * factor,
            throat_width=self.throat_width * factor,
            exit_opening=self.exit_opening * factor,
            spacing=self.spacing * factor,
            coordinate_scale=scale_name,
        )


@dataclass(frozen=True)
class DimensionalNozzleShapes:
    """Store dimensional ideal and boundary-layer-corrected stator passages.

    :ivar float total_throat_area: Total choked area of the complete stator row, m^2.
    :ivar float single_nozzle_throat_area: Choked area assigned to one nozzle passage, m^2.
    :ivar int nozzle_count: Number of identical stator passages.
    :ivar throat_height: Annulus height used by the planar MOC model, m, or ``None`` for a conical nozzle.
    :ivar ideal_throat_width: Planar full throat width, m, or ``None`` for a conical nozzle.
    :ivar ideal_throat_diameter: Conical throat diameter, m, or ``None`` for a planar MOC nozzle.
    :ivar float coordinate_scale_length: Metres represented by one stored coordinate unit.
    :ivar throat_half_width_scale: Planar throat half-width scale, m, or ``None`` for a conical nozzle.
    :ivar NozzleShape uncorrected: Ideal nozzle coordinates in metres.
    :ivar NozzleShape corrected: Boundary-layer-corrected nozzle coordinates in metres.
    """

    total_throat_area: float
    single_nozzle_throat_area: float
    nozzle_count: int
    throat_height: float | None
    ideal_throat_width: float | None
    ideal_throat_diameter: float | None
    coordinate_scale_length: float
    throat_half_width_scale: float | None
    uncorrected: NozzleShape
    corrected: NozzleShape
