"""Small result containers shared by rotor, stator, and BL modules."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SurfaceCoordinates:
    """Store coordinates and flow data along one blade or nozzle surface.

    :ivar x: Axial or local nozzle-axis coordinates in the active length scale.
    :ivar y: Tangential or local transverse coordinates in the active length scale.
    :ivar mach: Local ordinary Mach number at every surface station.
    :ivar tangent_angle_rad: Local surface-tangent angle in radians.
    """

    x: FloatArray
    y: FloatArray
    mach: FloatArray
    tangent_angle_rad: FloatArray

    def scaled(self, factor: float) -> SurfaceCoordinates:
        """Return a copy with only the coordinates multiplied by ``factor``.

        Mach number and tangent angle are dimensionless and are therefore copied without scaling.

        :param float factor: Coordinate scale multiplier.
        :return: New surface expressed in the requested length scale.
        :rtype: SurfaceCoordinates
        """

        return SurfaceCoordinates(
            x=np.asarray(self.x * factor, dtype=float),
            y=np.asarray(self.y * factor, dtype=float),
            mach=self.mach.copy(),
            tangent_angle_rad=self.tangent_angle_rad.copy(),
        )


@dataclass(frozen=True)
class BoundaryLayerResult:
    """Boundary-layer integral quantities along one surface.

    All distances are normalized by the geometry's reference chord.

    :ivar s_over_chord: Surface distance divided by reference chord, -.
    :ivar mach: Edge Mach number at every boundary-layer station, -.
    :ivar displacement_thickness_over_chord: Compressible displacement thickness divided by chord, -.
    :ivar momentum_thickness_over_chord: Compressible momentum thickness divided by chord, -.
    :ivar form_factor: Incompressible transformed boundary-layer form factor, -.
    :ivar regime: ``"laminar"`` or ``"turbulent"`` at every station.
    :ivar transition_index: First turbulent station, or ``None`` when transition does not occur.
    :ivar separation_index: First separated station, or ``None`` when separation does not occur.
    """

    s_over_chord: FloatArray
    mach: FloatArray
    displacement_thickness_over_chord: FloatArray
    momentum_thickness_over_chord: FloatArray
    form_factor: FloatArray
    regime: NDArray[np.str_]
    transition_index: int | None
    separation_index: int | None
