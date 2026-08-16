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
    Exactly one of ``absolute_flow_mach`` and ``relative_flow_mach`` is populated,
    so the reference frame remains explicit in the result API.

    :ivar absolute_flow_mach: Local absolute flow Mach, or ``None`` for a rotor surface.
    :ivar relative_flow_mach: Local rotor-relative flow Mach, or ``None`` for a stator surface.
    :ivar metal_angle: Local surface-tangent metal angle in degrees.
    """

    x: FloatArray
    y: FloatArray
    metal_angle: FloatArray
    absolute_flow_mach: FloatArray | None = None
    relative_flow_mach: FloatArray | None = None

    def __post_init__(self) -> None:
        """Require one explicitly framed flow-Mach array.

        :raises ValueError: If neither or both reference-frame arrays are supplied.
        """

        if (self.absolute_flow_mach is None) == (self.relative_flow_mach is None):
            raise ValueError("exactly one of absolute_flow_mach and relative_flow_mach must be supplied")

    def flow_mach_values(self) -> FloatArray:
        """Return the populated, explicitly framed flow-Mach array.

        :return: Absolute or rotor-relative flow Mach values for this surface.
        :rtype: numpy.ndarray
        """

        if self.absolute_flow_mach is not None:
            return self.absolute_flow_mach
        if self.relative_flow_mach is not None:
            return self.relative_flow_mach
        raise ValueError("surface has no flow-Mach data")

    def scaled(self, factor: float) -> SurfaceCoordinates:
        """Return a copy with only the coordinates multiplied by ``factor``.

        Flow Mach number and metal angle are therefore copied without scaling.

        :param float factor: Coordinate scale multiplier.
        :return: New surface expressed in the requested length scale.
        :rtype: SurfaceCoordinates
        """

        return SurfaceCoordinates(
            x=np.asarray(self.x * factor, dtype=float),
            y=np.asarray(self.y * factor, dtype=float),
            metal_angle=self.metal_angle.copy(),
            absolute_flow_mach=(None if self.absolute_flow_mach is None else self.absolute_flow_mach.copy()),
            relative_flow_mach=(None if self.relative_flow_mach is None else self.relative_flow_mach.copy()))


@dataclass(frozen=True)
class BoundaryLayerResult:
    """Boundary-layer integral quantities along one surface.

    All distances are normalized by the geometry's reference chord.

    :ivar s_over_chord: Surface distance divided by reference chord, -.
    Exactly one framed freestream-flow array is populated.

    :ivar freestream_absolute_flow_mach: Absolute freestream flow Mach, or ``None`` for a rotor boundary layer.
    :ivar freestream_relative_flow_mach: Rotor-relative freestream flow Mach, or ``None`` for a stator boundary layer.
    :ivar displacement_thickness_over_chord: Compressible displacement thickness divided by chord, -.
    :ivar momentum_thickness_over_chord: Compressible momentum thickness divided by chord, -.
    :ivar form_factor: Incompressible transformed boundary-layer form factor, -.
    :ivar regime: ``"laminar"`` or ``"turbulent"`` at every station.
    :ivar transition_index: First turbulent station, or ``None`` when transition does not occur.
    :ivar separation_index: First separated station, or ``None`` when separation does not occur.
    """

    s_over_chord: FloatArray
    displacement_thickness_over_chord: FloatArray
    momentum_thickness_over_chord: FloatArray
    form_factor: FloatArray
    regime: NDArray[np.str_]
    transition_index: int | None
    separation_index: int | None
    freestream_absolute_flow_mach: FloatArray | None = None
    freestream_relative_flow_mach: FloatArray | None = None

    def __post_init__(self) -> None:
        """Require one explicitly framed freestream-flow-Mach array.

        :raises ValueError: If neither or both reference-frame arrays are supplied.
        """

        if (self.freestream_absolute_flow_mach is None) == (self.freestream_relative_flow_mach is None):
            raise ValueError(
                "exactly one of freestream_absolute_flow_mach and freestream_relative_flow_mach must be supplied"
            )

    def freestream_flow_mach_values(self) -> FloatArray:
        """Return the populated, explicitly framed freestream-flow-Mach array.

        :return: Absolute or rotor-relative freestream flow Mach values.
        :rtype: numpy.ndarray
        """

        if self.freestream_absolute_flow_mach is not None:
            return self.freestream_absolute_flow_mach
        if self.freestream_relative_flow_mach is not None:
            return self.freestream_relative_flow_mach
        raise ValueError("boundary-layer result has no freestream flow-Mach data")
