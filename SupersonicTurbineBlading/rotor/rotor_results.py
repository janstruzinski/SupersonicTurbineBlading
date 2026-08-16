"""Result containers used only by the supersonic rotor designer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..common_results import SurfaceCoordinates


@dataclass(frozen=True)
class FlowStateTable:
    """Printable absolute/relative rotor flow-state comparison.

    :ivar rows: Flow-quantity label, absolute value, and rotor-relative value for each row.
    """

    rows: tuple[tuple[str, float, float], ...]
    headers: ClassVar[tuple[str, str, str]] = ("Flow quantity", "Absolute frame", "Relative frame")

    def __str__(self) -> str:
        """Format the flow states as an aligned plain-text table.

        :return: Table ready to pass to :func:`print`.
        :rtype: str
        """

        formatted_rows = tuple((label, f"{absolute:.6g}", f"{relative:.6g}") for label, absolute, relative in self.rows)
        columns = (self.headers,) + formatted_rows
        widths = tuple(max(len(row[index]) for row in columns) for index in range(3))

        def format_row(row: tuple[str, str, str]) -> str:
            return " | ".join(value.ljust(width) for value, width in zip(row, widths))

        separator = "-+-".join("-" * width for width in widths)
        return "\n".join((format_row(self.headers), separator, *(format_row(row) for row in formatted_rows)))


@dataclass(frozen=True)
class BladeShape:
    """A pressure/suction-side rotor passage in one length scale.

    ``inlet_pitch`` and ``outlet_pitch`` are open passage widths; they do not
    include leading- or trailing-edge blade metal thickness.

    :ivar pressure_surface: Pressure-side surface forming the upper boundary of the stored passage.
    :ivar suction_surface: Suction-side surface forming the lower boundary of the stored passage.
    :ivar float chord: Axial blade chord in the active length scale.
    :ivar float inlet_pitch: Open inlet passage width in the active length scale.
    :ivar float outlet_pitch: Open outlet passage width in the active length scale.
    :ivar float max_flow_turning_increment: Largest flow-turning increment between adjacent MOC nodes, deg.
    :ivar str coordinate_scale: Human-readable description of the active length scale.
    """

    pressure_surface: SurfaceCoordinates
    suction_surface: SurfaceCoordinates
    chord: float
    inlet_pitch: float
    outlet_pitch: float
    max_flow_turning_increment: float
    coordinate_scale: str

    @property
    def inlet_passage_pitch(self) -> float:
        """Return the open inlet passage width.

        :return: Alias of :attr:`inlet_pitch`.
        :rtype: float
        """

        return self.inlet_pitch

    @property
    def outlet_passage_pitch(self) -> float:
        """Return the open outlet passage width.

        :return: Alias of :attr:`outlet_pitch`.
        :rtype: float
        """

        return self.outlet_pitch

    def scaled(self, factor: float, scale_name: str) -> BladeShape:
        """Return a geometrically scaled copy.

        :param float factor: Length multiplier, for example the vortex sonic radius in metres.
        :param str scale_name: Description assigned to the returned coordinate scale.
        :return: New blade shape with all lengths scaled and flow variables unchanged.
        :rtype: BladeShape
        """

        return BladeShape(pressure_surface=self.pressure_surface.scaled(factor),
                          suction_surface=self.suction_surface.scaled(factor),
                          chord=self.chord * factor,
                          inlet_pitch=self.inlet_pitch * factor,
                          outlet_pitch=self.outlet_pitch * factor,
                          max_flow_turning_increment=self.max_flow_turning_increment,
                          coordinate_scale=scale_name)


@dataclass(frozen=True)
class StartingResult:
    """Store the supersonic-starting result calculated by NASA TN D-4421.

    :ivar float maximum_starting_ideal_inlet_relative_flow_mach: Largest inlet Mach that the passage can start, -.
    :ivar float maximum_starting_ideal_inlet_relative_prandtl_meyer_angle: Corresponding Prandtl--Meyer angle, deg.
    :ivar float specified_ideal_inlet_relative_flow_mach: Rotor-relative inlet Mach supplied to the starting test, -.
    :ivar bool starts_supersonically: Whether the specified design lies below the calculated starting limit.
    :ivar float critical_vortex_constant: Vortex constant at maximum swallowed flow.
    :ivar float two_dimensional_flow_reduction: NASA TN D-4421 flow-reduction factor, -.
    :ivar float weight_flow_parameter: NASA TN D-4421 weight-flow parameter, -.
    """

    maximum_starting_ideal_inlet_relative_flow_mach: float
    maximum_starting_ideal_inlet_relative_prandtl_meyer_angle: float
    specified_ideal_inlet_relative_flow_mach: float
    starts_supersonically: bool
    critical_vortex_constant: float
    two_dimensional_flow_reduction: float
    weight_flow_parameter: float


@dataclass(frozen=True)
class DimensionalBladeShapes:
    """Store ideal and boundary-layer-corrected rotor sections in metres.

    :ivar float mean_radius: Turbine mean radius, m.
    :ivar int blade_count: Number of equally spaced rotor blades.
    :ivar float sonic_radius_scale: Dimensional vortex sonic radius ``r*``, m.
    :ivar BladeShape uncorrected: Ideal MOC blade coordinates in metres.
    :ivar BladeShape corrected: Boundary-layer-corrected blade coordinates in metres.
    """

    mean_radius: float
    blade_count: int
    sonic_radius_scale: float
    uncorrected: BladeShape
    corrected: BladeShape
