"""Result containers used only by the supersonic rotor designer."""

from __future__ import annotations

from dataclasses import dataclass

from ..common_results import SurfaceCoordinates


@dataclass(frozen=True)
class BladeShape:
    """A pressure/suction-side rotor passage in one length scale.

    ``inlet_pitch`` and ``outlet_pitch`` are open passage widths; they do not
    include leading- or trailing-edge blade metal thickness.

    :ivar pressure: Pressure-side surface forming the upper boundary of the stored passage.
    :ivar suction: Suction-side surface forming the lower boundary of the stored passage.
    :ivar float chord: Axial blade chord in the active length scale.
    :ivar float inlet_pitch: Open inlet passage width in the active length scale.
    :ivar float outlet_pitch: Open outlet passage width in the active length scale.
    :ivar str coordinate_scale: Human-readable description of the active length scale.
    """

    pressure: SurfaceCoordinates
    suction: SurfaceCoordinates
    chord: float
    inlet_pitch: float
    outlet_pitch: float
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

        return BladeShape(
            pressure=self.pressure.scaled(factor),
            suction=self.suction.scaled(factor),
            chord=self.chord * factor,
            inlet_pitch=self.inlet_pitch * factor,
            outlet_pitch=self.outlet_pitch * factor,
            coordinate_scale=scale_name,
        )


@dataclass(frozen=True)
class StartingResult:
    """Store the supersonic-starting result calculated by NASA TN D-4421.

    :ivar float maximum_starting_inlet_mach: Largest inlet Mach number that the passage can start, -.
    :ivar float maximum_starting_inlet_prandtl_meyer_deg: Corresponding Prandtl--Meyer angle, deg.
    :ivar float specified_inlet_mach: Rotor-relative inlet Mach number supplied to the starting test, -.
    :ivar bool starts_supersonically: Whether the specified design lies below the calculated starting limit.
    :ivar float critical_vortex_constant: Vortex constant at maximum swallowed flow.
    :ivar float two_dimensional_flow_reduction: NASA TN D-4421 flow-reduction factor, -.
    :ivar float weight_flow_parameter: NASA TN D-4421 weight-flow parameter, -.
    """

    maximum_starting_inlet_mach: float
    maximum_starting_inlet_prandtl_meyer_deg: float
    specified_inlet_mach: float
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
