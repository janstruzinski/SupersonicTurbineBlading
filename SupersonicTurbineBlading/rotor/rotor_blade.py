"""Object-oriented public API for supersonic rotor blade design."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import brentq, fixed_point, least_squares, toms748

from ..boundary_layer.boundary_layer_solver import BoundaryLayerError, BoundaryLayerMode, solve_boundary_layer
from ..common_results import BoundaryLayerResult, SurfaceCoordinates
from ..fluid import Fluid, FluidState
from ..gas_dynamics import isentropic_area_ratio, mach_from_prandtl_meyer, mass_flow_parameter, prandtl_meyer_angle
from .rotor_geometry import GeometryError, design_ideal_geometry
from .rotor_results import BladeShape, DimensionalBladeShapes, FlowStateTable
from .rotor_starting import calculate_starting_limit

MixingSolution = Literal["supersonic", "subsonic"]
MixingSolutionOverride = Literal["subsonic"]
FlowInputReferenceFrame = Literal["absolute", "relative"]


@dataclass(frozen=True)
class _PhysicalScale:
    """Physical scale derived from one trial ideal blade geometry.

    :ivar float total_pitch: Circumferential blade pitch, m.
    :ivar float passage_pitch: Open flow-passage pitch, m.
    :ivar float leading_edge_thickness: Leading-edge metal thickness, m.
    :ivar float sonic_radius: Length represented by one nondimensional ``r*``.
    :ivar float chord: Dimensional axial chord, m.
    :ivar float chord_reynolds_number: Reynolds number based on ``chord``.
    """

    total_pitch: float
    passage_pitch: float
    leading_edge_thickness: float
    sonic_radius: float
    chord: float
    chord_reynolds_number: float


@dataclass(frozen=True)
class _RotorEvaluation:
    """Geometry, fixed-node BL marches, and MOC-station design results.

    This private record keeps every quantity produced by one angle/Mach trial
    together. It is cached because an outlet solve often evaluates the same
    trial more than once.

    :ivar BladeShape ideal: Inviscid MOC blade passage.
    :ivar BladeShape corrected: Passage after displacement-thickness correction.
    :ivar BoundaryLayerResult pressure_boundary_layer: Pressure-side BL at MOC stations.
    :ivar BoundaryLayerResult suction_boundary_layer: Suction-side BL at MOC stations.
    :ivar BoundaryLayerResult pressure_boundary_layer_marching: Alias of the pressure-side BL result.
    :ivar BoundaryLayerResult suction_boundary_layer_marching: Alias of the suction-side BL result.
    :ivar float leading_edge_thickness: Nondimensional leading-edge metal thickness.
    :ivar float trailing_edge_thickness: Nondimensional trailing-edge metal thickness.
    :ivar float trailing_edge_vertical_boundary_layer_height: Sum of the two vertical BL offsets at exit.
    :ivar float pitch_residual: Corrected outlet pitch minus corrected inlet pitch.
    :ivar dict mixing: Subsonic and supersonic aftermixing solutions.
    """

    ideal: BladeShape
    corrected: BladeShape
    pressure_boundary_layer: BoundaryLayerResult
    suction_boundary_layer: BoundaryLayerResult
    pressure_boundary_layer_marching: BoundaryLayerResult
    suction_boundary_layer_marching: BoundaryLayerResult
    leading_edge_thickness: float
    trailing_edge_thickness: float
    trailing_edge_vertical_boundary_layer_height: float
    pitch_residual: float
    mixing: dict[str, dict[str, float | bool]]


class DesignConvergenceError(RuntimeError):
    """Raised when an optional design iteration has no physical solution."""


class SupersonicRotorBlade:
    """Design one two-dimensional supersonic rotor blade section.

    Parameters are ordinary Mach numbers and signed angles in degrees. The inlet and outlet flow states can be
    supplied together in either the absolute frame or the rotor-relative frame. The two input families are mutually
    exclusive. Mean radius and rotational speed construct the corresponding states in the other frame. The two
    surface Mach inputs are always rotor-relative because they directly define the NASA TN D-4421 vortex arcs. No
    Mach input is a critical velocity ratio ``M*`` or a Prandtl--Meyer angle.

    The design is performed during initialization.  Ideal coordinates,
    boundary-layer results, corrected coordinates, outlet mixing, and the
    optional starting calculation are consequently available as object
    properties immediately after construction.

    :param float | None ideal_inlet_absolute_flow_mach: Absolute inlet Mach number.
    :param float | None ideal_inlet_absolute_flow_angle: Absolute inlet flow angle measured from
        the positive axial direction toward the direction of rotation.
    :param float | None requested_outlet_absolute_flow_angle: Absolute outlet flow-angle target
        measured from the positive axial direction, degrees; normally negative
        for the turbine-rotor convention used by NASA TN D-4421 and NASA TM X-2434.
        By default this is the ideal premixing direction. With
        ``iterate_outlet_metal_angle=True`` it is the desired real aftermixed direction.
    :param float | None requested_outlet_absolute_flow_mach: Absolute outlet Mach target. By default it
        is the uniform ideal value before aftermixing. With
        ``match_real_outlet_mach=True`` it is instead the desired
        mixed value. Omission selects the NASA TM X-2434 impulse-rotor assumption
        ``M_out,rel=M_in,rel``.
    :param float | None ideal_inlet_relative_flow_mach: Rotor-relative inlet Mach number. Supply this,
        ``ideal_inlet_relative_flow_angle``, and ``requested_outlet_relative_flow_angle`` instead of the absolute
        flow inputs to use the NASA TN D-4421 input convention.
    :param float | None ideal_inlet_relative_flow_angle: Rotor-relative inlet flow angle measured from the positive
        axial direction toward the direction of rotation, degrees.
    :param float | None requested_outlet_relative_flow_angle: Rotor-relative ideal outlet flow angle, degrees;
        normally negative for the NASA TN D-4421 turbine convention.
    :param float | None requested_outlet_relative_flow_mach: Rotor-relative outlet Mach target. By default it is
        the uniform ideal value before aftermixing. With ``match_real_outlet_mach=True`` it is instead the desired
        mixed value. Omission selects the impulse-rotor assumption ``M_out,rel=M_in,rel``.
    :param float lower_surface_relative_flow_mach: Rotor-relative constant-Mach pressure-
        surface arc value.
    :param float upper_surface_relative_flow_mach: Rotor-relative constant-Mach suction-
        surface arc value.
    :param int blade_count: Number of blades at the initialized mean radius.
    :param float mean_radius: Physical mean radius in the desired blade-length
        unit.  SI fluid properties require this value in metres.
    :param float rotational_speed_rpm: Rotor speed. Positive rotation is in
        the positive tangential direction used by ``ideal_inlet_absolute_flow_angle``.
    :param Fluid fluid: CoolProp-backed ideal-gas mixture.
    :param float inlet_total_temperature: Absolute inlet total temperature, K.
    :param float inlet_total_pressure: Absolute inlet total pressure, Pa.
    :param int number_of_nodes: Nodes used by every nonzero MOC transition and
        constant-Mach circular arc. The boundary-layer calculation marches on
        the assembled MOC surface without a separate mesh.
    :param bool iterate_outlet_metal_angle: If false, assume zero relative-flow
        deviation and convert the requested flow angle into the corresponding
        outlet metal angle. If true, iterate the metal angle until the selected
        mixed outlet flow angle in the input reference frame is obtained.
    :param bool match_real_outlet_mach: If true, the iterative design also varies
        the ideal relative outlet Mach until the selected mixed Mach equals the
        requested outlet Mach in the input reference frame. This requires
        ``iterate_outlet_metal_angle=True`` and a supplied requested outlet Mach.
    :param bool iterate_pitch_closure: If true, use the NASA TM X-2434 iteration to
        change the outlet metal angle until the BL-corrected outlet
        pitch equals the ideal inlet pitch. The supplied absolute outlet
        angle is only an initial estimate. This option is incompatible with
        either mixed-flow angle or Mach matching.
    :param float leading_edge_thickness_over_total_pitch: Leading-edge blade
        thickness divided by total blade pitch, ``t_LE/G*_total``. The
        default zero recovers the sharp-edge geometry exactly.
    :param bool use_leading_edge_entry_correction: If true, use the
        NACA RM L52B06 external-wave continuity and turning
        equations to transform the far-field rotor-relative inlet Mach and
        angle into the passage-entry values used by the MOC and BL models.
        A positive ratio with supersonic axial inflow emits a warning because
        NACA RM L52B06 derives this construction for subsonic axial inflow.
    :param bool calculate_starting: Run the NASA TN D-4421 starting feasibility test.
    :param BoundaryLayerMode boundary_layer_mode: ``"fully_turbulent"`` or
        ``"laminar_then_turbulent"``; both start at the rotor inlet.
    :param float | None initial_turbulent_displacement_thickness:
        Dimensional inlet displacement thickness in metres.  Required for a
        fully turbulent calculation and unused for a laminar inlet.
    :param float | None initial_turbulent_momentum_thickness:
        Dimensional inlet momentum thickness in metres.  Required for a fully
        turbulent calculation and unused for a laminar inlet.
    :param MixingSolutionOverride | None mixing_solution: Optional aftermixing-solution
        override. By default, subsonic premixing axial flow uses the subsonic
        solution, while supersonic axial flow uses the shockless solution when
        it is available. Set ``"subsonic"`` to force the subsonic solution.
    :ivar float max_flow_turning_increment: Largest turning increment between
        adjacent nodes in the final MOC transitions and circular arcs, degrees.
    :ivar FlowStateTable flow_state_table: Printable inlet-to-outlet flow-angle and Mach comparison in both frames.
    :ivar numpy.ndarray blade_profile_x_CAD: Corrected single-blade profile
        x coordinates in millimetres, ordered for direct CAD import. The first
        point is the lower-surface leading edge at zero.
    :ivar numpy.ndarray blade_profile_y_CAD: Corrected single-blade profile
        y coordinates in millimetres, ordered for direct CAD import. The first
        point is the lower-surface leading edge at zero.
    :ivar numpy.ndarray uncorrected_blade_profile_x_CAD: Ideal, uncorrected
        single-blade profile x coordinates in millimetres, with the same order
        and origin convention as :attr:`blade_profile_x_CAD`.
    :ivar numpy.ndarray uncorrected_blade_profile_y_CAD: Ideal, uncorrected
        single-blade profile y coordinates in millimetres, with the same order
        and origin convention as :attr:`blade_profile_y_CAD`.
    """

    def __init__(self, *, ideal_inlet_absolute_flow_mach: float | None = None,
        ideal_inlet_absolute_flow_angle: float | None = None,
        requested_outlet_absolute_flow_angle: float | None = None,
        requested_outlet_absolute_flow_mach: float | None = None,
        ideal_inlet_relative_flow_mach: float | None = None,
        ideal_inlet_relative_flow_angle: float | None = None,
        requested_outlet_relative_flow_angle: float | None = None,
        requested_outlet_relative_flow_mach: float | None = None,
        lower_surface_relative_flow_mach: float,
        upper_surface_relative_flow_mach: float,
        blade_count: int,
        mean_radius: float,
        rotational_speed_rpm: float,
        fluid: Fluid,
        inlet_total_temperature: float,
        inlet_total_pressure: float,
        number_of_nodes: int = 101,
        iterate_outlet_metal_angle: bool = False,
        match_real_outlet_mach: bool = False,
        iterate_pitch_closure: bool = False,
        leading_edge_thickness_over_total_pitch: float = 0.0,
        use_leading_edge_entry_correction: bool = True,
        calculate_starting: bool = True,
        boundary_layer_mode: BoundaryLayerMode = "laminar_then_turbulent",
        initial_turbulent_displacement_thickness: float | None = None,
        initial_turbulent_momentum_thickness: float | None = None,
        mixing_solution: MixingSolutionOverride | None = None) -> None:
        """Validate the user inputs and execute the complete rotor design.

        The constructor arguments and their units are documented on
        :class:`SupersonicRotorBlade`. Construction performs the velocity
        triangles, MOC geometry, BL marches, optional iterations, aftermixing,
        and optional starting check so the result attributes are immediately
        available to an engineering script.

        :raises TypeError: If an input has the wrong basic type.
        :raises ValueError: If an input or derived NASA-model quantity is outside its physical range.
        :raises DesignConvergenceError: If an optional design iteration cannot find a physical solution.
        """

        # Check raw user inputs before converting them to float. This produces
        # useful engineering errors rather than failures deep inside the MOC.
        self.flow_input_reference_frame = self._validate_inputs(
            ideal_inlet_absolute_flow_mach=ideal_inlet_absolute_flow_mach,
            ideal_inlet_absolute_flow_angle=ideal_inlet_absolute_flow_angle,
            requested_outlet_absolute_flow_angle=requested_outlet_absolute_flow_angle,
            requested_outlet_absolute_flow_mach=requested_outlet_absolute_flow_mach,
            ideal_inlet_relative_flow_mach=ideal_inlet_relative_flow_mach,
            ideal_inlet_relative_flow_angle=ideal_inlet_relative_flow_angle,
            requested_outlet_relative_flow_angle=requested_outlet_relative_flow_angle,
            requested_outlet_relative_flow_mach=requested_outlet_relative_flow_mach,
            lower_surface_relative_flow_mach=lower_surface_relative_flow_mach,
            upper_surface_relative_flow_mach=upper_surface_relative_flow_mach,
            blade_count=blade_count,
            mean_radius=mean_radius,
            rotational_speed_rpm=rotational_speed_rpm,
            fluid=fluid,
            inlet_total_temperature=inlet_total_temperature,
            inlet_total_pressure=inlet_total_pressure,
            number_of_nodes=number_of_nodes,
            iterate_outlet_metal_angle=iterate_outlet_metal_angle,
            match_real_outlet_mach=match_real_outlet_mach,
            iterate_pitch_closure=iterate_pitch_closure,
            leading_edge_thickness_over_total_pitch=leading_edge_thickness_over_total_pitch,
            use_leading_edge_entry_correction=use_leading_edge_entry_correction,
            boundary_layer_mode=boundary_layer_mode,
            initial_turbulent_displacement_thickness=initial_turbulent_displacement_thickness,
            initial_turbulent_momentum_thickness=initial_turbulent_momentum_thickness,
            mixing_solution=mixing_solution)
        self.lower_surface_relative_flow_mach = float(lower_surface_relative_flow_mach)
        self.upper_surface_relative_flow_mach = float(upper_surface_relative_flow_mach)
        self.blade_count = int(blade_count)
        self.mean_radius = float(mean_radius)
        self.rotational_speed_rpm = float(rotational_speed_rpm)
        self.fluid = fluid
        self.inlet_total_temperature = float(inlet_total_temperature)
        self.inlet_total_pressure = float(inlet_total_pressure)
        self.wheel_speed = 2.0 * math.pi * self.mean_radius * self.rotational_speed_rpm / 60.0
        self.requested_outlet_absolute_flow_angle = (None if requested_outlet_absolute_flow_angle is None
                                                     else float(requested_outlet_absolute_flow_angle))
        self.requested_outlet_relative_flow_angle = (None if requested_outlet_relative_flow_angle is None
                                                     else float(requested_outlet_relative_flow_angle))
        self._requested_outlet_absolute_flow_mach = (None if requested_outlet_absolute_flow_mach is None
                                                     else float(requested_outlet_absolute_flow_mach))
        self._requested_outlet_relative_flow_mach = (None if requested_outlet_relative_flow_mach is None
                                                     else float(requested_outlet_relative_flow_mach))

        # Keep the total-state properties as a useful diagnostic, but do not
        # use their gamma for the gas-dynamic design.  Gamma must represent
        # the actual static gas entering the rotor.
        self.inlet_total_fluid_state = self.fluid.properties(self.inlet_total_temperature, self.inlet_total_pressure)

        # Static temperature depends on gamma through the isentropic
        # total-to-static relation, while mixture gamma itself depends on that
        # static temperature through Cp(T).  Solve this small fixed-point
        # problem instead of evaluating gamma once at total temperature.
        if self.flow_input_reference_frame == "absolute":
            self.ideal_inlet_absolute_flow_mach = float(ideal_inlet_absolute_flow_mach)
            self.ideal_inlet_absolute_flow_angle = float(ideal_inlet_absolute_flow_angle)
            (self.inlet_static_temperature, self.inlet_static_pressure, self.inlet_static_fluid_state) = \
                self._solve_inlet_static_reference_state(initial_gamma=self.inlet_total_fluid_state.gamma)
        else:
            self.ideal_inlet_relative_flow_mach = float(ideal_inlet_relative_flow_mach)
            self.ideal_inlet_relative_flow_angle = float(ideal_inlet_relative_flow_angle)
            (self.inlet_static_temperature, self.inlet_static_pressure, self.inlet_static_fluid_state) = \
                self._solve_inlet_static_reference_state_from_relative(initial_gamma=self.inlet_total_fluid_state.gamma)
        self.gamma = float(self.inlet_static_fluid_state.gamma)
        self.prandtl_number = float(self.inlet_static_fluid_state.prandtl_number)

        # The static thermodynamic state and speed of sound are common to both frames; only the velocity vector
        # changes. Preserve the supplied frame exactly and derive the corresponding state in the other frame.
        inlet_speed_of_sound = self.inlet_static_fluid_state.speed_of_sound
        if self.flow_input_reference_frame == "absolute":
            self.absolute_inlet_speed = self.ideal_inlet_absolute_flow_mach * inlet_speed_of_sound
            absolute_inlet_flow_angle_rad = math.radians(self.ideal_inlet_absolute_flow_angle)
            self.absolute_inlet_axial_velocity = self.absolute_inlet_speed * math.cos(absolute_inlet_flow_angle_rad)
            self.absolute_inlet_tangential_velocity = self.absolute_inlet_speed * math.sin(
                absolute_inlet_flow_angle_rad)
            self.relative_inlet_axial_velocity = self.absolute_inlet_axial_velocity
            self.relative_inlet_tangential_velocity = self.absolute_inlet_tangential_velocity - self.wheel_speed
            self.relative_inlet_speed = math.hypot(self.relative_inlet_axial_velocity,
                                                   self.relative_inlet_tangential_velocity)
            self.ideal_inlet_relative_flow_mach = self.relative_inlet_speed / inlet_speed_of_sound
            self.ideal_inlet_relative_flow_angle = math.degrees(math.atan2(
                self.relative_inlet_tangential_velocity, self.relative_inlet_axial_velocity))
        else:
            self.relative_inlet_speed = self.ideal_inlet_relative_flow_mach * inlet_speed_of_sound
            relative_inlet_flow_angle_rad = math.radians(self.ideal_inlet_relative_flow_angle)
            self.relative_inlet_axial_velocity = self.relative_inlet_speed * math.cos(relative_inlet_flow_angle_rad)
            self.relative_inlet_tangential_velocity = self.relative_inlet_speed * math.sin(
                relative_inlet_flow_angle_rad)
            self.absolute_inlet_axial_velocity = self.relative_inlet_axial_velocity
            self.absolute_inlet_tangential_velocity = self.relative_inlet_tangential_velocity + self.wheel_speed
            self.absolute_inlet_speed = math.hypot(self.absolute_inlet_axial_velocity,
                                                   self.absolute_inlet_tangential_velocity)
            self.ideal_inlet_absolute_flow_mach = self.absolute_inlet_speed / inlet_speed_of_sound
            self.ideal_inlet_absolute_flow_angle = math.degrees(math.atan2(
                self.absolute_inlet_tangential_velocity, self.absolute_inlet_axial_velocity))
        self.leading_edge_thickness_over_total_pitch = float(leading_edge_thickness_over_total_pitch)
        self.use_leading_edge_entry_correction = bool(use_leading_edge_entry_correction)
        (self.real_inlet_relative_flow_mach, self.real_inlet_relative_flow_angle) = self._passage_entry_conditions()
        # Metal orientation is stored in the machine axial/tangential frame.
        # Zero incidence makes it numerically equal to the corrected entry
        # relative-flow angle, but it remains a separate geometry property.
        self.inlet_metal_angle = float(self.real_inlet_relative_flow_angle)
        self._validate_relative_design_inputs()

        relative_temperature_factor = 1.0 + 0.5 * (self.gamma - 1.0) * self.ideal_inlet_relative_flow_mach**2
        self.relative_inlet_total_temperature = self.inlet_static_temperature * relative_temperature_factor
        self.relative_inlet_total_pressure = self.inlet_static_pressure * relative_temperature_factor ** \
                                             (self.gamma / (self.gamma - 1.0))
        self.relative_inlet_total_fluid_state = self.fluid.properties(self.relative_inlet_total_temperature,
                                                                      self.relative_inlet_total_pressure)
        passage_temperature_factor = 1.0 + 0.5 * (self.gamma - 1.0) * self.real_inlet_relative_flow_mach**2
        self.passage_inlet_static_temperature = self.relative_inlet_total_temperature / passage_temperature_factor
        self.passage_inlet_static_pressure = self.relative_inlet_total_pressure / passage_temperature_factor ** \
                                             (self.gamma / (self.gamma - 1.0))
        self.passage_inlet_static_fluid_state = self.fluid.properties(self.passage_inlet_static_temperature,
                                                                      self.passage_inlet_static_pressure)
        self.passage_inlet_speed_of_sound = math.sqrt(self.gamma * self.fluid.specific_gas_constant
                                                      * self.passage_inlet_static_temperature)
        real_inlet_absolute_state = self._relative_flow_state_to_absolute(
            relative_flow_mach=self.real_inlet_relative_flow_mach,
            relative_flow_angle_rad=math.radians(self.real_inlet_relative_flow_angle))
        self.real_inlet_absolute_flow_mach = real_inlet_absolute_state["absolute_flow_mach"]
        self.real_inlet_absolute_flow_angle = real_inlet_absolute_state["absolute_flow_angle"]
        if self.flow_input_reference_frame == "absolute":
            self.requested_outlet_absolute_flow_mach = self._requested_outlet_absolute_flow_mach
            if self._requested_outlet_absolute_flow_mach is None:
                self.requested_outlet_relative_flow_mach = None
                self.requested_outlet_relative_flow_angle = self._absolute_outlet_to_relative_angle(
                    absolute_flow_angle=self.requested_outlet_absolute_flow_angle,
                    relative_flow_mach=self.ideal_inlet_relative_flow_mach)
            else:
                requested_outlet_state = self._absolute_outlet_state_to_relative(
                    absolute_flow_mach=self._requested_outlet_absolute_flow_mach,
                    absolute_flow_angle=self.requested_outlet_absolute_flow_angle)
                self.requested_outlet_relative_flow_mach = requested_outlet_state["relative_flow_mach"]
                self.requested_outlet_relative_flow_angle = requested_outlet_state["relative_flow_angle"]
        else:
            self.requested_outlet_relative_flow_mach = self._requested_outlet_relative_flow_mach
            requested_relative_flow_mach = (self.ideal_inlet_relative_flow_mach
                                            if self._requested_outlet_relative_flow_mach is None
                                            else self._requested_outlet_relative_flow_mach)
            requested_outlet_state = self._relative_flow_state_to_absolute(
                relative_flow_mach=requested_relative_flow_mach,
                relative_flow_angle_rad=math.radians(self.requested_outlet_relative_flow_angle))
            self.requested_outlet_absolute_flow_mach = (None if self._requested_outlet_relative_flow_mach is None
                                                        else requested_outlet_state["absolute_flow_mach"])
            self.requested_outlet_absolute_flow_angle = requested_outlet_state["absolute_flow_angle"]
        self.number_of_nodes = int(number_of_nodes)
        self.iterate_outlet_metal_angle = bool(iterate_outlet_metal_angle)
        self.match_real_outlet_mach = bool(match_real_outlet_mach)
        self.iterate_pitch_closure = bool(iterate_pitch_closure)
        self.calculate_starting = bool(calculate_starting)
        self.boundary_layer_mode = boundary_layer_mode
        self.initial_turbulent_displacement_thickness = (None if initial_turbulent_displacement_thickness is None
                                                         else float(initial_turbulent_displacement_thickness))
        self.initial_turbulent_momentum_thickness = (None if initial_turbulent_momentum_thickness is None
                                                     else float(initial_turbulent_momentum_thickness))
        self._mixing_solution_override = mixing_solution
        self._evaluation_cache: dict[tuple[float, float], _RotorEvaluation] = {}
        self.dimensional_shapes: DimensionalBladeShapes | None = None
        self.pitch_closure_iteration_count: int | None = None

        # The zero-deviation mode first converts the requested angle to the
        # matching relative direction and performs one design. The
        # iterative mode repeatedly rebuilds the geometry, both boundary
        # layers, and aftermixing solution because blockage changes whenever
        # the trial metal angle changes.
        if self.iterate_pitch_closure:
            if self.flow_input_reference_frame == "relative":
                ideal_outlet_relative_flow_mach = (self.ideal_inlet_relative_flow_mach
                                                   if self._requested_outlet_relative_flow_mach is None
                                                   else self._requested_outlet_relative_flow_mach)
                initial_outlet_metal_angle = self.requested_outlet_relative_flow_angle
            else:
                if self._requested_outlet_absolute_flow_mach is None:
                    ideal_outlet_relative_flow_mach = self.ideal_inlet_relative_flow_mach
                    initial_outlet_metal_angle = self._absolute_outlet_to_relative_angle(
                        absolute_flow_angle=self.requested_outlet_absolute_flow_angle,
                        relative_flow_mach=ideal_outlet_relative_flow_mach)
                else:
                    initial_outlet_state = self._absolute_outlet_state_to_relative(
                        absolute_flow_mach=self._requested_outlet_absolute_flow_mach,
                        absolute_flow_angle=self.requested_outlet_absolute_flow_angle)
                    ideal_outlet_relative_flow_mach = initial_outlet_state["relative_flow_mach"]
                    initial_outlet_metal_angle = initial_outlet_state["relative_flow_angle"]
            outlet_metal_angle = self._solve_outlet_metal_angle_for_pitch_closure(
                initial_outlet_metal_angle=initial_outlet_metal_angle,
                ideal_outlet_relative_flow_mach=ideal_outlet_relative_flow_mach)
            requested_angle_name = f"requested_outlet_{self.flow_input_reference_frame}_flow_angle"
            warnings.warn("legacy pitch closure changes the outlet metal angle from its "
                f"initial {initial_outlet_metal_angle:.6g} deg to "
                f"{outlet_metal_angle:.6g} deg; {requested_angle_name} is used only "
                "as the initial estimate",
                UserWarning,
                stacklevel=2)
        elif self.match_real_outlet_mach:
            outlet_metal_angle, ideal_outlet_relative_flow_mach = \
                self._solve_outlet_metal_angle_and_flow_mach_targets()
        elif self.iterate_outlet_metal_angle:
            outlet_metal_angle = self._solve_outlet_metal_angle_for_target_flow()
            ideal_outlet_relative_flow_mach = self._ideal_outlet_relative_flow_mach_for_metal_angle(outlet_metal_angle)
        else:
            if self.flow_input_reference_frame == "relative":
                ideal_outlet_relative_flow_mach = (self.ideal_inlet_relative_flow_mach
                                                   if self._requested_outlet_relative_flow_mach is None
                                                   else self._requested_outlet_relative_flow_mach)
                outlet_metal_angle = self.requested_outlet_relative_flow_angle
            else:
                if self._requested_outlet_absolute_flow_mach is None:
                    ideal_outlet_relative_flow_mach = self.ideal_inlet_relative_flow_mach
                    outlet_metal_angle = self._absolute_outlet_to_relative_angle(
                        absolute_flow_angle=self.requested_outlet_absolute_flow_angle,
                        relative_flow_mach=ideal_outlet_relative_flow_mach)
                else:
                    ideal_outlet_state = self._absolute_outlet_state_to_relative(
                        absolute_flow_mach=self._requested_outlet_absolute_flow_mach,
                        absolute_flow_angle=self.requested_outlet_absolute_flow_angle)
                    ideal_outlet_relative_flow_mach = ideal_outlet_state["relative_flow_mach"]
                    outlet_metal_angle = ideal_outlet_state["relative_flow_angle"]

        self._validate_surface_mach_ranges(
            ideal_outlet_relative_flow_mach=ideal_outlet_relative_flow_mach, outlet_metal_angle=outlet_metal_angle)
        evaluation = self._evaluate(ideal_outlet_relative_flow_mach, outlet_metal_angle)

        ideal_absolute_outlet_state = self._relative_flow_state_to_absolute(
            relative_flow_mach=ideal_outlet_relative_flow_mach,
            relative_flow_angle_rad=math.radians(outlet_metal_angle))
        # Absolute outlet properties use the stationary frame. The explicit
        # relative properties are the values passed to NASA TN D-4421 and
        # NASA TM X-2434.
        self.ideal_outlet_absolute_flow_mach = ideal_absolute_outlet_state["absolute_flow_mach"]
        self.ideal_outlet_relative_flow_mach = float(ideal_outlet_relative_flow_mach)
        self.ideal_outlet_absolute_flow_angle = ideal_absolute_outlet_state["absolute_flow_angle"]
        self.ideal_outlet_relative_flow_angle = float(outlet_metal_angle)
        self.ideal_outlet_absolute_axial_flow_mach = ideal_absolute_outlet_state["absolute_axial_flow_mach"]
        # Zero deviation makes the ideal relative-flow and metal angles
        # numerically equal; retain independent public quantities.
        self.outlet_metal_angle = float(outlet_metal_angle)
        self.pitch_closure_outlet_metal_angle = float(outlet_metal_angle) if self.iterate_pitch_closure else None
        self.uncorrected_shape = evaluation.ideal
        self.corrected_shape = evaluation.corrected
        self.max_flow_turning_increment = evaluation.ideal.max_flow_turning_increment
        self.pressure_boundary_layer = evaluation.pressure_boundary_layer
        self.suction_boundary_layer = evaluation.suction_boundary_layer
        self.pressure_boundary_layer_marching = evaluation.pressure_boundary_layer_marching
        self.suction_boundary_layer_marching = evaluation.suction_boundary_layer_marching
        self.boundary_layer_pressure_station_count = len(self.pressure_boundary_layer_marching.s_over_chord)
        self.boundary_layer_suction_station_count = len(self.suction_boundary_layer_marching.s_over_chord)
        self.corrected_pitch_residual = float(evaluation.pitch_residual)
        self.pitch_closure_residual = float(evaluation.corrected.outlet_pitch - evaluation.ideal.inlet_pitch)
        self.pitch_residual = self.pitch_closure_residual if self.iterate_pitch_closure \
            else self.corrected_pitch_residual
        self.inlet_passage_pitch = float(evaluation.ideal.inlet_pitch)
        self.inlet_total_pitch = self.inlet_passage_pitch / (1.0 - self.leading_edge_thickness_over_total_pitch)
        self.solidity = float(evaluation.ideal.chord / self.inlet_total_pitch)
        self.leading_edge_thickness = float(evaluation.leading_edge_thickness)
        self.trailing_edge_thickness = float(evaluation.trailing_edge_thickness)
        self.trailing_edge_vertical_boundary_layer_height = \
            float(evaluation.trailing_edge_vertical_boundary_layer_height)
        if (self.leading_edge_thickness > 0.0
            and self.trailing_edge_thickness == 0.0
            and self.trailing_edge_vertical_boundary_layer_height > self.leading_edge_thickness
            and not self.iterate_pitch_closure):
            warnings.warn("the summed vertical trailing-edge boundary-layer displacement "
                "exceeds t_LE; t_TE is limited to zero",
                RuntimeWarning,
                stacklevel=2)
        self.mixing_results = evaluation.mixing
        selected_solution, selected_mixing = self._select_mixing_result(evaluation.mixing)
        if not bool(selected_mixing["available"]):
            raise DesignConvergenceError("the selected rotor aftermixing solution is unavailable")
        self.mixing_solution = selected_solution
        self.real_outlet_absolute_flow_angle = float(selected_mixing["real_outlet_absolute_flow_angle"])
        self.real_outlet_absolute_flow_mach = float(selected_mixing["real_outlet_absolute_flow_mach"])
        self.real_outlet_absolute_axial_flow_mach = float(selected_mixing["real_outlet_absolute_axial_flow_mach"])
        self.real_outlet_relative_flow_angle = float(selected_mixing["real_outlet_relative_flow_angle"])
        self.real_outlet_relative_flow_mach = float(selected_mixing["real_outlet_relative_flow_mach"])
        self.real_outlet_relative_axial_flow_mach = float(selected_mixing["real_outlet_relative_axial_flow_mach"])
        self.ideal_outlet_relative_axial_flow_mach = float(selected_mixing["ideal_outlet_relative_axial_flow_mach"])
        self.supersonic_mixing_available = bool(self.mixing_results["supersonic"]["available"])
        self.flow_state_table = FlowStateTable(rows=(
            ("Ideal flow angle at the inlet upstream", self.ideal_inlet_absolute_flow_angle,
             self.ideal_inlet_relative_flow_angle),
            ("Ideal Mach number at the inlet upstream", self.ideal_inlet_absolute_flow_mach,
             self.ideal_inlet_relative_flow_mach),
            ("Real flow angle at the blade inlet", self.real_inlet_absolute_flow_angle,
             self.real_inlet_relative_flow_angle),
            ("Real Mach number at the blade inlet", self.real_inlet_absolute_flow_mach,
             self.real_inlet_relative_flow_mach),
            ("Ideal flow angle at the blade outlet", self.ideal_outlet_absolute_flow_angle,
             self.ideal_outlet_relative_flow_angle),
            ("Ideal Mach number at the blade outlet", self.ideal_outlet_absolute_flow_mach,
             self.ideal_outlet_relative_flow_mach),
            ("Real flow angle at the blade outlet", self.real_outlet_absolute_flow_angle,
             self.real_outlet_relative_flow_angle),
            ("Real Mach number at the blade outlet", self.real_outlet_absolute_flow_mach,
             self.real_outlet_relative_flow_mach)))

        # Store the dimensional and Reynolds scales that were used for the
        # final boundary-layer calculation.  In iterative outlet-angle mode,
        # trial geometries used their own independently recalculated scales.
        final_scale = self._physical_scale(evaluation.ideal)
        self.physical_total_pitch = final_scale.total_pitch
        self.physical_passage_pitch = final_scale.passage_pitch
        self.physical_leading_edge_thickness = final_scale.leading_edge_thickness
        self.physical_trailing_edge_thickness = (self.physical_leading_edge_thickness
                                                 if self.iterate_pitch_closure
                                                 else self.trailing_edge_thickness * final_scale.sonic_radius)
        # Backward-compatible name: machine pitch has always been 2*pi*r/Z.
        self.physical_pitch = self.physical_total_pitch
        self.sonic_radius_scale = final_scale.sonic_radius
        self.physical_chord = final_scale.chord
        self.chord_reynolds_number = final_scale.chord_reynolds_number
        self.blade_profile_x_CAD, self.blade_profile_y_CAD = self._assemble_cad_profile(self.corrected_shape)
        self.uncorrected_blade_profile_x_CAD, self.uncorrected_blade_profile_y_CAD = \
            self._assemble_cad_profile(self.uncorrected_shape)
        self.starting_result = (calculate_starting_limit(self.ideal_inlet_relative_flow_mach,
                                                         self.lower_surface_relative_flow_mach,
                                                         self.upper_surface_relative_flow_mach, self.gamma)
                                if self.calculate_starting else None)

    @staticmethod
    def _identify_flow_input_reference_frame(**values) -> FlowInputReferenceFrame:
        """Identify one complete, mutually exclusive rotor flow-input family.

        :param values: Constructor values indexed by their public argument name.
        :type values: dict[str, object]
        :return: Reference frame used by the supplied inlet and outlet flow inputs.
        :rtype: Literal["absolute", "relative"]
        :raises ValueError: If both, neither, or an incomplete flow-input family is supplied.
        """

        absolute_required = ("ideal_inlet_absolute_flow_mach",
                             "ideal_inlet_absolute_flow_angle",
        "requested_outlet_absolute_flow_angle")
        relative_required = ("ideal_inlet_relative_flow_mach",
                             "ideal_inlet_relative_flow_angle",
                             "requested_outlet_relative_flow_angle")
        absolute_names = absolute_required + ("requested_outlet_absolute_flow_mach",)
        relative_names = relative_required + ("requested_outlet_relative_flow_mach",)
        absolute_supplied = any(values[name] is not None for name in absolute_names)
        relative_supplied = any(values[name] is not None for name in relative_names)

        if absolute_supplied and relative_supplied:
            raise ValueError("absolute and relative rotor flow input sets are mutually exclusive")
        if not absolute_supplied and not relative_supplied:
            raise ValueError("supply either the absolute or the relative rotor flow input set")

        frame: FlowInputReferenceFrame = "absolute" if absolute_supplied else "relative"
        required = absolute_required if frame == "absolute" else relative_required
        missing = tuple(name for name in required if values[name] is None)
        if missing:
            raise ValueError(f"the {frame} rotor flow input set is incomplete; missing {', '.join(missing)}")
        return frame

    @staticmethod
    def _validate_inputs(**values) -> FlowInputReferenceFrame:
        """Validate public constructor values that do not need derived states.

        :param values: Constructor values indexed by their public argument name.
        :type values: dict[str, object]
        :return: Reference frame used by the supplied flow-input family.
        :rtype: Literal["absolute", "relative"]
        :raises TypeError: If a flag, count, mode, or fluid object has the wrong type.
        :raises ValueError: If a numerical value or option is outside the supported range.
        """

        flow_input_reference_frame = SupersonicRotorBlade._identify_flow_input_reference_frame(**values)
        if flow_input_reference_frame == "absolute":
            if (not math.isfinite(values["ideal_inlet_absolute_flow_mach"])
                or values["ideal_inlet_absolute_flow_mach"] <= 0.0):
                raise ValueError("absolute ideal_inlet_absolute_flow_mach must be positive and finite")
            if not 0.0 < values["ideal_inlet_absolute_flow_angle"] < 90.0:
                raise ValueError("absolute ideal_inlet_absolute_flow_angle must be between 0 and 90")
            if not -90.0 < values["requested_outlet_absolute_flow_angle"] <= 0.0:
                raise ValueError("absolute requested_outlet_absolute_flow_angle must be between -90 and 0")
        else:
            if (not math.isfinite(values["ideal_inlet_relative_flow_mach"])
                or values["ideal_inlet_relative_flow_mach"] <= 1.0):
                raise ValueError("relative ideal_inlet_relative_flow_mach must be finite and > 1")
            if not 0.0 < values["ideal_inlet_relative_flow_angle"] < 90.0:
                raise ValueError("relative ideal_inlet_relative_flow_angle must be between 0 and 90")
            if not -90.0 < values["requested_outlet_relative_flow_angle"] < 0.0:
                raise ValueError("relative requested_outlet_relative_flow_angle must be between -90 and 0")
        if (not math.isfinite(values["lower_surface_relative_flow_mach"])
            or values["lower_surface_relative_flow_mach"] < 1.0):
            raise ValueError("lower_surface_relative_flow_mach must be finite and >= 1")
        if (not math.isfinite(values["upper_surface_relative_flow_mach"])
            or values["upper_surface_relative_flow_mach"] < 1.0):
            raise ValueError("upper_surface_relative_flow_mach must be finite and >= 1")
        if values["upper_surface_relative_flow_mach"] <= values["lower_surface_relative_flow_mach"]:
            raise ValueError("upper_surface_relative_flow_mach must exceed lower_surface_relative_flow_mach")
        if values["requested_outlet_absolute_flow_mach"] is not None and (
            not math.isfinite(values["requested_outlet_absolute_flow_mach"])
            or values["requested_outlet_absolute_flow_mach"] <= 0.0):
            raise ValueError("absolute requested_outlet_absolute_flow_mach must be positive and finite")
        if values["requested_outlet_relative_flow_mach"] is not None and (
            not math.isfinite(values["requested_outlet_relative_flow_mach"])
            or values["requested_outlet_relative_flow_mach"] <= 1.0):
            raise ValueError("relative requested_outlet_relative_flow_mach must be finite and > 1")
        if not isinstance(values["fluid"], Fluid):
            raise TypeError("fluid must be an instance of Fluid")
        if not math.isfinite(values["inlet_total_temperature"]) or values["inlet_total_temperature"] <= 0.0:
            raise ValueError("inlet_total_temperature must be positive and finite")
        if not math.isfinite(values["inlet_total_pressure"]) or values["inlet_total_pressure"] <= 0.0:
            raise ValueError("inlet_total_pressure must be positive and finite")
        if not isinstance(values["blade_count"], int) or values["blade_count"] < 2:
            raise ValueError("blade_count must be an integer >= 2")
        if not math.isfinite(values["mean_radius"]) or values["mean_radius"] <= 0.0:
            raise ValueError("mean_radius must be positive and finite")
        if not math.isfinite(values["rotational_speed_rpm"]) or values["rotational_speed_rpm"] <= 0.0:
            raise ValueError("rotational_speed_rpm must be positive and finite")
        if (not isinstance(values["number_of_nodes"], int)
            or isinstance(values["number_of_nodes"], bool)
            or values["number_of_nodes"] < 20):
            raise ValueError("number_of_nodes must be an integer >= 20")
        if not isinstance(values["iterate_outlet_metal_angle"], bool):
            raise TypeError("iterate_outlet_metal_angle must be a bool")
        if not isinstance(values["match_real_outlet_mach"], bool):
            raise TypeError("match_real_outlet_mach must be a bool")
        if not isinstance(values["iterate_pitch_closure"], bool):
            raise TypeError("iterate_pitch_closure must be a bool")
        thickness_ratio = values["leading_edge_thickness_over_total_pitch"]
        if not math.isfinite(thickness_ratio) or not 0.0 <= thickness_ratio < 1.0:
            raise ValueError("leading_edge_thickness_over_total_pitch must be finite and in [0, 1)")
        if not isinstance(values["use_leading_edge_entry_correction"], bool):
            raise TypeError("use_leading_edge_entry_correction must be a bool")
        if values["iterate_pitch_closure"] and (
            values["iterate_outlet_metal_angle"] or values["match_real_outlet_mach"]):
            raise ValueError("iterate_pitch_closure=True is incompatible with mixed-flow angle or Mach matching")
        if values["match_real_outlet_mach"]:
            if not values["iterate_outlet_metal_angle"]:
                raise ValueError("match_real_outlet_mach=True requires iterate_outlet_metal_angle=True")
            requested_mach_name = f"requested_outlet_{flow_input_reference_frame}_flow_mach"
            if values[requested_mach_name] is None:
                raise ValueError(f"match_real_outlet_mach=True requires a {requested_mach_name} target")
        if values["boundary_layer_mode"] not in ("fully_turbulent", "laminar_then_turbulent"):
            raise ValueError("invalid boundary_layer_mode")

        initial_displacement = values["initial_turbulent_displacement_thickness"]
        initial_momentum = values["initial_turbulent_momentum_thickness"]
        if values["boundary_layer_mode"] == "fully_turbulent":
            if initial_displacement is None or initial_momentum is None:
                raise ValueError("fully_turbulent mode requires initial displacement and momentum thicknesses")
            if (not math.isfinite(initial_displacement)
                or initial_displacement <= 0.0
                or not math.isfinite(initial_momentum)
                or initial_momentum <= 0.0):
                raise ValueError("initial turbulent thicknesses must be positive and finite")
            if initial_displacement <= initial_momentum:
                raise ValueError("initial turbulent displacement thickness must exceed initial momentum thickness")
        elif initial_displacement is not None or initial_momentum is not None:
            raise ValueError("initial turbulent thicknesses are only used with boundary_layer_mode='fully_turbulent'")

        if values["mixing_solution"] not in (None, "subsonic"):
            raise ValueError("mixing_solution must be None or 'subsonic'")

        return flow_input_reference_frame

    def _select_mixing_result(self, mixing: dict[str, dict[str, float | bool]]) \
            -> tuple[MixingSolution, dict[str, float | bool]]:
        """Select one aftermixing solution for the current design trial.

        :param dict mixing: Subsonic and supersonic aftermixing results.
        :return: Selected solution name and its result dictionary.
        :rtype: tuple[MixingSolution, dict[str, float | bool]]
        """

        if self._mixing_solution_override is not None:
            solution = self._mixing_solution_override
        else:
            supersonic = mixing["supersonic"]
            ideal_outlet_relative_axial_flow_mach = float(supersonic["ideal_outlet_relative_axial_flow_mach"])
            solution = ("supersonic" if ideal_outlet_relative_axial_flow_mach >= 1.0
                        and bool(supersonic["available"]) else "subsonic")
        return solution, mixing[solution]

    def _passage_entry_conditions(self) -> tuple[float, float]:
        """Return finite-thickness MOC entrance Mach and angle.

        The external-wave construction in NACA RM L52B06, p. 11, combines

        ``Ae/Ai = (1 - t/G_total)*cos(beta_e)/cos(beta_i)``

        with the isentropic area--Mach relation and a Prandtl--Meyer
        turning compatibility relation. The NACA RM L52B06 inlet direction is
        signed opposite to this API's positive rotor-inlet angle. After
        translating that sign convention, the magnitude used here obeys

        ``beta_e = beta_i + nu_e - nu_i``.

        Of the possible mathematical intersections, the one closest to the
        far-field Mach is the NACA RM L52B06 weak-wave solution.

        :return: Passage-entry rotor-relative Mach number and signed flow angle, degrees.
        :rtype: tuple[float, float]
        :raises DesignConvergenceError: If no physical weak-wave intersection can be found.
        """

        ideal_inlet_relative_flow_mach = self.ideal_inlet_relative_flow_mach
        ideal_inlet_relative_flow_angle_rad = math.radians(self.ideal_inlet_relative_flow_angle)
        thickness_ratio = self.leading_edge_thickness_over_total_pitch
        if not self.use_leading_edge_entry_correction or thickness_ratio == 0.0:
            return ideal_inlet_relative_flow_mach, self.ideal_inlet_relative_flow_angle

        ideal_inlet_relative_axial_flow_mach = \
            ideal_inlet_relative_flow_mach * math.cos(ideal_inlet_relative_flow_angle_rad)
        if ideal_inlet_relative_axial_flow_mach >= 1.0:
            warnings.warn("the finite-leading-edge external-wave entry correction "
                "is being used with supersonic rotor-relative axial Mach "
                f"({ideal_inlet_relative_axial_flow_mach:.6g}); NACA RM L52B06 derives this method for "
                "subsonic axial inflow",
                RuntimeWarning,
                stacklevel=3)

        gamma = self.gamma
        ideal_inlet_relative_prandtl_meyer_angle = float(prandtl_meyer_angle(ideal_inlet_relative_flow_mach, gamma))
        ideal_inlet_relative_flow_area_ratio = float(isentropic_area_ratio(ideal_inlet_relative_flow_mach, gamma))

        def state(real_inlet_relative_flow_mach: float) -> tuple[float, float]:
            """Return continuity residual and passage angle for one Mach trial.

            :param float real_inlet_relative_flow_mach: Passage-entry relative flow Mach trial.
            :return: Area-continuity residual and corresponding signed angle in radians.
            :rtype: tuple[float, float]
            """

            real_inlet_relative_prandtl_meyer_angle = float(prandtl_meyer_angle(real_inlet_relative_flow_mach, gamma))
            real_inlet_relative_flow_angle_rad = (ideal_inlet_relative_flow_angle_rad
                                                  + real_inlet_relative_prandtl_meyer_angle
                                                  - ideal_inlet_relative_prandtl_meyer_angle)
            if not 0.0 < real_inlet_relative_flow_angle_rad < 0.5 * math.pi:
                return math.nan, real_inlet_relative_flow_angle_rad
            area_ratio = (float(isentropic_area_ratio(real_inlet_relative_flow_mach, gamma))
                          / ideal_inlet_relative_flow_area_ratio)
            geometric_ratio = (1.0 - thickness_ratio) * math.cos(real_inlet_relative_flow_angle_rad) \
                              / math.cos(ideal_inlet_relative_flow_angle_rad)
            return area_ratio - geometric_ratio, real_inlet_relative_flow_angle_rad

        # Locate all physical sign changes, then select the one closest to
        # Mi. This deliberately retains the small-disturbance/weak-wave root.
        samples = np.linspace(math.nextafter(1.0, math.inf), ideal_inlet_relative_flow_mach, 600)
        brackets: list[tuple[float, float]] = []
        previous_real_inlet_relative_flow_mach: float | None = None
        previous_residual: float | None = None
        for sample in samples:
            residual, _ = state(float(sample))
            if not math.isfinite(residual):
                continue
            if previous_residual is not None:
                if residual == 0.0:
                    brackets.append((float(sample), float(sample)))
                elif previous_residual * residual < 0.0:
                    brackets.append((previous_real_inlet_relative_flow_mach, float(sample)))
            previous_real_inlet_relative_flow_mach = float(sample)
            previous_residual = residual

        if not brackets:
            raise DesignConvergenceError(
                "the finite-leading-edge external-wave equations have no "
                "physical weak-wave passage-entry solution for the supplied "
                "t_LE/G_total and far-field rotor-relative inlet state")

        lower_real_inlet_relative_flow_mach, upper_real_inlet_relative_flow_mach = brackets[-1]
        if lower_real_inlet_relative_flow_mach == upper_real_inlet_relative_flow_mach:
            real_inlet_relative_flow_mach = lower_real_inlet_relative_flow_mach
        else:
            # The scan above chooses the weak-wave branch; TOMS 748 then refines only that physical intersection.
            real_inlet_relative_flow_mach = toms748(lambda mach: state(mach)[0],
                lower_real_inlet_relative_flow_mach,
                upper_real_inlet_relative_flow_mach, xtol=1.0e-13 * ideal_inlet_relative_flow_mach,
                rtol=1.0e-14, maxiter=100)
        residual, real_inlet_relative_flow_angle_rad = state(real_inlet_relative_flow_mach)
        if not math.isfinite(residual):
            raise DesignConvergenceError("finite-leading-edge entry solution left the physical flow-angle domain")
        return real_inlet_relative_flow_mach, math.degrees(real_inlet_relative_flow_angle_rad)

    def _validate_relative_design_inputs(self) -> None:
        """Check the NASA TN D-4421 domain after the velocity triangle.

        :raises ValueError: If a rotor-relative Mach, angle, or surface Mach lies outside the NASA TN D-4421 domain.
        """

        if self.ideal_inlet_relative_flow_mach <= 1.0:
            raise ValueError(
                "the calculated rotor-relative inlet Mach must be "
                "supersonic (> 1); adjust absolute inlet conditions or RPM")
        if not 0.0 < self.ideal_inlet_relative_flow_angle < 90.0:
            raise ValueError(
                "the calculated rotor-relative inlet flow angle must lie "
                "between 0 and 90 degrees; the selected RPM reverses or "
                "eliminates the required tangential relative component")
        if self.real_inlet_relative_flow_mach <= 1.0:
            raise ValueError(
                "the finite-thickness passage-entry Mach must remain supersonic (> 1) for the rotor MOC construction")
        if not 0.0 < self.real_inlet_relative_flow_angle < 90.0:
            raise ValueError("the finite-thickness passage-entry flow angle must remain between 0 and 90 degrees")
        if self.lower_surface_relative_flow_mach > self.real_inlet_relative_flow_mach:
            raise ValueError(
                "lower_surface_relative_flow_mach is outside its feasible NASA TN D-4421 range: "
                "1 <= lower_surface_relative_flow_mach <= "
                f"{self.real_inlet_relative_flow_mach:.6g} at the passage inlet")
        if self.upper_surface_relative_flow_mach < self.real_inlet_relative_flow_mach:
            raise ValueError(
                "upper_surface_relative_flow_mach is outside its feasible NASA TN D-4421 range: "
                "upper_surface_relative_flow_mach >= "
                f"{self.real_inlet_relative_flow_mach:.6g} at the passage inlet")

        # NASA TN D-4421 equations (6a) and (7a) also require the inlet transition
        # turns nu_in-nu_lower and nu_upper-nu_in not to exceed beta_in.
        # Check those limits before an optional outlet-angle search so an
        # impossible surface-Mach selection is not misreported as an angle
        # iteration failure.
        self._validate_surface_mach_ranges(
            ideal_outlet_relative_flow_mach=self.ideal_inlet_relative_flow_mach, outlet_metal_angle=None)

    def _surface_mach_ranges(self, *, ideal_outlet_relative_flow_mach: float,
        outlet_metal_angle: float | None) -> tuple[float, float, float, float]:
        """Return NASA TN D-4421 feasible lower/upper surface-Mach intervals.

        The ordering constraints come from the stated surface acceleration
        pattern. The additional bounds come directly from equations (6a-b)
        and (7a-b): no transition arc may turn through more than the
        corresponding inlet or outlet relative flow angle.

        When ``outlet_metal_angle`` is omitted, only the inlet limits and
        the broad requirement that an outlet angle below 90 degrees must
        exist are applied. This form is used before angle iteration.

        :param float ideal_outlet_relative_flow_mach: Trial ideal rotor-relative outlet Mach number.
        :param float | None outlet_metal_angle: Trial outlet metal angle, degrees.
        :return: Minimum and maximum feasible Mach numbers for the lower and upper surfaces.
        :rtype: tuple[float, float, float, float]
        """

        nu_in = float(prandtl_meyer_angle(self.real_inlet_relative_flow_mach, self.gamma))
        nu_out = float(prandtl_meyer_angle(ideal_outlet_relative_flow_mach, self.gamma))
        real_inlet_relative_flow_angle_rad = math.radians(self.real_inlet_relative_flow_angle)
        if outlet_metal_angle is None:
            ideal_outlet_relative_flow_angle_rad = math.nextafter(math.pi / 2.0, 0.0)
        else:
            ideal_outlet_relative_flow_angle_rad = abs(math.radians(outlet_metal_angle))

        lower_nu_minimum = max(0.0,
                               nu_in - real_inlet_relative_flow_angle_rad,
                               nu_out - ideal_outlet_relative_flow_angle_rad)
        lower_nu_maximum = min(nu_in, nu_out)
        upper_nu_minimum = max(nu_in, nu_out)
        upper_nu_maximum = min(nu_in + real_inlet_relative_flow_angle_rad,
                               nu_out + ideal_outlet_relative_flow_angle_rad)

        maximum_prandtl_meyer = 0.5 * math.pi * (math.sqrt((self.gamma + 1.0) / (self.gamma - 1.0)) - 1.0)
        lower_minimum = float(mach_from_prandtl_meyer(lower_nu_minimum, self.gamma))
        lower_maximum = float(mach_from_prandtl_meyer(lower_nu_maximum, self.gamma))
        upper_minimum = float(mach_from_prandtl_meyer(upper_nu_minimum, self.gamma))
        upper_maximum = (math.inf if upper_nu_maximum >= maximum_prandtl_meyer - 1.0e-12
                         else float(mach_from_prandtl_meyer(upper_nu_maximum, self.gamma)))
        return (lower_minimum, lower_maximum, upper_minimum, upper_maximum)

    def _validate_surface_mach_ranges(self, *, ideal_outlet_relative_flow_mach: float,
                                      outlet_metal_angle: float | None) -> None:
        """Raise a designer-facing error for infeasible surface Mach inputs.

        :param float ideal_outlet_relative_flow_mach: Trial ideal rotor-relative outlet Mach number.
        :param float | None outlet_metal_angle: Trial outlet metal angle, degrees.
        :raises ValueError: If either constant surface Mach lies outside the NASA TN D-4421 bounds.
        """

        lower_minimum, lower_maximum, upper_minimum, upper_maximum = self._surface_mach_ranges(
            ideal_outlet_relative_flow_mach=ideal_outlet_relative_flow_mach, outlet_metal_angle=outlet_metal_angle)
        tolerance = 1.0e-10
        angle_context = ("the inlet relative flow angle and some physical outlet metal angle"
                         if outlet_metal_angle is None
                         else (f"the inlet flow angle and outlet metal angle {outlet_metal_angle:.6g} deg"))
        if not (lower_minimum - tolerance <= self.lower_surface_relative_flow_mach <= lower_maximum + tolerance):
            raise ValueError("lower_surface_relative_flow_mach is outside the feasible NASA TN D-4421 range "
                             f"[{lower_minimum:.6g}, {lower_maximum:.6g}] for "
                             f"{angle_context}")
        if not (self.upper_surface_relative_flow_mach >= upper_minimum - tolerance
                and (math.isinf(upper_maximum) or self.upper_surface_relative_flow_mach <= upper_maximum + tolerance)):
            upper_text = "infinity" if math.isinf(upper_maximum) else f"{upper_maximum:.6g}"
            raise ValueError(
                "upper_surface_relative_flow_mach is outside the feasible NASA TN D-4421 range "
                f"[{upper_minimum:.6g}, {upper_text}] for "
                f"{angle_context}")

    def _solve_inlet_static_reference_state(self, *, initial_gamma: float) -> tuple[float, float, FluidState]:
        """Find the self-consistent actual inlet static state.

        The user supplies the stationary-frame total state and absolute Mach.
        The iteration is required because the selected mixture's ideal-gas
        Cp, and hence gamma, varies with the resulting static temperature.

        :param float initial_gamma: First heat-capacity-ratio estimate from the total state.
        :return: Static temperature, static pressure, and converged fluid state.
        :rtype: tuple[float, float, FluidState]
        :raises DesignConvergenceError: If mixture gamma does not converge.
        """

        def static_state(gamma: float) -> tuple[float, float, FluidState]:
            """Return the inlet static state implied by one heat-capacity-ratio trial."""

            temperature_factor = 1.0 + 0.5 * (gamma - 1.0) * self.ideal_inlet_absolute_flow_mach**2
            static_temperature = self.inlet_total_temperature / temperature_factor
            static_pressure = self.inlet_total_pressure / temperature_factor ** (gamma / (gamma - 1.0))
            return static_temperature, static_pressure, self.fluid.properties(static_temperature, static_pressure)

        # Accelerate the self-consistency between the isentropic state and CoolProp gamma with SciPy.
        try:
            gamma = float(fixed_point(lambda value: static_state(value)[2].gamma, initial_gamma,
                                      xtol=1.0e-12, maxiter=100))
        except RuntimeError as error:
            raise DesignConvergenceError("mixture gamma did not converge at the inlet static state") from error
        return static_state(gamma)

    def _solve_inlet_static_reference_state_from_relative(
            self, *, initial_gamma: float) -> tuple[float, float, FluidState]:
        """Find the inlet static state when the rotor-relative flow is supplied.

        Absolute stagnation temperature is retained as the thermodynamic reference. For each gamma trial, the
        absolute/relative velocity triangle and total-to-static temperature relation reduce to a quadratic in
        ``sqrt(T)``. The resulting static pressure updates the CoolProp mixture state and gamma.

        :param float initial_gamma: First heat-capacity-ratio estimate from the absolute total state.
        :return: Static temperature, static pressure, and converged fluid state.
        :rtype: tuple[float, float, FluidState]
        :raises DesignConvergenceError: If the temperature root is nonphysical or mixture gamma does not converge.
        """

        gas_constant = self.fluid.specific_gas_constant
        relative_flow_mach = self.ideal_inlet_relative_flow_mach
        relative_flow_angle_rad = math.radians(self.ideal_inlet_relative_flow_angle)

        def static_state(gamma: float) -> tuple[float, float, FluidState]:
            """Return the inlet static state implied by one heat-capacity-ratio trial."""

            gm = gamma - 1.0
            coefficient_a = 1.0 + 0.5 * gm * relative_flow_mach**2
            coefficient_b = (gm * relative_flow_mach * self.wheel_speed * math.sin(relative_flow_angle_rad)
                             / math.sqrt(gamma * gas_constant))
            coefficient_c = 0.5 * gm * self.wheel_speed**2 / (gamma * gas_constant)
            coefficient_c -= self.inlet_total_temperature
            discriminant = coefficient_b**2 - 4.0 * coefficient_a * coefficient_c
            if discriminant < 0.0:
                raise DesignConvergenceError(
                    "the supplied relative inlet state has no physical static temperature at this wheel speed")
            root_temperature = (-coefficient_b + math.sqrt(discriminant)) / (2.0 * coefficient_a)
            if root_temperature <= 0.0:
                raise DesignConvergenceError("the supplied relative inlet state gives a non-positive temperature")

            static_temperature = root_temperature**2
            sound_speed = math.sqrt(gamma * gas_constant * static_temperature)
            relative_speed = relative_flow_mach * sound_speed
            absolute_axial_velocity = relative_speed * math.cos(relative_flow_angle_rad)
            absolute_tangential_velocity = relative_speed * math.sin(relative_flow_angle_rad) + self.wheel_speed
            absolute_flow_mach = math.hypot(absolute_axial_velocity, absolute_tangential_velocity) / sound_speed
            temperature_factor = 1.0 + 0.5 * gm * absolute_flow_mach**2
            static_pressure = self.inlet_total_pressure / temperature_factor ** (gamma / gm)
            return static_temperature, static_pressure, self.fluid.properties(static_temperature, static_pressure)

        # The accelerated fixed point couples the velocity triangle, isentropic state, and CoolProp gamma.
        try:
            gamma = float(fixed_point(lambda value: static_state(value)[2].gamma, initial_gamma,
                                      xtol=1.0e-12, maxiter=100))
        except RuntimeError as error:
            raise DesignConvergenceError("mixture gamma did not converge at the inlet static state") from error
        return static_state(gamma)

    def _absolute_outlet_to_relative_angle(self, *, absolute_flow_angle: float, relative_flow_mach: float) -> float:
        """Return the relative direction represented by an absolute angle.

        The inviscid exit Mach is a relative Mach in the NASA TN D-4421 formulation.
        At fixed radius, the axial velocity is unchanged by the frame
        transformation and the wheel speed is added to the relative
        tangential component:

        ``V_x = W*cos(beta)`` and ``V_theta = W*sin(beta) + U``.

        Solving those equations for ``beta`` provides the zero-deviation metal
        angle used when outlet-angle iteration is disabled.

        :param float absolute_flow_angle: Requested stationary-frame outlet angle, degrees.
        :param float relative_flow_mach: Ideal rotor-relative outlet Mach number.
        :return: Corresponding rotor-relative outlet angle, degrees.
        :rtype: float
        :raises DesignConvergenceError: If the velocity triangle has no angle in the NASA TN D-4421 range.
        """

        temperature_factor = 1.0 + 0.5 * (self.gamma - 1.0) * relative_flow_mach**2
        static_temperature = self.relative_inlet_total_temperature / temperature_factor
        sound_speed = math.sqrt(self.gamma * self.fluid.specific_gas_constant * static_temperature)
        relative_speed = relative_flow_mach * sound_speed
        absolute_flow_angle_rad = math.radians(absolute_flow_angle)
        sine_difference = -self.wheel_speed * math.cos(absolute_flow_angle_rad) / relative_speed
        if abs(sine_difference) > 1.0 + 1.0e-12:
            raise DesignConvergenceError(
                "the requested absolute outlet angle cannot be produced by "
                "the specified relative outlet Mach and wheel speed")
        relative_flow_angle_rad = absolute_flow_angle_rad + math.asin(min(max(sine_difference, -1.0), 1.0))
        relative_flow_angle = math.degrees(relative_flow_angle_rad)
        if not -90.0 < relative_flow_angle < 0.0:
            raise DesignConvergenceError(
                "the requested absolute outlet state converts to a relative "
                "angle outside the NASA TN D-4421 geometry range (-90, 0 degrees)")
        return relative_flow_angle

    def _absolute_outlet_state_to_relative(self, *, absolute_flow_mach: float,
                                           absolute_flow_angle: float) -> dict[str, float]:
        """Transform a specified ideal absolute outlet state to the rotor.

        The input Mach fixes ``V/a`` but not the dimensional velocity because
        outlet static temperature is not an independent API input. At constant
        mean radius, relative total temperature (rothalpy) is conserved. With
        frozen gamma this makes ``sqrt(T)`` the positive root of a quadratic,
        after which the ordinary velocity triangle gives ``W``, ``M_rel``,
        and ``beta``.

        :param float absolute_flow_mach: Specified ideal stationary-frame outlet Mach number.
        :param float absolute_flow_angle: Specified stationary-frame outlet angle, degrees.
        :return: Absolute and relative state quantities required by the design.
        :rtype: dict[str, float]
        :raises DesignConvergenceError: If the rothalpy relation or velocity triangle has no physical root.
        """

        gamma = self.gamma
        gas_constant = self.fluid.specific_gas_constant
        absolute_flow_angle_rad = math.radians(absolute_flow_angle)
        temperature_coefficient = 1.0 + 0.5 * (gamma - 1.0) * absolute_flow_mach**2
        cross_coefficient = (absolute_flow_mach * self.wheel_speed * math.sin(absolute_flow_angle_rad) * (gamma - 1.0)
                             / math.sqrt(gamma * gas_constant))
        wheel_temperature = self.wheel_speed**2 * (gamma - 1.0) / (2.0 * gamma * gas_constant)
        discriminant = cross_coefficient**2 - 4.0 * temperature_coefficient * (
            wheel_temperature - self.relative_inlet_total_temperature)
        if discriminant < -1.0e-10:
            raise DesignConvergenceError(
                "the specified absolute outlet Mach and angle have no "
                "physical state at this wheel speed and relative total "
                "temperature")
        root_temperature = (cross_coefficient + math.sqrt(max(discriminant, 0.0))) / (2.0 * temperature_coefficient)
        if root_temperature <= 0.0:
            raise DesignConvergenceError("the specified absolute outlet state gives a non-positive static temperature")

        static_temperature = root_temperature**2
        sound_speed = math.sqrt(gamma * gas_constant * static_temperature)
        absolute_speed = absolute_flow_mach * sound_speed
        absolute_axial_velocity = absolute_speed * math.cos(absolute_flow_angle_rad)
        absolute_tangential_velocity = absolute_speed * math.sin(absolute_flow_angle_rad)
        relative_axial_velocity = absolute_axial_velocity
        relative_tangential_velocity = absolute_tangential_velocity - self.wheel_speed
        relative_speed = math.hypot(relative_axial_velocity, relative_tangential_velocity)
        relative_flow_angle_rad = math.atan2(relative_tangential_velocity, relative_axial_velocity)
        relative_flow_angle = math.degrees(relative_flow_angle_rad)
        if not -90.0 < relative_flow_angle < 0.0:
            raise DesignConvergenceError("the specified absolute outlet state converts to a relative "
                                         "angle outside the NASA TN D-4421 geometry range (-90, 0 degrees)")
        return {"absolute_flow_mach": absolute_flow_mach,
                "absolute_flow_angle": absolute_flow_angle,
                "relative_flow_mach": relative_speed / sound_speed,
                "relative_flow_angle": relative_flow_angle,
                "static_temperature": static_temperature,
                "sound_speed": sound_speed}

    def _ideal_outlet_relative_flow_mach_for_metal_angle(self, outlet_metal_angle: float) -> float:
        """Resolve the ideal relative outlet Mach for one metal-angle trial.

        A relative-frame target directly fixes the ideal relative Mach. For an
        absolute-frame target, changing the trial metal angle changes the ideal
        relative flow direction, velocity triangle, and relative Mach. The
        frozen-gamma rothalpy relation reduces that conversion to a quadratic
        in relative speed. The supersonic physical branch accepted by the NASA
        TN D-4421 geometry is returned.

        :param float outlet_metal_angle: Trial outlet metal angle, degrees.
        :return: Ideal rotor-relative Mach for the selected input frame.
        :rtype: float
        :raises DesignConvergenceError: If no admissible supersonic relative state exists.
        """

        if self.flow_input_reference_frame == "relative":
            if self._requested_outlet_relative_flow_mach is None:
                return self.ideal_inlet_relative_flow_mach
            return self._requested_outlet_relative_flow_mach

        if self._requested_outlet_absolute_flow_mach is None:
            return self.ideal_inlet_relative_flow_mach

        absolute_flow_mach = self._requested_outlet_absolute_flow_mach
        gamma = self.gamma
        gas_constant = self.fluid.specific_gas_constant
        outlet_metal_angle_rad = math.radians(outlet_metal_angle)
        total_sound_speed_squared = gamma * gas_constant * self.relative_inlet_total_temperature
        coefficient_a = 1.0 + 0.5 * (gamma - 1.0) * absolute_flow_mach**2
        coefficient_b = 2.0 * self.wheel_speed * math.sin(outlet_metal_angle_rad)
        coefficient_c = self.wheel_speed**2 - absolute_flow_mach**2 * total_sound_speed_squared
        discriminant = coefficient_b**2 - 4.0 * coefficient_a * coefficient_c
        if discriminant < -1.0e-10:
            raise DesignConvergenceError(
                "trial outlet metal angle has no physical relative state for the specified absolute outlet Mach")

        square_root = math.sqrt(max(discriminant, 0.0))
        relative_speed_candidates = ((-coefficient_b + square_root) / (2.0 * coefficient_a),
                                     (-coefficient_b - square_root) / (2.0 * coefficient_a))
        relative_flow_mach_candidates: list[float] = []
        for relative_speed in relative_speed_candidates:
            if relative_speed <= 0.0:
                continue
            static_temperature = self.relative_inlet_total_temperature - (gamma - 1.0) * relative_speed**2 / (
                2.0 * gamma * gas_constant)
            if static_temperature <= 0.0:
                continue
            sound_speed = math.sqrt(gamma * gas_constant * static_temperature)
            relative_flow_mach = relative_speed / sound_speed
            if (relative_flow_mach >= 1.0
                    and self.lower_surface_relative_flow_mach
                    <= relative_flow_mach
                    <= self.upper_surface_relative_flow_mach):
                relative_flow_mach_candidates.append(relative_flow_mach)

        if not relative_flow_mach_candidates:
            raise DesignConvergenceError("the specified absolute outlet Mach does not produce a "
                                         "supersonic relative outlet Mach within the selected "
                                         "surface-Mach interval at this outlet metal angle")
        return max(relative_flow_mach_candidates)

    def _relative_flow_state_to_absolute(self, *, relative_flow_mach: float,
                                         relative_flow_angle_rad: float) -> dict[str, float]:
        """Transform one rotor-relative state to the fixed frame.

        :param float relative_flow_mach: Rotor-relative Mach number.
        :param float relative_flow_angle_rad: Rotor-relative flow angle, radians.
        :return: Relative and absolute Mach numbers and flow angles.
        :rtype: dict[str, float]
        """

        temperature_factor = 1.0 + 0.5 * (self.gamma - 1.0) * relative_flow_mach**2
        static_temperature = self.relative_inlet_total_temperature / temperature_factor
        sound_speed = math.sqrt(self.gamma * self.fluid.specific_gas_constant * static_temperature)
        relative_speed = relative_flow_mach * sound_speed
        relative_axial_velocity = relative_speed * math.cos(relative_flow_angle_rad)
        relative_tangential_velocity = relative_speed * math.sin(relative_flow_angle_rad)
        absolute_axial_velocity = relative_axial_velocity
        absolute_tangential_velocity = relative_tangential_velocity + self.wheel_speed
        absolute_speed = math.hypot(absolute_axial_velocity, absolute_tangential_velocity)
        absolute_flow_angle_rad = math.atan2(absolute_tangential_velocity, absolute_axial_velocity)
        return {"absolute_flow_mach": absolute_speed / sound_speed,
                "absolute_axial_flow_mach": absolute_axial_velocity / sound_speed,
                "absolute_flow_angle": math.degrees(absolute_flow_angle_rad),
                "relative_flow_mach": relative_flow_mach,
                "relative_axial_flow_mach": (relative_axial_velocity / sound_speed),
                "relative_flow_angle": math.degrees(relative_flow_angle_rad)}

    def _physical_scale(self, ideal: BladeShape) -> _PhysicalScale:
        """Dimensionalize one ideal trial and calculate its chord Reynolds number.

        NASA TM X-2434 supplied physical chord ``XMAX`` to the boundary-layer
        program. Here ``XMAX`` is recovered from mean radius, blade count, the
        ideal nondimensional open-passage pitch, and ``t_LE/G_total`` before
        any viscous correction:

        ``G_total = 2*pi*mean_radius/blade_count``
        ``G_total* = G_passage*/(1 - t_LE/G_total)``
        ``r_star = G_total/G_total*``
        ``chord = ideal_chord*r_star``

        The chord Reynolds number uses the passage-entry static mixture state
        and rotor-relative velocity.

        :param BladeShape ideal: Trial nondimensional MOC geometry.
        :return: Dimensional pitch, thickness, chord, and Reynolds-number scale.
        :rtype: _PhysicalScale
        :raises ValueError: If the derived chord Reynolds number is not positive.
        """

        total_pitch = 2.0 * math.pi * self.mean_radius / self.blade_count
        total_pitch_star = ideal.inlet_pitch / (1.0 - self.leading_edge_thickness_over_total_pitch)
        sonic_radius = total_pitch / total_pitch_star
        passage_pitch = ideal.inlet_pitch * sonic_radius
        leading_edge_thickness = total_pitch - passage_pitch
        chord = ideal.chord * sonic_radius

        inlet_relative_velocity = self.real_inlet_relative_flow_mach * self.passage_inlet_speed_of_sound
        chord_reynolds_number = \
            inlet_relative_velocity * chord / self.passage_inlet_static_fluid_state.kinematic_viscosity
        if not math.isfinite(chord_reynolds_number) or chord_reynolds_number <= 0.0:
            raise ValueError("calculated chord Reynolds number is not positive")

        return _PhysicalScale(total_pitch=total_pitch, passage_pitch=passage_pitch,
                              leading_edge_thickness=leading_edge_thickness, sonic_radius=sonic_radius,
                              chord=chord, chord_reynolds_number=chord_reynolds_number)

    def _evaluate(self, ideal_outlet_relative_flow_mach: float, outlet_metal_angle: float) -> _RotorEvaluation:
        """Evaluate one ideal relative outlet-flow Mach and metal-angle trial.

        One evaluation includes the inviscid MOC construction, BL marches on
        its fixed-node surfaces, displacement correction, finite trailing-edge
        thickness, and both aftermixing solutions.

        :param float ideal_outlet_relative_flow_mach: Trial ideal rotor-relative outlet flow Mach number.
        :param float outlet_metal_angle: Trial outlet metal angle, degrees.
        :return: All inviscid, viscous, thickness, and mixing results for the trial.
        :rtype: _RotorEvaluation
        """

        # Optional outlet iterations can revisit a trial. Rounding prevents
        # insignificant floating-point noise from defeating the cache key.
        key = (round(float(ideal_outlet_relative_flow_mach), 10), round(float(outlet_metal_angle), 10))
        if key in self._evaluation_cache:
            return self._evaluation_cache[key]

        ideal = design_ideal_geometry(real_inlet_relative_flow_mach=self.real_inlet_relative_flow_mach,
            ideal_outlet_relative_flow_mach=ideal_outlet_relative_flow_mach,
            lower_surface_relative_flow_mach=self.lower_surface_relative_flow_mach,
            upper_surface_relative_flow_mach=self.upper_surface_relative_flow_mach,
            real_inlet_relative_flow_angle=self.real_inlet_relative_flow_angle,
            ideal_outlet_relative_flow_angle=outlet_metal_angle,
            inlet_metal_angle=self.inlet_metal_angle,
            outlet_metal_angle=outlet_metal_angle,
            number_of_nodes=self.number_of_nodes,
            gamma=self.gamma)

        # Boundary-layer blockage depends on physical chord through Reynolds
        # number.  Recalculate the scale for every outlet-angle trial because
        # its ideal chord can change even though mean radius and blade count
        # remain fixed.
        physical_scale = self._physical_scale(ideal)

        # The BL equations use thicknesses normalized by the trial chord. Convert the optional dimensional inlet
        # thicknesses once so both surfaces start from exactly the same specified state.
        initial_displacement_ratio = (None if self.initial_turbulent_displacement_thickness is None
                                      else self.initial_turbulent_displacement_thickness / physical_scale.chord)
        initial_momentum_ratio = (None if self.initial_turbulent_momentum_thickness is None
                                  else self.initial_turbulent_momentum_thickness / physical_scale.chord)
        pressure_bl_marching = solve_boundary_layer(surface=ideal.pressure_surface,
            chord=ideal.chord,
            inlet_edge_flow_mach=self.real_inlet_relative_flow_mach,
            chord_reynolds_number=physical_scale.chord_reynolds_number,
            gamma=self.gamma,
            fluid=self.fluid,
            inlet_total_temperature=self.relative_inlet_total_temperature,
            inlet_total_pressure=self.relative_inlet_total_pressure,
            mode=self.boundary_layer_mode,
            initial_turbulent_displacement_thickness_over_chord=initial_displacement_ratio,
            initial_turbulent_momentum_thickness_over_chord=initial_momentum_ratio,
            laminar_correlation_limit=0.50)
        suction_bl_marching = solve_boundary_layer(surface=ideal.suction_surface,
            chord=ideal.chord,
            inlet_edge_flow_mach=self.real_inlet_relative_flow_mach,
            chord_reynolds_number=physical_scale.chord_reynolds_number,
            gamma=self.gamma,
            fluid=self.fluid,
            inlet_total_temperature=self.relative_inlet_total_temperature,
            inlet_total_pressure=self.relative_inlet_total_pressure,
            mode=self.boundary_layer_mode,
            initial_turbulent_displacement_thickness_over_chord=initial_displacement_ratio,
            initial_turbulent_momentum_thickness_over_chord=initial_momentum_ratio,
            laminar_correlation_limit=0.50)

        pressure_bl = pressure_bl_marching
        suction_bl = suction_bl_marching
        corrected = self._correct_shape(ideal, pressure_bl, suction_bl)
        leading_edge_thickness = (self.leading_edge_thickness_over_total_pitch
                                  / (1.0 - self.leading_edge_thickness_over_total_pitch) * ideal.inlet_pitch)
        trailing_edge_vertical_boundary_layer_height = \
            abs(corrected.pressure_surface.y[-1] - ideal.pressure_surface.y[-1]) \
            + abs(corrected.suction_surface.y[-1] - ideal.suction_surface.y[-1])
        # In legacy pitch closure, NASA TM X-2434 carries the leading-edge thickness
        # through to the trailing edge. Without closure, the two vertical BL
        # offsets consume part of that metal thickness in the corrected plot.
        if self.iterate_pitch_closure:
            trailing_edge_thickness = leading_edge_thickness
        else:
            trailing_edge_thickness = max(0.0, leading_edge_thickness - trailing_edge_vertical_boundary_layer_height)
        pitch_residual = corrected.outlet_pitch - corrected.inlet_pitch
        mixing = self._aftermixing(ideal_outlet_relative_flow_mach=ideal_outlet_relative_flow_mach,
            ideal_outlet_relative_flow_angle=outlet_metal_angle,
            ideal=ideal,
            corrected=corrected,
            pressure_bl=pressure_bl,
            suction_bl=suction_bl,
            trailing_edge_thickness=trailing_edge_thickness)
        result = _RotorEvaluation(ideal=ideal, corrected=corrected,
            pressure_boundary_layer=pressure_bl,
            suction_boundary_layer=suction_bl,
            pressure_boundary_layer_marching=pressure_bl_marching,
            suction_boundary_layer_marching=suction_bl_marching,
            leading_edge_thickness=leading_edge_thickness,
            trailing_edge_thickness=trailing_edge_thickness,
            trailing_edge_vertical_boundary_layer_height=trailing_edge_vertical_boundary_layer_height,
            pitch_residual=pitch_residual,
            mixing=mixing)
        self._evaluation_cache[key] = result
        return result

    @staticmethod
    def _correct_shape(ideal: BladeShape, pressure_bl: BoundaryLayerResult,
                       suction_bl: BoundaryLayerResult) -> BladeShape:
        """Offset both MOC surfaces by the calculated displacement thickness.

        The BL solver reports a thickness normal to the local flow direction.
        ``NOZZLC``-style geometry correction needs its vertical component, so
        the offset is divided by the local tangent cosine. The pressure and
        suction surfaces move away from the open passage in opposite directions.

        :param BladeShape ideal: Inviscid MOC geometry.
        :param BoundaryLayerResult pressure_bl: Pressure-side BL at MOC stations.
        :param BoundaryLayerResult suction_bl: Suction-side BL at MOC stations.
        :return: BL-corrected passage with recalculated inlet and outlet pitch.
        :rtype: BladeShape
        """

        pressure_offset = pressure_bl.displacement_thickness_over_chord * ideal.chord
        suction_offset = suction_bl.displacement_thickness_over_chord * ideal.chord
        pressure_cosine = np.maximum(np.abs(np.cos(np.radians(ideal.pressure_surface.metal_angle))), 1.0e-6)
        suction_cosine = np.maximum(np.abs(np.cos(np.radians(ideal.suction_surface.metal_angle))), 1.0e-6)
        pressure = SurfaceCoordinates(x=ideal.pressure_surface.x.copy(),
            y=ideal.pressure_surface.y + np.abs(pressure_offset / pressure_cosine),
            relative_flow_mach=ideal.pressure_surface.relative_flow_mach.copy(),
            metal_angle=ideal.pressure_surface.metal_angle.copy())
        suction = SurfaceCoordinates(x=ideal.suction_surface.x.copy(),
            y=ideal.suction_surface.y - np.abs(suction_offset / suction_cosine),
            relative_flow_mach=ideal.suction_surface.relative_flow_mach.copy(),
            metal_angle=ideal.suction_surface.metal_angle.copy())

        inlet_metal_angle_rad = math.radians(float(ideal.suction_surface.metal_angle[0]))
        # Tangent arrays store magnitudes; the outlet direction is negative.
        outlet_metal_angle_rad = math.radians(-abs(float(ideal.suction_surface.metal_angle[-1])))
        inlet_pitch = pressure.y[0] - (suction.y[0] + math.tan(inlet_metal_angle_rad)
                                       * (pressure.x[0] - suction.x[0]))
        outlet_pitch = pressure.y[-1] - (suction.y[-1] + math.tan(outlet_metal_angle_rad)
                                         * (pressure.x[-1] - suction.x[-1]))
        return BladeShape(pressure_surface=pressure,
                          suction_surface=suction,
                          chord=ideal.chord,
                          inlet_pitch=float(inlet_pitch),
                          outlet_pitch=float(outlet_pitch),
                          max_flow_turning_increment=ideal.max_flow_turning_increment,
                          coordinate_scale=ideal.coordinate_scale)

    def _solve_outlet_metal_angle_for_pitch_closure(self, *, initial_outlet_metal_angle: float,
                                                    ideal_outlet_relative_flow_mach: float) -> float:
        """Reproduce the NASA TM X-2434 ``BETAT`` pitch-closure iteration.

        The first unbracketed update comes from the NASA TM X-2434 continuity
        relation with outlet displacement blockage. Once trials exist on
        both sides of equal pitch, SciPy's Brent scheme refines the angle.
        NASA TM X-2434 uses a tolerance of 0.0001 physical length unit. This
        API applies a tighter SI tolerance of 0.000001 m.

        :param float initial_outlet_metal_angle: Initial outlet metal-angle estimate, degrees.
        :param float ideal_outlet_relative_flow_mach: Ideal relative outlet Mach held during pitch closure.
        :return: Outlet metal angle giving equal corrected outlet and ideal inlet pitch.
        :rtype: float
        :raises DesignConvergenceError: If the NASA TM X-2434 iteration stagnates or leaves its physical range.
        """

        outlet_metal_angle = float(initial_outlet_metal_angle)
        angle_above_target: float | None = None
        angle_below_target: float | None = None
        inlet_mass_flow = float(mass_flow_parameter(self.real_inlet_relative_flow_mach, self.gamma))
        outlet_mass_flow = float(mass_flow_parameter(ideal_outlet_relative_flow_mach, self.gamma))
        evaluation_count = 0
        evaluation: _RotorEvaluation | None = None

        def pitch_residual(angle: float) -> float:
            """Return corrected outlet pitch minus ideal inlet pitch for one angle trial."""

            nonlocal evaluation_count, evaluation
            self._validate_surface_mach_ranges(
                ideal_outlet_relative_flow_mach=ideal_outlet_relative_flow_mach, outlet_metal_angle=angle)
            evaluation = self._evaluate(ideal_outlet_relative_flow_mach, angle)
            evaluation_count += 1
            return evaluation.corrected.outlet_pitch - evaluation.ideal.inlet_pitch

        # Use the NASA continuity update only to discover the second side of the physical pitch-closure bracket.
        for _ in range(50):
            residual = pitch_residual(outlet_metal_angle)
            physical_scale = self._physical_scale(evaluation.ideal)
            pitch_tolerance = 1.0e-6 * evaluation.ideal.chord / physical_scale.chord
            if abs(residual) <= pitch_tolerance:
                self.pitch_closure_iteration_count = evaluation_count
                return outlet_metal_angle

            if residual >= 0.0:
                angle_above_target = outlet_metal_angle
            else:
                angle_below_target = outlet_metal_angle

            if angle_above_target is not None and angle_below_target is not None:
                break

            # The continuity update uses the combined pressure- and suction-side exit blockage.
            pressure_displacement = evaluation.pressure_boundary_layer.displacement_thickness_over_chord[-1]
            suction_displacement = evaluation.suction_boundary_layer.displacement_thickness_over_chord[-1]
            total_displacement = (pressure_displacement + suction_displacement) * evaluation.ideal.chord
            cosine_argument = math.cos(math.radians(self.real_inlet_relative_flow_angle)) \
                              * inlet_mass_flow / outlet_mass_flow \
                              + total_displacement / evaluation.ideal.inlet_pitch
            if not -1.0 <= cosine_argument <= 1.0:
                raise DesignConvergenceError(
                    "legacy pitch closure produced no physical outlet angle from the continuity update")
            candidate = -math.degrees(math.acos(cosine_argument))

            if not -90.0 < candidate < 0.0:
                raise DesignConvergenceError(
                    "legacy pitch closure moved the outlet metal angle outside (-90, 0) degrees")
            if abs(candidate - outlet_metal_angle) <= 1.0e-12:
                raise DesignConvergenceError("legacy pitch closure stagnated before equal spacing")
            outlet_metal_angle = candidate
        else:
            raise DesignConvergenceError("pitch closure could not bracket equal spacing within 50 evaluations")

        lower_angle, upper_angle = sorted((angle_above_target, angle_below_target))
        try:
            outlet_metal_angle = brentq(pitch_residual, lower_angle, upper_angle, xtol=1.0e-12,
                                        rtol=1.0e-14, maxiter=50)
        except (RuntimeError, ValueError) as error:
            raise DesignConvergenceError("pitch closure did not converge within the physical angle bracket") from error

        residual = pitch_residual(outlet_metal_angle)
        physical_scale = self._physical_scale(evaluation.ideal)
        pitch_tolerance = 1.0e-6 * evaluation.ideal.chord / physical_scale.chord
        if abs(residual) > pitch_tolerance:
            raise DesignConvergenceError(
                "pitch closure angle converged without satisfying the physical pitch tolerance")
        self.pitch_closure_iteration_count = evaluation_count
        return outlet_metal_angle

    def _flow_residual_for_outlet_metal_angle(self, outlet_metal_angle: float, *,
        ideal_outlet_relative_flow_mach: float | None = None) -> float:
        """Return selected mixed-flow angle minus the requested angle.

        :param float outlet_metal_angle: Trial outlet metal angle, degrees.
        :param float | None ideal_outlet_relative_flow_mach: Fixed ideal relative outlet Mach, or ``None`` to derive it
            from the requested outlet Mach in the selected input frame.
        :return: Outlet flow-angle residual in the selected input frame, degrees.
        :rtype: float
        :raises DesignConvergenceError: If the selected aftermixing solution is unavailable.
        """

        if ideal_outlet_relative_flow_mach is None:
            ideal_outlet_relative_flow_mach = self._ideal_outlet_relative_flow_mach_for_metal_angle(outlet_metal_angle)
        mixing = self._evaluate(ideal_outlet_relative_flow_mach, outlet_metal_angle).mixing
        _, selected = self._select_mixing_result(mixing)
        if not bool(selected["available"]):
            raise DesignConvergenceError("selected aftermixing solution is unavailable")
        frame = self.flow_input_reference_frame
        selected_angle = float(selected[f"real_outlet_{frame}_flow_angle"])
        requested_angle = float(getattr(self, f"requested_outlet_{frame}_flow_angle"))
        return selected_angle - requested_angle

    def _solve_outlet_metal_angle_for_target_flow(
            self, *, ideal_outlet_relative_flow_mach: float | None = None) -> float:
        """Find the metal angle that matches the requested mixed outlet direction.

        SciPy's bounded nonlinear least-squares solver starts from the
        requested relative outlet direction and follows the fixed-mesh
        residual.

        :param float | None ideal_outlet_relative_flow_mach: Fixed ideal relative Mach, or ``None`` to recompute it for
            each angle from the requested outlet Mach in the selected input frame.
        :return: Converged outlet metal angle, degrees.
        :rtype: float
        :raises DesignConvergenceError: If the requested mixed angle is unattainable.
        """

        angle_tolerance = 1.0e-4

        def scaled_residual(values: np.ndarray) -> np.ndarray:
            """Return the angle residual scaled by its convergence tolerance."""

            return np.asarray([self._flow_residual_for_outlet_metal_angle(float(values[0]),
                              ideal_outlet_relative_flow_mach=ideal_outlet_relative_flow_mach) / angle_tolerance],
                              dtype=float)

        initial = np.asarray([float(np.clip(self.requested_outlet_relative_flow_angle, -88.5, -1.5))], dtype=float)
        try:
            solution = least_squares(scaled_residual, initial, bounds=([-88.5], [-1.5]), diff_step=5.0e-3,
                                     xtol=1.0e-10, ftol=1.0e-10, gtol=1.0e-10, max_nfev=60)
            outlet_metal_angle = float(solution.x[0])
            final_residual = self._flow_residual_for_outlet_metal_angle(outlet_metal_angle,
                ideal_outlet_relative_flow_mach=ideal_outlet_relative_flow_mach)
        except (GeometryError, BoundaryLayerError, DesignConvergenceError, ValueError, OverflowError) as error:
            raise DesignConvergenceError("rotor outlet flow-angle solve encountered an infeasible trial") from error
        if solution.success and abs(final_residual) <= angle_tolerance:
            return outlet_metal_angle
        raise DesignConvergenceError(
            f"target mixed outlet angle did not converge; final residual was {final_residual:.4f} deg")

    def _solve_outlet_metal_angle_and_flow_mach_targets(self) -> tuple[float, float]:
        """Match mixed outlet angle and Mach in the selected input frame.

        SciPy's bounded nonlinear least-squares solver varies outlet metal
        angle and ideal relative outlet Mach together. Thus the two mixed-flow
        constraints are paired with two independent construction variables
        without nesting two expensive global searches.

        :return: Outlet metal angle in degrees and ideal relative outlet Mach.
        :rtype: tuple[float, float]
        :raises DesignConvergenceError: If no physical coupled solution converges.
        """

        frame = self.flow_input_reference_frame
        target_real_outlet_flow_mach = getattr(self, f"_requested_outlet_{frame}_flow_mach")
        if target_real_outlet_flow_mach is None:
            raise DesignConvergenceError("a mixed outlet Mach target was not supplied")
        target_real_outlet_flow_mach = float(target_real_outlet_flow_mach)
        target_real_outlet_flow_angle = float(getattr(self, f"requested_outlet_{frame}_flow_angle"))
        lower_ideal_outlet_relative_flow_mach = max(1.0 + 1.0e-5, self.lower_surface_relative_flow_mach + 1.0e-5)
        upper_ideal_outlet_relative_flow_mach = self.upper_surface_relative_flow_mach - 1.0e-5
        if upper_ideal_outlet_relative_flow_mach <= lower_ideal_outlet_relative_flow_mach:
            raise DesignConvergenceError(
                "surface-Mach interval leaves no relative outlet Mach available for the coupled outlet solve")

        # Treating the requested mixed state temporarily as an ideal state gives
        # a physically informed initial trial. BL mixing corrections are normally
        # small enough for this to lie in the basin of the coupled solution.
        if frame == "absolute":
            initial_state = self._absolute_outlet_state_to_relative(absolute_flow_mach=target_real_outlet_flow_mach,
                absolute_flow_angle=target_real_outlet_flow_angle)
            initial_outlet_metal_angle = initial_state["relative_flow_angle"]
            initial_ideal_outlet_relative_flow_mach = initial_state["relative_flow_mach"]
        else:
            initial_outlet_metal_angle = target_real_outlet_flow_angle
            initial_ideal_outlet_relative_flow_mach = target_real_outlet_flow_mach
        variables = np.asarray([min(max(initial_outlet_metal_angle, -88.5), -1.5),
                                min(max(initial_ideal_outlet_relative_flow_mach,
                                        lower_ideal_outlet_relative_flow_mach),
                                    upper_ideal_outlet_relative_flow_mach)], dtype=float)

        def residual(values: np.ndarray) -> np.ndarray:
            """Evaluate angle and Mach residuals for one solver trial.

            :param numpy.ndarray values: Relative metal angle and ideal relative Mach trial.
            :return: Mixed-flow angle and Mach residuals in the selected input frame.
            :rtype: numpy.ndarray
            """

            outlet_metal_angle = float(values[0])
            ideal_outlet_relative_flow_mach = float(values[1])
            evaluation = self._evaluate(ideal_outlet_relative_flow_mach, outlet_metal_angle)
            _, selected = self._select_mixing_result(evaluation.mixing)
            if not bool(selected["available"]):
                raise DesignConvergenceError("selected aftermixing solution is unavailable")
            result = np.asarray([
                float(selected[f"real_outlet_{frame}_flow_angle"]) - target_real_outlet_flow_angle,
                float(selected[f"real_outlet_{frame}_flow_mach"]) - target_real_outlet_flow_mach], dtype=float)
            if not np.all(np.isfinite(result)):
                raise DesignConvergenceError("selected aftermixing solution is not physical")
            return result

        angle_tolerance = 2.0e-3
        mach_tolerance = 1.0e-4

        def scaled_residual(values: np.ndarray) -> np.ndarray:
            """Scale both residuals by their independently checked tolerances."""

            result = residual(values)
            return result / np.asarray([angle_tolerance, mach_tolerance], dtype=float)

        try:
            solution = least_squares(scaled_residual, variables,
                                     bounds=([-88.5, lower_ideal_outlet_relative_flow_mach],
                                             [-1.5, upper_ideal_outlet_relative_flow_mach]),
                                     diff_step=5.0e-3, x_scale=np.asarray([10.0, 0.5], dtype=float),
                                     xtol=1.0e-10, ftol=1.0e-10, gtol=1.0e-10, max_nfev=100)
            final_residual = residual(solution.x)
        except (GeometryError, BoundaryLayerError, DesignConvergenceError, ValueError, OverflowError) as error:
            raise DesignConvergenceError("coupled rotor outlet solve encountered an infeasible trial; "
                                         "adjust surface Mach numbers or the requested outlet state") from error
        if (solution.success and abs(float(final_residual[0])) <= angle_tolerance
                and abs(float(final_residual[1])) <= mach_tolerance):
            return float(solution.x[0]), float(solution.x[1])
        raise DesignConvergenceError(
            "coupled outlet angle/Mach solve did not converge; final "
            f"{frame} angle residual={final_residual[0]:.6g} deg and "
            f"Mach residual={final_residual[1]:.6g}")

    def _aftermixing(self, *, ideal_outlet_relative_flow_mach: float,
        ideal_outlet_relative_flow_angle: float,
        ideal: BladeShape,
        corrected: BladeShape,
        pressure_bl: BoundaryLayerResult,
        suction_bl: BoundaryLayerResult,
        trailing_edge_thickness: float) -> dict[str, dict[str, float | bool]]:
        """Evaluate NASA TM X-2434 ``AFMIX`` with finite trailing-edge blockage.

        :param float ideal_outlet_relative_flow_mach: Ideal premixing rotor-relative outlet flow Mach.
        :param float ideal_outlet_relative_flow_angle: Ideal premixing rotor-relative flow angle, degrees.
        :param BladeShape ideal: Inviscid geometry providing the chord scale.
        :param BladeShape corrected: BL-corrected geometry providing outlet pitch.
        :param BoundaryLayerResult pressure_bl: Pressure-side BL result.
        :param BoundaryLayerResult suction_bl: Suction-side BL result.
        :param float trailing_edge_thickness: Nondimensional finite trailing-edge thickness.
        :return: Subsonic and supersonic mixing solutions with relative and absolute quantities.
        :rtype: dict[str, dict[str, float | bool]]
        :raises BoundaryLayerError: If blockage closes the passage or the conservation equation has no real solution.
        """

        gamma = self.gamma
        gp = gamma + 1.0
        gm = gamma - 1.0
        relative_flow_angle_rad = math.radians(ideal_outlet_relative_flow_angle)
        velocity_ratio = math.sqrt((0.5 * gp * ideal_outlet_relative_flow_mach**2)\
                                   / (1.0 + 0.5 * gm * ideal_outlet_relative_flow_mach**2))
        spacing = corrected.outlet_pitch / ideal.chord
        projected_spacing = spacing * math.cos(relative_flow_angle_rad)
        if projected_spacing <= 0.0:
            raise BoundaryLayerError("non-positive projected outlet spacing")
        delta_sum = pressure_bl.displacement_thickness_over_chord[-1] + suction_bl.displacement_thickness_over_chord[-1]
        theta_sum = pressure_bl.momentum_thickness_over_chord[-1] + suction_bl.momentum_thickness_over_chord[-1]
        displacement_ratio = delta_sum / projected_spacing
        momentum_ratio = theta_sum / projected_spacing
        # Metal thickness occupies the same projected exit area as the BL
        # displacement thickness and therefore enters both AFMIX area factors.
        trailing_edge_blockage_ratio = trailing_edge_thickness / ideal.chord / projected_spacing
        area_momentum = 1.0 - displacement_ratio - trailing_edge_blockage_ratio - momentum_ratio
        area = 1.0 - displacement_ratio - trailing_edge_blockage_ratio
        if area_momentum <= 0.0 or area <= 0.0:
            raise BoundaryLayerError("boundary-layer blockage closes the outlet")
        afs = gm / gp * velocity_ratio**2
        c_value = ((1.0 - afs) * gp / (2.0 * gamma)
                   + math.cos(relative_flow_angle_rad) ** 2 * area_momentum * velocity_ratio**2)\
                  / (math.cos(relative_flow_angle_rad) * area * velocity_ratio)
        d_value = velocity_ratio * math.sin(relative_flow_angle_rad) * area_momentum / area
        radical = (gamma * c_value / gp) ** 2 - 1.0 + gm / gp * d_value**2
        if radical < -1.0e-10:
            raise BoundaryLayerError("aftermixing equation has no real solution")
        square_root = math.sqrt(max(radical, 0.0))
        ideal_outlet_relative_axial_flow_mach = ideal_outlet_relative_flow_mach * math.cos(relative_flow_angle_rad)

        results: dict[str, dict[str, float | bool]] = {}
        for name, axial_velocity_ratio in (
            ("subsonic", gamma * c_value / gp - square_root),
            ("supersonic", gamma * c_value / gp + square_root)):
            available = name == "subsonic" or ideal_outlet_relative_axial_flow_mach >= 1.0 - 1.0e-12
            total_velocity_ratio = math.hypot(d_value, axial_velocity_ratio)
            denominator = 1.0 - gm / gp * total_velocity_ratio**2
            if not available or denominator <= 0.0 or axial_velocity_ratio <= 0.0:
                results[name] = {
                    "available": False,
                    "ideal_outlet_relative_axial_flow_mach": ideal_outlet_relative_axial_flow_mach,
                    "trailing_edge_blockage_ratio": (trailing_edge_blockage_ratio),
                    "real_outlet_absolute_flow_mach": math.nan,
                    "real_outlet_absolute_axial_flow_mach": math.nan,
                    "real_outlet_absolute_flow_angle": math.nan,
                    "real_outlet_relative_flow_mach": math.nan,
                    "real_outlet_relative_axial_flow_mach": math.nan,
                    "real_outlet_relative_flow_angle": math.nan}
                continue
            real_outlet_relative_flow_mach = math.sqrt((2.0 / gp * total_velocity_ratio**2) / denominator)
            real_outlet_relative_flow_angle_rad = math.atan2(d_value, axial_velocity_ratio)
            mixed_state = self._relative_flow_state_to_absolute(relative_flow_mach=real_outlet_relative_flow_mach,
                relative_flow_angle_rad=real_outlet_relative_flow_angle_rad)
            results[name] = {
                "available": True,
                "ideal_outlet_relative_axial_flow_mach": ideal_outlet_relative_axial_flow_mach,
                "trailing_edge_blockage_ratio": trailing_edge_blockage_ratio,
                "real_outlet_absolute_flow_mach": mixed_state["absolute_flow_mach"],
                "real_outlet_absolute_axial_flow_mach": mixed_state["absolute_axial_flow_mach"],
                "real_outlet_absolute_flow_angle": mixed_state["absolute_flow_angle"],
                "real_outlet_relative_flow_mach": mixed_state["relative_flow_mach"],
                "real_outlet_relative_axial_flow_mach": mixed_state["relative_axial_flow_mach"],
                "real_outlet_relative_flow_angle": mixed_state["relative_flow_angle"]}
        return results

    def _assemble_cad_profile(self, shape: BladeShape) -> tuple[np.ndarray, np.ndarray]:
        """Assemble one dimensional single-blade profile for CAD import.

        The stored pressure and suction surfaces bound opposite sides of one
        passage. A single top blade is therefore formed from the stored
        pressure surface and the periodically translated suction surface, as
        in :meth:`plot`. The lower-surface leading edge defines the origin.

        :param BladeShape shape: Corrected or ideal nondimensional passage geometry.
        :return: x- and y-coordinate arrays in millimetres. The lower
            surface runs from leading to trailing edge, followed by the upper
            surface in reverse order from trailing to leading edge. The leading
            edge is not closed by repeating the first point.
        :rtype: tuple[numpy.ndarray, numpy.ndarray]
        """

        # Scale the selected ideal or BL-corrected geometry to metres before
        # assembling the profile so both exports use exactly the same physical
        # rotor scale. The returned arrays are converted to millimetres below.
        shape = shape.scaled(self.sonic_radius_scale, "dimensional [m]")
        translation_x = shape.pressure_surface.x[0] - shape.suction_surface.x[0]
        translation_y = shape.pressure_surface.y[0] - shape.suction_surface.y[0]

        # The pressure surface is the lower side of the top blade. Translate
        # the suction surface to form its upper side, then add the leading-edge
        # metal thickness without moving the passage boundary.
        lower_x = shape.pressure_surface.x - shape.pressure_surface.x[0]
        lower_y = shape.pressure_surface.y - shape.pressure_surface.y[0]
        upper_x = shape.suction_surface.x + translation_x - shape.pressure_surface.x[0]
        upper_y = (shape.suction_surface.y + translation_y + self.physical_leading_edge_thickness
            - shape.pressure_surface.y[0])

        # Reverse the upper surface so consecutive points trace one profile.
        # Adjacent trailing-edge endpoints are joined by the imported CAD
        # polyline. Do not append the lower leading edge again: finite-thickness
        # profiles end at the upper leading edge, while a coincident upper
        # leading-edge point is omitted for zero thickness. Consequently,
        # exactly one point in the exported arrays has coordinates (0, 0).
        upper_x_reversed = upper_x[::-1]
        upper_y_reversed = upper_y[::-1]
        if self.physical_leading_edge_thickness == 0.0:
            upper_x_reversed = upper_x_reversed[:-1]
            upper_y_reversed = upper_y_reversed[:-1]
        profile_x = np.concatenate((lower_x, upper_x_reversed))
        profile_y = np.concatenate((lower_y, upper_y_reversed))

        return 1000.0 * profile_x, 1000.0 * profile_y

    def dimensionalize(self) -> DimensionalBladeShapes:
        """Scale both shapes using the initialized mean radius and blade count.

        The NASA TN D-4421 coordinate divisor is the vortex sonic radius ``r*``.
        The same ``r*`` used to calculate physical chord and Reynolds number
        during initialization is reused here, preventing scale inconsistency.

        :return: Ideal and corrected surfaces scaled to metres.
        :rtype: DimensionalBladeShapes
        """

        result = DimensionalBladeShapes(mean_radius=self.mean_radius,
            blade_count=self.blade_count,
            sonic_radius_scale=self.sonic_radius_scale,
            uncorrected=self.uncorrected_shape.scaled(self.sonic_radius_scale, "dimensional"),
            corrected=self.corrected_shape.scaled(self.sonic_radius_scale, "dimensional"))
        self.dimensional_shapes = result
        return result

    def plot(self, *, dimensional: bool = False,
        corrected: bool = True,
        show_two_blades: bool = True,
        ax=None,
        show: bool = True):
        """Plot the passage and two complete adjacent rotor blades.

        If ``dimensional=True``, the mean radius supplied during initialization
        is used and the plotted coordinates are expressed in millimetres.
        ``corrected=True`` selects the BL-corrected shape, while
        ``corrected=False`` selects the ideal uncorrected shape. The two stored
        surfaces bound one passage: pressure is its upper boundary and suction
        is its lower boundary. With ``show_two_blades=True``, a translated
        suction surface completes the upper blade and a translated pressure
        surface completes the lower blade. The upper blade's outer surface is
        shifted upward by ``t_LE`` and the lower blade's outer surface is
        shifted downward by ``t_LE``; the two central passage boundaries are
        unchanged. A straight segment joins the two leading-edge endpoints of
        each blade, so finite leading-edge thickness is visible as a closed
        profile rather than a gap. A second segment joins the two trailing-edge
        endpoints of each blade, including designs without legacy pitch
        closure. Edge closures use the selected geometry's solid BL-corrected
        or dashed uncorrected line style. The created Matplotlib ``(figure, ax)``
        pair is returned.

        :param bool dimensional: Plot in millimetres instead of NASA TN D-4421 coordinates divided by ``r*``.
        :param bool corrected: Select the BL-corrected rather than ideal geometry.
        :param bool show_two_blades: Complete the two blades surrounding the stored passage.
        :param ax: Existing Matplotlib axes, or ``None`` to create a figure.
        :type ax: matplotlib.axes.Axes | None
        :param bool show: Call ``matplotlib.pyplot.show`` before returning.
        :return: Matplotlib figure and axes.
        :rtype: tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        :raises TypeError: If ``corrected`` is not boolean.
        :raises ImportError: If Matplotlib is unavailable.
        """

        if not isinstance(corrected, bool):
            raise TypeError("corrected must be a bool")
        try:
            import matplotlib.pyplot as plt
        except ImportError as error:
            raise ImportError("plotting requires matplotlib; install the project dependencies") from error

        if dimensional:
            shapes = self.dimensional_shapes if self.dimensional_shapes is not None else self.dimensionalize()
            dimensional_shape = shapes.corrected if corrected else shapes.uncorrected
            # Keep the stored dimensional geometry in metres and scale only
            # the local plotting copy. This preserves the public physical
            # dimensions while presenting engineering drawings in millimetres.
            shape = dimensional_shape.scaled(1000.0, "dimensional [mm]")
            leading_edge_thickness = 1000.0 * self.physical_leading_edge_thickness
            axis_label = "length [mm]"
        else:
            shape = self.corrected_shape if corrected else self.uncorrected_shape
            leading_edge_thickness = self.leading_edge_thickness
            axis_label = r"coordinate / $r^*$"

        if ax is None:
            figure, ax = plt.subplots()
        else:
            figure = ax.figure

        def plot_shape(shape: BladeShape, *, linestyle: str, color: str, linewidth: float, label: str) -> None:
            """Draw the selected passage or the four visible blade surfaces.

            :param BladeShape shape: Ideal or corrected blade passage.
            :param str linestyle: Matplotlib line style.
            :param str color: Matplotlib line color.
            :param float linewidth: Line width in points.
            :param str label: Legend label applied to the first surface.
            """

            if show_two_blades:
                # The stored pressure and suction surfaces are opposite
                # boundaries of the central passage, not the two sides of one
                # blade. Their leading-edge separation is the periodic
                # translation vector. Cross-pairing the surface types produces
                # the physical order:
                #
                #   shifted suction -- upper blade upper surface
                #   pressure        -- upper blade lower surface
                #                    -- central flow passage --
                #   suction         -- lower blade upper surface
                #   shifted pressure-- lower blade lower surface
                #
                # Calculate the passage translation from coordinates instead
                # of reusing a scalar pitch. The extra outer-surface offsets
                # add the specified leading-edge metal without moving either
                # central passage boundary.
                translation_x = shape.pressure_surface.x[0] - shape.suction_surface.x[0]
                translation_y = shape.pressure_surface.y[0] - shape.suction_surface.y[0]
                surfaces = ((shape.suction_surface, translation_x, translation_y + leading_edge_thickness),
                    (shape.pressure_surface, 0.0, 0.0),
                    (shape.suction_surface, 0.0, 0.0),
                    (shape.pressure_surface, -translation_x, -translation_y - leading_edge_thickness))
            else:
                # Retain the concise passage-only view when requested.
                surfaces = ((shape.pressure_surface, 0.0, 0.0), (shape.suction_surface, 0.0, 0.0))
            for index, (surface, x_offset, y_offset) in enumerate(surfaces):
                ax.plot(surface.x + x_offset,
                    surface.y + y_offset,
                    linestyle,
                    color=color,
                    linewidth=linewidth,
                    label=label if index == 0 else None)

            if show_two_blades:
                # Close each finite-thickness leading edge independently. The
                # first pair belongs to the upper blade and the second pair to
                # the lower blade; no line is drawn across the flow passage.
                for first_index, second_index in ((0, 1), (2, 3)):
                    first_surface, first_x_offset, first_y_offset = surfaces[first_index]
                    second_surface, second_x_offset, second_y_offset = surfaces[second_index]
                    ax.plot([first_surface.x[0] + first_x_offset, second_surface.x[0] + second_x_offset],
                        [first_surface.y[0] + first_y_offset, second_surface.y[0] + second_y_offset],
                        linestyle,
                        color=color,
                        linewidth=linewidth)

                # Close both trailing edges even when legacy pitch closure is
                # disabled. This makes each plotted blade a complete profile;
                # the connector spans the actual separation produced by the
                # selected ideal or BL-corrected surface coordinates.
                for first_index, second_index in ((0, 1), (2, 3)):
                    first_surface, first_x_offset, first_y_offset = surfaces[first_index]
                    second_surface, second_x_offset, second_y_offset = surfaces[second_index]
                    ax.plot([first_surface.x[-1] + first_x_offset, second_surface.x[-1] + second_x_offset],
                        [first_surface.y[-1] + first_y_offset, second_surface.y[-1] + second_y_offset],
                        linestyle,
                        color=color,
                        linewidth=linewidth)

        if corrected:
            plot_shape(shape, linestyle="-", color="#0068b5", linewidth=1.8, label="BL corrected")
        else:
            plot_shape(shape, linestyle="--", color="0.35", linewidth=1.4, label="uncorrected")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel(axis_label)
        ax.set_ylabel(axis_label)
        ax.grid(True, alpha=0.25)
        ax.legend()
        figure.tight_layout()
        if show:
            plt.show()
        return figure, ax
