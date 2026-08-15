"""Object-oriented two-dimensional supersonic stator-nozzle design."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import least_squares

from ..boundary_layer.boundary_layer_solver import (
    BoundaryLayerError,
    BoundaryLayerMode,
    solve_boundary_layer,
)
from ..common_results import BoundaryLayerResult, SurfaceCoordinates
from ..fluid import Fluid, FluidState
from ..gas_dynamics import isentropic_area_ratio
from .stator_geometry import (
    ContourMethod,
    IdealNozzleConstruction,
    StatorGeometryError,
    design_conical_stator_nozzle,
    design_ideal_stator_nozzle,
)
from .stator_results import DimensionalNozzleShapes, NozzleShape

MixingSolution = Literal["subsonic", "supersonic"]
MixingSolutionOverride = Literal["subsonic"]


class StatorDesignConvergenceError(RuntimeError):
    """Raised when a throat-state or optional nozzle-angle solve fails."""


@dataclass(frozen=True)
class _CorrectedExit:
    """Extrapolated suction-side values created by NASA TM X-2343 ``AFMIX``.

    :ivar float displacement: Final displacement thickness in nozzle coordinates.
    :ivar float momentum: Final momentum thickness in nozzle coordinates.
    :ivar float first_extension: First added suction-wall straight length.
    :ivar float second_extension: Second added suction-wall straight length.
    :ivar float spacing_increment: Exit-pitch increase produced by BL growth.
    """

    displacement: float
    momentum: float
    first_extension: float
    second_extension: float
    spacing_increment: float


@dataclass(frozen=True)
class _StatorEvaluation:
    """Complete result of one ideal exit-flow Mach and outlet-metal-angle trial.

    :ivar float ideal_outlet_absolute_flow_mach: Uniform inviscid Mach used to construct the nozzle.
    :ivar float ideal_outlet_absolute_flow_angle: Uniform inviscid flow angle before aftermixing.
    :ivar float outlet_metal_angle: Nozzle outlet metal angle in the machine frame.
    :ivar IdealNozzleConstruction construction: MOC or conical construction metadata.
    :ivar NozzleShape ideal: Inviscid nozzle contour.
    :ivar NozzleShape corrected: BL-corrected nozzle contour.
    :ivar BoundaryLayerResult pressure_boundary_layer: Pressure-side BL at geometry stations.
    :ivar BoundaryLayerResult suction_boundary_layer: Suction-side BL at geometry stations.
    :ivar BoundaryLayerResult pressure_boundary_layer_marching: Alias of the pressure-side BL result.
    :ivar BoundaryLayerResult suction_boundary_layer_marching: Alias of the suction-side BL result.
    :ivar _CorrectedExit corrected_exit: BL quantities extrapolated through the exit correction.
    :ivar dict uncorrected_mixing: AFMIX solutions using inviscid geometry and wall values.
    :ivar dict corrected_mixing: AFMIX solutions using the corrected exit.
    :ivar float physical_chord: Dimensional suction-wall chord, m.
    :ivar float chord_reynolds_number: Reynolds number based on ``physical_chord``.
    """

    ideal_outlet_absolute_flow_mach: float
    ideal_outlet_absolute_flow_angle: float
    outlet_metal_angle: float
    construction: IdealNozzleConstruction
    ideal: NozzleShape
    corrected: NozzleShape
    pressure_boundary_layer: BoundaryLayerResult
    suction_boundary_layer: BoundaryLayerResult
    pressure_boundary_layer_marching: BoundaryLayerResult
    suction_boundary_layer_marching: BoundaryLayerResult
    corrected_exit: _CorrectedExit
    uncorrected_mixing: dict[str, dict[str, float | bool]]
    corrected_mixing: dict[str, dict[str, float | bool]]
    physical_chord: float
    chord_reynolds_number: float


def _surface(x: np.ndarray, y: np.ndarray, absolute_flow_mach: np.ndarray) -> SurfaceCoordinates:
    """Create a surface after displacement correction or extrapolation.

    :param numpy.ndarray x: Axial coordinates in the current scale.
    :param numpy.ndarray y: Transverse coordinates in the current scale.
    :param numpy.ndarray absolute_flow_mach: Local inviscid absolute flow Mach numbers.
    :return: Surface with tangent angles recalculated from the new coordinates.
    :rtype: SurfaceCoordinates
    :raises BoundaryLayerError: If the correction creates coincident stations.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    absolute_flow_mach = np.asarray(absolute_flow_mach, dtype=float)
    if np.any(np.hypot(np.diff(x), np.diff(y)) <= 1.0e-12):
        raise BoundaryLayerError("boundary-layer correction created duplicate nozzle points")
    return SurfaceCoordinates(
        x=x,
        y=y,
        absolute_flow_mach=absolute_flow_mach,
        metal_angle=np.asarray(np.degrees(np.arctan2(np.gradient(y), np.gradient(x))), dtype=float),
    )


def _slice_boundary_layer(result: BoundaryLayerResult, count: int) -> BoundaryLayerResult:
    """Return the pressure-surface prefix of the common nozzle-wall march.

    :param BoundaryLayerResult result: Complete suction-wall boundary-layer result.
    :param int count: Number of stations ending at the pressure-side exit.
    :return: Independent copy of the requested prefix.
    :rtype: BoundaryLayerResult
    """

    transition = result.transition_index
    separation = result.separation_index
    return BoundaryLayerResult(
        s_over_chord=result.s_over_chord[:count].copy(),
        displacement_thickness_over_chord=(result.displacement_thickness_over_chord[:count].copy()),
        momentum_thickness_over_chord=(result.momentum_thickness_over_chord[:count].copy()),
        form_factor=result.form_factor[:count].copy(),
        regime=result.regime[:count].copy(),
        transition_index=(transition if transition is not None and transition < count else None),
        separation_index=(separation if separation is not None and separation < count else None),
        freestream_absolute_flow_mach=result.freestream_absolute_flow_mach[:count].copy(),
    )


class SupersonicStatorNozzle:
    """Design a two-dimensional sharp-throat supersonic turbine stator.

    The ideal diverging contour can be either the
    method-of-characteristics construction from NASA TM X-1502 or an
    axisymmetric straight-wall (conical) de Laval nozzle sized with the
    perfect-gas area--Mach relation. Boundary-layer displacement, the extra corrected
    suction-side straight length, and aftermixing use NASA TM X-2343.
    Construction is completed during initialization, so all geometry,
    thermodynamic, boundary-layer, mixing, and dimensional results are
    immediately available as object properties.

    Angles are measured from the machine axial direction.  This differs from
    the legacy FORTRAN input ``ALP1``, which was measured from the tangential
    direction.  ``requested_outlet_absolute_flow_mach`` is the absolute Mach number because a stator is
    stationary.

    The mass-flow calculation assumes a choked, calorically perfect ideal gas
    at the geometric throat. For the MOC contour, ``throat_height`` is the
    out-of-plane blade span and the calculated ``throat_width`` is the opening
    of one rectangular two-dimensional passage:

    ``total area = nozzle_count * throat_height * throat_width``.

    For the axisymmetric conical contour, one-nozzle throat area is
    ``pi * throat_diameter**2 / 4``. No ``throat_height`` is used.

    :param float requested_outlet_absolute_flow_mach: Absolute Mach target. By default this is the
        uniform ideal Mach after the diverging characteristic region and
        before aftermixing, so it must exceed one. With
        ``match_real_outlet_absolute_flow_mach=True`` it is the desired mixed Mach.
    :param float requested_outlet_absolute_flow_angle: Absolute outlet flow-angle target
        measured from the machine axial direction, degrees. By default this is
        the ideal premixing direction. With ``iterate_outlet_metal_angle=True``
        it is the desired real aftermixed direction.
    :param float mass_flow_rate: Total stator mass flow, kg/s.
    :param int nozzle_count: Number of equal stator nozzles.
    :param Fluid fluid: CoolProp-backed ideal-gas mixture.
    :param float upstream_total_temperature: Stator upstream total
        temperature, K.
    :param float upstream_total_pressure: Stator upstream total pressure, Pa.
    :param float | None throat_height: Out-of-plane throat height, m. Required
        only for the rectangular MOC passage and invalid for the circular
        conical nozzle.
    :param float trailing_edge_thickness: Physical trailing-edge thickness,
        m.  NASA TM X-2343 treats this as blockage in ``AFMIX``; it does not
        modify the method-of-characteristics contour.  The default zero
        retains the sharp trailing-edge mixing model.
    :param ContourMethod contour_method: ``"moc"`` for the NASA TM X-1502
        characteristic contour or ``"conical"`` for an axisymmetric
        straight-wall de Laval contour.
    :param float | None half_cone_metal_angle: Divergent-wall half angle from
        the nozzle axis, degrees. Required only for
        ``contour_method="conical"``.
    :param int number_of_nodes: Nodes used by each MOC, conical, or straight-wall
        nozzle segment. The boundary-layer calculation marches directly on
        the assembled nozzle surface without a separate mesh.
    :param bool iterate_outlet_metal_angle: If false, the outlet metal angle equals
        ``requested_outlet_absolute_flow_angle``.  If true, it is iterated until the selected
        corrected aftermixing solution matches that requested direction.
    :param bool match_real_outlet_absolute_flow_mach: If true, jointly vary ideal
        supersonic construction Mach and outlet metal angle until the selected
        aftermixing solution matches both ``requested_outlet_absolute_flow_mach`` and
        ``requested_outlet_absolute_flow_angle``. Requires ``iterate_outlet_metal_angle=True``.
    :param BoundaryLayerMode boundary_layer_mode: ``"fully_turbulent"`` or
        ``"laminar_then_turbulent"``.  Both calculations start at the throat;
        the converging subsonic contour is outside this class.
    :param float | None initial_turbulent_displacement_thickness:
        Physical turbulent displacement thickness at the throat, m.
    :param float | None initial_turbulent_momentum_thickness:
        Physical turbulent momentum thickness at the throat, m.
    :param MixingSolutionOverride | None mixing_solution: Optional mixed-flow-solution
        override. By default, subsonic premixing axial flow uses the subsonic
        solution, while supersonic axial flow uses the shockless solution when
        it is available. Set ``"subsonic"`` to force the subsonic solution.

    As in NASA TM X-2343, trailing-edge thickness affects only the mixed-out
    conservation calculation. The stored pressure and suction coordinates
    remain the selected ideal and boundary-layer-corrected contours.
    """

    def __init__(
        self,
        *,
        requested_outlet_absolute_flow_mach: float,
        requested_outlet_absolute_flow_angle: float,
        mass_flow_rate: float,
        nozzle_count: int,
        fluid: Fluid,
        upstream_total_temperature: float,
        upstream_total_pressure: float,
        throat_height: float | None = None,
        trailing_edge_thickness: float = 0.0,
        contour_method: ContourMethod = "moc",
        half_cone_metal_angle: float | None = None,
        number_of_nodes: int = 101,
        iterate_outlet_metal_angle: bool = False,
        match_real_outlet_absolute_flow_mach: bool = False,
        boundary_layer_mode: BoundaryLayerMode = "laminar_then_turbulent",
        initial_turbulent_displacement_thickness: float | None = None,
        initial_turbulent_momentum_thickness: float | None = None,
        mixing_solution: MixingSolutionOverride | None = None,
    ) -> None:
        """Validate inputs and execute the complete stator-nozzle design.

        Constructor arguments and units are documented on
        :class:`SupersonicStatorNozzle`. Initialization calculates the choked
        throat scale, creates the inviscid contour, marches the BL, applies the
        NASA TM X-2343 correction, evaluates mixing, and stores dimensional geometry.

        :raises TypeError: If an input has the wrong basic type.
        :raises ValueError: If an input is outside the selected contour model's range.
        :raises StatorDesignConvergenceError: If a throat-state or optional outlet solve fails.
        """

        # Validate the mode-dependent arguments before converting values. This
        # keeps MOC-only and conical-only input errors close to the user call.
        self._validate_inputs(
            requested_outlet_absolute_flow_mach=requested_outlet_absolute_flow_mach,
            requested_outlet_absolute_flow_angle=requested_outlet_absolute_flow_angle,
            mass_flow_rate=mass_flow_rate,
            nozzle_count=nozzle_count,
            throat_height=throat_height,
            fluid=fluid,
            upstream_total_temperature=upstream_total_temperature,
            upstream_total_pressure=upstream_total_pressure,
            trailing_edge_thickness=trailing_edge_thickness,
            contour_method=contour_method,
            half_cone_metal_angle=half_cone_metal_angle,
            number_of_nodes=number_of_nodes,
            iterate_outlet_metal_angle=iterate_outlet_metal_angle,
            match_real_outlet_absolute_flow_mach=match_real_outlet_absolute_flow_mach,
            boundary_layer_mode=boundary_layer_mode,
            initial_turbulent_displacement_thickness=initial_turbulent_displacement_thickness,
            initial_turbulent_momentum_thickness=initial_turbulent_momentum_thickness,
            mixing_solution=mixing_solution,
        )
        self.requested_outlet_absolute_flow_mach = float(requested_outlet_absolute_flow_mach)
        self.requested_outlet_absolute_flow_angle = float(requested_outlet_absolute_flow_angle)
        self.mass_flow_rate = float(mass_flow_rate)
        self.nozzle_count = int(nozzle_count)
        self.throat_height = None if throat_height is None else float(throat_height)
        self.fluid = fluid
        self.upstream_total_temperature = float(upstream_total_temperature)
        self.upstream_total_pressure = float(upstream_total_pressure)
        self.trailing_edge_thickness = float(trailing_edge_thickness)
        self.contour_method = contour_method
        self.half_cone_metal_angle = None if half_cone_metal_angle is None else float(half_cone_metal_angle)
        self.number_of_nodes = int(number_of_nodes)
        self.iterate_outlet_metal_angle = bool(iterate_outlet_metal_angle)
        self.match_real_outlet_absolute_flow_mach = bool(match_real_outlet_absolute_flow_mach)
        self.boundary_layer_mode = boundary_layer_mode
        self.initial_turbulent_displacement_thickness = (
            None
            if initial_turbulent_displacement_thickness is None
            else float(initial_turbulent_displacement_thickness)
        )
        self.initial_turbulent_momentum_thickness = (
            None if initial_turbulent_momentum_thickness is None else float(initial_turbulent_momentum_thickness)
        )
        self._mixing_solution_override = mixing_solution

        self.upstream_total_fluid_state = self.fluid.properties(
            self.upstream_total_temperature, self.upstream_total_pressure
        )
        (self.throat_static_temperature, self.throat_static_pressure, self.throat_static_fluid_state) = (
            self._solve_throat_static_reference_state(initial_gamma=self.upstream_total_fluid_state.gamma)
        )
        self.gamma = float(self.throat_static_fluid_state.gamma)
        self.prandtl_number = float(self.throat_static_fluid_state.prandtl_number)

        # NASA's choked mass-flow relation is written as mass flux times area.
        # The operating point fixes total throat area. Dividing by nozzle
        # count then gives either a rectangular MOC passage area or the
        # circular area of one axisymmetric conical nozzle.
        choked_mass_flux = (
            self.upstream_total_pressure
            / math.sqrt(self.upstream_total_temperature)
            * math.sqrt(self.gamma / self.fluid.specific_gas_constant)
            * (2.0 / (self.gamma + 1.0)) ** ((self.gamma + 1.0) / (2.0 * (self.gamma - 1.0)))
        )
        self.total_throat_area = self.mass_flow_rate / choked_mass_flux
        self.single_nozzle_throat_area = self.total_throat_area / self.nozzle_count
        if self.contour_method == "moc":
            self.throat_width = self.single_nozzle_throat_area / self.throat_height
            self.throat_diameter = None
            self.throat_radius = None
            self.throat_half_width_scale = 0.5 * self.throat_width
            self.coordinate_scale_length = self.throat_half_width_scale
            self.trailing_edge_thickness_over_throat_half_width = (
                self.trailing_edge_thickness / self.throat_half_width_scale
            )
            self.trailing_edge_thickness_over_throat_diameter = None
        else:
            self.throat_width = None
            self.throat_diameter = math.sqrt(4.0 * self.single_nozzle_throat_area / math.pi)
            self.throat_radius = 0.5 * self.throat_diameter
            self.throat_half_width_scale = None
            # Conical coordinates are deliberately divided by D*, so every
            # nondimensional coordinate is multiplied by D* here.
            self.coordinate_scale_length = self.throat_diameter
            self.trailing_edge_thickness_over_throat_half_width = None
            self.trailing_edge_thickness_over_throat_diameter = self.trailing_edge_thickness / self.throat_diameter
        self.trailing_edge_thickness_over_coordinate_scale = self.trailing_edge_thickness / self.coordinate_scale_length
        self.mass_flux_at_throat = choked_mass_flux

        self._evaluation_cache: dict[tuple[float, float], _StatorEvaluation] = {}
        if self.match_real_outlet_absolute_flow_mach:
            (outlet_metal_angle, ideal_outlet_absolute_flow_mach) = self._solve_outlet_metal_angle_and_flow_mach_targets()
        else:
            ideal_outlet_absolute_flow_mach = self.requested_outlet_absolute_flow_mach
            outlet_metal_angle = (
                self._solve_outlet_metal_angle_for_target_flow(ideal_outlet_absolute_flow_mach=ideal_outlet_absolute_flow_mach)
                if self.iterate_outlet_metal_angle
                else self.requested_outlet_absolute_flow_angle
            )
        evaluation = self._evaluate(outlet_metal_angle, ideal_outlet_absolute_flow_mach=ideal_outlet_absolute_flow_mach)

        selected_solution, selected = self._select_mixing_result(evaluation.corrected_mixing)
        if not bool(selected["available"]):
            raise StatorDesignConvergenceError("the selected stator aftermixing solution is unavailable")

        self.outlet_metal_angle = float(evaluation.outlet_metal_angle)
        self.ideal_outlet_absolute_flow_mach = float(ideal_outlet_absolute_flow_mach)
        # The uniform premixing stator flow is aligned with the nozzle metal,
        # but both quantities remain separately addressable.
        self.ideal_outlet_absolute_flow_angle = float(evaluation.ideal_outlet_absolute_flow_angle)
        if self.contour_method == "conical":
            self.required_exit_area_ratio = float(isentropic_area_ratio(self.ideal_outlet_absolute_flow_mach, self.gamma))
            self.ideal_exit_area_ratio = self.required_exit_area_ratio
        else:
            # The MOC wall is set by its characteristic net, not by a
            # quasi-one-dimensional area ratio.
            self.required_exit_area_ratio = None
            self.ideal_exit_area_ratio = None
        self.uncorrected_shape = evaluation.ideal
        self.corrected_shape = evaluation.corrected
        self.pressure_boundary_layer = evaluation.pressure_boundary_layer
        self.suction_boundary_layer = evaluation.suction_boundary_layer
        self.pressure_boundary_layer_marching = evaluation.pressure_boundary_layer_marching
        self.suction_boundary_layer_marching = evaluation.suction_boundary_layer_marching
        self.boundary_layer_pressure_station_count = len(self.pressure_boundary_layer_marching.s_over_chord)
        self.boundary_layer_suction_station_count = len(self.suction_boundary_layer_marching.s_over_chord)
        self.uncorrected_mixing_results = evaluation.uncorrected_mixing
        self.mixing_results = evaluation.corrected_mixing
        self.mixing_solution = selected_solution
        self.real_outlet_absolute_flow_angle = float(selected["real_outlet_absolute_flow_angle"])
        self.real_outlet_absolute_flow_mach = float(selected["real_outlet_absolute_flow_mach"])
        self.ideal_outlet_absolute_axial_flow_mach = self.ideal_outlet_absolute_flow_mach * math.cos(
            math.radians(self.ideal_outlet_absolute_flow_angle)
        )
        self.supersonic_mixing_available = bool(self.mixing_results["supersonic"]["available"])
        self.contour_point_count = evaluation.construction.contour_point_count
        self.pressure_number_of_nodes = evaluation.construction.pressure_point_count
        self.actual_flow_turning_increment = evaluation.construction.actual_flow_turning_increment
        if self.contour_method == "conical":
            self.conical_divergent_length_over_throat_diameter = float(self.uncorrected_shape.pressure_surface.x[-1])
            self.conical_divergent_length = self.conical_divergent_length_over_throat_diameter * self.throat_diameter
        else:
            self.conical_divergent_length_over_throat_diameter = None
            self.conical_divergent_length = None
        self.physical_chord = evaluation.physical_chord
        self.chord_reynolds_number = evaluation.chord_reynolds_number
        self.corrected_exit_displacement_thickness = (
            evaluation.corrected_exit.displacement * self.coordinate_scale_length
        )
        self.corrected_exit_momentum_thickness = evaluation.corrected_exit.momentum * self.coordinate_scale_length

        # The physical scale is known at initialization, unlike the rotor
        # scale that can be supplied later.  Store both dimensional shapes
        # immediately while retaining dimensionalize() as a convenient,
        # idempotent public method.
        self.dimensional_shapes: DimensionalNozzleShapes
        self.dimensionalize()

    @staticmethod
    def _validate_inputs(**values) -> None:
        """Validate public inputs before any thermodynamic or geometric solve.

        :param values: Constructor values indexed by their public argument name.
        :type values: dict[str, object]
        :raises TypeError: If a flag or fluid object has the wrong type.
        :raises ValueError: If a value or model-dependent option is invalid.
        """

        positive_floats = ("mass_flow_rate", "upstream_total_temperature", "upstream_total_pressure")
        for name in positive_floats:
            value = values[name]
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        trailing_edge_thickness = values["trailing_edge_thickness"]
        if not math.isfinite(trailing_edge_thickness) or trailing_edge_thickness < 0.0:
            raise ValueError("trailing_edge_thickness must be nonnegative and finite")
        if not math.isfinite(values["requested_outlet_absolute_flow_mach"]) or values["requested_outlet_absolute_flow_mach"] <= 0.0:
            raise ValueError("requested_outlet_absolute_flow_mach must be positive and finite")
        if not values["match_real_outlet_absolute_flow_mach"] and values["requested_outlet_absolute_flow_mach"] <= 1.0:
            raise ValueError("ideal pre-mixing requested_outlet_absolute_flow_mach must be supersonic (> 1)")
        if not (math.isfinite(values["requested_outlet_absolute_flow_angle"]) and 0.0 < values["requested_outlet_absolute_flow_angle"] < 90.0):
            raise ValueError("requested_outlet_absolute_flow_angle must be between 0 and 90")
        if not isinstance(values["nozzle_count"], int) or values["nozzle_count"] < 1:
            raise ValueError("nozzle_count must be an integer >= 1")
        if not isinstance(values["fluid"], Fluid):
            raise TypeError("fluid must be an instance of Fluid")
        contour_method = values["contour_method"]
        half_cone_metal_angle = values["half_cone_metal_angle"]
        if contour_method not in ("moc", "conical"):
            raise ValueError("contour_method must be 'moc' or 'conical'")
        if contour_method == "moc":
            throat_height = values["throat_height"]
            if throat_height is None or not math.isfinite(throat_height) or throat_height <= 0.0:
                raise ValueError("contour_method='moc' requires a positive finite throat_height")
            if half_cone_metal_angle is not None:
                raise ValueError("half_cone_metal_angle is only valid for contour_method='conical'")
        else:
            if values["throat_height"] is not None:
                raise ValueError(
                    "throat_height is only valid for "
                    "contour_method='moc'; circular conical nozzles "
                    "derive throat_diameter from area"
                )
            if half_cone_metal_angle is None:
                raise ValueError("contour_method='conical' requires half_cone_metal_angle")
            if not (
                math.isfinite(half_cone_metal_angle) and 0.0 < half_cone_metal_angle < 90.0
            ):
                raise ValueError("half_cone_metal_angle must be between 0 and 90")
        if (
            not isinstance(values["number_of_nodes"], int)
            or isinstance(values["number_of_nodes"], bool)
            or values["number_of_nodes"] < 20
        ):
            raise ValueError("number_of_nodes must be an integer >= 20")
        if not isinstance(values["iterate_outlet_metal_angle"], bool):
            raise TypeError("iterate_outlet_metal_angle must be a bool")
        if not isinstance(values["match_real_outlet_absolute_flow_mach"], bool):
            raise TypeError("match_real_outlet_absolute_flow_mach must be a bool")
        if values["match_real_outlet_absolute_flow_mach"] and not values["iterate_outlet_metal_angle"]:
            raise ValueError("match_real_outlet_absolute_flow_mach=True requires iterate_outlet_metal_angle=True")
        if values["boundary_layer_mode"] not in ("fully_turbulent", "laminar_then_turbulent"):
            raise ValueError("invalid boundary_layer_mode")
        if values["mixing_solution"] not in (None, "subsonic"):
            raise ValueError("mixing_solution must be None or 'subsonic'")

        displacement = values["initial_turbulent_displacement_thickness"]
        momentum = values["initial_turbulent_momentum_thickness"]
        if values["boundary_layer_mode"] == "fully_turbulent":
            if displacement is None or momentum is None:
                raise ValueError(
                    "fully_turbulent mode requires initial displacement and momentum thicknesses at the throat"
                )
            if not math.isfinite(displacement) or displacement <= 0.0 or not math.isfinite(momentum) or momentum <= 0.0:
                raise ValueError("initial turbulent thicknesses must be positive and finite")
            if displacement <= momentum:
                raise ValueError("initial turbulent displacement thickness must exceed initial momentum thickness")
        elif displacement is not None or momentum is not None:
            raise ValueError("initial turbulent thicknesses are only used with boundary_layer_mode='fully_turbulent'")

    def _select_mixing_result(
        self, mixing: dict[str, dict[str, float | bool]]
    ) -> tuple[MixingSolution, dict[str, float | bool]]:
        """Select one aftermixing solution for the current design trial.

        :param dict mixing: Subsonic and supersonic aftermixing results.
        :return: Selected solution name and its result dictionary.
        :rtype: tuple[MixingSolution, dict[str, float | bool]]
        """

        if self._mixing_solution_override is not None:
            solution = self._mixing_solution_override
        else:
            supersonic = mixing["supersonic"]
            ideal_outlet_absolute_axial_flow_mach = float(supersonic["ideal_outlet_absolute_axial_flow_mach"])
            solution = (
                "supersonic"
                if ideal_outlet_absolute_axial_flow_mach >= 1.0 and bool(supersonic["available"])
                else "subsonic"
            )
        return solution, mixing[solution]

    def _solve_throat_static_reference_state(self, *, initial_gamma: float) -> tuple[float, float, FluidState]:
        """Find gamma at the self-consistent choked static throat state.

        :param float initial_gamma: First heat-capacity-ratio estimate from the total state.
        :return: Choked static temperature, static pressure, and converged fluid state.
        :rtype: tuple[float, float, FluidState]
        :raises StatorDesignConvergenceError: If mixture gamma does not converge.
        """

        gamma = float(initial_gamma)
        for _ in range(100):
            temperature_ratio = 2.0 / (gamma + 1.0)
            static_temperature = self.upstream_total_temperature * temperature_ratio
            static_pressure = self.upstream_total_pressure * temperature_ratio ** (gamma / (gamma - 1.0))
            static_state = self.fluid.properties(static_temperature, static_pressure)
            updated_gamma = float(static_state.gamma)
            if abs(updated_gamma - gamma) <= 1.0e-12:
                return static_temperature, static_pressure, static_state
            gamma = updated_gamma
        raise StatorDesignConvergenceError("mixture gamma did not converge at the choked throat state")

    def _evaluate(
        self, outlet_metal_angle: float, *, ideal_outlet_absolute_flow_mach: float | None = None
    ) -> _StatorEvaluation:
        """Build one angle/Mach trial, including BL correction and aftermixing.

        :param float outlet_metal_angle: Trial outlet metal angle from the machine axis, degrees.
        :param float | None ideal_outlet_absolute_flow_mach: Trial inviscid exit Mach, or the requested value when omitted.
        :return: Geometry, BL results, corrected exit, dimensional scale, and mixing solutions.
        :rtype: _StatorEvaluation
        """

        if ideal_outlet_absolute_flow_mach is None:
            ideal_outlet_absolute_flow_mach = self.requested_outlet_absolute_flow_mach
        # Zero deviation aligns ideal premixing flow with the outlet metal,
        # while retaining separate flow and geometry quantities.
        ideal_outlet_absolute_flow_angle = float(outlet_metal_angle)
        key = (round(float(ideal_outlet_absolute_flow_mach), 10), round(float(outlet_metal_angle), 10))
        # Coupled outlet iterations can revisit a trial. The rounded key avoids
        # repeating its relatively expensive geometry and BL calculations.
        if key in self._evaluation_cache:
            return self._evaluation_cache[key]

        if self.contour_method == "moc":
            construction = design_ideal_stator_nozzle(
                ideal_outlet_absolute_flow_mach=ideal_outlet_absolute_flow_mach,
                outlet_metal_angle=outlet_metal_angle,
                number_of_nodes=self.number_of_nodes,
                gamma=self.gamma,
            )
        else:
            # Every coupled Mach/angle trial reaches this branch with its
            # current ideal_outlet_absolute_flow_mach. The area ratio and cone length are
            # therefore rebuilt when Mach is varied to meet a mixed-Mach
            # target, rather than being frozen from the requested value.
            construction = design_conical_stator_nozzle(
                ideal_outlet_absolute_flow_mach=ideal_outlet_absolute_flow_mach,
                outlet_metal_angle=outlet_metal_angle,
                half_cone_metal_angle=float(self.half_cone_metal_angle),
                number_of_nodes=self.number_of_nodes,
                gamma=self.gamma,
            )
        ideal = construction.shape

        physical_chord = ideal.chord * self.coordinate_scale_length
        throat_velocity = self.throat_static_fluid_state.speed_of_sound
        chord_reynolds_number = throat_velocity * physical_chord / self.throat_static_fluid_state.kinematic_viscosity

        # Both walls of the unrotated NASA TM X-1502 nozzle share the same symmetric
        # contour.  The original program marched one BL along the upper wall;
        # the pressure-side exit is an earlier station, while the suction side
        # continues over the fixed-node straight segment. Reusing one march here is
        # therefore more faithful than solving two independent BLs.
        suction_boundary_layer = solve_boundary_layer(
            surface=ideal.suction_surface,
            chord=ideal.chord,
            inlet_edge_flow_mach=1.0,
            chord_reynolds_number=chord_reynolds_number,
            gamma=self.gamma,
            fluid=self.fluid,
            inlet_total_temperature=self.upstream_total_temperature,
            inlet_total_pressure=self.upstream_total_pressure,
            mode=self.boundary_layer_mode,
            initial_turbulent_displacement_thickness_over_chord=(
                None
                if self.initial_turbulent_displacement_thickness is None
                else self.initial_turbulent_displacement_thickness / physical_chord
            ),
            initial_turbulent_momentum_thickness_over_chord=(
                None
                if self.initial_turbulent_momentum_thickness is None
                else self.initial_turbulent_momentum_thickness / physical_chord
            ),
            laminar_correlation_limit=0.16,
        )

        pressure_boundary_layer = _slice_boundary_layer(suction_boundary_layer, construction.pressure_point_count)
        pressure_boundary_layer_marching = pressure_boundary_layer
        suction_boundary_layer_marching = suction_boundary_layer
        (corrected, corrected_exit) = self._correct_shape(
            ideal=ideal,
            construction=construction,
            boundary_layer=suction_boundary_layer,
            outlet_metal_angle=outlet_metal_angle,
        )

        pressure_index = construction.pressure_point_count - 1
        displacement = suction_boundary_layer.displacement_thickness_over_chord * ideal.chord
        momentum = suction_boundary_layer.momentum_thickness_over_chord * ideal.chord
        uncorrected_mixing = self._aftermixing(
            ideal_outlet_absolute_flow_mach=ideal_outlet_absolute_flow_mach,
            ideal_outlet_absolute_flow_angle=ideal_outlet_absolute_flow_angle,
            spacing=ideal.spacing,
            trailing_edge_thickness=self.trailing_edge_thickness_over_coordinate_scale,
            pressure_displacement=float(displacement[pressure_index]),
            suction_displacement=float(displacement[-1]),
            pressure_momentum=float(momentum[pressure_index]),
            suction_momentum=float(momentum[-1]),
        )
        corrected_mixing = self._aftermixing(
            ideal_outlet_absolute_flow_mach=ideal_outlet_absolute_flow_mach,
            ideal_outlet_absolute_flow_angle=ideal_outlet_absolute_flow_angle,
            spacing=corrected.spacing,
            trailing_edge_thickness=self.trailing_edge_thickness_over_coordinate_scale,
            pressure_displacement=float(displacement[pressure_index]),
            suction_displacement=corrected_exit.displacement,
            pressure_momentum=float(momentum[pressure_index]),
            suction_momentum=corrected_exit.momentum,
        )
        evaluation = _StatorEvaluation(
            ideal_outlet_absolute_flow_mach=float(ideal_outlet_absolute_flow_mach),
            ideal_outlet_absolute_flow_angle=ideal_outlet_absolute_flow_angle,
            outlet_metal_angle=float(outlet_metal_angle),
            construction=construction,
            ideal=ideal,
            corrected=corrected,
            pressure_boundary_layer=pressure_boundary_layer,
            suction_boundary_layer=suction_boundary_layer,
            pressure_boundary_layer_marching=pressure_boundary_layer_marching,
            suction_boundary_layer_marching=suction_boundary_layer_marching,
            corrected_exit=corrected_exit,
            uncorrected_mixing=uncorrected_mixing,
            corrected_mixing=corrected_mixing,
            physical_chord=physical_chord,
            chord_reynolds_number=chord_reynolds_number,
        )
        self._evaluation_cache[key] = evaluation
        return evaluation

    @staticmethod
    def _correct_shape(
        *,
        ideal: NozzleShape,
        construction: IdealNozzleConstruction,
        boundary_layer: BoundaryLayerResult,
        outlet_metal_angle: float,
    ) -> tuple[NozzleShape, _CorrectedExit]:
        """Apply ``NOZZLC`` displacement and ``AFMIX`` exit extrapolation.

        :param NozzleShape ideal: Inviscid nozzle geometry.
        :param IdealNozzleConstruction construction: Geometry metadata locating the pressure-side exit.
        :param BoundaryLayerResult boundary_layer: Common wall BL result at geometry stations.
        :param float outlet_metal_angle: Outlet metal angle from the machine axis, degrees.
        :return: Corrected nozzle and BL values extrapolated to its final suction-side station.
        :rtype: tuple[NozzleShape, _CorrectedExit]
        :raises BoundaryLayerError: If the corrected exit geometry becomes negative or singular.
        """

        displacement = boundary_layer.displacement_thickness_over_chord * ideal.chord
        momentum = boundary_layer.momentum_thickness_over_chord * ideal.chord
        pressure_count = construction.pressure_point_count
        pressure_index = pressure_count - 1
        outlet_metal_angle_rad = math.radians(outlet_metal_angle)

        # First add the displacement thickness vertically, exactly as
        # subroutine NOZZLC does.  This is not a generic normal-offset
        # operation; it is part of the NASA TM X-2343 axis-horizontal nozzle model.
        pressure_x = ideal.pressure_surface.x.copy()
        pressure_y = ideal.pressure_surface.y - displacement[:pressure_count]
        pressure_absolute_flow_mach = ideal.pressure_surface.absolute_flow_mach.copy()
        suction_x = ideal.suction_surface.x.copy()
        suction_y = ideal.suction_surface.y + displacement
        suction_absolute_flow_mach = ideal.suction_surface.absolute_flow_mach.copy()

        # AFMIX lengthens the suction-side straight twice.  The first extension
        # restores the specified outlet metal angle after delta* is added at the
        # pressure exit.  The second accounts for continued boundary-layer
        # growth along that newly created straight segment.
        pressure_exit_y = float(ideal.suction_surface.y[pressure_index])
        required_straight = 2.0 * (pressure_exit_y + displacement[pressure_index]) * math.tan(
            outlet_metal_angle_rad
        )
        existing_straight = float(ideal.suction_surface.x[-1] - ideal.suction_surface.x[pressure_index])
        first_extension = required_straight - existing_straight
        if first_extension < -1.0e-10:
            raise BoundaryLayerError("boundary-layer correction shortened the nozzle straight")
        first_extension = max(first_extension, 0.0)

        arc = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(ideal.suction_surface.x), np.diff(ideal.suction_surface.y)))))
        delta_gradient = (displacement[-1] - displacement[-3]) / (arc[-1] - arc[-3])
        momentum_gradient = (momentum[-1] - momentum[-3]) / (arc[-1] - arc[-3])
        displacement_1 = float(displacement[-1]) + first_extension * delta_gradient
        momentum_1 = float(momentum[-1]) + first_extension * momentum_gradient

        growth_angle = math.atan(delta_gradient)
        momentum_growth_angle = math.atan(momentum_gradient)
        closing_angle = math.pi / 2.0 - outlet_metal_angle_rad - growth_angle
        closing_sine = math.sin(closing_angle)
        if abs(closing_sine) <= 1.0e-10:
            raise BoundaryLayerError("corrected nozzle exit geometry is singular")
        displacement_difference = displacement_1 - displacement[pressure_index]
        spacing_increment = math.cos(growth_angle) * displacement_difference / closing_sine
        auxiliary = math.sin(outlet_metal_angle_rad) * displacement_difference / closing_sine
        second_extension = auxiliary * math.cos(growth_angle)
        displacement_2 = displacement_1 + second_extension * math.tan(growth_angle)
        momentum_2 = momentum_1 + second_extension * math.tan(momentum_growth_angle)
        if second_extension < -1.0e-10:
            raise BoundaryLayerError("boundary-layer growth produced a negative exit extension")
        second_extension = max(second_extension, 0.0)

        # Retain the two extrapolated stations printed by the FORTRAN.  The
        # inviscid Mach remains the specified uniform exit value.
        x_1 = float(suction_x[-1] + first_extension)
        x_2 = x_1 + second_extension
        suction_x = np.concatenate((suction_x, [x_1, x_2]))
        suction_y = np.concatenate(
            (suction_y, [float(ideal.suction_surface.y[-1] + displacement_1), float(ideal.suction_surface.y[-1] + displacement_2)])
        )
        suction_absolute_flow_mach = np.concatenate(
            (
                suction_absolute_flow_mach,
                [suction_absolute_flow_mach[-1], suction_absolute_flow_mach[-1]],
            )
        )

        corrected_spacing = (
            2.0 * (pressure_exit_y + displacement[pressure_index]) / math.cos(outlet_metal_angle_rad)
            + spacing_increment
        )
        corrected = NozzleShape(
            pressure_surface=_surface(pressure_x, pressure_y, pressure_absolute_flow_mach),
            suction_surface=_surface(suction_x, suction_y, suction_absolute_flow_mach),
            chord=float(suction_x[-1]),
            throat_width=(ideal.throat_width + 2.0 * displacement[0]),
            exit_opening=2.0 * (pressure_exit_y + displacement[pressure_index]),
            spacing=corrected_spacing,
            coordinate_scale=ideal.coordinate_scale,
        )
        return corrected, _CorrectedExit(
            displacement=float(displacement_2),
            momentum=float(momentum_2),
            first_extension=float(first_extension),
            second_extension=float(second_extension),
            spacing_increment=float(spacing_increment),
        )

    def _aftermixing(
        self,
        *,
        ideal_outlet_absolute_flow_mach: float,
        ideal_outlet_absolute_flow_angle: float,
        spacing: float,
        trailing_edge_thickness: float,
        pressure_displacement: float,
        suction_displacement: float,
        pressure_momentum: float,
        suction_momentum: float,
    ) -> dict[str, dict[str, float | bool]]:
        """Evaluate the NASA TM X-2343 ``AFMIX`` conservation model.

        :param float ideal_outlet_absolute_flow_mach: Uniform inviscid exit Mach before mixing.
        :param float ideal_outlet_absolute_flow_angle: Ideal premixing absolute flow angle, degrees.
        :param float spacing: Pitch between corresponding nozzles in the current coordinate scale.
        :param float trailing_edge_thickness: Trailing-edge thickness in the same coordinate scale.
        :param float pressure_displacement: Pressure-side exit displacement thickness.
        :param float suction_displacement: Suction-side exit displacement thickness.
        :param float pressure_momentum: Pressure-side exit momentum thickness.
        :param float suction_momentum: Suction-side exit momentum thickness.
        :return: Subsonic and shockless-supersonic mixed states.
        :rtype: dict[str, dict[str, float | bool]]
        :raises BoundaryLayerError: If blockage closes the exit or the conservation equation has no real solution.
        """

        gamma = self.gamma
        gp = gamma + 1.0
        gm = gamma - 1.0
        absolute_flow_angle_rad = math.radians(ideal_outlet_absolute_flow_angle)
        velocity_ratio = math.sqrt((0.5 * gp * ideal_outlet_absolute_flow_mach**2) / (1.0 + 0.5 * gm * ideal_outlet_absolute_flow_mach**2))
        projected_spacing = spacing * math.cos(absolute_flow_angle_rad)
        if projected_spacing <= 0.0:
            raise BoundaryLayerError("nozzle exit spacing has no positive axial projection")

        displacement_ratio = (pressure_displacement + suction_displacement) / projected_spacing
        momentum_ratio = (pressure_momentum + suction_momentum) / projected_spacing

        # These are the FORTRAN AFMIX variables DTE, A, and A1.  TE and SP
        # are expressed in the same units, and XX = SP*cos(ALPH1) is the
        # pitch projected onto the plane normal to the axial direction.
        # The trailing edge removes both flow area (A1) and momentum area
        # (A); boundary-layer momentum thickness is additionally removed
        # only from A.
        trailing_edge_blockage_ratio = trailing_edge_thickness / projected_spacing
        effective_momentum_area = 1.0 - displacement_ratio - trailing_edge_blockage_ratio - momentum_ratio
        effective_area = 1.0 - displacement_ratio - trailing_edge_blockage_ratio
        if effective_momentum_area <= 0.0 or effective_area <= 0.0:
            raise BoundaryLayerError("boundary-layer and trailing-edge blockage close the stator exit")

        afs = gm / gp * velocity_ratio**2
        c_value = (
            (1.0 - afs) * gp / (2.0 * gamma)
            + math.cos(absolute_flow_angle_rad) ** 2 * effective_momentum_area * velocity_ratio**2
        ) / (math.cos(absolute_flow_angle_rad) * effective_area * velocity_ratio)
        d_value = velocity_ratio * math.sin(absolute_flow_angle_rad) * effective_momentum_area / effective_area
        radical = (gamma * c_value / gp) ** 2 - 1.0 + gm / gp * d_value**2
        if radical < -1.0e-10:
            raise BoundaryLayerError("stator aftermixing equation has no real solution")
        square_root = math.sqrt(max(radical, 0.0))
        ideal_outlet_absolute_axial_flow_mach = ideal_outlet_absolute_flow_mach * math.cos(
            absolute_flow_angle_rad
        )

        results: dict[str, dict[str, float | bool]] = {}
        for name, axial_velocity_ratio in (
            ("subsonic", gamma * c_value / gp - square_root),
            ("supersonic", gamma * c_value / gp + square_root),
        ):
            available = name == "subsonic" or ideal_outlet_absolute_axial_flow_mach >= 1.0 - 1.0e-12
            total_velocity_ratio = math.hypot(d_value, axial_velocity_ratio)
            denominator = 1.0 - gm / gp * total_velocity_ratio**2
            if not available or denominator <= 0.0 or axial_velocity_ratio <= 0.0:
                results[name] = {
                    "available": False,
                    "real_outlet_absolute_flow_mach": math.nan,
                    "real_outlet_absolute_axial_flow_mach": math.nan,
                    "real_outlet_absolute_flow_angle": math.nan,
                    "ideal_outlet_absolute_axial_flow_mach": ideal_outlet_absolute_axial_flow_mach,
                    "trailing_edge_blockage_ratio": (trailing_edge_blockage_ratio),
                }
                continue
            real_outlet_absolute_flow_mach = math.sqrt((2.0 / gp * total_velocity_ratio**2) / denominator)
            real_outlet_absolute_flow_angle_rad = math.atan2(d_value, axial_velocity_ratio)
            results[name] = {
                "available": True,
                "real_outlet_absolute_flow_mach": real_outlet_absolute_flow_mach,
                "real_outlet_absolute_axial_flow_mach": (
                    real_outlet_absolute_flow_mach * math.cos(real_outlet_absolute_flow_angle_rad)
                ),
                "real_outlet_absolute_flow_angle": math.degrees(real_outlet_absolute_flow_angle_rad),
                "ideal_outlet_absolute_axial_flow_mach": ideal_outlet_absolute_axial_flow_mach,
                "trailing_edge_blockage_ratio": (trailing_edge_blockage_ratio),
            }
        return results

    def _flow_residual_for_outlet_metal_angle(
        self, outlet_metal_angle: float, *, ideal_outlet_absolute_flow_mach: float | None = None
    ) -> float:
        """Return selected mixed angle minus the requested outlet angle.

        :param float outlet_metal_angle: Trial outlet metal angle, degrees.
        :param float | None ideal_outlet_absolute_flow_mach: Fixed trial inviscid Mach, or the requested Mach when omitted.
        :return: Corrected mixed-flow angle residual, degrees.
        :rtype: float
        :raises StatorDesignConvergenceError: If the selected mixing solution is unavailable.
        """

        mixing = self._evaluate(outlet_metal_angle, ideal_outlet_absolute_flow_mach=ideal_outlet_absolute_flow_mach).corrected_mixing
        _, result = self._select_mixing_result(mixing)
        if not bool(result["available"]):
            raise StatorDesignConvergenceError("selected aftermixing solution is unavailable at this angle")
        return float(result["real_outlet_absolute_flow_angle"]) - self.requested_outlet_absolute_flow_angle

    def _solve_outlet_metal_angle_for_target_flow(self, *, ideal_outlet_absolute_flow_mach: float | None = None) -> float:
        """Match the requested corrected mixed-flow direction with SciPy.

        :param float | None ideal_outlet_absolute_flow_mach: Fixed inviscid exit Mach, or the requested value when omitted.
        :return: Outlet metal angle giving the requested corrected mixed-flow direction, degrees.
        :rtype: float
        :raises StatorDesignConvergenceError: If no physical solution converges.
        """

        angle_tolerance = 1.0e-4

        def scaled_residual(values: np.ndarray) -> np.ndarray:
            """Return the angle residual scaled by its convergence tolerance."""

            return np.asarray(
                [
                    self._flow_residual_for_outlet_metal_angle(
                        float(values[0]),
                        ideal_outlet_absolute_flow_mach=ideal_outlet_absolute_flow_mach,
                    )
                    / angle_tolerance
                ],
                dtype=float,
            )

        initial = np.asarray(
            [min(max(self.requested_outlet_absolute_flow_angle, 0.75), 89.25)],
            dtype=float,
        )
        try:
            solution = least_squares(
                scaled_residual,
                initial,
                bounds=([0.5], [89.5]),
                diff_step=5.0e-3,
                xtol=1.0e-10,
                ftol=1.0e-10,
                gtol=1.0e-10,
                max_nfev=60,
            )
            outlet_metal_angle = float(solution.x[0])
            final_residual = self._flow_residual_for_outlet_metal_angle(
                outlet_metal_angle,
                ideal_outlet_absolute_flow_mach=ideal_outlet_absolute_flow_mach,
            )
        except (
            BoundaryLayerError,
            StatorGeometryError,
            StatorDesignConvergenceError,
            ValueError,
            OverflowError,
        ) as error:
            raise StatorDesignConvergenceError(
                "stator outlet flow-angle solve encountered an infeasible trial"
            ) from error
        if solution.success and abs(final_residual) <= angle_tolerance:
            return outlet_metal_angle
        raise StatorDesignConvergenceError(
            f"target mixed outlet angle did not converge; final residual was {final_residual:.4f} deg"
        )

    def _solve_outlet_metal_angle_and_flow_mach_targets(self) -> tuple[float, float]:
        """Match mixed absolute flow angle and Mach with two variables.

        Stator-relative and absolute frames are identical. SciPy's bounded
        nonlinear least-squares solver therefore varies the ideal supersonic
        exit Mach and nozzle angle directly. Every residual evaluation rebuilds
        the contour, dimensional Reynolds scale, BL march, correction, and
        aftermixing.

        :return: Outlet metal angle in degrees and ideal premixing exit Mach.
        :rtype: tuple[float, float]
        :raises StatorDesignConvergenceError: If the coupled solve cannot find a physical state.
        """

        target_real_outlet_absolute_flow_mach = self.requested_outlet_absolute_flow_mach
        lower_ideal_outlet_absolute_flow_mach = 1.0 + 1.0e-4
        upper_ideal_outlet_absolute_flow_mach = max(
            10.0, 3.0 * target_real_outlet_absolute_flow_mach
        )
        initial_ideal_outlet_absolute_flow_mach = max(target_real_outlet_absolute_flow_mach, 1.05)
        initial_outlet_metal_angle = self.requested_outlet_absolute_flow_angle
        variables = np.asarray(
            [
                min(max(initial_outlet_metal_angle, 0.75), 89.25),
                min(
                    max(
                        initial_ideal_outlet_absolute_flow_mach,
                        lower_ideal_outlet_absolute_flow_mach,
                    ),
                    upper_ideal_outlet_absolute_flow_mach,
                ),
            ],
            dtype=float,
        )

        def residual(values: np.ndarray) -> np.ndarray:
            """Evaluate mixed angle and Mach residuals for one solver trial.

            :param numpy.ndarray values: Outlet metal angle and ideal exit-flow Mach trial.
            :return: Corrected mixed angle and Mach residuals.
            :rtype: numpy.ndarray
            """

            outlet_metal_angle = float(values[0])
            ideal_outlet_absolute_flow_mach = float(values[1])
            mixing = self._evaluate(
                outlet_metal_angle,
                ideal_outlet_absolute_flow_mach=ideal_outlet_absolute_flow_mach,
            ).corrected_mixing
            _, selected = self._select_mixing_result(mixing)
            if not bool(selected["available"]):
                raise StatorDesignConvergenceError("selected aftermixing solution is unavailable")
            result = np.asarray(
                [
                    float(selected["real_outlet_absolute_flow_angle"])
                    - self.requested_outlet_absolute_flow_angle,
                    float(selected["real_outlet_absolute_flow_mach"])
                    - target_real_outlet_absolute_flow_mach,
                ],
                dtype=float,
            )
            if not np.all(np.isfinite(result)):
                raise StatorDesignConvergenceError("selected aftermixing solution is not physical")
            return result

        angle_tolerance = 2.0e-3
        mach_tolerance = 1.0e-4

        def scaled_residual(values: np.ndarray) -> np.ndarray:
            """Scale both residuals by their independently checked tolerances."""

            result = residual(values)
            return result / np.asarray([angle_tolerance, mach_tolerance], dtype=float)

        try:
            solution = least_squares(
                scaled_residual,
                variables,
                bounds=(
                    [0.5, lower_ideal_outlet_absolute_flow_mach],
                    [89.5, upper_ideal_outlet_absolute_flow_mach],
                ),
                diff_step=5.0e-3,
                x_scale=np.asarray([10.0, 0.5], dtype=float),
                xtol=1.0e-10,
                ftol=1.0e-10,
                gtol=1.0e-10,
                max_nfev=100,
            )
            final_residual = residual(solution.x)
        except (
            BoundaryLayerError,
            StatorGeometryError,
            StatorDesignConvergenceError,
            ValueError,
            OverflowError,
        ) as error:
            raise StatorDesignConvergenceError(
                "coupled stator outlet solve encountered an infeasible trial; "
                "adjust the requested outlet state or mixing solution"
            ) from error
        if (
            solution.success
            and abs(float(final_residual[0])) <= angle_tolerance
            and abs(float(final_residual[1])) <= mach_tolerance
        ):
            return float(solution.x[0]), float(solution.x[1])
        raise StatorDesignConvergenceError(
            "coupled stator outlet angle/Mach solve did not converge; "
            f"final absolute angle residual={final_residual[0]:.6g} deg and "
            f"Mach residual={final_residual[1]:.6g}"
        )

    def dimensionalize(self) -> DimensionalNozzleShapes:
        """Store both shapes in metres using the mode-specific throat scale.

        :return: Ideal and corrected nozzle surfaces in metres with throat metadata.
        :rtype: DimensionalNozzleShapes
        """

        result = DimensionalNozzleShapes(
            total_throat_area=self.total_throat_area,
            single_nozzle_throat_area=self.single_nozzle_throat_area,
            nozzle_count=self.nozzle_count,
            throat_height=self.throat_height,
            ideal_throat_width=self.throat_width,
            ideal_throat_diameter=self.throat_diameter,
            coordinate_scale_length=self.coordinate_scale_length,
            throat_half_width_scale=self.throat_half_width_scale,
            uncorrected=self.uncorrected_shape.scaled(self.coordinate_scale_length, "dimensional [m]"),
            corrected=self.corrected_shape.scaled(self.coordinate_scale_length, "dimensional [m]"),
        )
        self.dimensional_shapes = result
        self.uncorrected_dimensional_shape = result.uncorrected
        self.corrected_dimensional_shape = result.corrected
        return result

    @staticmethod
    def _rotate(surface: SurfaceCoordinates, angle_rad: float) -> tuple[np.ndarray, np.ndarray]:
        """Rotate nozzle-axis coordinates into axial/tangential coordinates.

        :param SurfaceCoordinates surface: Surface expressed along the nozzle axis.
        :param float angle_rad: Nozzle-axis rotation from the machine axis, radians.
        :return: Axial and tangential coordinate arrays.
        :rtype: tuple[numpy.ndarray, numpy.ndarray]
        """

        cosine = math.cos(angle_rad)
        sine = math.sin(angle_rad)
        return (surface.x * cosine - surface.y * sine, surface.x * sine + surface.y * cosine)

    def plot(self, *, dimensional: bool = False, ax=None, show: bool = True):
        """Plot both shapes after rotation by the final outlet metal angle.

        :param bool dimensional: Plot in millimetres instead of throat-based NASA TM X-1502 coordinates.
        :param ax: Existing Matplotlib axes, or ``None`` to create a figure.
        :type ax: matplotlib.axes.Axes | None
        :param bool show: Call ``matplotlib.pyplot.show`` before returning.
        :return: Matplotlib figure and axes.
        :rtype: tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        :raises ImportError: If Matplotlib is unavailable.
        """

        try:
            import matplotlib.pyplot as plt
        except ImportError as error:
            raise ImportError("plotting requires matplotlib; install project dependencies") from error

        if dimensional:
            # Dimensional stator shapes remain stored in metres. Only the
            # plotting copies are converted to millimetres so calculations and
            # public dimensional properties retain their original units.
            ideal = self.dimensional_shapes.uncorrected.scaled(1000.0, "dimensional [mm]")
            corrected = self.dimensional_shapes.corrected.scaled(1000.0, "dimensional [mm]")
            axis_label = "length [mm]"
        else:
            ideal = self.uncorrected_shape
            corrected = self.corrected_shape
            axis_label = (
                "coordinate / throat diameter" if self.contour_method == "conical" else "coordinate / throat half-width"
            )

        if ax is None:
            figure, ax = plt.subplots()
        else:
            figure = ax.figure
        outlet_metal_angle_rad = math.radians(self.outlet_metal_angle)
        for surface in (ideal.pressure_surface, ideal.suction_surface):
            axial, tangential = self._rotate(surface, outlet_metal_angle_rad)
            ax.plot(
                axial,
                tangential,
                "--",
                color="0.35",
                linewidth=1.4,
                label=("uncorrected" if surface is ideal.pressure_surface else None),
            )
        for surface in (corrected.pressure_surface, corrected.suction_surface):
            axial, tangential = self._rotate(surface, outlet_metal_angle_rad)
            ax.plot(
                axial,
                tangential,
                "-",
                color="#b23a48",
                linewidth=1.8,
                label=("BL corrected" if surface is corrected.pressure_surface else None),
            )
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel(f"axial {axis_label}")
        ax.set_ylabel(f"tangential {axis_label}")
        ax.grid(True, alpha=0.25)
        ax.legend()
        figure.tight_layout()
        if show:
            plt.show()
        return figure, ax
