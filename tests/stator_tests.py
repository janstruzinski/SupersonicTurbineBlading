import math

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from SupersonicTurbineBlading import Fluid, StatorDesignConvergenceError, SupersonicStatorNozzle
from SupersonicTurbineBlading.gas_dynamics import isentropic_area_ratio, supersonic_mach_from_area_ratio
from SupersonicTurbineBlading.stator import stator_nozzle as stator_nozzle_module
from SupersonicTurbineBlading.stator.stator_geometry import design_conical_stator_nozzle, design_ideal_stator_nozzle


def make_stator(**overrides):
    inputs = dict(
        requested_outlet_absolute_flow_mach=1.77,
        requested_outlet_absolute_flow_angle=70.0,
        mass_flow_rate=5.0,
        nozzle_count=30,
        throat_height=0.05,
        fluid=Fluid(["Air"], [1.0]),
        upstream_total_temperature=900.0,
        upstream_total_pressure=1.0e6,
        flow_turning_increment=0.5,
        number_of_stations=81,
        iterate_outlet_metal_angle=False,
        boundary_layer_mode="fully_turbulent",
        initial_turbulent_displacement_thickness=2.0e-5,
        initial_turbulent_momentum_thickness=5.0e-6,
    )
    inputs.update(overrides)
    return SupersonicStatorNozzle(**inputs)


def test_stator_passes_nasa_tm_x_2343_correlation_limit(monkeypatch):
    observed_limits = []
    original_solver = stator_nozzle_module.solve_boundary_layer

    def recording_solver(**kwargs):
        observed_limits.append(kwargs["laminar_correlation_limit"])
        return original_solver(**kwargs)

    monkeypatch.setattr(stator_nozzle_module, "solve_boundary_layer", recording_solver)
    make_stator()

    assert observed_limits == [0.16]


def test_nasa_tm_x_1502_ideal_nozzle_endpoint_regression():
    # NASA TM X-1502 table II prints x=64.15586 and y=12.97935 for
    # M_e=4.05, gamma=1.36, and requested delta-v=0.1 degree.  The Python
    # inverse Prandtl--Meyer solve is substantially tighter than the FORTRAN
    # tolerance, so a small difference from the rounded legacy values remains.
    construction = design_ideal_stator_nozzle(
        ideal_outlet_absolute_flow_mach=4.05,
        outlet_metal_angle=70.0,
        flow_turning_increment=0.1,
        gamma=1.36,
    )
    assert math.isclose(construction.shape.pressure.x[-1], 64.15586, rel_tol=1.0e-3)
    assert math.isclose(-construction.shape.pressure.y[-1], 12.97935, rel_tol=1.0e-3)
    assert math.isclose(construction.actual_flow_turning_increment, 0.09990, rel_tol=1.0e-4)


def test_conical_nozzle_uses_nasa_area_mach_relation():
    ideal_outlet_absolute_flow_mach = 2.4
    gamma = 1.36
    half_cone_metal_angle = 12.0
    outlet_metal_angle = 65.0
    construction = design_conical_stator_nozzle(
        ideal_outlet_absolute_flow_mach=ideal_outlet_absolute_flow_mach,
        outlet_metal_angle=outlet_metal_angle,
        half_cone_metal_angle=half_cone_metal_angle,
        gamma=gamma,
    )
    shape = construction.shape

    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    expected_area_ratio = (
        2.0 / (gamma + 1.0) * (1.0 + 0.5 * (gamma - 1.0) * ideal_outlet_absolute_flow_mach**2)
    ) ** exponent / ideal_outlet_absolute_flow_mach
    expected_exit_diameter_ratio = math.sqrt(expected_area_ratio)
    expected_divergent_length = 0.5 * (expected_exit_diameter_ratio - 1.0) / math.tan(math.radians(half_cone_metal_angle))
    expected_straight_length = expected_exit_diameter_ratio * math.tan(math.radians(outlet_metal_angle))

    assert math.isclose(
        isentropic_area_ratio(ideal_outlet_absolute_flow_mach, gamma), expected_area_ratio, rel_tol=1.0e-14
    )
    assert math.isclose(
        supersonic_mach_from_area_ratio(expected_area_ratio, gamma),
        ideal_outlet_absolute_flow_mach,
        rel_tol=1.0e-11,
    )
    assert math.isclose((shape.exit_opening / shape.throat_width) ** 2, expected_area_ratio, rel_tol=1.0e-14)
    assert math.isclose(shape.pressure.x[-1], expected_divergent_length, rel_tol=1.0e-14)
    assert math.isclose(shape.suction.x[-1] - shape.pressure.x[-1], expected_straight_length, rel_tol=1.0e-14)
    assert np.allclose(shape.suction.y[1:], 0.5 * expected_exit_diameter_ratio)
    assert shape.throat_width == 1.0
    assert shape.coordinate_scale == "throat diameter"
    assert construction.pressure_point_count == 2
    assert construction.actual_flow_turning_increment is None


def test_gamma_is_evaluated_at_self_consistent_static_throat():
    stator = make_stator()
    expected_temperature = stator.upstream_total_temperature * 2.0 / (stator.gamma + 1.0)
    expected_pressure = stator.upstream_total_pressure * (2.0 / (stator.gamma + 1.0)) ** (
        stator.gamma / (stator.gamma - 1.0)
    )
    assert math.isclose(stator.throat_static_temperature, expected_temperature, rel_tol=1.0e-12)
    assert math.isclose(stator.throat_static_pressure, expected_pressure, rel_tol=1.0e-12)
    assert stator.gamma == stator.throat_static_fluid_state.gamma


def test_choked_area_and_width_follow_mass_flow_equation():
    stator = make_stator()
    gamma = stator.gamma
    gas_constant = stator.fluid.specific_gas_constant
    expected_flux = (
        stator.upstream_total_pressure
        / math.sqrt(stator.upstream_total_temperature)
        * math.sqrt(gamma / gas_constant)
        * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
    )
    expected_area = stator.mass_flow_rate / expected_flux
    expected_width = expected_area / (stator.nozzle_count * stator.throat_height)
    assert math.isclose(stator.total_throat_area, expected_area, rel_tol=1.0e-12)
    assert math.isclose(stator.throat_width, expected_width, rel_tol=1.0e-12)
    assert math.isclose(stator.dimensional_shapes.uncorrected.throat_width, expected_width, rel_tol=1.0e-12)
    assert stator.throat_diameter is None
    assert stator.dimensional_shapes.ideal_throat_diameter is None
    assert math.isclose(stator.coordinate_scale_length, 0.5 * expected_width, rel_tol=1.0e-12)


def test_stores_corrected_uncorrected_and_dimensional_shapes():
    stator = make_stator()
    assert stator.uncorrected_shape.coordinate_scale == "throat half-width"
    assert stator.corrected_shape.coordinate_scale == "throat half-width"
    assert len(stator.suction_boundary_layer.s_over_chord) == len(stator.uncorrected_shape.suction.x)
    assert len(stator.suction_boundary_layer_marching.s_over_chord) == 81
    assert len(stator.uncorrected_shape.pressure.x) == stator.pressure_number_of_stations
    assert len(stator.corrected_shape.suction.x) == (len(stator.uncorrected_shape.suction.x) + 2)
    assert stator.corrected_shape.spacing > stator.uncorrected_shape.spacing
    assert stator.corrected_dimensional_shape.throat_width > stator.uncorrected_dimensional_shape.throat_width
    assert math.isclose(
        stator.physical_chord, stator.uncorrected_shape.chord * stator.throat_half_width_scale, rel_tol=1.0e-12
    )
    assert stator.outlet_metal_angle == stator.ideal_outlet_absolute_flow_angle
    assert stator.uncorrected_shape.pressure.absolute_flow_mach is not None
    assert stator.uncorrected_shape.pressure.relative_flow_mach is None
    assert stator.pressure_boundary_layer.freestream_absolute_flow_mach is not None
    assert stator.pressure_boundary_layer.freestream_relative_flow_mach is None


def test_conical_contour_reuses_bl_mixing_and_plotting_pipeline():
    stator = make_stator(
        throat_height=None,
        contour_method="conical",
        flow_turning_increment=None,
        half_cone_metal_angle=15.0,
        trailing_edge_thickness=1.0e-4,
    )
    expected_area_ratio = float(isentropic_area_ratio(stator.ideal_outlet_absolute_flow_mach, stator.gamma))

    assert stator.contour_method == "conical"
    assert stator.actual_flow_turning_increment is None
    assert math.isclose(stator.required_exit_area_ratio, expected_area_ratio, rel_tol=1.0e-13)
    assert math.isclose(
        (stator.uncorrected_shape.exit_opening / stator.uncorrected_shape.throat_width) ** 2,
        expected_area_ratio,
        rel_tol=1.0e-13,
    )
    expected_single_area = stator.total_throat_area / stator.nozzle_count
    expected_diameter = math.sqrt(4.0 * expected_single_area / math.pi)
    assert stator.throat_height is None
    assert stator.throat_width is None
    assert math.isclose(stator.single_nozzle_throat_area, expected_single_area, rel_tol=1.0e-14)
    assert math.isclose(stator.throat_diameter, expected_diameter, rel_tol=1.0e-14)
    assert math.isclose(stator.coordinate_scale_length, expected_diameter, rel_tol=1.0e-14)
    assert math.isclose(math.pi * stator.throat_diameter**2 / 4.0, expected_single_area, rel_tol=1.0e-14)
    assert math.isclose(stator.uncorrected_dimensional_shape.throat_width, expected_diameter, rel_tol=1.0e-14)
    assert stator.dimensional_shapes.ideal_throat_width is None
    assert math.isclose(stator.dimensional_shapes.ideal_throat_diameter, expected_diameter, rel_tol=1.0e-14)
    assert math.isclose(stator.dimensional_shapes.coordinate_scale_length, expected_diameter, rel_tol=1.0e-14)
    assert math.isclose(stator.physical_chord, stator.uncorrected_shape.chord * expected_diameter, rel_tol=1.0e-14)
    assert math.isclose(
        stator.conical_divergent_length,
        stator.conical_divergent_length_over_throat_diameter * expected_diameter,
        rel_tol=1.0e-14,
    )
    assert math.isclose(
        stator.trailing_edge_thickness_over_throat_diameter,
        stator.trailing_edge_thickness / expected_diameter,
        rel_tol=1.0e-14,
    )
    assert stator.trailing_edge_thickness_over_throat_half_width is None
    expected_te_blockage = stator.trailing_edge_thickness_over_throat_diameter / (
        stator.corrected_shape.spacing * math.cos(math.radians(stator.outlet_metal_angle))
    )
    assert math.isclose(
        stator.mixing_results["subsonic"]["trailing_edge_blockage_ratio"], expected_te_blockage, rel_tol=1.0e-13
    )
    assert len(stator.uncorrected_shape.pressure.x) == 2
    assert len(stator.uncorrected_shape.suction.x) == 12
    assert len(stator.corrected_shape.suction.x) == 14
    assert stator.boundary_layer_suction_station_count == 81
    assert math.isclose(
        stator.suction_boundary_layer_marching.freestream_absolute_flow_mach[-1],
        stator.ideal_outlet_absolute_flow_mach,
        rel_tol=1.0e-11,
    )
    assert math.isfinite(stator.real_outlet_absolute_flow_mach)
    figure, axes = stator.plot(dimensional=True, show=False)
    assert figure is axes.figure
    assert len(axes.lines) == 4


def test_conical_contour_supports_laminar_transition_mode():
    stator = make_stator(
        throat_height=None,
        contour_method="conical",
        flow_turning_increment=None,
        half_cone_metal_angle=15.0,
        boundary_layer_mode="laminar_then_turbulent",
        initial_turbulent_displacement_thickness=None,
        initial_turbulent_momentum_thickness=None,
    )
    assert stator.suction_boundary_layer.regime[0] == "laminar"
    assert math.isfinite(stator.real_outlet_absolute_flow_mach)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"throat_height": None}, "requires a positive finite throat_height"),
        ({"flow_turning_increment": None}, "requires flow_turning_increment"),
        (
            {"contour_method": "conical", "throat_height": None, "flow_turning_increment": None},
            "requires half_cone_metal_angle",
        ),
        ({"half_cone_metal_angle": 15.0}, "only valid"),
        (
            {"contour_method": "conical", "throat_height": None, "half_cone_metal_angle": 15.0},
            "flow_turning_increment is only valid",
        ),
        (
            {"contour_method": "conical", "flow_turning_increment": None, "half_cone_metal_angle": 15.0},
            "throat_height is only valid",
        ),
    ],
)
def test_contour_modes_require_only_their_own_inputs(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_stator(**overrides)


def test_stator_station_count_only_controls_dense_bl_march():
    coarse = make_stator(number_of_stations=57)
    fine = make_stator(number_of_stations=151)

    assert np.array_equal(coarse.uncorrected_shape.pressure.x, fine.uncorrected_shape.pressure.x)
    assert np.array_equal(coarse.uncorrected_shape.suction.x, fine.uncorrected_shape.suction.x)
    assert len(coarse.corrected_shape.suction.x) == (len(coarse.uncorrected_shape.suction.x) + 2)
    assert coarse.boundary_layer_suction_station_count == 57
    assert fine.boundary_layer_suction_station_count == 151
    assert len(fine.suction_boundary_layer.s_over_chord) == len(fine.uncorrected_shape.suction.x)


def test_subsonic_axial_flow_still_has_subsonic_mixing_solution():
    stator = make_stator()
    assert stator.ideal_outlet_absolute_axial_flow_mach < 1.0
    assert stator.mixing_solution == "subsonic"
    assert stator.mixing_results["subsonic"]["available"]
    assert not stator.mixing_results["supersonic"]["available"]
    assert math.isfinite(stator.real_outlet_absolute_flow_angle)


def test_supersonic_axial_flow_automatically_uses_supersonic_mixing_solution():
    stator = make_stator(requested_outlet_absolute_flow_mach=2.0, requested_outlet_absolute_flow_angle=30.0)

    assert stator.ideal_outlet_absolute_axial_flow_mach >= 1.0
    assert stator.supersonic_mixing_available
    assert stator.mixing_solution == "supersonic"
    assert stator.real_outlet_absolute_flow_mach == stator.mixing_results["supersonic"]["real_outlet_absolute_flow_mach"]


def test_stator_subsonic_mixing_solution_overrides_automatic_selection():
    stator = make_stator(requested_outlet_absolute_flow_mach=2.0, requested_outlet_absolute_flow_angle=30.0, mixing_solution="subsonic")

    assert stator.ideal_outlet_absolute_axial_flow_mach >= 1.0
    assert stator.supersonic_mixing_available
    assert stator.mixing_solution == "subsonic"
    assert stator.real_outlet_absolute_flow_mach == stator.mixing_results["subsonic"]["real_outlet_absolute_flow_mach"]


def test_trailing_edge_thickness_uses_nasa_tm_x_2343_afmix_blockage():
    sharp = make_stator(trailing_edge_thickness=0.0)
    finite = make_stator(trailing_edge_thickness=1.0e-4)

    # AFMIX treats TE as a downstream blockage/loss input. It must therefore
    # change the mixed state without silently modifying the MOC geometry,
    # boundary-layer correction, throat sizing, or dimensional scale.
    assert np.array_equal(finite.uncorrected_shape.pressure.x, sharp.uncorrected_shape.pressure.x)
    assert np.array_equal(finite.corrected_shape.suction.y, sharp.corrected_shape.suction.y)
    assert finite.throat_width == sharp.throat_width
    assert finite.real_outlet_absolute_flow_mach != sharp.real_outlet_absolute_flow_mach
    assert finite.real_outlet_absolute_flow_angle != sharp.real_outlet_absolute_flow_angle

    # NASA TM X-2343 defines DTE = TE / (SP*cos(ALPH1)). Both TE and SP below
    # are nondimensionalized by the same throat-half-width scale.
    expected_dte = finite.trailing_edge_thickness_over_throat_half_width / (
        finite.corrected_shape.spacing * math.cos(math.radians(finite.outlet_metal_angle))
    )
    assert math.isclose(
        finite.mixing_results["subsonic"]["trailing_edge_blockage_ratio"], expected_dte, rel_tol=1.0e-12
    )
    assert math.isclose(
        finite.trailing_edge_thickness_over_throat_half_width,
        finite.trailing_edge_thickness / finite.throat_half_width_scale,
        rel_tol=1.0e-12,
    )

    # Independently reconstruct the subsonic root from the variables named in
    # the FORTRAN listing. This guards the actual A/A1 use, rather than merely
    # checking that a nonzero input perturbs the answer.
    gamma = finite.gamma
    gp = gamma + 1.0
    gm = gamma - 1.0
    angle = math.radians(finite.outlet_metal_angle)
    velocity_ratio = math.sqrt((0.5 * gp * finite.ideal_outlet_absolute_flow_mach**2) / (1.0 + 0.5 * gm * finite.ideal_outlet_absolute_flow_mach**2))
    projected_spacing = finite.corrected_shape.spacing * math.cos(angle)
    pressure_displacement = (
        finite.pressure_boundary_layer.displacement_thickness_over_chord[-1] * finite.uncorrected_shape.chord
    )
    pressure_momentum = (
        finite.pressure_boundary_layer.momentum_thickness_over_chord[-1] * finite.uncorrected_shape.chord
    )
    displacement_ratio = (
        pressure_displacement + finite.corrected_exit_displacement_thickness / finite.throat_half_width_scale
    ) / projected_spacing
    momentum_ratio = (
        pressure_momentum + finite.corrected_exit_momentum_thickness / finite.throat_half_width_scale
    ) / projected_spacing
    area_momentum = 1.0 - displacement_ratio - expected_dte - momentum_ratio
    area = 1.0 - displacement_ratio - expected_dte
    afs = gm / gp * velocity_ratio**2
    c_value = ((1.0 - afs) * gp / (2.0 * gamma) + math.cos(angle) ** 2 * area_momentum * velocity_ratio**2) / (
        math.cos(angle) * area * velocity_ratio
    )
    d_value = velocity_ratio * math.sin(angle) * area_momentum / area
    radical = (gamma * c_value / gp) ** 2 - 1.0 + gm / gp * d_value**2
    axial_velocity_ratio = gamma * c_value / gp - math.sqrt(radical)
    total_velocity_ratio = math.hypot(d_value, axial_velocity_ratio)
    expected_mach = math.sqrt((2.0 / gp * total_velocity_ratio**2) / (1.0 - gm / gp * total_velocity_ratio**2))
    expected_angle = math.degrees(math.atan2(d_value, axial_velocity_ratio))
    assert math.isclose(finite.mixing_results["subsonic"]["real_outlet_absolute_flow_mach"], expected_mach, rel_tol=1.0e-12)
    assert math.isclose(finite.mixing_results["subsonic"]["real_outlet_absolute_flow_angle"], expected_angle, rel_tol=1.0e-12)


def test_trailing_edge_thickness_must_be_nonnegative():
    with pytest.raises(ValueError, match="trailing_edge_thickness"):
        make_stator(trailing_edge_thickness=-1.0e-6)


def test_supersonic_mixing_solution_override_is_rejected():
    with pytest.raises(ValueError, match="mixing_solution"):
        make_stator(mixing_solution="supersonic")


def test_iterated_outlet_metal_angle_matches_requested_real_flow_angle():
    stator = make_stator(iterate_outlet_metal_angle=True, trailing_edge_thickness=1.0e-4)
    assert abs(stator.real_outlet_absolute_flow_angle - stator.requested_outlet_absolute_flow_angle) < 2.0e-3
    assert abs(stator.outlet_metal_angle - stator.requested_outlet_absolute_flow_angle) > 0.05


def test_coupled_iteration_matches_real_outlet_absolute_flow_mach_and_angle():
    stator = make_stator(requested_outlet_absolute_flow_mach=1.77, iterate_outlet_metal_angle=True, match_real_outlet_absolute_flow_mach=True)
    assert abs(stator.real_outlet_absolute_flow_angle - stator.requested_outlet_absolute_flow_angle) < 2.0e-3
    assert abs(stator.real_outlet_absolute_flow_mach - stator.requested_outlet_absolute_flow_mach) < 1.0e-4
    assert not math.isclose(stator.ideal_outlet_absolute_flow_mach, stator.requested_outlet_absolute_flow_mach, rel_tol=1.0e-3)
    assert math.isclose(
        stator.ideal_outlet_absolute_axial_flow_mach,
        stator.ideal_outlet_absolute_flow_mach * math.cos(math.radians(stator.outlet_metal_angle)),
        rel_tol=1.0e-12,
    )


def test_coupled_stator_flag_requires_outlet_metal_angle_iteration():
    with pytest.raises(ValueError, match="iterate_outlet_metal_angle"):
        make_stator(match_real_outlet_absolute_flow_mach=True)


def test_coupled_stator_iteration_supports_supersonic_root():
    stator = make_stator(
        requested_outlet_absolute_flow_mach=2.0,
        requested_outlet_absolute_flow_angle=30.0,
        iterate_outlet_metal_angle=True,
        match_real_outlet_absolute_flow_mach=True,
    )
    assert abs(stator.real_outlet_absolute_flow_mach - 2.0) < 1.0e-4
    assert abs(stator.real_outlet_absolute_flow_angle - 30.0) < 2.0e-3
    assert stator.supersonic_mixing_available
    assert stator.mixing_solution == "supersonic"


def test_coupled_conical_iteration_varies_ideal_absolute_flow_mach():
    stator = make_stator(
        throat_height=None,
        contour_method="conical",
        flow_turning_increment=None,
        half_cone_metal_angle=15.0,
        iterate_outlet_metal_angle=True,
        match_real_outlet_absolute_flow_mach=True,
    )
    expected_area_ratio = float(isentropic_area_ratio(stator.ideal_outlet_absolute_flow_mach, stator.gamma))

    assert abs(stator.real_outlet_absolute_flow_mach - stator.requested_outlet_absolute_flow_mach) < 1.0e-4
    assert abs(stator.real_outlet_absolute_flow_angle - stator.requested_outlet_absolute_flow_angle) < 2.0e-3
    assert not math.isclose(stator.ideal_outlet_absolute_flow_mach, stator.requested_outlet_absolute_flow_mach, rel_tol=1.0e-3)
    assert math.isclose(stator.required_exit_area_ratio, expected_area_ratio, rel_tol=1.0e-13)
    assert math.isclose(
        (stator.uncorrected_shape.exit_opening / stator.uncorrected_shape.throat_width) ** 2,
        expected_area_ratio,
        rel_tol=1.0e-13,
    )


def test_plot_uses_rotated_geometry_and_returns_four_lines():
    stator = make_stator()
    figure, axes = stator.plot(dimensional=True, show=False)
    assert figure is axes.figure
    assert len(axes.lines) == 4

    pressure = stator.dimensional_shapes.uncorrected.pressure
    angle = math.radians(stator.outlet_metal_angle)
    expected_axial = 1000.0 * (pressure.x[0] * math.cos(angle) - pressure.y[0] * math.sin(angle))
    assert math.isclose(axes.lines[0].get_xdata()[0], expected_axial, rel_tol=1.0e-12)
    assert axes.get_xlabel() == "axial length [mm]"
    assert axes.get_ylabel() == "tangential length [mm]"
