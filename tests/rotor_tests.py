import math

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from SupersonicTurbineBlading import Fluid, SupersonicRotorBlade
from SupersonicTurbineBlading.gas_dynamics import isentropic_area_ratio, mach_from_prandtl_meyer, prandtl_meyer_angle
from SupersonicTurbineBlading.rotor import rotor_blade as rotor_blade_module
from SupersonicTurbineBlading.rotor.rotor_geometry import design_ideal_geometry


def make_blade(**overrides):
    inputs = dict(
        inlet_mach=2.80,
        inlet_flow_angle_deg=70.0,
        outlet_flow_angle_deg=-61.0,
        lower_surface_mach=1.75,
        upper_surface_mach=2.95,
        blade_count=36,
        mean_radius=0.20,
        rotational_speed_rpm=6000.0,
        fluid=Fluid(["Air"], [1.0]),
        inlet_total_temperature=1000.0,
        inlet_total_pressure=1.0e6,
        turning_increment_deg=0.5,
        number_of_stations=81,
        iterate_outlet_blade_angle=False,
        calculate_starting=False,
        boundary_layer_mode="fully_turbulent",
        initial_turbulent_displacement_thickness=2.4e-4,
        initial_turbulent_momentum_thickness=6.0e-5,
    )
    inputs.update(overrides)
    return SupersonicRotorBlade(**inputs)


def test_fluid_uses_explicit_ideal_gas_mixing_rules():
    fluid = Fluid(["Nitrogen", "Oxygen"], [0.75, 0.25])
    state = fluid.properties(500.0, 2.0e5)

    expected_density = state.pressure / (fluid.specific_gas_constant * state.temperature)
    expected_cp = sum(
        fraction * component_cp
        for fraction, component_cp in zip(fluid.mass_fractions, state.component_specific_heats_cp)
    )
    expected_viscosity = sum(
        fraction * component_viscosity
        for fraction, component_viscosity in zip(fluid.mass_fractions, state.component_dynamic_viscosities)
    )

    assert math.isclose(sum(fluid.mole_fractions), 1.0, rel_tol=1.0e-14)
    assert math.isclose(state.density, expected_density, rel_tol=1.0e-14)
    assert math.isclose(state.specific_heat_cp, expected_cp, rel_tol=1.0e-14)
    assert math.isclose(state.dynamic_viscosity, expected_viscosity, rel_tol=1.0e-14)
    assert math.isclose(state.specific_heat_cv, state.specific_heat_cp - fluid.specific_gas_constant, rel_tol=1.0e-14)


def test_fluid_rejects_invalid_composition():
    try:
        Fluid(["Nitrogen", "Oxygen"], [0.7, 0.2])
    except ValueError as error:
        assert "sum to one" in str(error)
    else:
        raise AssertionError("invalid mass fractions were accepted")


def test_prandtl_meyer_round_trip():
    for mach in (1.01, 1.5, 2.5, 5.0):
        recovered = mach_from_prandtl_meyer(prandtl_meyer_angle(mach, 1.4), 1.4)
        assert math.isclose(recovered, mach, rel_tol=1.0e-10)


def test_nasa_tm_x_2434_example_geometry_regression():
    # Table II input: nu_in=nu_out=39.1203 deg, nu_low=19.1203 deg,
    # nu_up=49.1203 deg, beta_in=70 deg.  The listed final beta_out is
    # approximately -61.13 deg.
    gamma = 1.4
    shape = design_ideal_geometry(
        inlet_mach=mach_from_prandtl_meyer(math.radians(39.1203), gamma),
        outlet_mach=mach_from_prandtl_meyer(math.radians(39.1203), gamma),
        lower_surface_mach=mach_from_prandtl_meyer(math.radians(19.1203), gamma),
        upper_surface_mach=mach_from_prandtl_meyer(math.radians(49.1203), gamma),
        inlet_flow_angle_deg=70.0,
        outlet_blade_angle_deg=-61.13,
        turning_increment_deg=0.1,
        gamma=gamma,
    )
    assert math.isclose(shape.inlet_pitch, 0.59436, rel_tol=3.0e-4)
    assert math.isclose(shape.chord / shape.inlet_pitch, 2.7603, rel_tol=3.0e-4)
    assert math.isclose(shape.chord / shape.outlet_pitch, 3.8966, rel_tol=3.0e-4)


def test_object_stores_ideal_and_corrected_shapes():
    blade = make_blade()
    assert blade.uncorrected_shape.coordinate_scale == "vortex sonic radius r*"
    assert blade.corrected_shape.coordinate_scale == "vortex sonic radius r*"
    assert len(blade.pressure_boundary_layer.s_over_chord) == len(blade.uncorrected_shape.pressure.x)
    assert len(blade.suction_boundary_layer.s_over_chord) == len(blade.uncorrected_shape.suction.x)
    assert len(blade.pressure_boundary_layer_marching.s_over_chord) == 81
    assert len(blade.suction_boundary_layer_marching.s_over_chord) == 81
    assert np.allclose(blade.uncorrected_shape.pressure.x, blade.corrected_shape.pressure.x)
    assert not np.allclose(blade.uncorrected_shape.pressure.y, blade.corrected_shape.pressure.y)
    assert blade.pressure_boundary_layer.regime[0] == "turbulent"
    assert blade.gamma == blade.inlet_static_fluid_state.gamma
    assert not math.isclose(blade.gamma, blade.inlet_total_fluid_state.gamma, rel_tol=1.0e-4)
    assert blade.prandtl_number == blade.inlet_static_fluid_state.prandtl_number
    expected_static_temperature = blade.inlet_total_temperature / (
        1.0 + 0.5 * (blade.gamma - 1.0) * blade.inlet_mach**2
    )
    assert math.isclose(blade.inlet_static_temperature, expected_static_temperature, rel_tol=1.0e-10)
    assert math.isclose(
        blade.pressure_boundary_layer.displacement_thickness_over_chord[0],
        2.4e-4 / blade.physical_chord,
        rel_tol=1.0e-12,
    )
    assert math.isclose(
        blade.pressure_boundary_layer.momentum_thickness_over_chord[0], 6.0e-5 / blade.physical_chord, rel_tol=1.0e-12
    )


def test_rotor_passes_nasa_tm_x_2434_correlation_limit(monkeypatch):
    observed_limits = []
    original_solver = rotor_blade_module.solve_boundary_layer

    def recording_solver(**kwargs):
        observed_limits.append(kwargs["laminar_correlation_limit"])
        return original_solver(**kwargs)

    monkeypatch.setattr(rotor_blade_module, "solve_boundary_layer", recording_solver)
    make_blade()

    assert observed_limits == [0.50, 0.50]


def test_fully_turbulent_mode_requires_both_inlet_thicknesses():
    with pytest.raises(ValueError, match="requires initial"):
        make_blade(initial_turbulent_displacement_thickness=None, initial_turbulent_momentum_thickness=None)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [({"lower_surface_mach": math.nan}, "finite"), ({"upper_surface_mach": math.inf}, "finite")],
)
def test_surface_mach_inputs_must_be_finite(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_blade(**overrides)


def test_surface_mach_inputs_obey_nasa_tn_d_4421_transition_ranges():
    # With this relatively axial inlet, dropping the pressure-side Mach too
    # far would require a Prandtl--Meyer transition turn larger than beta_in.
    with pytest.raises(ValueError, match="lower_surface_mach.*NASA TN D-4421 range"):
        make_blade(inlet_flow_angle_deg=30.0, lower_surface_mach=1.50)

    # This value satisfies M_upper > M_in but violates the upper transition
    # turning limit for the specified inlet/outlet angles.
    with pytest.raises(ValueError, match="upper_surface_mach.*NASA TN D-4421 range"):
        make_blade(upper_surface_mach=12.0)


def test_absolute_inlet_is_converted_with_velocity_triangle():
    blade = make_blade()
    sound_speed = blade.inlet_static_fluid_state.speed_of_sound
    absolute_speed = blade.inlet_mach * sound_speed
    absolute_angle = math.radians(blade.inlet_flow_angle_deg)
    expected_axial = absolute_speed * math.cos(absolute_angle)
    expected_tangential = (
        absolute_speed * math.sin(absolute_angle)
        - 2.0 * math.pi * blade.mean_radius * blade.rotational_speed_rpm / 60.0
    )
    expected_relative_speed = math.hypot(expected_axial, expected_tangential)

    assert math.isclose(blade.relative_inlet_mach, expected_relative_speed / sound_speed, rel_tol=1.0e-12)
    assert math.isclose(
        blade.relative_inlet_flow_angle_deg,
        math.degrees(math.atan2(expected_tangential, expected_axial)),
        rel_tol=1.0e-12,
    )
    assert not math.isclose(blade.relative_inlet_mach, blade.inlet_mach, rel_tol=1.0e-3)


def test_zero_leading_edge_thickness_preserves_passage_entry_state():
    blade = make_blade()

    assert blade.leading_edge_thickness_over_total_pitch == 0.0
    assert blade.use_leading_edge_entry_correction
    assert blade.passage_inlet_mach == blade.relative_inlet_mach
    assert blade.passage_inlet_flow_angle_deg == blade.relative_inlet_flow_angle_deg
    assert blade.leading_edge_thickness == 0.0
    assert blade.trailing_edge_thickness == 0.0
    assert blade.physical_leading_edge_thickness == 0.0
    assert blade.physical_trailing_edge_thickness == 0.0
    assert blade.inlet_passage_pitch == blade.inlet_total_pitch
    assert blade.uncorrected_shape.inlet_passage_pitch == (blade.uncorrected_shape.inlet_pitch)


def test_external_wave_entry_correction_satisfies_naca_rm_l52b06_equations():
    ratio = 0.05
    with pytest.warns(RuntimeWarning, match="limited to zero"):
        blade = make_blade(leading_edge_thickness_over_total_pitch=ratio)

    mach_i = blade.relative_inlet_mach
    mach_e = blade.passage_inlet_mach
    beta_i = math.radians(blade.relative_inlet_flow_angle_deg)
    beta_e = math.radians(blade.passage_inlet_flow_angle_deg)
    nu_i = prandtl_meyer_angle(mach_i, blade.gamma)
    nu_e = prandtl_meyer_angle(mach_e, blade.gamma)
    geometric_area_ratio = (1.0 - ratio) * math.cos(beta_e) / math.cos(beta_i)
    isentropic_ratio = isentropic_area_ratio(mach_e, blade.gamma) / isentropic_area_ratio(mach_i, blade.gamma)

    # The library's positive inlet angle is the negative of the signed NACA RM L52B06
    # direction, hence beta_e-beta_i = nu_e-nu_i in this API convention.
    assert math.isclose(beta_e - beta_i, nu_e - nu_i, abs_tol=1.0e-12)
    assert math.isclose(geometric_area_ratio, isentropic_ratio, rel_tol=1.0e-12)
    assert mach_e < mach_i
    assert beta_e < beta_i


def test_leading_edge_entry_correction_can_be_disabled():
    baseline = make_blade()
    with pytest.warns(RuntimeWarning, match="limited to zero"):
        finite = make_blade(leading_edge_thickness_over_total_pitch=0.05, use_leading_edge_entry_correction=False)

    assert finite.passage_inlet_mach == finite.relative_inlet_mach
    assert finite.passage_inlet_flow_angle_deg == finite.relative_inlet_flow_angle_deg
    assert np.array_equal(finite.uncorrected_shape.pressure.x, baseline.uncorrected_shape.pressure.x)
    assert np.array_equal(finite.uncorrected_shape.pressure.y, baseline.uncorrected_shape.pressure.y)


@pytest.mark.parametrize("ratio", [-0.01, 1.0, math.inf])
def test_leading_edge_thickness_ratio_is_bounded(ratio):
    with pytest.raises(ValueError, match="thickness_over_total_pitch"):
        make_blade(leading_edge_thickness_over_total_pitch=ratio)


def test_external_wave_correction_warns_for_supersonic_axial_inflow():
    with pytest.warns(RuntimeWarning, match="supersonic rotor-relative axial Mach"):
        blade = make_blade(
            inlet_flow_angle_deg=60.0, upper_surface_mach=3.2, leading_edge_thickness_over_total_pitch=0.20
        )

    assert blade.relative_inlet_mach * math.cos(math.radians(blade.relative_inlet_flow_angle_deg)) > 1.0


def test_absolute_outlet_angle_uses_exit_velocity_triangle():
    blade = make_blade()
    temperature_factor = 1.0 + 0.5 * (blade.gamma - 1.0) * blade.relative_outlet_mach**2
    static_temperature = blade.relative_inlet_total_temperature / temperature_factor
    sound_speed = math.sqrt(blade.gamma * blade.fluid.specific_gas_constant * static_temperature)
    relative_speed = blade.relative_outlet_mach * sound_speed
    relative_angle = math.radians(blade.outlet_blade_angle_deg)
    absolute_angle = math.degrees(
        math.atan2(
            relative_speed * math.sin(relative_angle) + blade.wheel_speed, relative_speed * math.cos(relative_angle)
        )
    )

    # In zero-deviation mode the inviscid relative exit direction equals the
    # metal angle, and its fixed-frame transform equals the requested angle.
    assert math.isclose(absolute_angle, blade.outlet_flow_angle_deg, abs_tol=1.0e-10)
    assert not math.isclose(blade.outlet_blade_angle_deg, blade.outlet_flow_angle_deg, abs_tol=1.0e-3)
    selected = blade.mixing_results[blade.mixing_solution]
    assert selected["flow_angle_deg"] == selected["absolute_flow_angle_deg"]
    assert selected["mach"] == selected["absolute_mach"]
    assert not math.isclose(selected["absolute_flow_angle_deg"], selected["relative_flow_angle_deg"], abs_tol=1.0e-3)


def test_rotor_default_mixing_solution_is_subsonic():
    blade = make_blade()

    assert blade.mixing_solution == "subsonic"
    assert blade.mixing_results["subsonic"]["available"]


def test_rotor_station_count_only_controls_dense_bl_march():
    coarse = make_blade(number_of_stations=57)
    fine = make_blade(number_of_stations=151)

    # MOC geometry is independent of the temporary BL marching resolution.
    assert np.array_equal(coarse.uncorrected_shape.pressure.x, fine.uncorrected_shape.pressure.x)
    assert np.array_equal(coarse.uncorrected_shape.suction.x, fine.uncorrected_shape.suction.x)
    assert len(coarse.pressure_boundary_layer.s_over_chord) == len(coarse.uncorrected_shape.pressure.x)
    assert len(fine.suction_boundary_layer.s_over_chord) == len(fine.uncorrected_shape.suction.x)
    assert coarse.boundary_layer_pressure_station_count == max(57, len(coarse.uncorrected_shape.pressure.x))
    assert fine.boundary_layer_pressure_station_count == 151
    assert fine.boundary_layer_suction_station_count == 151


def test_optional_absolute_outlet_mach_controls_exit_construction():
    impulse = make_blade()
    asymmetric = make_blade(outlet_mach=2.0, mixing_solution="subsonic")

    assert math.isclose(impulse.relative_outlet_mach, impulse.relative_inlet_mach, rel_tol=1.0e-12)
    assert math.isclose(asymmetric.outlet_mach, 2.0, rel_tol=1.0e-12)
    assert math.isclose(asymmetric.ideal_outlet_flow_angle_deg, asymmetric.outlet_flow_angle_deg, abs_tol=1.0e-10)
    assert not math.isclose(asymmetric.relative_outlet_mach, asymmetric.outlet_mach, rel_tol=1.0e-3)
    assert math.isclose(
        asymmetric.uncorrected_shape.pressure.mach[-1], asymmetric.relative_outlet_mach, rel_tol=1.0e-12
    )
    assert math.isclose(asymmetric.uncorrected_shape.suction.mach[-1], asymmetric.relative_outlet_mach, rel_tol=1.0e-12)
    assert not math.isclose(asymmetric.uncorrected_shape.chord, impulse.uncorrected_shape.chord, rel_tol=1.0e-3)


def test_iterated_angle_keeps_specified_absolute_ideal_outlet_mach():
    blade = make_blade(outlet_mach=2.2, outlet_flow_angle_deg=-56.0, iterate_outlet_blade_angle=True)
    assert math.isclose(blade.outlet_mach, 2.2, rel_tol=1.0e-12)
    assert abs(blade.obtained_outlet_flow_angle_deg + 56.0) < 2.0e-3


def test_coupled_iteration_matches_absolute_mach_and_angle_after_mixing():
    blade = make_blade(
        outlet_mach=2.1,
        outlet_flow_angle_deg=-56.0,
        iterate_outlet_blade_angle=True,
        match_outlet_mach_after_mixing=True,
        mixing_solution="supersonic",
    )
    assert abs(blade.obtained_outlet_flow_angle_deg + 56.0) < 2.0e-3
    assert abs(blade.obtained_outlet_mach - 2.1) < 1.0e-4
    assert not math.isclose(blade.ideal_outlet_mach, blade.outlet_mach, rel_tol=1.0e-3)


def test_coupled_iteration_flag_requires_angle_iteration_and_mach():
    with pytest.raises(ValueError, match="iterate_outlet_blade_angle"):
        make_blade(outlet_mach=2.1, match_outlet_mach_after_mixing=True)


def test_legacy_pitch_closure_changes_angle_and_closes_nasa_tm_x_2434_pitch():
    with pytest.warns(UserWarning, match="changes the outlet.*angle"):
        blade = make_blade(iterate_pitch_closure=True, mixing_solution="subsonic")

    assert blade.pitch_closure_iteration_count is not None
    assert blade.pitch_closure_outlet_angle_deg == (blade.outlet_blade_angle_deg)
    assert not math.isclose(blade.ideal_outlet_flow_angle_deg, blade.outlet_flow_angle_deg, abs_tol=1.0e-3)
    assert abs(blade.pitch_closure_residual * blade.sonic_radius_scale) <= 1.0e-4
    assert blade.pitch_residual == blade.pitch_closure_residual
    assert not math.isclose(blade.corrected_pitch_residual, blade.pitch_closure_residual, abs_tol=1.0e-4)


def test_pitch_closure_keeps_trailing_edge_as_thick_as_leading_edge():
    with pytest.warns(UserWarning, match="changes the outlet.*angle"):
        blade = make_blade(iterate_pitch_closure=True, leading_edge_thickness_over_total_pitch=0.05)

    assert blade.trailing_edge_thickness == blade.leading_edge_thickness
    assert blade.physical_trailing_edge_thickness == blade.physical_leading_edge_thickness


@pytest.mark.parametrize(
    "matching_flags",
    [
        {"iterate_outlet_blade_angle": True},
        {"iterate_outlet_blade_angle": True, "match_outlet_mach_after_mixing": True, "outlet_mach": 2.1},
    ],
)
def test_pitch_closure_rejects_mixed_flow_matching(matching_flags):
    with pytest.raises(ValueError, match="incompatible"):
        make_blade(iterate_pitch_closure=True, **matching_flags)


def test_subsonic_premixing_axial_mach_disables_supersonic_root():
    blade = make_blade(outlet_flow_angle_deg=-65.0, mixing_solution="subsonic")

    assert blade.premixing_axial_mach < 1.0
    assert blade.mixing_results["subsonic"]["available"]
    assert not blade.mixing_results["supersonic"]["available"]
    assert not blade.supersonic_mixing_available
    assert math.isnan(blade.mixing_results["supersonic"]["mach"])

    with pytest.raises(RuntimeError, match="axial Mach"):
        make_blade(outlet_flow_angle_deg=-65.0, mixing_solution="supersonic")
    with pytest.raises(ValueError, match="outlet_mach target"):
        make_blade(outlet_mach=None, iterate_outlet_blade_angle=True, match_outlet_mach_after_mixing=True)


def test_dimensionalization_uses_mean_radius_and_blade_count():
    blade = make_blade()
    result = blade.dimensionalize()
    expected_pitch = 2.0 * math.pi * 0.20 / 36
    scaled_pitch = blade.uncorrected_shape.inlet_pitch * result.sonic_radius_scale
    assert math.isclose(scaled_pitch, expected_pitch, rel_tol=1.0e-12)
    assert math.isclose(
        result.corrected.chord, blade.corrected_shape.chord * result.sonic_radius_scale, rel_tol=1.0e-12
    )


def test_finite_leading_edge_separates_total_and_passage_pitch():
    ratio = 0.20
    blade = make_blade(leading_edge_thickness_over_total_pitch=ratio)
    result = blade.dimensionalize()
    expected_total_pitch = 2.0 * math.pi * 0.20 / 36

    assert math.isclose(
        blade.inlet_total_pitch, blade.inlet_passage_pitch + blade.leading_edge_thickness, rel_tol=1.0e-12
    )
    assert math.isclose(blade.leading_edge_thickness / blade.inlet_total_pitch, ratio, rel_tol=1.0e-12)
    assert math.isclose(blade.physical_total_pitch, expected_total_pitch, rel_tol=1.0e-12)
    assert math.isclose(
        blade.physical_passage_pitch + blade.physical_leading_edge_thickness, expected_total_pitch, rel_tol=1.0e-12
    )
    assert math.isclose(result.uncorrected.inlet_pitch, blade.physical_passage_pitch, rel_tol=1.0e-12)


def test_nonclosure_trailing_edge_and_aftermixing_include_metal_blockage():
    blade = make_blade(leading_edge_thickness_over_total_pitch=0.20)
    expected_trailing_edge = max(0.0, blade.leading_edge_thickness - blade.trailing_edge_vertical_boundary_layer_height)
    projected_outlet_pitch = blade.corrected_shape.outlet_pitch * math.cos(math.radians(blade.outlet_blade_angle_deg))
    expected_blockage = blade.trailing_edge_thickness / projected_outlet_pitch

    assert math.isclose(blade.trailing_edge_thickness, expected_trailing_edge, rel_tol=1.0e-12)
    assert blade.trailing_edge_thickness > 0.0
    for solution in ("subsonic", "supersonic"):
        assert math.isclose(
            blade.mixing_results[solution]["trailing_edge_blockage_ratio"], expected_blockage, rel_tol=1.0e-12
        )


def test_chord_reynolds_number_is_derived_from_dimensional_ideal_chord():
    blade = make_blade()
    expected_velocity = blade.relative_inlet_mach * blade.inlet_static_fluid_state.speed_of_sound
    expected_reynolds_number = (
        expected_velocity * blade.physical_chord / blade.inlet_static_fluid_state.kinematic_viscosity
    )

    assert math.isclose(blade.chord_reynolds_number, expected_reynolds_number, rel_tol=1.0e-12)
    assert math.isclose(blade.physical_chord, blade.uncorrected_shape.chord * blade.sonic_radius_scale, rel_tol=1.0e-12)


def test_reynolds_number_scales_with_initialized_mean_radius():
    reference = make_blade(mean_radius=0.20)
    # Keep wheel speed, and therefore the inlet velocity triangle, fixed while
    # doubling radius by halving RPM.
    doubled = make_blade(mean_radius=0.40, rotational_speed_rpm=3000.0)

    assert math.isclose(doubled.physical_chord, 2.0 * reference.physical_chord, rel_tol=1.0e-12)
    assert math.isclose(doubled.chord_reynolds_number, 2.0 * reference.chord_reynolds_number, rel_tol=1.0e-12)


def test_starting_flag():
    skipped = make_blade(calculate_starting=False)
    calculated = make_blade(calculate_starting=True)
    assert skipped.starting_result is None
    assert calculated.starting_result is not None
    assert calculated.starting_result.maximum_starting_inlet_mach > 1.0


def test_iterated_blade_angle_matches_requested_mixed_angle():
    blade = make_blade(outlet_flow_angle_deg=-57.5, iterate_outlet_blade_angle=True)
    assert abs(blade.obtained_outlet_flow_angle_deg + 57.5) < 2.0e-3
    assert abs(blade.outlet_blade_angle_deg + 57.5) > 0.1


def test_plot_pairs_opposite_surfaces_at_common_leading_edges():
    blade = make_blade()

    # The corrected flag selects exactly one geometry. Each one contributes
    # four surface lines in physical top-to-bottom order: upper-blade
    # suction/pressure, then lower-blade suction/pressure. Four additional
    # lines close the upper and lower leading and trailing edges independently.
    for corrected, shape in ((True, blade.corrected_shape), (False, blade.uncorrected_shape)):
        figure, axes = blade.plot(corrected=corrected, show=False)
        assert figure is axes.figure
        assert len(axes.lines) == 8
        upper_suction = axes.lines[0]
        upper_pressure = axes.lines[1]
        lower_suction = axes.lines[2]
        lower_pressure = axes.lines[3]
        upper_leading_edge = axes.lines[4]
        lower_leading_edge = axes.lines[5]
        upper_trailing_edge = axes.lines[6]
        lower_trailing_edge = axes.lines[7]

        assert math.isclose(upper_suction.get_xdata()[0], upper_pressure.get_xdata()[0], abs_tol=1.0e-14)
        assert math.isclose(upper_suction.get_ydata()[0], upper_pressure.get_ydata()[0], abs_tol=1.0e-14)
        assert math.isclose(lower_suction.get_xdata()[0], lower_pressure.get_xdata()[0], abs_tol=1.0e-14)
        assert math.isclose(lower_suction.get_ydata()[0], lower_pressure.get_ydata()[0], abs_tol=1.0e-14)

        translation_x = shape.pressure.x[0] - shape.suction.x[0]
        translation_y = shape.pressure.y[0] - shape.suction.y[0]
        assert np.allclose(upper_suction.get_xdata(), shape.suction.x + translation_x)
        assert np.allclose(upper_suction.get_ydata(), shape.suction.y + translation_y)
        assert np.array_equal(upper_pressure.get_xdata(), shape.pressure.x)
        assert np.array_equal(lower_suction.get_xdata(), shape.suction.x)
        assert np.allclose(lower_pressure.get_xdata(), shape.pressure.x - translation_x)
        assert np.allclose(lower_pressure.get_ydata(), shape.pressure.y - translation_y)
        assert np.allclose(
            upper_leading_edge.get_xdata(), [upper_suction.get_xdata()[0], upper_pressure.get_xdata()[0]]
        )
        assert np.allclose(
            upper_leading_edge.get_ydata(), [upper_suction.get_ydata()[0], upper_pressure.get_ydata()[0]]
        )
        assert np.allclose(
            lower_leading_edge.get_xdata(), [lower_suction.get_xdata()[0], lower_pressure.get_xdata()[0]]
        )
        assert np.allclose(
            lower_leading_edge.get_ydata(), [lower_suction.get_ydata()[0], lower_pressure.get_ydata()[0]]
        )
        assert np.allclose(
            upper_trailing_edge.get_xdata(), [upper_suction.get_xdata()[-1], upper_pressure.get_xdata()[-1]]
        )
        assert np.allclose(
            upper_trailing_edge.get_ydata(), [upper_suction.get_ydata()[-1], upper_pressure.get_ydata()[-1]]
        )
        assert np.allclose(
            lower_trailing_edge.get_xdata(), [lower_suction.get_xdata()[-1], lower_pressure.get_xdata()[-1]]
        )
        assert np.allclose(
            lower_trailing_edge.get_ydata(), [lower_suction.get_ydata()[-1], lower_pressure.get_ydata()[-1]]
        )

    passage_figure, passage_axes = blade.plot(corrected=False, show_two_blades=False, show=False)
    assert passage_figure is passage_axes.figure
    assert len(passage_axes.lines) == 2


def test_plot_adds_leading_edge_thickness_only_to_outer_surfaces():
    blade = make_blade(leading_edge_thickness_over_total_pitch=0.20)
    shape = blade.corrected_shape
    figure, axes = blade.plot(corrected=True, show=False)
    upper_suction, upper_pressure, lower_suction, lower_pressure, upper_leading_edge, lower_leading_edge = axes.lines[
        :6
    ]
    translation_x = shape.pressure.x[0] - shape.suction.x[0]
    translation_y = shape.pressure.y[0] - shape.suction.y[0]

    assert np.array_equal(upper_pressure.get_ydata(), shape.pressure.y)
    assert np.array_equal(lower_suction.get_ydata(), shape.suction.y)
    assert np.allclose(upper_suction.get_xdata(), shape.suction.x + translation_x)
    assert np.allclose(upper_suction.get_ydata(), shape.suction.y + translation_y + blade.leading_edge_thickness)
    assert np.allclose(lower_pressure.get_xdata(), shape.pressure.x - translation_x)
    assert np.allclose(lower_pressure.get_ydata(), shape.pressure.y - translation_y - blade.leading_edge_thickness)
    assert np.allclose(
        upper_leading_edge.get_ydata(), [shape.pressure.y[0] + blade.leading_edge_thickness, shape.pressure.y[0]]
    )
    assert np.allclose(
        lower_leading_edge.get_ydata(), [shape.suction.y[0], shape.suction.y[0] - blade.leading_edge_thickness]
    )

    dimensional_figure, dimensional_axes = blade.plot(dimensional=True, corrected=True, show=False)
    dimensional_shape = blade.dimensional_shapes.corrected
    dimensional_upper_suction = dimensional_axes.lines[0]
    dimensional_upper_leading_edge = dimensional_axes.lines[4]
    dimensional_translation_y = dimensional_shape.pressure.y[0] - dimensional_shape.suction.y[0]
    assert np.allclose(
        dimensional_upper_suction.get_ydata(),
        1000.0 * (dimensional_shape.suction.y + dimensional_translation_y + blade.physical_leading_edge_thickness),
    )
    assert math.isclose(
        abs(float(np.diff(dimensional_upper_leading_edge.get_ydata())[0])),
        1000.0 * blade.physical_leading_edge_thickness,
    )
    assert dimensional_axes.get_xlabel() == "length [mm]"
    assert dimensional_axes.get_ylabel() == "length [mm]"


@pytest.mark.parametrize("thickness_ratio", [0.0, 0.20])
def test_cad_profiles_store_corrected_and_uncorrected_geometry_in_millimetres(thickness_ratio):
    blade = make_blade(leading_edge_thickness_over_total_pitch=thickness_ratio)

    profiles = (
        (blade.corrected_shape, blade.blade_profile_x_CAD, blade.blade_profile_y_CAD),
        (blade.uncorrected_shape, blade.uncorrected_blade_profile_x_CAD, blade.uncorrected_blade_profile_y_CAD),
    )
    for nondimensional_shape, profile_x_CAD, profile_y_CAD in profiles:
        shape = nondimensional_shape.scaled(blade.sonic_radius_scale, "dimensional [m]")
        translation_x = shape.pressure.x[0] - shape.suction.x[0]
        translation_y = shape.pressure.y[0] - shape.suction.y[0]

        lower_x = shape.pressure.x - shape.pressure.x[0]
        lower_y = shape.pressure.y - shape.pressure.y[0]
        upper_x = shape.suction.x + translation_x - shape.pressure.x[0]
        upper_y = shape.suction.y + translation_y + blade.physical_leading_edge_thickness - shape.pressure.y[0]
        upper_x_reversed = upper_x[::-1]
        upper_y_reversed = upper_y[::-1]
        if thickness_ratio == 0.0:
            upper_x_reversed = upper_x_reversed[:-1]
            upper_y_reversed = upper_y_reversed[:-1]
        expected_x = np.concatenate((lower_x, upper_x_reversed))
        expected_y = np.concatenate((lower_y, upper_y_reversed))

        assert isinstance(profile_x_CAD, np.ndarray)
        assert isinstance(profile_y_CAD, np.ndarray)
        assert np.array_equal(profile_x_CAD[:1], [0.0])
        assert np.array_equal(profile_y_CAD[:1], [0.0])
        origin_count = np.count_nonzero(
            np.logical_and(
                np.isclose(profile_x_CAD, 0.0, rtol=0.0, atol=1.0e-12),
                np.isclose(profile_y_CAD, 0.0, rtol=0.0, atol=1.0e-12),
            )
        )
        assert origin_count == 1
        assert np.allclose(profile_x_CAD, 1000.0 * expected_x)
        assert np.allclose(profile_y_CAD, 1000.0 * expected_y)
        if thickness_ratio > 0.0:
            assert math.isclose(profile_x_CAD[-1], 0.0, abs_tol=1.0e-12)
            assert math.isclose(profile_y_CAD[-1], 1000.0 * blade.physical_leading_edge_thickness, rel_tol=1.0e-12)


def test_plot_connectors_follow_corrected_or_uncorrected_line_style_without_pitch_closure():
    blade = make_blade(iterate_pitch_closure=False, leading_edge_thickness_over_total_pitch=0.20)

    for corrected in (True, False):
        figure, axes = blade.plot(corrected=corrected, show=False)
        assert figure is axes.figure
        assert len(axes.lines) == 8
        upper_suction, upper_pressure, lower_suction, lower_pressure = axes.lines[:4]
        upper_leading_edge, lower_leading_edge = axes.lines[4:6]
        upper_trailing_edge, lower_trailing_edge = axes.lines[6:]

        expected_linestyle = "-" if corrected else "--"
        for closure in (upper_leading_edge, lower_leading_edge, upper_trailing_edge, lower_trailing_edge):
            assert closure.get_linestyle() == expected_linestyle
        assert np.allclose(
            upper_trailing_edge.get_xdata(), [upper_suction.get_xdata()[-1], upper_pressure.get_xdata()[-1]]
        )
        assert np.allclose(
            upper_trailing_edge.get_ydata(), [upper_suction.get_ydata()[-1], upper_pressure.get_ydata()[-1]]
        )
        assert np.allclose(
            lower_trailing_edge.get_xdata(), [lower_suction.get_xdata()[-1], lower_pressure.get_xdata()[-1]]
        )
        assert np.allclose(
            lower_trailing_edge.get_ydata(), [lower_suction.get_ydata()[-1], lower_pressure.get_ydata()[-1]]
        )


def test_plot_rejects_non_boolean_corrected_flag():
    blade = make_blade()
    with pytest.raises(TypeError, match="corrected"):
        blade.plot(corrected="yes", show=False)
