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
        ideal_inlet_absolute_flow_mach=2.80,
        ideal_inlet_absolute_flow_angle=70.0,
        requested_outlet_absolute_flow_angle=-61.0,
        lower_surface_relative_flow_mach=1.75,
        upper_surface_relative_flow_mach=2.95,
        blade_count=36,
        mean_radius=0.20,
        rotational_speed_rpm=6000.0,
        fluid=Fluid(["Air"], [1.0]),
        inlet_total_temperature=1000.0,
        inlet_total_pressure=1.0e6,
        flow_turning_increment=0.5,
        number_of_stations=81,
        iterate_outlet_metal_angle=False,
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
        real_inlet_relative_flow_mach=mach_from_prandtl_meyer(math.radians(39.1203), gamma),
        ideal_outlet_relative_flow_mach=mach_from_prandtl_meyer(math.radians(39.1203), gamma),
        lower_surface_relative_flow_mach=mach_from_prandtl_meyer(math.radians(19.1203), gamma),
        upper_surface_relative_flow_mach=mach_from_prandtl_meyer(math.radians(49.1203), gamma),
        real_inlet_relative_flow_angle=70.0,
        ideal_outlet_relative_flow_angle=-61.13,
        inlet_metal_angle=70.0,
        outlet_metal_angle=-61.13,
        flow_turning_increment=0.1,
        gamma=gamma,
    )
    assert math.isclose(shape.inlet_pitch, 0.59436, rel_tol=3.0e-4)
    assert math.isclose(shape.chord / shape.inlet_pitch, 2.7603, rel_tol=3.0e-4)
    assert math.isclose(shape.chord / shape.outlet_pitch, 3.8966, rel_tol=3.0e-4)


@pytest.mark.parametrize(
    (
        "case_name",
        "gamma",
        "inlet_prandtl_meyer_angle",
        "lower_prandtl_meyer_angle",
        "upper_prandtl_meyer_angle",
        "total_flow_turning_angle",
        "reported_solidity",
    ),
    [
        ("8a", 1.4, 12.0, 0.0, 26.0, 130.0, 2.26),
        ("8b", 1.4, 12.0, 4.0, 26.0, 130.0, 3.06),
        ("8c", 1.4, 12.0, 8.0, 26.0, 130.0, 4.02),
        ("8d", 1.4, 12.0, 0.0, 77.0, 130.0, 1.96),
        ("8e", 1.4, 12.0, 0.0, 26.0, 110.0, 2.96),
        ("8f", 1.4, 12.0, 0.0, 26.0, 150.0, 1.39),
        ("9a", 1.4, 39.0, 0.0, 59.0, 130.0, 1.92),
        ("9b", 1.4, 39.0, 12.0, 59.0, 130.0, 2.56),
        ("9c", 1.4, 39.0, 18.0, 59.0, 130.0, 3.07),
        ("9d", 1.4, 39.0, 18.0, 104.0, 130.0, 2.95),
        ("9e", 1.4, 39.0, 21.0, 59.0, 120.0, 4.05),
        ("9f", 1.4, 39.0, 21.0, 59.0, 140.0, 2.76),
        ("10a", 1.4, 59.0, 12.0, 100.0, 130.0, 2.30),
        ("10b", 1.4, 59.0, 34.0, 100.0, 130.0, 3.49),
        ("10c", 1.4, 59.0, 45.0, 100.0, 130.0, 5.56),
        ("10d", 1.4, 59.0, 34.0, 59.0, 130.0, 3.97),
        ("10e", 1.4, 59.0, 40.0, 77.0, 130.0, 4.48),
        ("10f", 1.4, 59.0, 40.0, 77.0, 150.0, 2.65),
        ("11a", 1.4, 77.0, 39.0, 110.0, 130.0, 3.11),
        ("11b", 1.4, 77.0, 45.0, 110.0, 130.0, 3.48),
        ("11c", 1.4, 77.0, 58.0, 110.0, 130.0, 5.57),
        ("11d", 1.4, 77.0, 58.0, 77.0, 130.0, 6.32),
        ("11e", 1.4, 77.0, 62.0, 91.0, 140.0, 5.78),
        ("11f", 1.4, 77.0, 62.0, 91.0, 160.0, 2.81),
        ("12a", 1.3, 43.0, 0.0, 68.0, 130.0, 1.85),
        ("12b", 1.3, 43.0, 13.0, 68.0, 130.0, 2.38),
        ("12c", 1.3, 43.0, 19.0, 68.0, 130.0, 2.77),
        ("12d", 1.3, 43.0, 19.0, 108.0, 130.0, 2.70),
        ("12e", 1.3, 43.0, 25.0, 59.0, 120.0, 4.16),
        ("12f", 1.3, 43.0, 25.0, 59.0, 140.0, 2.85),
        ("13a", 1.66, 32.0, 0.0, 45.0, 130.0, 2.11),
        ("13b", 1.66, 32.0, 10.0, 45.0, 130.0, 3.01),
        ("13c", 1.66, 32.0, 15.0, 45.0, 130.0, 3.72),
        ("13d", 1.66, 32.0, 15.0, 89.0, 130.0, 3.47),
        ("13e", 1.66, 32.0, 15.0, 45.0, 120.0, 4.39),
        ("13f", 1.66, 32.0, 15.0, 45.0, 140.0, 3.00),
    ],
    ids=lambda value: str(value),
)
def test_nasa_tn_d_4422_impulse_blade_solidities(
    case_name,
    gamma,
    inlet_prandtl_meyer_angle,
    lower_prandtl_meyer_angle,
    upper_prandtl_meyer_angle,
    total_flow_turning_angle,
    reported_solidity,
):
    """Reproduce the ideal zero-edge-thickness cases in NASA TN D-4422, figures 8--13."""

    mach = lambda angle: mach_from_prandtl_meyer(math.radians(angle), gamma)
    inlet_flow_angle = 0.5 * total_flow_turning_angle
    shape = design_ideal_geometry(
        real_inlet_relative_flow_mach=mach(inlet_prandtl_meyer_angle),
        ideal_outlet_relative_flow_mach=mach(inlet_prandtl_meyer_angle),
        lower_surface_relative_flow_mach=mach(lower_prandtl_meyer_angle),
        upper_surface_relative_flow_mach=mach(upper_prandtl_meyer_angle),
        real_inlet_relative_flow_angle=inlet_flow_angle,
        ideal_outlet_relative_flow_angle=-inlet_flow_angle,
        inlet_metal_angle=inlet_flow_angle,
        outlet_metal_angle=-inlet_flow_angle,
        flow_turning_increment=0.1,
        gamma=gamma,
    )

    assert case_name
    assert abs(shape.chord / shape.inlet_pitch - reported_solidity) <= 6.0e-3
    assert np.all(np.isfinite(shape.pressure.x))
    assert np.all(np.isfinite(shape.pressure.y))
    assert np.all(np.isfinite(shape.suction.x))
    assert np.all(np.isfinite(shape.suction.y))


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
    assert blade.inlet_metal_angle == blade.real_inlet_relative_flow_angle
    assert blade.outlet_metal_angle == blade.ideal_outlet_relative_flow_angle
    assert blade.uncorrected_shape.pressure.absolute_flow_mach is None
    assert blade.uncorrected_shape.pressure.relative_flow_mach is not None
    assert blade.pressure_boundary_layer.freestream_absolute_flow_mach is None
    assert blade.pressure_boundary_layer.freestream_relative_flow_mach is not None
    expected_static_temperature = blade.inlet_total_temperature / (
        1.0 + 0.5 * (blade.gamma - 1.0) * blade.ideal_inlet_absolute_flow_mach**2
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


def test_rotor_scalar_flow_results_have_absolute_and_relative_pairs():
    with pytest.warns(RuntimeWarning, match="limited to zero"):
        blade = make_blade(leading_edge_thickness_over_total_pitch=0.05)

    for station in ("ideal_inlet", "real_inlet", "ideal_outlet", "real_outlet"):
        for quantity in ("flow_angle", "flow_mach"):
            assert hasattr(blade, f"{station}_absolute_{quantity}")
            assert hasattr(blade, f"{station}_relative_{quantity}")
    assert hasattr(blade, "ideal_outlet_absolute_axial_flow_mach")
    assert hasattr(blade, "ideal_outlet_relative_axial_flow_mach")
    assert hasattr(blade, "real_outlet_absolute_axial_flow_mach")
    assert hasattr(blade, "real_outlet_relative_axial_flow_mach")
    assert not math.isclose(
        blade.real_inlet_absolute_flow_mach, blade.real_inlet_relative_flow_mach, rel_tol=1.0e-3
    )


def test_flow_state_table_is_ordered_from_upstream_inlet_to_aftermixed_outlet():
    blade = make_blade()
    labels = tuple(row[0] for row in blade.flow_state_table.rows)

    assert labels == (
        "Ideal flow angle at the inlet upstream",
        "Ideal Mach number at the inlet upstream",
        "Real flow angle at the blade inlet",
        "Real Mach number at the blade inlet",
        "Ideal flow angle at the blade outlet",
        "Ideal Mach number at the blade outlet",
        "Real flow angle at the blade outlet",
        "Real Mach number at the blade outlet",
    )
    expected_values = (
        (blade.ideal_inlet_absolute_flow_angle, blade.ideal_inlet_relative_flow_angle),
        (blade.ideal_inlet_absolute_flow_mach, blade.ideal_inlet_relative_flow_mach),
        (blade.real_inlet_absolute_flow_angle, blade.real_inlet_relative_flow_angle),
        (blade.real_inlet_absolute_flow_mach, blade.real_inlet_relative_flow_mach),
        (blade.ideal_outlet_absolute_flow_angle, blade.ideal_outlet_relative_flow_angle),
        (blade.ideal_outlet_absolute_flow_mach, blade.ideal_outlet_relative_flow_mach),
        (blade.real_outlet_absolute_flow_angle, blade.real_outlet_relative_flow_angle),
        (blade.real_outlet_absolute_flow_mach, blade.real_outlet_relative_flow_mach),
    )
    assert tuple(row[1:] for row in blade.flow_state_table.rows) == expected_values
    printed = str(blade.flow_state_table)
    assert "Flow quantity" in printed
    assert "Absolute frame" in printed
    assert "Relative frame" in printed
    assert printed.splitlines()[2].startswith(labels[0])


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
    [({"lower_surface_relative_flow_mach": math.nan}, "finite"), ({"upper_surface_relative_flow_mach": math.inf}, "finite")],
)
def test_surface_mach_inputs_must_be_finite(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_blade(**overrides)


def test_surface_mach_inputs_obey_nasa_tn_d_4421_transition_ranges():
    # With this relatively axial inlet, dropping the pressure-side Mach too
    # far would require a Prandtl--Meyer transition turn larger than beta_in.
    with pytest.raises(ValueError, match="lower_surface_relative_flow_mach.*NASA TN D-4421 range"):
        make_blade(ideal_inlet_absolute_flow_angle=30.0, lower_surface_relative_flow_mach=1.50)

    # This value satisfies M_upper > M_in but violates the upper transition
    # turning limit for the specified inlet/outlet angles.
    with pytest.raises(ValueError, match="upper_surface_relative_flow_mach.*NASA TN D-4421 range"):
        make_blade(upper_surface_relative_flow_mach=12.0)


def test_absolute_inlet_is_converted_with_velocity_triangle():
    blade = make_blade()
    sound_speed = blade.inlet_static_fluid_state.speed_of_sound
    absolute_speed = blade.ideal_inlet_absolute_flow_mach * sound_speed
    absolute_angle = math.radians(blade.ideal_inlet_absolute_flow_angle)
    expected_axial = absolute_speed * math.cos(absolute_angle)
    expected_tangential = (
        absolute_speed * math.sin(absolute_angle)
        - 2.0 * math.pi * blade.mean_radius * blade.rotational_speed_rpm / 60.0
    )
    expected_relative_speed = math.hypot(expected_axial, expected_tangential)

    assert math.isclose(blade.ideal_inlet_relative_flow_mach, expected_relative_speed / sound_speed, rel_tol=1.0e-12)
    assert math.isclose(
        blade.ideal_inlet_relative_flow_angle,
        math.degrees(math.atan2(expected_tangential, expected_axial)),
        rel_tol=1.0e-12,
    )
    assert not math.isclose(blade.ideal_inlet_relative_flow_mach, blade.ideal_inlet_absolute_flow_mach, rel_tol=1.0e-3)


def test_relative_flow_input_set_reproduces_the_same_velocity_triangles_and_geometry():
    absolute = make_blade()
    relative = make_blade(
        ideal_inlet_absolute_flow_mach=None,
        ideal_inlet_absolute_flow_angle=None,
        requested_outlet_absolute_flow_angle=None,
        ideal_inlet_relative_flow_mach=absolute.ideal_inlet_relative_flow_mach,
        ideal_inlet_relative_flow_angle=absolute.ideal_inlet_relative_flow_angle,
        requested_outlet_relative_flow_angle=absolute.ideal_outlet_relative_flow_angle,
    )

    assert absolute.flow_input_reference_frame == "absolute"
    assert relative.flow_input_reference_frame == "relative"
    assert math.isclose(
        relative.requested_outlet_absolute_flow_angle,
        absolute.requested_outlet_absolute_flow_angle,
        abs_tol=1.0e-12,
    )
    for name in (
        "ideal_inlet_absolute_flow_mach",
        "ideal_inlet_absolute_flow_angle",
        "ideal_inlet_relative_flow_mach",
        "ideal_inlet_relative_flow_angle",
        "real_inlet_absolute_flow_mach",
        "real_inlet_absolute_flow_angle",
        "real_inlet_relative_flow_mach",
        "real_inlet_relative_flow_angle",
        "ideal_outlet_absolute_flow_mach",
        "ideal_outlet_absolute_flow_angle",
        "ideal_outlet_relative_flow_mach",
        "ideal_outlet_relative_flow_angle",
    ):
        assert math.isclose(getattr(relative, name), getattr(absolute, name), rel_tol=1.0e-12, abs_tol=1.0e-12)
    assert np.allclose(relative.uncorrected_shape.pressure.x, absolute.uncorrected_shape.pressure.x, atol=1.0e-13)
    assert np.allclose(relative.uncorrected_shape.pressure.y, absolute.uncorrected_shape.pressure.y, atol=1.0e-13)
    assert np.allclose(relative.uncorrected_shape.suction.x, absolute.uncorrected_shape.suction.x, atol=1.0e-13)
    assert np.allclose(relative.uncorrected_shape.suction.y, absolute.uncorrected_shape.suction.y, atol=1.0e-13)


def test_relative_flow_input_set_accepts_an_explicit_outlet_mach():
    absolute = make_blade(requested_outlet_absolute_flow_mach=2.0, mixing_solution="subsonic")
    relative = make_blade(
        ideal_inlet_absolute_flow_mach=None,
        ideal_inlet_absolute_flow_angle=None,
        requested_outlet_absolute_flow_angle=None,
        requested_outlet_absolute_flow_mach=None,
        ideal_inlet_relative_flow_mach=absolute.ideal_inlet_relative_flow_mach,
        ideal_inlet_relative_flow_angle=absolute.ideal_inlet_relative_flow_angle,
        requested_outlet_relative_flow_angle=absolute.ideal_outlet_relative_flow_angle,
        requested_outlet_relative_flow_mach=absolute.ideal_outlet_relative_flow_mach,
        mixing_solution="subsonic",
    )

    assert relative.requested_outlet_relative_flow_mach == absolute.ideal_outlet_relative_flow_mach
    assert math.isclose(relative.requested_outlet_absolute_flow_mach, 2.0, rel_tol=1.0e-12)
    assert math.isclose(relative.ideal_outlet_absolute_flow_mach, 2.0, rel_tol=1.0e-12)
    assert np.allclose(relative.uncorrected_shape.pressure.x, absolute.uncorrected_shape.pressure.x, atol=1.0e-13)
    assert np.allclose(relative.uncorrected_shape.suction.y, absolute.uncorrected_shape.suction.y, atol=1.0e-13)


def test_absolute_and_relative_flow_input_sets_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        make_blade(
            ideal_inlet_relative_flow_mach=2.5,
            ideal_inlet_relative_flow_angle=67.0,
            requested_outlet_relative_flow_angle=-64.0,
        )


def test_relative_flow_input_set_must_be_complete():
    with pytest.raises(ValueError, match="relative rotor flow input set is incomplete"):
        make_blade(
            ideal_inlet_absolute_flow_mach=None,
            ideal_inlet_absolute_flow_angle=None,
            requested_outlet_absolute_flow_angle=None,
            ideal_inlet_relative_flow_mach=2.5,
        )


def test_relative_flow_inputs_support_real_outlet_flow_angle_matching():
    reference = make_blade()
    matched = make_blade(
        ideal_inlet_absolute_flow_mach=None,
        ideal_inlet_absolute_flow_angle=None,
        requested_outlet_absolute_flow_angle=None,
        ideal_inlet_relative_flow_mach=reference.ideal_inlet_relative_flow_mach,
        ideal_inlet_relative_flow_angle=reference.ideal_inlet_relative_flow_angle,
        requested_outlet_relative_flow_angle=reference.real_outlet_relative_flow_angle,
        iterate_outlet_metal_angle=True,
    )

    assert matched.flow_input_reference_frame == "relative"
    assert abs(
        matched.real_outlet_relative_flow_angle - matched.requested_outlet_relative_flow_angle
    ) < 2.0e-3
    assert math.isclose(
        matched.ideal_outlet_relative_flow_mach,
        matched.ideal_inlet_relative_flow_mach,
        rel_tol=1.0e-12,
    )


def test_zero_leading_edge_thickness_preserves_passage_entry_state():
    blade = make_blade()

    assert blade.leading_edge_thickness_over_total_pitch == 0.0
    assert blade.use_leading_edge_entry_correction
    assert blade.real_inlet_relative_flow_mach == blade.ideal_inlet_relative_flow_mach
    assert blade.real_inlet_relative_flow_angle == blade.ideal_inlet_relative_flow_angle
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

    mach_i = blade.ideal_inlet_relative_flow_mach
    mach_e = blade.real_inlet_relative_flow_mach
    beta_i = math.radians(blade.ideal_inlet_relative_flow_angle)
    beta_e = math.radians(blade.real_inlet_relative_flow_angle)
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

    assert finite.real_inlet_relative_flow_mach == finite.ideal_inlet_relative_flow_mach
    assert finite.real_inlet_relative_flow_angle == finite.ideal_inlet_relative_flow_angle
    assert np.array_equal(finite.uncorrected_shape.pressure.x, baseline.uncorrected_shape.pressure.x)
    assert np.array_equal(finite.uncorrected_shape.pressure.y, baseline.uncorrected_shape.pressure.y)


@pytest.mark.parametrize("ratio", [-0.01, 1.0, math.inf])
def test_leading_edge_thickness_ratio_is_bounded(ratio):
    with pytest.raises(ValueError, match="thickness_over_total_pitch"):
        make_blade(leading_edge_thickness_over_total_pitch=ratio)


def test_external_wave_correction_warns_for_supersonic_axial_inflow():
    with pytest.warns(RuntimeWarning, match="supersonic rotor-relative axial Mach"):
        blade = make_blade(
            ideal_inlet_absolute_flow_angle=60.0, upper_surface_relative_flow_mach=3.2, leading_edge_thickness_over_total_pitch=0.20
        )

    assert blade.ideal_inlet_relative_flow_mach * math.cos(math.radians(blade.ideal_inlet_relative_flow_angle)) > 1.0


def test_absolute_outlet_flow_angle_uses_exit_velocity_triangle():
    blade = make_blade()
    temperature_factor = 1.0 + 0.5 * (blade.gamma - 1.0) * blade.ideal_outlet_relative_flow_mach**2
    static_temperature = blade.relative_inlet_total_temperature / temperature_factor
    sound_speed = math.sqrt(blade.gamma * blade.fluid.specific_gas_constant * static_temperature)
    relative_speed = blade.ideal_outlet_relative_flow_mach * sound_speed
    relative_angle = math.radians(blade.outlet_metal_angle)
    absolute_angle = math.degrees(
        math.atan2(
            relative_speed * math.sin(relative_angle) + blade.wheel_speed, relative_speed * math.cos(relative_angle)
        )
    )

    # In zero-deviation mode the inviscid relative exit direction equals the
    # metal angle, and its fixed-frame transform equals the requested angle.
    assert math.isclose(absolute_angle, blade.requested_outlet_absolute_flow_angle, abs_tol=1.0e-10)
    assert not math.isclose(blade.outlet_metal_angle, blade.requested_outlet_absolute_flow_angle, abs_tol=1.0e-3)
    selected = blade.mixing_results[blade.mixing_solution]
    assert selected["real_outlet_absolute_flow_angle"] == blade.real_outlet_absolute_flow_angle
    assert selected["real_outlet_absolute_flow_mach"] == blade.real_outlet_absolute_flow_mach
    assert not math.isclose(
        selected["real_outlet_absolute_flow_angle"],
        selected["real_outlet_relative_flow_angle"],
        abs_tol=1.0e-3,
    )


def test_rotor_default_mixing_solution_follows_premixing_axial_mach():
    blade = make_blade()

    assert blade.ideal_outlet_relative_axial_flow_mach >= 1.0
    assert blade.mixing_results["supersonic"]["available"]
    assert blade.mixing_solution == "supersonic"
    assert (
        blade.real_outlet_absolute_flow_mach
        == blade.mixing_results["supersonic"]["real_outlet_absolute_flow_mach"]
    )


def test_rotor_subsonic_mixing_solution_overrides_automatic_selection():
    blade = make_blade(mixing_solution="subsonic")

    assert blade.ideal_outlet_relative_axial_flow_mach >= 1.0
    assert blade.mixing_results["supersonic"]["available"]
    assert blade.mixing_solution == "subsonic"
    assert (
        blade.real_outlet_absolute_flow_mach
        == blade.mixing_results["subsonic"]["real_outlet_absolute_flow_mach"]
    )


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
    asymmetric = make_blade(requested_outlet_absolute_flow_mach=2.0, mixing_solution="subsonic")

    assert math.isclose(impulse.ideal_outlet_relative_flow_mach, impulse.ideal_inlet_relative_flow_mach, rel_tol=1.0e-12)
    assert math.isclose(asymmetric.requested_outlet_absolute_flow_mach, 2.0, rel_tol=1.0e-12)
    assert math.isclose(asymmetric.ideal_outlet_absolute_flow_angle, asymmetric.requested_outlet_absolute_flow_angle, abs_tol=1.0e-10)
    assert not math.isclose(asymmetric.ideal_outlet_relative_flow_mach, asymmetric.requested_outlet_absolute_flow_mach, rel_tol=1.0e-3)
    assert math.isclose(
        asymmetric.uncorrected_shape.pressure.relative_flow_mach[-1],
        asymmetric.ideal_outlet_relative_flow_mach,
        rel_tol=1.0e-12,
    )
    assert math.isclose(
        asymmetric.uncorrected_shape.suction.relative_flow_mach[-1],
        asymmetric.ideal_outlet_relative_flow_mach,
        rel_tol=1.0e-12,
    )
    assert not math.isclose(asymmetric.uncorrected_shape.chord, impulse.uncorrected_shape.chord, rel_tol=1.0e-3)


def test_iterated_outlet_metal_angle_keeps_specified_ideal_absolute_flow_mach():
    blade = make_blade(requested_outlet_absolute_flow_mach=2.2, requested_outlet_absolute_flow_angle=-56.0, iterate_outlet_metal_angle=True)
    assert math.isclose(blade.requested_outlet_absolute_flow_mach, 2.2, rel_tol=1.0e-12)
    assert abs(blade.real_outlet_absolute_flow_angle + 56.0) < 2.0e-3


def test_coupled_iteration_matches_real_absolute_flow_mach_and_angle():
    blade = make_blade(
        requested_outlet_absolute_flow_mach=2.1,
        requested_outlet_absolute_flow_angle=-56.0,
        iterate_outlet_metal_angle=True,
        match_real_outlet_mach=True,
    )
    assert abs(blade.real_outlet_absolute_flow_angle + 56.0) < 2.0e-3
    assert abs(blade.real_outlet_absolute_flow_mach - 2.1) < 1.0e-4
    assert not math.isclose(blade.ideal_outlet_absolute_flow_mach, blade.requested_outlet_absolute_flow_mach, rel_tol=1.0e-3)


def test_coupled_iteration_flag_requires_metal_angle_iteration_and_flow_mach():
    with pytest.raises(ValueError, match="iterate_outlet_metal_angle"):
        make_blade(requested_outlet_absolute_flow_mach=2.1, match_real_outlet_mach=True)


def test_coupled_iteration_matches_real_relative_flow_mach_and_angle():
    reference = make_blade()
    matched = make_blade(
        ideal_inlet_absolute_flow_mach=None,
        ideal_inlet_absolute_flow_angle=None,
        requested_outlet_absolute_flow_angle=None,
        ideal_inlet_relative_flow_mach=reference.ideal_inlet_relative_flow_mach,
        ideal_inlet_relative_flow_angle=reference.ideal_inlet_relative_flow_angle,
        requested_outlet_relative_flow_angle=reference.real_outlet_relative_flow_angle,
        requested_outlet_relative_flow_mach=reference.real_outlet_relative_flow_mach,
        iterate_outlet_metal_angle=True,
        match_real_outlet_mach=True,
    )

    assert matched.flow_input_reference_frame == "relative"
    assert abs(
        matched.real_outlet_relative_flow_angle - matched.requested_outlet_relative_flow_angle
    ) < 2.0e-3
    assert abs(
        matched.real_outlet_relative_flow_mach - matched.requested_outlet_relative_flow_mach
    ) < 1.0e-4
    assert not math.isclose(
        matched.ideal_outlet_relative_flow_mach,
        matched.requested_outlet_relative_flow_mach,
        rel_tol=1.0e-3,
    )


def test_legacy_pitch_closure_changes_metal_angle_and_closes_nasa_tm_x_2434_pitch():
    with pytest.warns(UserWarning, match="changes the outlet.*angle"):
        blade = make_blade(
            iterate_pitch_closure=True, mixing_solution="subsonic", flow_turning_increment=0.1
        )

    assert blade.pitch_closure_iteration_count is not None
    assert blade.pitch_closure_outlet_metal_angle == (blade.outlet_metal_angle)
    assert not math.isclose(blade.ideal_outlet_absolute_flow_angle, blade.requested_outlet_absolute_flow_angle, abs_tol=1.0e-3)
    assert abs(blade.pitch_closure_residual * blade.sonic_radius_scale) <= 1.0e-6
    assert blade.pitch_residual == blade.pitch_closure_residual
    assert not math.isclose(blade.corrected_pitch_residual, blade.pitch_closure_residual, abs_tol=1.0e-4)


def test_relative_flow_inputs_support_nasa_tm_x_2434_pitch_closure():
    initial = make_blade(flow_turning_increment=0.1)
    with pytest.warns(UserWarning, match="changes the outlet.*angle"):
        blade = make_blade(
            ideal_inlet_absolute_flow_mach=None,
            ideal_inlet_absolute_flow_angle=None,
            requested_outlet_absolute_flow_angle=None,
            ideal_inlet_relative_flow_mach=initial.ideal_inlet_relative_flow_mach,
            ideal_inlet_relative_flow_angle=initial.ideal_inlet_relative_flow_angle,
            requested_outlet_relative_flow_angle=initial.ideal_outlet_relative_flow_angle,
            iterate_pitch_closure=True,
            mixing_solution="subsonic",
            flow_turning_increment=0.1,
        )

    assert blade.flow_input_reference_frame == "relative"
    assert blade.pitch_closure_iteration_count is not None
    assert abs(blade.pitch_closure_residual * blade.sonic_radius_scale) <= 1.0e-6


def test_pitch_closure_keeps_trailing_edge_as_thick_as_leading_edge():
    with pytest.warns(UserWarning, match="changes the outlet.*angle"):
        blade = make_blade(
            iterate_pitch_closure=True, leading_edge_thickness_over_total_pitch=0.05, flow_turning_increment=0.1
        )

    assert blade.trailing_edge_thickness == blade.leading_edge_thickness
    assert blade.physical_trailing_edge_thickness == blade.physical_leading_edge_thickness


@pytest.mark.parametrize(
    "matching_flags",
    [
        {"iterate_outlet_metal_angle": True},
        {
            "iterate_outlet_metal_angle": True,
            "match_real_outlet_mach": True,
            "requested_outlet_absolute_flow_mach": 2.1,
        },
    ],
)
def test_pitch_closure_rejects_mixed_flow_matching(matching_flags):
    with pytest.raises(ValueError, match="incompatible"):
        make_blade(iterate_pitch_closure=True, **matching_flags)


def test_subsonic_premixing_axial_mach_selects_subsonic_root():
    blade = make_blade(requested_outlet_absolute_flow_angle=-65.0)

    assert blade.ideal_outlet_relative_axial_flow_mach < 1.0
    assert blade.mixing_solution == "subsonic"
    assert blade.mixing_results["subsonic"]["available"]
    assert not blade.mixing_results["supersonic"]["available"]
    assert not blade.supersonic_mixing_available
    assert math.isnan(blade.mixing_results["supersonic"]["real_outlet_absolute_flow_mach"])

    with pytest.raises(ValueError, match="requested_outlet_absolute_flow_mach target"):
        make_blade(
            requested_outlet_absolute_flow_mach=None,
            iterate_outlet_metal_angle=True,
            match_real_outlet_mach=True,
        )


def test_relative_coupled_iteration_requires_relative_flow_mach_target():
    with pytest.raises(ValueError, match="requested_outlet_relative_flow_mach target"):
        make_blade(
            ideal_inlet_absolute_flow_mach=None,
            ideal_inlet_absolute_flow_angle=None,
            requested_outlet_absolute_flow_angle=None,
            ideal_inlet_relative_flow_mach=2.5,
            ideal_inlet_relative_flow_angle=67.0,
            requested_outlet_relative_flow_angle=-64.0,
            iterate_outlet_metal_angle=True,
            match_real_outlet_mach=True,
        )


def test_supersonic_mixing_solution_override_is_rejected():
    with pytest.raises(ValueError, match="mixing_solution"):
        make_blade(mixing_solution="supersonic")


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
    projected_outlet_pitch = blade.corrected_shape.outlet_pitch * math.cos(math.radians(blade.outlet_metal_angle))
    expected_blockage = blade.trailing_edge_thickness / projected_outlet_pitch

    assert math.isclose(blade.trailing_edge_thickness, expected_trailing_edge, rel_tol=1.0e-12)
    assert blade.trailing_edge_thickness > 0.0
    for solution in ("subsonic", "supersonic"):
        assert math.isclose(
            blade.mixing_results[solution]["trailing_edge_blockage_ratio"], expected_blockage, rel_tol=1.0e-12
        )


def test_chord_reynolds_number_is_derived_from_dimensional_ideal_chord():
    blade = make_blade()
    expected_velocity = blade.ideal_inlet_relative_flow_mach * blade.inlet_static_fluid_state.speed_of_sound
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
    assert calculated.starting_result.maximum_starting_ideal_inlet_relative_flow_mach > 1.0


def test_iterated_outlet_metal_angle_matches_requested_real_flow_angle():
    blade = make_blade(requested_outlet_absolute_flow_angle=-57.5, iterate_outlet_metal_angle=True)
    assert abs(blade.real_outlet_absolute_flow_angle + 57.5) < 2.0e-3
    assert abs(blade.outlet_metal_angle + 57.5) > 0.1


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
