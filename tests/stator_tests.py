import math

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from SupersonicTurbineBlading import Fluid, SupersonicStatorNozzle
from SupersonicTurbineBlading.gas_dynamics import isentropic_area_ratio, supersonic_mach_from_area_ratio
from SupersonicTurbineBlading.stator import stator_nozzle as stator_nozzle_module
from SupersonicTurbineBlading.stator.stator_geometry import design_conical_stator_nozzle, design_ideal_stator_nozzle


def make_stator(**overrides):
    inputs = dict(
        requested_outlet_absolute_flow_mach=1.77,
        requested_outlet_absolute_flow_angle=70.0,
        reference_length=0.04,
        fluid=Fluid(["Air"], [1.0]),
        upstream_total_temperature=900.0,
        upstream_total_pressure=1.0e6,
        trailing_edge_thickness=0.002,
        number_of_nodes=41,
        iterate_outlet_metal_angle=False,
        boundary_layer_mode="fully_turbulent",
        initial_turbulent_displacement_thickness=2.0e-5,
        initial_turbulent_momentum_thickness=5.0e-6,
    )
    if overrides.get("contour_method") == "conical":
        inputs.update(boundary_layer_mode=None,
                      initial_turbulent_displacement_thickness=None,
                      initial_turbulent_momentum_thickness=None)
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


def test_stator_uses_rocket_turbines_gas():
    IdealGas = pytest.importorskip("IdealGas").IdealGas
    gas = IdealGas(T_min=300.0, T_max=1100.0, species=["Nitrogen", "Oxygen"],
                   mass_fractions=[0.767, 0.233], cea_species=["N2", "O2"])
    stator = make_stator(fluid=Fluid(ideal_gas=gas))
    assert stator.gamma == gas.gamma
    assert stator.throat_static_fluid_state.specific_heat_cp == gas.Cp
    assert stator.upstream_total_fluid_state.dynamic_viscosity == gas.calculate_dynamic_viscosity(900.0)
    assert stator.upstream_total_fluid_state.thermal_conductivity == gas.calculate_thermal_conductivity(900.0)
    assert np.all(np.isfinite(stator.suction_boundary_layer.displacement_thickness_over_chord))


def test_nasa_tm_x_1502_ideal_nozzle_endpoint_regression():
    # NASA TM X-1502 table II prints x=64.15586 and y=12.97935 for
    # M_e=4.05, gamma=1.36, and requested delta-v=0.1 degree. The equivalent
    # fixed mesh has 351 wall nodes including the throat and exit. The Python
    # inverse Prandtl--Meyer solve is substantially tighter than the FORTRAN
    # tolerance, so a small difference from the rounded legacy values remains.
    construction = design_ideal_stator_nozzle(
        ideal_outlet_absolute_flow_mach=4.05,
        outlet_metal_angle=70.0,
        number_of_nodes=351,
        gamma=1.36,
    )
    assert math.isclose(construction.shape.pressure_surface.x[-1], 64.15586, rel_tol=1.0e-3)
    assert math.isclose(-construction.shape.pressure_surface.y[-1], 12.97935, rel_tol=1.0e-3)
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
        number_of_nodes=51,
        gamma=gamma,
    )
    shape = construction.shape

    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    expected_area_ratio = (
        2.0 / (gamma + 1.0) * (1.0 + 0.5 * (gamma - 1.0) * ideal_outlet_absolute_flow_mach**2)
    ) ** exponent / ideal_outlet_absolute_flow_mach
    expected_exit_diameter_ratio = math.sqrt(expected_area_ratio)
    expected_divergent_length = (
        0.5 * (expected_exit_diameter_ratio - 1.0) / math.tan(math.radians(half_cone_metal_angle)))
    expected_straight_length = expected_exit_diameter_ratio * math.tan(math.radians(outlet_metal_angle))

    assert math.isclose(
        isentropic_area_ratio(ideal_outlet_absolute_flow_mach, gamma), expected_area_ratio, rel_tol=1.0e-14
    )
    assert math.isclose(
        supersonic_mach_from_area_ratio(expected_area_ratio, gamma),
        ideal_outlet_absolute_flow_mach,
        rel_tol=1.0e-11,
    )
    assert math.isclose((shape.nozzle_exit_width / shape.throat_width) ** 2,
                        expected_area_ratio, rel_tol=1.0e-14)
    assert math.isclose(shape.pressure_surface.x[-1], expected_divergent_length, rel_tol=1.0e-14)
    assert math.isclose(shape.suction_surface.x[-1] - shape.pressure_surface.x[-1], expected_straight_length,
                        rel_tol=1.0e-14)
    assert np.allclose(shape.suction_surface.y[50:], 0.5 * expected_exit_diameter_ratio)
    assert np.allclose(
        shape.pressure_surface.absolute_flow_mach,
        [supersonic_mach_from_area_ratio((2.0 * radius) ** 2, gamma) for radius in -shape.pressure_surface.y],
    )
    assert shape.throat_width == 1.0
    assert shape.coordinate_scale == "throat diameter"
    assert construction.pressure_point_count == 51
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


@pytest.mark.parametrize("contour_method", ["moc", "conical"])
def test_reference_length_sets_geometry_scale_and_metal_is_added_to_open_pitch(contour_method):
    options = {} if contour_method == "moc" else {"half_cone_metal_angle": 15.0}
    stator = make_stator(contour_method=contour_method, **options)
    nondimensional = stator.nondimensional_shapes.uncorrected
    dimensional = stator.dimensional_shapes.uncorrected
    if contour_method == "moc":
        assert math.isclose(dimensional.nozzle_passage_pitch, stator.reference_length, rel_tol=1.0e-12)
    else:
        assert math.isclose(dimensional.throat_width, stator.reference_length, rel_tol=1.0e-12)
    expected_open_pitch = dimensional.nozzle_exit_width / math.cos(math.radians(stator.outlet_metal_angle))
    expected_total_pitch = expected_open_pitch + stator.trailing_edge_thickness
    assert math.isclose(dimensional.nozzle_passage_pitch, expected_open_pitch, rel_tol=1.0e-12)
    assert math.isclose(stator.dimensional_total_pitch, expected_total_pitch, rel_tol=1.0e-12)
    assert math.isclose(dimensional.total_pitch, expected_total_pitch, rel_tol=1.0e-12)
    assert math.isclose(dimensional.trailing_edge_thickness, stator.trailing_edge_thickness, rel_tol=1.0e-12)
    assert math.isclose(stator.dimensional_scale_factor,
                        expected_total_pitch / nondimensional.total_pitch, rel_tol=1.0e-12)
    assert math.isclose(stator.dimensional_shapes.reference_length, stator.reference_length, rel_tol=1.0e-12)
    if contour_method == "moc":
        assert np.allclose(stator.dimensional_shapes.corrected.suction_surface.x,
                           stator.nondimensional_shapes.corrected.suction_surface.x * stator.dimensional_scale_factor)
    else:
        assert stator.dimensional_shapes.corrected is None


@pytest.mark.parametrize("contour_method", ["moc", "conical"])
def test_reference_length_scales_ideal_contour_independently_of_trailing_edge_metal(contour_method):
    options = {} if contour_method == "moc" else {"half_cone_metal_angle": 15.0}
    original = make_stator(contour_method=contour_method, trailing_edge_thickness=0.004, **options)
    doubled = make_stator(contour_method=contour_method, reference_length=0.08,
                          trailing_edge_thickness=0.004, **options)

    assert math.isclose(doubled.dimensional_scale_factor, 2.0 * original.dimensional_scale_factor, rel_tol=1.0e-12)
    assert np.allclose(doubled.dimensional_shapes.uncorrected.pressure_surface.x,
                       2.0 * original.dimensional_shapes.uncorrected.pressure_surface.x)
    assert math.isclose(doubled.dimensional_shapes.uncorrected.trailing_edge_thickness,
                        original.dimensional_shapes.uncorrected.trailing_edge_thickness, rel_tol=1.0e-12)


@pytest.mark.parametrize("options, chord, reynolds_number, real_flow_angle, real_flow_mach", [
    ({"reference_length": 0.03979350694547071}, 0.05432331459426887, 1997057.033343444,
     70.91712074713297, 1.7303884409586916),
])
def test_equivalent_dimensions_preserve_existing_bl_and_aftermixing_results(
        options, chord, reynolds_number, real_flow_angle, real_flow_mach):
    # Preserve the previous pitch-sized design's physical dimensions to compare both scaling routes at the same
    # operating point. Its throat Reynolds scale, BL correction, and aftermixed state should remain unchanged.
    stator = make_stator(trailing_edge_thickness=0.002094395102393196, **options)
    assert math.isclose(stator.dimensional_chord, chord, rel_tol=1.0e-12)
    assert math.isclose(stator.chord_reynolds_number, reynolds_number, rel_tol=1.0e-12)
    assert math.isclose(stator.real_outlet_absolute_flow_angle, real_flow_angle, rel_tol=1.0e-10)
    assert math.isclose(stator.real_outlet_absolute_flow_mach, real_flow_mach, rel_tol=1.0e-10)


def test_moc_stores_ideal_uncorrected_and_corrected_area_ratios():
    stator = make_stator()
    uncorrected = stator.nondimensional_shapes.uncorrected
    corrected = stator.nondimensional_shapes.corrected
    expected_ideal = isentropic_area_ratio(stator.ideal_outlet_absolute_flow_mach, stator.gamma)

    assert math.isclose(stator.nondimensional_ideal_exit_area_ratio, expected_ideal, rel_tol=1.0e-13)
    assert math.isclose(stator.nondimensional_uncorrected_exit_area_ratio,
                        uncorrected.nozzle_exit_width / uncorrected.throat_width, rel_tol=1.0e-13)
    assert math.isclose(stator.nondimensional_corrected_exit_area_ratio,
                        corrected.nozzle_exit_width / corrected.throat_width, rel_tol=1.0e-13)


def test_stores_corrected_uncorrected_and_dimensional_shapes():
    stator = make_stator()
    uncorrected = stator.nondimensional_shapes.uncorrected
    corrected = stator.nondimensional_shapes.corrected
    dimensional_uncorrected = stator.dimensional_shapes.uncorrected
    dimensional_corrected = stator.dimensional_shapes.corrected

    assert uncorrected.coordinate_scale == "throat half-width"
    assert corrected.coordinate_scale == "throat half-width"
    assert len(stator.suction_boundary_layer.s_over_chord) == len(uncorrected.suction_surface.x)
    assert len(stator.suction_boundary_layer.s_over_chord) == 2 * stator.number_of_nodes - 1
    assert len(uncorrected.pressure_surface.x) == stator.pressure_number_of_nodes
    assert len(corrected.suction_surface.x) == len(uncorrected.suction_surface.x) + 2
    assert corrected.nozzle_passage_pitch > uncorrected.nozzle_passage_pitch
    assert dimensional_corrected.throat_width > dimensional_uncorrected.throat_width
    assert math.isclose(stator.dimensional_chord,
                        uncorrected.chord * stator.dimensional_scale_factor, rel_tol=1.0e-12)
    assert stator.outlet_metal_angle == stator.ideal_outlet_absolute_flow_angle
    assert uncorrected.pressure_surface.absolute_flow_mach is not None
    assert uncorrected.pressure_surface.relative_flow_mach is None
    assert stator.pressure_boundary_layer.freestream_absolute_flow_mach is not None
    assert stator.pressure_boundary_layer.freestream_relative_flow_mach is None
    assert not hasattr(stator, "uncorrected_shape")
    assert not hasattr(stator, "uncorrected_dimensional_shape")
    assert not hasattr(stator, "uncorrected_mixing_results")
    assert not hasattr(stator, "pressure_boundary_layer_marching")
    assert not hasattr(stator, "suction_boundary_layer_marching")
    assert not hasattr(stator, "boundary_layer_pressure_station_count")
    assert not hasattr(stator, "boundary_layer_suction_station_count")


@pytest.mark.parametrize("calculate_boundary_layer", [True, False])
def test_conical_contour_is_inviscid_and_skips_bl_correction_and_mixing(monkeypatch, calculate_boundary_layer):
    def unexpected_call(*args, **kwargs):
        pytest.fail("an inviscid conical nozzle must not call BL, shape correction, or aftermixing")

    monkeypatch.setattr(stator_nozzle_module, "solve_boundary_layer", unexpected_call)
    monkeypatch.setattr(SupersonicStatorNozzle, "_correct_shape", unexpected_call)
    monkeypatch.setattr(SupersonicStatorNozzle, "_aftermixing", unexpected_call)
    monkeypatch.setattr(SupersonicStatorNozzle, "_select_mixing_result", unexpected_call)
    stator = make_stator(
        contour_method="conical",
        half_cone_metal_angle=15.0,
        calculate_boundary_layer=calculate_boundary_layer,
    )
    expected_area_ratio = float(isentropic_area_ratio(stator.ideal_outlet_absolute_flow_mach, stator.gamma))
    nondimensional_uncorrected = stator.nondimensional_shapes.uncorrected
    dimensional_uncorrected = stator.dimensional_shapes.uncorrected

    assert stator.contour_method == "conical"
    assert stator.calculate_boundary_layer is False
    assert stator.actual_flow_turning_increment is None
    assert math.isclose(stator.nondimensional_ideal_exit_area_ratio, expected_area_ratio, rel_tol=1.0e-13)
    assert math.isclose(
        (nondimensional_uncorrected.nozzle_exit_width / nondimensional_uncorrected.throat_width) ** 2,
        expected_area_ratio,
        rel_tol=1.0e-13,
    )
    assert math.isclose(dimensional_uncorrected.throat_width, stator.reference_length, rel_tol=1.0e-14)
    assert math.isclose(stator.dimensional_chord,
                        nondimensional_uncorrected.chord * stator.dimensional_scale_factor, rel_tol=1.0e-14)
    assert math.isclose(
        stator.dimensional_conical_divergent_length,
        stator.nondimensional_conical_divergent_length * stator.dimensional_scale_factor,
        rel_tol=1.0e-14,
    )
    assert len(nondimensional_uncorrected.pressure_surface.x) == stator.number_of_nodes
    assert len(nondimensional_uncorrected.suction_surface.x) == 2 * stator.number_of_nodes - 1
    assert math.isclose(nondimensional_uncorrected.pressure_surface.absolute_flow_mach[-1],
                        stator.ideal_outlet_absolute_flow_mach, rel_tol=1.0e-11)
    assert stator.outlet_metal_angle == stator.requested_outlet_absolute_flow_angle
    assert stator.ideal_outlet_absolute_flow_mach == stator.requested_outlet_absolute_flow_mach
    for name in ("boundary_layer_mode", "pressure_boundary_layer", "suction_boundary_layer", "mixing_results",
                 "mixing_solution", "real_outlet_absolute_flow_angle", "real_outlet_absolute_flow_mach",
                 "supersonic_mixing_available", "nondimensional_corrected_exit_area_ratio",
                 "dimensional_corrected_exit_displacement_thickness", "dimensional_corrected_exit_momentum_thickness"):
        assert getattr(stator, name) is None
    assert stator.nondimensional_shapes.corrected is None
    assert stator.dimensional_shapes.corrected is None
    assert stator.dimensionalize().corrected is None
    figure, axes = stator.plot(dimensional=True, show=False)
    assert figure is axes.figure
    assert len(axes.lines) == 5
    assert axes.get_legend_handles_labels()[1] == ["ideal"]


@pytest.mark.parametrize("options, message", [
    ({"boundary_layer_mode": "laminar_then_turbulent"}, "boundary_layer_mode"),
    ({"boundary_layer_mode": "fully_turbulent"}, "boundary_layer_mode"),
    ({"initial_turbulent_displacement_thickness": 2.0e-5}, "initial_turbulent_displacement_thickness"),
    ({"initial_turbulent_momentum_thickness": 5.0e-6}, "initial_turbulent_momentum_thickness"),
    ({"mixing_solution": "subsonic"}, "mixing_solution"),
    ({"iterate_outlet_metal_angle": True}, "iterate_outlet_metal_angle"),
    ({"iterate_outlet_metal_angle": True, "match_real_outlet_absolute_flow_mach": True}, "iterate_outlet_metal_angle"),
])
def test_conical_contour_rejects_viscous_and_mixed_flow_controls(options, message):
    with pytest.raises(ValueError, match=message):
        make_stator(contour_method="conical", half_cone_metal_angle=15.0, **options)


def test_moc_default_still_calculates_boundary_layer_correction_and_mixing():
    stator = make_stator(boundary_layer_mode=None,
                         initial_turbulent_displacement_thickness=None,
                         initial_turbulent_momentum_thickness=None)
    assert stator.calculate_boundary_layer is True
    assert stator.boundary_layer_mode == "laminar_then_turbulent"
    assert stator.pressure_boundary_layer is not None
    assert stator.suction_boundary_layer is not None
    assert stator.nondimensional_shapes.corrected is not None
    assert stator.mixing_results is not None
    assert math.isfinite(stator.real_outlet_absolute_flow_mach)


def test_moc_without_boundary_layer_keeps_only_ideal_geometry(monkeypatch):
    import matplotlib.pyplot as plt

    reference = make_stator()

    def unexpected_call(*args, **kwargs):
        pytest.fail("disabled BL must skip the BL solver, correction, aftermixing and outlet iterations")

    monkeypatch.setattr(stator_nozzle_module, "solve_boundary_layer", unexpected_call)
    for name in ("_correct_shape", "_aftermixing", "_select_mixing_result",
                 "_solve_outlet_metal_angle_for_target_flow", "_solve_outlet_metal_angle_and_flow_mach_targets"):
        monkeypatch.setattr(SupersonicStatorNozzle, name, unexpected_call)
    stator = make_stator(calculate_boundary_layer=False,
                         initial_turbulent_displacement_thickness=None, initial_turbulent_momentum_thickness=None)

    assert stator.calculate_boundary_layer is False
    assert stator.contour_method == "moc"
    for name in ("boundary_layer_mode", "initial_turbulent_displacement_thickness",
                 "initial_turbulent_momentum_thickness", "pressure_boundary_layer", "suction_boundary_layer",
                 "mixing_results", "mixing_solution", "real_outlet_absolute_flow_angle",
                 "real_outlet_absolute_flow_mach", "supersonic_mixing_available",
                 "nondimensional_corrected_exit_area_ratio", "dimensional_corrected_exit_displacement_thickness",
                 "dimensional_corrected_exit_momentum_thickness"):
        assert getattr(stator, name) is None
    assert stator.nondimensional_shapes.corrected is None
    assert stator.dimensional_shapes.corrected is None
    assert stator.dimensionalize().corrected is None
    assert stator.outlet_metal_angle == stator.requested_outlet_absolute_flow_angle
    assert stator.ideal_outlet_absolute_flow_mach == stator.requested_outlet_absolute_flow_mach
    assert stator.dimensional_chord == reference.dimensional_chord
    assert stator.dimensional_total_pitch == reference.dimensional_total_pitch
    assert stator.nondimensional_uncorrected_exit_area_ratio == reference.nondimensional_uncorrected_exit_area_ratio
    assert stator.dimensional_shapes.uncorrected.trailing_edge_thickness == stator.trailing_edge_thickness
    for name in ("pressure_surface", "suction_surface"):
        surface = getattr(stator.nondimensional_shapes.uncorrected, name)
        expected = getattr(reference.nondimensional_shapes.uncorrected, name)
        assert np.array_equal(surface.x, expected.x)
        assert np.array_equal(surface.y, expected.y)
    for dimensional in (False, True):
        figure, axes = stator.plot(dimensional=dimensional, show=False)
        assert axes.get_legend_handles_labels()[1] == ["ideal"]
        assert len(axes.lines) == 5
        plt.close(figure)


@pytest.mark.parametrize("options", [
    {"iterate_outlet_metal_angle": True},
    {"match_real_outlet_absolute_flow_mach": True},
    {"mixing_solution": "subsonic"},
    {"iterate_outlet_metal_angle": True, "match_real_outlet_absolute_flow_mach": True,
     "mixing_solution": "subsonic"},
])
def test_stator_without_boundary_layer_overrides_dependent_controls_with_warning(monkeypatch, options):
    def unexpected_call(*args, **kwargs):
        pytest.fail("disabled BL controls must never run outlet iterations")

    for name in ("_solve_outlet_metal_angle_for_target_flow", "_solve_outlet_metal_angle_and_flow_mach_targets"):
        monkeypatch.setattr(SupersonicStatorNozzle, name, unexpected_call)
    with pytest.warns(UserWarning, match="calculate_boundary_layer=False disables") as captured:
        stator = make_stator(calculate_boundary_layer=False, **options)
    assert len(captured) == 1
    for name in options:
        assert name in str(captured[0].message)
    assert stator.iterate_outlet_metal_angle is False
    assert stator.match_real_outlet_absolute_flow_mach is False
    assert stator.mixing_solution is None
    assert stator.outlet_metal_angle == stator.requested_outlet_absolute_flow_angle


@pytest.mark.parametrize("value", [None, 0, "False"])
def test_stator_requires_boolean_boundary_layer_setting(value):
    with pytest.raises(TypeError, match="calculate_boundary_layer must be a bool"):
        make_stator(calculate_boundary_layer=value)


def test_stator_without_boundary_layer_requires_supersonic_ideal_mach():
    with pytest.raises(ValueError, match="must be supersonic"):
        make_stator(calculate_boundary_layer=False, requested_outlet_absolute_flow_mach=0.9,
                    match_real_outlet_absolute_flow_mach=True)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"contour_method": "conical"}, "requires half_cone_metal_angle"),
        ({"half_cone_metal_angle": 15.0}, "only valid"),
        (
            {"contour_method": "conical", "half_cone_metal_angle": 90.0},
            "half_cone_metal_angle must be between",
        ),
    ],
)
def test_contour_modes_require_only_their_own_inputs(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_stator(**overrides)


def test_stator_boundary_layer_uses_the_fixed_geometry_mesh():
    coarse = make_stator(number_of_nodes=31)
    fine = make_stator(number_of_nodes=61)

    coarse_uncorrected = coarse.nondimensional_shapes.uncorrected
    coarse_corrected = coarse.nondimensional_shapes.corrected
    assert len(coarse_corrected.suction_surface.x) == len(coarse_uncorrected.suction_surface.x) + 2
    assert coarse.actual_flow_turning_increment > fine.actual_flow_turning_increment


@pytest.mark.parametrize("number_of_nodes", [19, 20.5, True])
def test_stator_number_of_nodes_must_be_an_integer_at_least_twenty(number_of_nodes):
    with pytest.raises(ValueError, match="number_of_nodes"):
        make_stator(number_of_nodes=number_of_nodes)


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
    assert stator.real_outlet_absolute_flow_mach == \
        stator.mixing_results["supersonic"]["real_outlet_absolute_flow_mach"]


def test_stator_subsonic_mixing_solution_overrides_automatic_selection():
    stator = make_stator(requested_outlet_absolute_flow_mach=2.0, requested_outlet_absolute_flow_angle=30.0,
                         mixing_solution="subsonic")

    assert stator.ideal_outlet_absolute_axial_flow_mach >= 1.0
    assert stator.supersonic_mixing_available
    assert stator.mixing_solution == "subsonic"
    assert stator.real_outlet_absolute_flow_mach == stator.mixing_results["subsonic"]["real_outlet_absolute_flow_mach"]


def test_trailing_edge_thickness_uses_nasa_tm_x_2343_afmix_blockage():
    sharp = make_stator(trailing_edge_thickness=0.0)
    finite = make_stator(trailing_edge_thickness=0.002)
    uncorrected = finite.nondimensional_shapes.uncorrected
    corrected = finite.nondimensional_shapes.corrected

    # Dimensional trailing-edge metal is added to the open passage after scaling. It changes the total pitch and
    # mixed-out state without changing either the ideal contour scale or the boundary-layer march.
    assert np.array_equal(uncorrected.pressure_surface.x,
                          sharp.nondimensional_shapes.uncorrected.pressure_surface.x)
    assert uncorrected.total_pitch > sharp.nondimensional_shapes.uncorrected.total_pitch
    assert finite.dimensional_scale_factor == sharp.dimensional_scale_factor
    assert np.array_equal(finite.suction_boundary_layer.displacement_thickness_over_chord,
                          sharp.suction_boundary_layer.displacement_thickness_over_chord)
    assert finite.real_outlet_absolute_flow_mach != sharp.real_outlet_absolute_flow_mach
    assert finite.real_outlet_absolute_flow_angle != sharp.real_outlet_absolute_flow_angle

    # BL correction widens the open nozzle pitch and consumes the same amount of trailing-edge metal. AFMIX projects
    # both circumferential quantities onto the plane normal to the flow, so the cosine cancels from their ratio.
    expected_corrected_thickness = max(
        0.0,
        uncorrected.trailing_edge_thickness
        - (corrected.nozzle_passage_pitch - uncorrected.nozzle_passage_pitch))
    expected_dte = corrected.trailing_edge_thickness / corrected.nozzle_passage_pitch
    assert math.isclose(corrected.trailing_edge_thickness, expected_corrected_thickness, rel_tol=1.0e-12)
    assert math.isclose(
        finite.mixing_results["subsonic"]["trailing_edge_blockage_ratio"], expected_dte, rel_tol=1.0e-12
    )

    # Independently reconstruct the subsonic solution from the variables named in
    # the FORTRAN listing. This guards the actual A/A1 use, rather than merely
    # checking that a nonzero input perturbs the answer.
    gamma = finite.gamma
    gp = gamma + 1.0
    gm = gamma - 1.0
    angle = math.radians(finite.outlet_metal_angle)
    velocity_ratio = math.sqrt((0.5 * gp * finite.ideal_outlet_absolute_flow_mach**2)
                               / (1.0 + 0.5 * gm * finite.ideal_outlet_absolute_flow_mach**2))
    projected_nozzle_passage_pitch = corrected.nozzle_passage_pitch * math.cos(angle)
    pressure_displacement = (
        finite.pressure_boundary_layer.displacement_thickness_over_chord[-1] * uncorrected.chord
    )
    pressure_momentum = (
        finite.pressure_boundary_layer.momentum_thickness_over_chord[-1] * uncorrected.chord
    )
    displacement_ratio = (
        pressure_displacement
        + finite.dimensional_corrected_exit_displacement_thickness / finite.dimensional_scale_factor
    ) / projected_nozzle_passage_pitch
    momentum_ratio = (
        pressure_momentum
        + finite.dimensional_corrected_exit_momentum_thickness / finite.dimensional_scale_factor
    ) / projected_nozzle_passage_pitch
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
    assert math.isclose(finite.mixing_results["subsonic"]["real_outlet_absolute_flow_mach"], expected_mach,
                        rel_tol=1.0e-12)
    assert math.isclose(finite.mixing_results["subsonic"]["real_outlet_absolute_flow_angle"], expected_angle,
                        rel_tol=1.0e-12)


def test_boundary_layer_correction_warns_when_it_consumes_all_trailing_edge_metal():
    with pytest.warns(RuntimeWarning, match="consumed the complete stator trailing-edge thickness"):
        stator = make_stator(trailing_edge_thickness=1.0e-5)

    assert stator.nondimensional_shapes.corrected.trailing_edge_thickness == 0.0
    assert stator.mixing_results["subsonic"]["trailing_edge_blockage_ratio"] == 0.0


@pytest.mark.parametrize("trailing_edge_thickness", [-1.0e-6, math.inf, math.nan])
def test_trailing_edge_thickness_must_be_nonnegative_and_finite(trailing_edge_thickness):
    with pytest.raises(ValueError, match="trailing_edge_thickness"):
        make_stator(trailing_edge_thickness=trailing_edge_thickness)


@pytest.mark.parametrize("reference_length", [0.0, -0.01, math.inf, math.nan])
def test_reference_length_must_be_positive_and_finite(reference_length):
    with pytest.raises(ValueError, match="reference_length"):
        make_stator(reference_length=reference_length)


def test_supersonic_mixing_solution_override_is_rejected():
    with pytest.raises(ValueError, match="mixing_solution"):
        make_stator(mixing_solution="supersonic")


def test_iterated_outlet_metal_angle_matches_requested_real_flow_angle():
    stator = make_stator(iterate_outlet_metal_angle=True)
    assert abs(stator.real_outlet_absolute_flow_angle - stator.requested_outlet_absolute_flow_angle) < 2.0e-3
    assert abs(stator.outlet_metal_angle - stator.requested_outlet_absolute_flow_angle) > 0.05


def test_coupled_iteration_matches_real_outlet_absolute_flow_mach_and_angle():
    stator = make_stator(requested_outlet_absolute_flow_mach=1.77, iterate_outlet_metal_angle=True,
                         match_real_outlet_absolute_flow_mach=True)
    assert abs(stator.real_outlet_absolute_flow_angle - stator.requested_outlet_absolute_flow_angle) < 2.0e-3
    assert abs(stator.real_outlet_absolute_flow_mach - stator.requested_outlet_absolute_flow_mach) < 1.0e-4
    assert not math.isclose(stator.ideal_outlet_absolute_flow_mach, stator.requested_outlet_absolute_flow_mach,
                            rel_tol=1.0e-3)
    assert math.isclose(
        stator.ideal_outlet_absolute_axial_flow_mach,
        stator.ideal_outlet_absolute_flow_mach * math.cos(math.radians(stator.outlet_metal_angle)),
        rel_tol=1.0e-12,
    )
    for evaluation in stator._evaluation_cache.values():
        assert math.isclose(evaluation.ideal.nozzle_passage_pitch * evaluation.dimensional_scale_factor,
                            stator.reference_length, rel_tol=1.0e-12)
        assert math.isclose(evaluation.ideal.trailing_edge_thickness * evaluation.dimensional_scale_factor,
                            stator.trailing_edge_thickness, rel_tol=1.0e-12)


def test_coupled_stator_flag_requires_outlet_metal_angle_iteration():
    with pytest.raises(ValueError, match="iterate_outlet_metal_angle"):
        make_stator(match_real_outlet_absolute_flow_mach=True)


def test_coupled_stator_iteration_supports_supersonic_solution():
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


def test_conical_design_changes_exit_diameter_and_length_with_requested_ideal_mach():
    low_mach = make_stator(contour_method="conical", half_cone_metal_angle=15.0)
    high_mach = make_stator(contour_method="conical", half_cone_metal_angle=15.0,
                            requested_outlet_absolute_flow_mach=2.4)
    low_shape = low_mach.dimensional_shapes.uncorrected
    high_shape = high_mach.dimensional_shapes.uncorrected
    expected_ratio = float(isentropic_area_ratio(2.4, high_mach.gamma))
    assert high_mach.ideal_outlet_absolute_flow_mach == 2.4
    assert math.isclose((high_shape.nozzle_exit_width / high_shape.throat_width) ** 2,
                        expected_ratio, rel_tol=1.0e-13)
    assert high_shape.nozzle_exit_width > low_shape.nozzle_exit_width
    assert high_mach.dimensional_conical_divergent_length > low_mach.dimensional_conical_divergent_length
    assert high_shape.throat_width == low_shape.throat_width


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"contour_method": "conical", "half_cone_metal_angle": 15.0},
    ],
)
def test_plot_uses_total_pitch_and_closes_trailing_edge(overrides):
    stator = make_stator(**overrides)
    figure, axes = stator.plot(dimensional=True, show=False)
    assert figure is axes.figure
    assert len(axes.lines) == (10 if stator.contour_method == "moc" else 5)

    pressure = stator.dimensional_shapes.uncorrected.pressure_surface
    angle = math.radians(stator.outlet_metal_angle)
    expected_axial = 1000.0 * (pressure.x[0] * math.cos(angle) - pressure.y[0] * math.sin(angle))
    assert math.isclose(axes.lines[0].get_xdata()[0], expected_axial, rel_tol=1.0e-12)
    assert axes.get_xlabel() == "axial length [mm]"
    assert axes.get_ylabel() == "tangential length [mm]"

    for first_line, shape in ((0, stator.dimensional_shapes.uncorrected),
                              (5, stator.dimensional_shapes.corrected)):
        if shape is None:
            continue
        plotted_lines = axes.lines[first_line:first_line + 5]
        lower_pressure, lower_suction, upper_pressure, upper_suction, trailing_edge = plotted_lines
        total_pitch = 1000.0 * shape.total_pitch
        trailing_edge_thickness = 1000.0 * shape.trailing_edge_thickness
        assert np.allclose(upper_pressure.get_xdata(), lower_pressure.get_xdata())
        assert np.allclose(upper_suction.get_xdata(), lower_suction.get_xdata())
        assert np.allclose(upper_pressure.get_ydata() - lower_pressure.get_ydata(), total_pitch)
        assert np.allclose(upper_suction.get_ydata() - lower_suction.get_ydata(), total_pitch)
        assert math.isclose(trailing_edge.get_xdata()[0], trailing_edge.get_xdata()[1], abs_tol=1.0e-14)
        assert math.isclose(trailing_edge.get_ydata()[0], lower_suction.get_ydata()[-1], rel_tol=1.0e-12)
        assert math.isclose(trailing_edge.get_ydata()[1], upper_pressure.get_ydata()[-1], rel_tol=1.0e-12)
        assert math.isclose(abs(np.diff(trailing_edge.get_ydata())[0]), trailing_edge_thickness, rel_tol=1.0e-12)


@pytest.mark.parametrize("options", [{}, {"contour_method": "conical", "half_cone_metal_angle": 15.0}])
def test_plot_can_retain_single_nozzle_passage_view(options):
    stator = make_stator(**options)
    figure, axes = stator.plot(show_two_nozzles=False, show=False)
    assert figure is axes.figure
    assert len(axes.lines) == (4 if stator.contour_method == "moc" else 2)
