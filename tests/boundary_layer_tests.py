import math
import warnings

import numpy as np
import pytest

from SupersonicTurbineBlading import Fluid
from SupersonicTurbineBlading.common_results import SurfaceCoordinates
from SupersonicTurbineBlading.boundary_layer import boundary_layer_solver
from SupersonicTurbineBlading.boundary_layer.boundary_layer_solver import (
    _laminar_solution,
    _transition_incompressible_form_factor,
)


@pytest.mark.parametrize("transition_reynolds_number", [0.0, -1.0])
@pytest.mark.parametrize("transition_model, expected", [("rotor", 1.4), ("stator", 1.411515979131044)])
def test_transition_form_factor_uses_nasa_row_separation_fallback(
        transition_reynolds_number, transition_model, expected):
    # NASA TM X-2434 p. 40 branches directly to FORMS=1.4 for RTRAN<=0.
    # NASA TM X-2343 p. 39 instead evaluates the correlation at RTRAN=1000.
    laminar_form_factor = 2.4

    assert math.isclose(
        _transition_incompressible_form_factor(laminar_form_factor, transition_reynolds_number,
                                               transition_model=transition_model),
        expected,
        rel_tol=1.0e-14,
    )


@pytest.mark.parametrize("transition_model", ["rotor", "stator"])
def test_transition_form_factor_uses_calculated_transition_reynolds_number(transition_model):
    laminar_form_factor = 2.4
    transition_reynolds_number = 1750.0
    log_rtran = math.log(transition_reynolds_number)
    expected = laminar_form_factor - 0.59389 - 0.06591 * log_rtran + 0.001272 * log_rtran**2

    assert math.isclose(
        _transition_incompressible_form_factor(laminar_form_factor, transition_reynolds_number,
                                               transition_model=transition_model),
        expected,
        rel_tol=1.0e-14,
    )


@pytest.mark.parametrize("transition_model, transition_reynolds_number, separated, expected", [
    ("rotor", 0.0, True, 1.4),
    ("stator", 0.0, True, 1.411515979131044),
    ("rotor", -1.0, True, 1.4),
    ("stator", -1.0, True, 1.411515979131044),
    ("rotor", 1750.0, False, 1.384864367167583),
    ("stator", 1750.0, False, 1.384864367167583),
    ("rotor", 1750.0, True, 1.384864367167583),
    ("stator", 1750.0, True, 1.384864367167583),
])
def test_transition_handoff_uses_nasa_row_initialization(
        monkeypatch, transition_model, transition_reynolds_number, separated, expected):
    gamma = 1.4
    prandtl_number = 0.72
    transition_relative_flow_mach = 2.0
    laminar_incompressible_form = 2.4
    temperature_factor = 1.0 + 0.5 * (gamma - 1.0) * transition_relative_flow_mach**2
    # FORMI in LAMNAR uses SQRT(PR), distinct from the turbulent Pr^(1/3).
    laminar_compressible_form = laminar_incompressible_form * temperature_factor + math.sqrt(prandtl_number) * (
        temperature_factor - 1.0
    )
    state = {
        "s": np.array([0.0, 0.5, 1.0]),
        "edge_flow_mach": np.array([1.0, transition_relative_flow_mach, transition_relative_flow_mach]),
        "pr": prandtl_number,
    }
    laminar = {
        "theta": np.array([0.01, 0.02, 0.025]),
        "displacement": np.array([0.025, 0.02 * laminar_compressible_form, 0.025 * laminar_compressible_form]),
        "form": np.array([2.5, laminar_compressible_form, laminar_compressible_form]),
        "transition_index": 1,
        "separation_index": 1 if separated else None,
        "transition_reynolds_number": transition_reynolds_number,
    }
    captured = {}

    monkeypatch.setattr(boundary_layer_solver, "_surface_state", lambda *args, **kwargs: state)
    monkeypatch.setattr(boundary_layer_solver, "_laminar_solution", lambda supplied_state, correlation_limit: laminar)

    def fake_turbulent_solution(supplied_state, start_index, initial_theta, initial_incompressible_form):
        captured["initial_theta"] = initial_theta
        captured["initial_incompressible_form"] = initial_incompressible_form
        turbulent_form = (initial_incompressible_form * temperature_factor
                          + prandtl_number ** (1.0 / 3.0) * (temperature_factor - 1.0))
        return {
            "theta": np.array([np.nan, initial_theta, initial_theta]),
            "displacement": np.array([np.nan, initial_theta * turbulent_form, initial_theta * turbulent_form]),
            "form": np.array([np.nan, turbulent_form, turbulent_form]),
            "separation_index": None,
        }

    monkeypatch.setattr(boundary_layer_solver, "_turbulent_solution", fake_turbulent_solution)

    surface = SurfaceCoordinates(
        x=np.array([0.0, 0.5, 1.0]),
        y=np.array([0.0, 0.0, 0.0]),
        relative_flow_mach=state["edge_flow_mach"],
        metal_angle=np.zeros(3),
    )
    result = boundary_layer_solver.solve_boundary_layer(
        surface=surface,
        chord=1.0,
        inlet_edge_flow_mach=1.0,
        chord_reynolds_number=1.0e6,
        gamma=gamma,
        fluid=None,
        inlet_total_temperature=300.0,
        inlet_total_pressure=1.0e5,
        mode="laminar_then_turbulent",
        initial_turbulent_displacement_thickness_over_chord=None,
        initial_turbulent_momentum_thickness_over_chord=None,
        laminar_correlation_limit=0.50,
        transition_model=transition_model,
    )

    assert math.isclose(
        captured["initial_incompressible_form"],
        expected,
        rel_tol=1.0e-14,
    )
    assert captured["initial_theta"] == laminar["theta"][1]
    assert result.momentum_thickness_over_chord[1] == laminar["theta"][1]
    expected_compressible_form = expected * temperature_factor + prandtl_number ** (1.0 / 3.0) * (
        temperature_factor - 1.0)
    assert math.isclose(result.form_factor[1], expected_compressible_form, rel_tol=1.0e-14)
    assert math.isclose(result.displacement_thickness_over_chord[1],
                        laminar["theta"][1] * expected_compressible_form, rel_tol=1.0e-14)
    assert result.transition_index == 1
    assert result.separation_index is None
    assert list(result.regime) == ["laminar", "turbulent", "turbulent"]


def test_laminar_separation_retains_previously_calculated_transition_reynolds_number(monkeypatch):
    s = np.linspace(0.0, 1.0, 5)
    mach = np.full_like(s, 2.0)
    temperature_factor = 1.0 + 0.2 * mach**2
    velocity = mach / np.sqrt(temperature_factor)
    viscosity = 1.0e-6
    ff = 1.0 + 0.1599 * mach**2 + 0.0114 * mach**4
    state = {
        "s": s, "sol": s, "edge_flow_mach": mach, "arcl": 1.0, "gamma": 1.4, "pr": 0.72,
        "nu_total": viscosity, "dmdl": np.zeros_like(s), "sw": np.zeros_like(s),
        "nu": np.full_like(s, viscosity), "velocity": velocity, "duds": np.zeros_like(s), "ff": ff,
    }
    original_polyfit = boundary_layer_solver._polyfit
    shear_stations = []

    def controlled_instability_and_separation(coefficients, x, y=0.0):
        if coefficients[0] == 0.224488:
            shear_stations.append(x)
            return -1.0 if len(shear_stations) == 4 else 1.0
        if coefficients[0] == 5.47073:
            return 0.0  # RCRIT=1: instability at the first station with nonzero theta.
        return original_polyfit(coefficients, x, y)

    monkeypatch.setattr(boundary_layer_solver, "_polyfit", controlled_instability_and_separation)
    result = _laminar_solution(state, 0.50)
    assert result["instability_index"] == 1
    assert result["separation_index"] == result["transition_index"] == 3
    r_instability = velocity[1] * result["theta"][1] / viscosity / ff[1] / math.sqrt(temperature_factor[1])
    # With zero pressure gradient, KBAR=0 and the NASA CDIF fit gives DIFF=903.785.
    assert math.isclose(result["transition_reynolds_number"], r_instability + 903.785, rel_tol=1.0e-14)


def test_application_correlation_limits_warn_only_when_extrapolating():
    s = np.linspace(0.0, 1.0, 21)
    relative_edge_flow_mach = np.linspace(2.0, 1.5, 21)
    temperature = 1.0 / (1.0 + 0.2 * relative_edge_flow_mach**2)
    velocity = relative_edge_flow_mach * np.sqrt(temperature)
    state = {
        "s": s,
        "sol": s,
        "edge_flow_mach": relative_edge_flow_mach,
        "arcl": 1.0,
        "gamma": 1.4,
        "pr": 0.72,
        "nu_total": 1.0e-6,
        "dmdl": np.gradient(relative_edge_flow_mach, s),
        "sw": np.zeros_like(s),
        "nu": np.full_like(s, 1.0e-6),
        "velocity": velocity,
        "duds": np.gradient(velocity, s),
        "ff": 1.0 + 0.1599 * relative_edge_flow_mach**2 + 0.0114 * relative_edge_flow_mach**4,
    }

    with pytest.warns(RuntimeWarning, match="0.16.*extrapolating"):
        _laminar_solution(state, 0.16)
    with warnings.catch_warnings(record=True) as rotor_warnings:
        warnings.simplefilter("always")
        _laminar_solution(state, 0.50)
    assert not rotor_warnings


@pytest.mark.parametrize("transition_model", ["rotor", "stator"])
def test_transition_at_final_station_returns_turbulent_initial_values(monkeypatch, transition_model):
    # Real correlation case from the audit: transition is first reached at
    # station 20 of 21. Keep the real laminar and turbulent initialization paths.
    surface = SurfaceCoordinates(x=np.linspace(0.0, 1.0, 21), y=np.zeros(21), metal_angle=np.zeros(21),
                                 **{("relative_flow_mach" if transition_model == "rotor" else "absolute_flow_mach"):
                                    np.full(21, 2.0)})
    arguments = dict(surface=surface, chord=1.0, inlet_edge_flow_mach=2.0,
                     chord_reynolds_number=1.65e8, gamma=1.4, fluid=Fluid(["Air"], [1.0]),
                     inlet_total_temperature=1000.0, inlet_total_pressure=1.0e6,
                     mode="laminar_then_turbulent",
                     initial_turbulent_displacement_thickness_over_chord=None,
                     initial_turbulent_momentum_thickness_over_chord=None,
                     laminar_correlation_limit=0.50 if transition_model == "rotor" else 0.16,
                     transition_model=transition_model)
    state = boundary_layer_solver._surface_state(
        surface, 1.0, 2.0, arguments["chord_reynolds_number"], 1.4, arguments["fluid"], 1000.0, 1.0e6)
    laminar = _laminar_solution(state, arguments["laminar_correlation_limit"])
    assert laminar["transition_index"] == len(surface.x) - 1

    def unexpected_integration(*args, **kwargs):
        pytest.fail("a final-station transition has no interval to integrate")

    monkeypatch.setattr(boundary_layer_solver, "solve_ivp", unexpected_integration)
    result = boundary_layer_solver.solve_boundary_layer(**arguments)
    temperature_factor = 1.8
    pr = state["pr"]
    laminar_incompressible_form = (
        (laminar["form"][-1] - math.sqrt(pr) * (temperature_factor - 1.0)) / temperature_factor)
    turbulent_incompressible_form = _transition_incompressible_form_factor(
        laminar_incompressible_form, laminar["transition_reynolds_number"], transition_model=transition_model)
    expected_form = (turbulent_incompressible_form * temperature_factor
                     + pr ** (1.0 / 3.0) * (temperature_factor - 1.0))
    np.testing.assert_array_equal(result.momentum_thickness_over_chord, laminar["theta"])
    np.testing.assert_array_equal(result.displacement_thickness_over_chord[:-1], laminar["displacement"][:-1])
    np.testing.assert_array_equal(result.form_factor[:-1], laminar["form"][:-1])
    assert math.isclose(result.form_factor[-1], expected_form, rel_tol=1.0e-14)
    assert math.isclose(result.displacement_thickness_over_chord[-1],
                        laminar["theta"][-1] * expected_form, rel_tol=1.0e-14)
    assert np.all(np.isfinite(result.displacement_thickness_over_chord))
    assert list(result.regime) == ["laminar"] * 20 + ["turbulent"]
    assert result.transition_index == 20
    assert result.separation_index is None


@pytest.mark.parametrize("transition_model", ["rotor", "stator"])
@pytest.mark.parametrize("initial_form_factor, separated", [(2.79, False), (2.800001, True), (3.0, True)])
def test_fully_turbulent_initial_separation_error(monkeypatch, transition_model, initial_form_factor, separated):
    surface = SurfaceCoordinates(x=np.linspace(0.0, 1.0, 3), y=np.zeros(3), metal_angle=np.zeros(3),
                                 **{("relative_flow_mach" if transition_model == "rotor" else "absolute_flow_mach"):
                                    np.full(3, 2.0)})
    fluid = Fluid(["Air"], [1.0])
    state = boundary_layer_solver._surface_state(surface, 1.0, 2.0, 1.0e6, 1.4, fluid, 1000.0, 1.0e6)
    temperature_factor = 1.8
    theta = 1.0e-4
    displacement = theta * (initial_form_factor * temperature_factor
                            + state["pr"] ** (1.0 / 3.0) * (temperature_factor - 1.0))
    arguments = dict(surface=surface, chord=1.0, inlet_edge_flow_mach=2.0, chord_reynolds_number=1.0e6,
                     gamma=1.4, fluid=fluid, inlet_total_temperature=1000.0, inlet_total_pressure=1.0e6,
                     mode="fully_turbulent", initial_turbulent_displacement_thickness_over_chord=displacement,
                     initial_turbulent_momentum_thickness_over_chord=theta,
                     laminar_correlation_limit=0.50 if transition_model == "rotor" else 0.16,
                     transition_model=transition_model)
    if separated:
        def unexpected_integration(*args, **kwargs):
            pytest.fail("an already-separated initial state must be rejected before integration")

        monkeypatch.setattr(boundary_layer_solver, "solve_ivp", unexpected_integration)
        with pytest.raises(boundary_layer_solver.BoundaryLayerError,
                           match="^initial turbulent state is already separated$"):
            boundary_layer_solver.solve_boundary_layer(**arguments)
    else:
        result = boundary_layer_solver.solve_boundary_layer(**arguments)
        assert result.separation_index is None
        assert np.all(result.regime == "turbulent")
        assert np.all(np.isfinite(result.displacement_thickness_over_chord))
