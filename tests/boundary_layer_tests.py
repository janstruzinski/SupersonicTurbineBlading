import math
import warnings

import numpy as np
import pytest

from SupersonicTurbineBlading.common_results import SurfaceCoordinates
from SupersonicTurbineBlading.boundary_layer import boundary_layer_solver
from SupersonicTurbineBlading.boundary_layer.boundary_layer_solver import (
    _laminar_solution,
    _transition_incompressible_form_factor,
)


@pytest.mark.parametrize("transition_reynolds_number", [0.0, -1.0])
def test_transition_form_factor_uses_legacy_separation_fallback(transition_reynolds_number):
    laminar_form_factor = 2.4
    log_rtran = math.log(1000.0)
    expected = laminar_form_factor - 0.59389 - 0.06591 * log_rtran + 0.001272 * log_rtran**2

    assert math.isclose(
        _transition_incompressible_form_factor(laminar_form_factor, transition_reynolds_number),
        expected,
        rel_tol=1.0e-14,
    )


def test_transition_form_factor_uses_calculated_transition_reynolds_number():
    laminar_form_factor = 2.4
    transition_reynolds_number = 1750.0
    log_rtran = math.log(transition_reynolds_number)
    expected = laminar_form_factor - 0.59389 - 0.06591 * log_rtran + 0.001272 * log_rtran**2

    assert math.isclose(
        _transition_incompressible_form_factor(laminar_form_factor, transition_reynolds_number),
        expected,
        rel_tol=1.0e-14,
    )


def test_natural_transition_handoff_uses_corrected_form_factor(monkeypatch):
    gamma = 1.4
    prandtl_number = 0.72
    transition_relative_flow_mach = 2.0
    transition_reynolds_number = 1750.0
    laminar_incompressible_form = 2.4
    temperature_factor = 1.0 + 0.5 * (gamma - 1.0) * transition_relative_flow_mach**2
    laminar_compressible_form = laminar_incompressible_form * temperature_factor + prandtl_number ** (1.0 / 3.0) * (
        temperature_factor - 1.0
    )
    state = {
        "s": np.array([0.0, 1.0]),
        "edge_flow_mach": np.array([1.0, transition_relative_flow_mach]),
        "pr": prandtl_number,
    }
    laminar = {
        "theta": np.array([0.01, 0.02]),
        "displacement": np.array([0.025, 0.02 * laminar_compressible_form]),
        "form": np.array([2.5, laminar_compressible_form]),
        "transition_index": 1,
        "separation_index": None,
        "transition_reynolds_number": transition_reynolds_number,
    }
    captured = {}

    monkeypatch.setattr(boundary_layer_solver, "_surface_state", lambda *args, **kwargs: state)
    monkeypatch.setattr(boundary_layer_solver, "_laminar_solution", lambda supplied_state, correlation_limit: laminar)

    def fake_turbulent_solution(supplied_state, start_index, initial_theta, initial_incompressible_form):
        captured["initial_incompressible_form"] = initial_incompressible_form
        return {
            "theta": np.array([np.nan, initial_theta]),
            "displacement": np.array([np.nan, 0.03]),
            "form": np.array([np.nan, 1.5]),
            "separation_index": None,
        }

    monkeypatch.setattr(boundary_layer_solver, "_turbulent_solution", fake_turbulent_solution)

    surface = SurfaceCoordinates(
        x=np.array([0.0, 1.0]),
        y=np.array([0.0, 0.0]),
        relative_flow_mach=np.array([1.0, transition_relative_flow_mach]),
        metal_angle=np.array([0.0, 0.0]),
    )
    boundary_layer_solver.solve_boundary_layer(
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
    )

    assert math.isclose(
        captured["initial_incompressible_form"],
        _transition_incompressible_form_factor(laminar_incompressible_form, transition_reynolds_number),
        rel_tol=1.0e-14,
    )


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
