"""Property consistency across the optional RocketTurbines adapter."""

from types import SimpleNamespace

import CoolProp.CoolProp as CP
import pytest

from SupersonicTurbineBlading import Fluid, FluidPropertyError


@pytest.fixture
def ideal_gas():
    # Match the upstream interface, which does not retain mass_fractions.
    # Distinct thermodynamic constants catch accidental CoolProp recomputation.
    return SimpleNamespace(species=("Nitrogen", "Oxygen"),
                           molar_masses=(0.02801348, 0.0319988),
                           molar_fractions=(0.6, 0.4),
                           R=300.0, Cp=1200.0, Cv=900.0, gamma=4.0 / 3.0,
                           calculate_dynamic_viscosity=lambda T: 3.0e-5 * (T / 500.0)**0.7,
                           calculate_thermal_conductivity=lambda T: 0.045 * (T / 500.0)**0.8)


def test_ideal_gas_retains_thermodynamics_transport_and_composition(ideal_gas, monkeypatch):
    def unexpected_coolprop_property(*args):
        pytest.fail("IdealGas properties must not be recomputed through CoolProp")

    monkeypatch.setattr(CP, "PropsSI", unexpected_coolprop_property)
    fluid = Fluid(ideal_gas=ideal_gas)
    expected_molar_mass = 0.6 * 0.02801348 + 0.4 * 0.0319988
    expected_mass_fractions = (0.6 * 0.02801348 / expected_molar_mass,
                               0.4 * 0.0319988 / expected_molar_mass)
    assert fluid.coolprop_names == ideal_gas.species
    assert fluid.molar_masses == ideal_gas.molar_masses
    assert fluid.mole_fractions == pytest.approx(ideal_gas.molar_fractions)
    assert fluid.mass_fractions == pytest.approx(expected_mass_fractions)
    assert fluid.molar_mass == pytest.approx(expected_molar_mass)
    assert fluid.specific_gas_constant == ideal_gas.R

    for temperature, pressure in ((500.0, 2.0e5), (900.0, 1.0e6)):
        state = fluid.properties(temperature, pressure)
        assert state.specific_heat_cp == ideal_gas.Cp
        assert state.specific_heat_cv == ideal_gas.Cv
        assert state.gamma == ideal_gas.gamma
        assert state.density == pytest.approx(pressure / (ideal_gas.R * temperature))
        assert state.speed_of_sound == pytest.approx((ideal_gas.gamma * ideal_gas.R * temperature)**0.5)
        assert state.dynamic_viscosity == ideal_gas.calculate_dynamic_viscosity(temperature)
        assert state.kinematic_viscosity == pytest.approx(state.dynamic_viscosity / state.density)
        assert state.component_partial_pressures == pytest.approx((0.6 * pressure, 0.4 * pressure))
        expected_conductivity = ideal_gas.calculate_thermal_conductivity(temperature)
        assert state.thermal_conductivity == expected_conductivity
        assert state.prandtl_number == pytest.approx(ideal_gas.Cp * state.dynamic_viscosity / expected_conductivity)
        assert state.component_specific_heats_cp == ()
        assert state.component_dynamic_viscosities == ()
        assert state.component_thermal_conductivities == ()


def test_composition_based_fluid_retains_coolprop_conductivity():
    fluid = Fluid(["Nitrogen", "Oxygen"], [0.767, 0.233])
    state = fluid.properties(500.0, 2.0e5)
    component_conductivities = tuple(CP.PropsSI("CONDUCTIVITY", "T", 500.0, "P", x * 2.0e5, name)
                                     for x, name in zip(fluid.mole_fractions, fluid.coolprop_names))
    assert state.component_thermal_conductivities == pytest.approx(component_conductivities)
    assert state.thermal_conductivity == pytest.approx(sum(w * k for w, k in zip(fluid.mass_fractions,
                                                                             component_conductivities)))


@pytest.mark.parametrize("composition", [dict(coolprop_names=["Nitrogen"]), dict(mass_fractions=[1.0])])
def test_ideal_gas_rejects_conflicting_composition(ideal_gas, composition):
    with pytest.raises(ValueError, match="either ideal_gas"):
        Fluid(ideal_gas=ideal_gas, **composition)


def test_fluid_requires_composition_or_ideal_gas():
    with pytest.raises(TypeError, match="provide coolprop_names"):
        Fluid()
    with pytest.raises(TypeError, match="IdealGas interface"):
        Fluid(ideal_gas=object())


def test_ideal_gas_requires_conductivity_method(ideal_gas):
    del ideal_gas.calculate_thermal_conductivity
    with pytest.raises(TypeError, match="including calculate_thermal_conductivity"):
        Fluid(ideal_gas=ideal_gas)


@pytest.mark.parametrize("attribute, value", [("Cp", float("nan")), ("Cv", -1.0), ("gamma", 1.4),
                                            ("molar_fractions", (0.5, 0.4)), ("molar_masses", (0.028,))])
def test_ideal_gas_rejects_invalid_model(ideal_gas, attribute, value):
    setattr(ideal_gas, attribute, value)
    with pytest.raises(ValueError, match="IdealGas"):
        Fluid(ideal_gas=ideal_gas)


@pytest.mark.parametrize("property_name, method_name", [("viscosity", "calculate_dynamic_viscosity"),
                                                       ("thermal conductivity", "calculate_thermal_conductivity")])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_ideal_gas_rejects_invalid_transport_property(ideal_gas, property_name, method_name, value):
    setattr(ideal_gas, method_name, lambda T: value)
    with pytest.raises(FluidPropertyError, match=f"invalid {property_name}"):
        Fluid(ideal_gas=ideal_gas).properties(500.0, 1.0e5)


@pytest.mark.parametrize("property_name, method_name", [("viscosity", "calculate_dynamic_viscosity"),
                                                       ("thermal conductivity", "calculate_thermal_conductivity")])
def test_ideal_gas_wraps_transport_failure(ideal_gas, property_name, method_name):
    def unavailable(T):
        raise ValueError("transport fit unavailable")

    setattr(ideal_gas, method_name, unavailable)
    with pytest.raises(FluidPropertyError, match=f"IdealGas could not evaluate {property_name}") as error:
        Fluid(ideal_gas=ideal_gas).properties(500.0, 1.0e5)
    assert isinstance(error.value.__cause__, ValueError)


@pytest.mark.parametrize("species, fractions, cea_species, temperatures", [
    (("Nitrogen", "Oxygen", "Hydrogen"), (0.767, 0.233, 0.0), ("N2", "O2", "H2"), (250.0, 600.0, 1000.0)),
    (("Water", "Oxygen"), (0.54, 0.46), ("H2O", "O2"), (600.0, 1000.0)),
])
def test_rocket_turbines_ideal_gas_integration(species, fractions, cea_species, temperatures):
    IdealGas = pytest.importorskip("IdealGas").IdealGas
    gas = IdealGas(T_min=500.0, T_max=1100.0, species=species,
                   mass_fractions=fractions, cea_species=cea_species)
    fluid = Fluid(ideal_gas=gas)
    assert fluid.mass_fractions == pytest.approx(tuple(w for w in fractions if w > 0.0))
    for temperature in temperatures:
        state = fluid.properties(temperature, 2.0e5)
        assert state.specific_heat_cp == gas.Cp
        assert state.specific_heat_cv == gas.Cv
        assert state.gamma == gas.gamma
        assert state.density == gas.calculate_density(2.0e5, temperature)
        assert state.speed_of_sound == pytest.approx(gas.calculate_sound_velocity(temperature))
        assert state.dynamic_viscosity == gas.calculate_dynamic_viscosity(temperature)
        assert state.thermal_conductivity == gas.calculate_thermal_conductivity(temperature)
        assert state.prandtl_number == pytest.approx(gas.Cp * gas.calculate_dynamic_viscosity(temperature)
                                                    / gas.calculate_thermal_conductivity(temperature))
