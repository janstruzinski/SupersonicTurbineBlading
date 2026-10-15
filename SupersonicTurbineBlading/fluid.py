"""Shared gas properties from pure-fluid CoolProp calls or RocketTurbines IdealGas.

CoolProp can evaluate many mixtures directly.  This package deliberately
does not use that interface, because a failed mixture flash would make otherwise simple
preliminary designs impossible. Objects with mixture fluids evaluate each pure
gas species separately and apply the mixing rules below.
Using IdealGas from RocketTurbines to construct Fluid retains IdealGas'
supplied thermodynamics and transport models.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol, runtime_checkable

import CoolProp.CoolProp as CP


@runtime_checkable
class _IdealGas(Protocol):
    """RocketTurbines interface, without requiring it as an installed dependency."""

    species: Sequence[str]
    molar_masses: Sequence[float]
    molar_fractions: Sequence[float]
    R: float
    Cp: float
    Cv: float
    gamma: float

    def calculate_dynamic_viscosity(self, T: float) -> float: ...

    def calculate_thermal_conductivity(self, T: float) -> float: ...


class FluidPropertyError(RuntimeError):
    """Raised when a property provider cannot evaluate a required gas property."""


@dataclass(frozen=True)
class FluidState:
    """Properties of an ideal-gas mixture represented by FluidState at a specific temperature and pressure are recalled
     below.

    All values use SI units, irrespective of the eventual units used to
    dimensionalize the blade coordinates.

    :param float temperature: Mixture temperature, K.
    :param float pressure: Mixture absolute pressure, Pa.
    :param float density: Ideal-gas mixture density, kg/m³.
    :param float specific_heat_cp: Constant-pressure specific heat, J/(kg K).
    :param float specific_heat_cv: Ideal-gas constant-volume heat, J/(kg K).
    :param float gamma: Ratio of specific heats, dimensionless.
    :param float dynamic_viscosity: Mixture viscosity, Pa s.
    :param float kinematic_viscosity: ``dynamic_viscosity / density``, m²/s.
    :param float thermal_conductivity: Mixture conductivity, W/(m K).
    :param float prandtl_number: ``cp * dynamic_viscosity / conductivity``.
    :param float speed_of_sound: Ideal-gas speed of sound, m/s.
    :param tuple component_partial_pressures: Dalton partial pressures, Pa.
    :param tuple component_specific_heats_cp: Pure-component ideal-gas Cp;
        empty when an IdealGas supplies the mixture heat capacity.
    :param tuple component_dynamic_viscosities: Pure-component viscosities;
        empty when an IdealGas supplies the mixture viscosity.
    :param tuple component_thermal_conductivities: Pure-component conductivities;
        empty when an IdealGas supplies the mixture conductivity.
    """

    temperature: float
    pressure: float
    density: float
    specific_heat_cp: float
    specific_heat_cv: float
    gamma: float
    dynamic_viscosity: float
    kinematic_viscosity: float
    thermal_conductivity: float
    prandtl_number: float
    speed_of_sound: float
    component_partial_pressures: tuple[float, ...]
    component_specific_heats_cp: tuple[float, ...]
    component_dynamic_viscosities: tuple[float, ...]
    component_thermal_conductivities: tuple[float, ...]


class Fluid:
    """Represent a CoolProp mixture or adapt a RocketTurbines IdealGas.

    The composition is fixed at initialization.  State-dependent properties
    are obtained later with :meth:`properties`.

    The class is intended for gases.  A component that CoolProp identifies as
    liquid or two-phase at its Dalton partial pressure is rejected instead of
    silently applying gas equations to an invalid state.

    :param Sequence[str] coolprop_names: CoolProp names of mixture components.
    :param Sequence[float] mass_fractions: Component mass fractions; these
        must be positive and sum to one.
    :param ideal_gas: Optional RocketTurbines ``IdealGas`` object, used instead
        of composition inputs. Its heat capacities, gas constant, specific-heat
        ratio, viscosity and thermal-conductivity models are retained.
    """

    def __init__(self, coolprop_names: Sequence[str] | None = None,
                 mass_fractions: Sequence[float] | None = None, *, ideal_gas: _IdealGas | None = None) -> None:
        """Create a gas mixture with fixed chemical composition.

        Composition and any supplied IdealGas constants are stored here. State inputs go to :meth:`properties`, making
        one ``Fluid`` object reusable at all stations of the rotor, stator, and boundary-layer calculations.

        :param Sequence[str] coolprop_names: CoolProp names of all pure-fluid components.
        :param Sequence[float] mass_fractions: Positive component mass fractions that sum to one, -.
        :param ideal_gas: RocketTurbines gas to adapt, mutually exclusive with composition inputs.
        :raises TypeError: If composition inputs are missing or ideal_gas does not expose the required interface.
        :raises ValueError: If names and fractions are empty, inconsistent,
            duplicated, non-positive, or do not sum to one, or the IdealGas constants are inconsistent.
        :raises FluidPropertyError: If CoolProp cannot provide molar mass or gas constant for a component.
        """

        self._ideal_gas = ideal_gas
        if ideal_gas is not None:
            if coolprop_names is not None or mass_fractions is not None:
                raise ValueError("provide either ideal_gas or coolprop_names and mass_fractions, not both")
            if not isinstance(ideal_gas, _IdealGas):
                raise TypeError("ideal_gas must expose the RocketTurbines IdealGas interface, "
                                "including calculate_thermal_conductivity")
            coolprop_names, mass_fractions = self._ideal_gas_composition(ideal_gas)
        elif coolprop_names is None or mass_fractions is None:
            raise TypeError("provide coolprop_names and mass_fractions, or ideal_gas")

        # Copy the inputs to immutable tuples.  This prevents a caller from
        # changing a list after initialization and invalidating cached states.
        names = tuple(str(name).strip() for name in coolprop_names)
        fractions = tuple(float(value) for value in mass_fractions)

        if not names:
            raise ValueError("coolprop_names must contain at least one fluid")
        if len(names) != len(fractions):
            raise ValueError("coolprop_names and mass_fractions must have the same length")
        if any(not name for name in names):
            raise ValueError("CoolProp fluid names must not be empty")
        if len(set(names)) != len(names):
            raise ValueError("each CoolProp fluid name must appear only once")
        if any(not math.isfinite(value) or value <= 0.0 for value in fractions):
            raise ValueError("all mass fractions must be positive and finite")

        fraction_sum = math.fsum(fractions)
        if not math.isclose(fraction_sum, 1.0, rel_tol=0.0, abs_tol=1.0e-8):
            raise ValueError("mass fractions must sum to one")

        # Remove harmless floating-point summation error.  A materially
        # incorrect composition was rejected above and is never normalized
        # silently.
        fractions = tuple(value / fraction_sum for value in fractions)

        # Molar mass is a composition constant, so it can be queried without
        # a temperature-pressure state.  CoolProp returns kg/mol.
        molar_masses = (tuple(self._trivial_property("MOLAR_MASS", name) for name in names)
                        if ideal_gas is None else tuple(float(value) for value in ideal_gas.molar_masses))

        # Convert the user-facing mass fractions to mole fractions.  These are
        # required only for Dalton partial pressures:
        #
        #       x_i = (w_i / M_i) / sum_j(w_j / M_j)
        #
        mole_amounts = tuple(fraction / molar_mass for fraction, molar_mass in zip(fractions, molar_masses))
        mole_amount_sum = math.fsum(mole_amounts)
        mole_fractions = tuple(value / mole_amount_sum for value in mole_amounts)

        # For an ideal mixture, 1/M_mix = sum(w_i/M_i).  The mass-specific gas
        # constant then follows from R_mix = R_universal/M_mix.
        mixture_molar_mass = 1.0 / mole_amount_sum
        universal_gas_constant = (self._trivial_property("GAS_CONSTANT", names[0])
                                  if ideal_gas is None else float(ideal_gas.R) * mixture_molar_mass)

        self.coolprop_names = names
        self.mass_fractions = fractions
        self.molar_masses = molar_masses
        self.mole_fractions = mole_fractions
        self.molar_mass = mixture_molar_mass
        self.universal_gas_constant = universal_gas_constant
        self.specific_gas_constant = universal_gas_constant / mixture_molar_mass

        if ideal_gas is not None:
            # Copy the calorically perfect constants rather than recomputing Cp
            # at local blade temperatures. IdealGas does not store mass fractions.
            self.specific_gas_constant = float(ideal_gas.R)
            self._specific_heat_cp = float(ideal_gas.Cp)
            self._specific_heat_cv = float(ideal_gas.Cv)
            self._gamma = float(ideal_gas.gamma)
            constants = (self.specific_gas_constant, self._specific_heat_cp, self._specific_heat_cv, self._gamma)
            if any(not math.isfinite(value) or value <= 0.0 for value in constants):
                raise ValueError("IdealGas R, Cp, Cv and gamma must be positive and finite")
            if (self._gamma <= 1.0
                    or not math.isclose(self._specific_heat_cp - self.specific_gas_constant,
                                        self._specific_heat_cv, rel_tol=1.0e-10)
                    or not math.isclose(self._specific_heat_cp / self._specific_heat_cv,
                                        self._gamma, rel_tol=1.0e-10)):
                raise ValueError("IdealGas must satisfy Cv = Cp - R and gamma = Cp / Cv > 1")

    @staticmethod
    def _ideal_gas_composition(ideal_gas: _IdealGas) -> tuple[tuple[str, ...], tuple[float, ...]]:
        """Recover mass fractions from IdealGas mole fractions and molar masses."""

        names = tuple(ideal_gas.species)
        molar_masses = tuple(float(value) for value in ideal_gas.molar_masses)
        mole_fractions = tuple(float(value) for value in ideal_gas.molar_fractions)
        if not names or len(names) != len(molar_masses) or len(names) != len(mole_fractions):
            raise ValueError("IdealGas species, molar_masses and molar_fractions must have the same nonzero length")
        if any(not math.isfinite(value) or value <= 0.0 for value in (*molar_masses, *mole_fractions)):
            raise ValueError("IdealGas molar masses and mole fractions must be positive and finite")
        if not math.isclose(math.fsum(mole_fractions), 1.0, rel_tol=0.0, abs_tol=1.0e-8):
            raise ValueError("IdealGas mole fractions must sum to one")
        mass_amounts = tuple(x * molar_mass for x, molar_mass in zip(mole_fractions, molar_masses))
        mixture_molar_mass = math.fsum(mass_amounts)
        return names, tuple(value / mixture_molar_mass for value in mass_amounts)

    @staticmethod
    def _trivial_property(output: str, fluid_name: str) -> float:
        """Read a state-independent CoolProp property with a clear error.

        :param str output: CoolProp output key, for example ``"MOLAR_MASS"``.
        :param str fluid_name: CoolProp pure-fluid name.
        :return: Positive property value in CoolProp SI units.
        :rtype: float
        :raises FluidPropertyError: If CoolProp fails or returns a non-positive value.
        """

        try:
            value = float(CP.PropsSI(output, fluid_name))
        except Exception as error:
            raise FluidPropertyError(f"CoolProp could not evaluate {output!r} for {fluid_name!r}") from error
        if not math.isfinite(value) or value <= 0.0:
            raise FluidPropertyError(f"CoolProp returned an invalid {output!r} for {fluid_name!r}: {value}")
        return value

    @staticmethod
    def _state_property(output: str, temperature: float, pressure: float, fluid_name: str) -> float:
        """Read one pure-fluid property at a specified SI state.

        :param str output: CoolProp output key.
        :param float temperature: Component temperature, K.
        :param float pressure: Component Dalton partial pressure, Pa.
        :param str fluid_name: CoolProp pure-fluid name.
        :return: Positive property value in CoolProp SI units.
        :rtype: float
        :raises FluidPropertyError: If CoolProp fails or returns a non-positive value.
        """

        try:
            value = float(CP.PropsSI(output, "T", temperature, "P", pressure, fluid_name))
        except Exception as error:
            raise FluidPropertyError(f"CoolProp could not evaluate {output!r} for {fluid_name!r} at "
                                     f"T={temperature:.6g} K and P={pressure:.6g} Pa") from error
        if not math.isfinite(value) or value <= 0.0:
            raise FluidPropertyError(f"CoolProp returned an invalid {output!r} for {fluid_name!r}: {value}")
        return value

    @staticmethod
    def _check_gas_phase(temperature: float, pressure: float, fluid_name: str) -> None:
        """Reject pure-component states incompatible with the gas model.

        :param float temperature: Component temperature, K.
        :param float pressure: Component Dalton partial pressure, Pa.
        :param str fluid_name: CoolProp pure-fluid name.
        :raises FluidPropertyError: If the phase cannot be evaluated or is not gas-like.
        """

        try:
            phase = str(CP.PhaseSI("T", temperature, "P", pressure, fluid_name))
        except Exception as error:
            raise FluidPropertyError(f"CoolProp could not identify the phase of {fluid_name!r}") from error

        gas_phases = {"gas", "supercritical_gas", "supercritical"}
        if phase not in gas_phases:
            raise FluidPropertyError(f"{fluid_name!r} is {phase!r} at its partial-pressure state; "
                                     "the Fluid model is restricted to gases")

    def properties(self, temperature: float, pressure: float) -> FluidState:
        """Return mixture properties at a temperature and absolute pressure.

        Each pure component is evaluated at the mixture temperature and its
        Dalton partial pressure ``x_i * pressure``.  This is consistent with
        the ideal-gas density assumption and avoids asking CoolProp to solve a
        potentially unsupported mixture flash.

        With ``ideal_gas``, heat capacities and gamma remain fixed at the
        supplied values, and viscosity and thermal conductivity come from its
        temperature-dependent methods. Only gas-phase checks use CoolProp. Treat the
        supplied IdealGas as fixed after construction, as states are cached.

        :param float temperature: Mixture temperature, K.
        :param float pressure: Mixture absolute pressure, Pa.
        :return: Immutable SI property bundle for the requested state.
        :rtype: FluidState
        :raises ValueError: If temperature or pressure is not positive and finite.
        :raises FluidPropertyError: If a component is not gaseous or a required property cannot be evaluated.
        """

        temperature = float(temperature)
        pressure = float(pressure)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be positive and finite")
        if not math.isfinite(pressure) or pressure <= 0.0:
            raise ValueError("pressure must be positive and finite")
        return self._properties_cached(temperature, pressure)

    @lru_cache(maxsize=1024)
    def _properties_cached(self, temperature: float, pressure: float) -> FluidState:
        """Evaluate and cache one mixture state.

        The public :meth:`properties` method validates the inputs before this helper is called. ``lru_cache`` avoids
        repeating relatively expensive CoolProp calls when design iterations revisit the same trial state.

        :param float temperature: Validated mixture temperature, K.
        :param float pressure: Validated mixture absolute pressure, Pa.
        :return: Immutable SI property bundle for the requested state.
        :rtype: FluidState
        """

        partial_pressures = tuple(mole_fraction * pressure for mole_fraction in self.mole_fractions)
        component_cp: list[float] = []
        component_viscosity: list[float] = []
        component_conductivity: list[float] = []

        for name, partial_pressure in zip(self.coolprop_names, partial_pressures):
            self._check_gas_phase(temperature, partial_pressure, name)

            if self._ideal_gas is None:
                # CP0MASS excludes real-fluid residual effects, consistent
                # with the ideal-gas density and Cv used below.
                component_cp.append(self._state_property("CP0MASS", temperature, partial_pressure, name))

                # Composition-based fluids use a mass-fraction viscosity average.
                component_viscosity.append(self._state_property("VISCOSITY", temperature, partial_pressure, name))

                component_conductivity.append(self._state_property("CONDUCTIVITY", temperature, partial_pressure, name))

        if self._ideal_gas is None:
            specific_heat_cp = math.fsum(fraction * value for fraction, value in zip(self.mass_fractions, component_cp))
            specific_heat_cv = specific_heat_cp - self.specific_gas_constant
            if specific_heat_cv <= 0.0:
                raise FluidPropertyError("mixture Cp is not greater than its specific gas constant")
            gamma = specific_heat_cp / specific_heat_cv
            dynamic_viscosity = math.fsum(fraction * value
                                          for fraction, value in zip(self.mass_fractions, component_viscosity))
            thermal_conductivity = math.fsum(fraction * value
                                             for fraction, value in zip(self.mass_fractions, component_conductivity))
        else:
            specific_heat_cp = self._specific_heat_cp
            specific_heat_cv = self._specific_heat_cv
            gamma = self._gamma
            try:
                dynamic_viscosity = float(self._ideal_gas.calculate_dynamic_viscosity(temperature))
            except Exception as error:
                raise FluidPropertyError(f"IdealGas could not evaluate viscosity at T={temperature:.6g} K") from error
            if not math.isfinite(dynamic_viscosity) or dynamic_viscosity <= 0.0:
                raise FluidPropertyError(f"IdealGas returned an invalid viscosity: {dynamic_viscosity}")
            try:
                thermal_conductivity = float(self._ideal_gas.calculate_thermal_conductivity(temperature))
            except Exception as error:
                raise FluidPropertyError(f"IdealGas could not evaluate thermal conductivity at "
                                         f"T={temperature:.6g} K") from error
            if not math.isfinite(thermal_conductivity) or thermal_conductivity <= 0.0:
                raise FluidPropertyError(f"IdealGas returned an invalid thermal conductivity: {thermal_conductivity}")

        # The following properties do not require another CoolProp call.
        # They are consequences of the ideal-gas equation of state and the
        # calorically-perfect relation Cv = Cp - R_mix at this local state.
        density = pressure / (self.specific_gas_constant * temperature)
        kinematic_viscosity = dynamic_viscosity / density
        prandtl_number = specific_heat_cp * dynamic_viscosity / thermal_conductivity
        speed_of_sound = math.sqrt(gamma * self.specific_gas_constant * temperature)

        return FluidState(temperature=temperature,
                          pressure=pressure,
                          density=density,
                          specific_heat_cp=specific_heat_cp,
                          specific_heat_cv=specific_heat_cv,
                          gamma=gamma,
                          dynamic_viscosity=dynamic_viscosity,
                          kinematic_viscosity=kinematic_viscosity,
                          thermal_conductivity=thermal_conductivity,
                          prandtl_number=prandtl_number,
                          speed_of_sound=speed_of_sound,
                          component_partial_pressures=partial_pressures,
                          component_specific_heats_cp=tuple(component_cp),
                          component_dynamic_viscosities=tuple(component_viscosity),
                          component_thermal_conductivities=tuple(component_conductivity))
