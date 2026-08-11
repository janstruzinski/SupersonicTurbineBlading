"""Shared ideal-gas mixture properties assembled from pure-fluid CoolProp calls.

CoolProp can evaluate many mixtures directly.  This package deliberately
does not use that interface, because some component pairs do not have binary
interaction data and a failed mixture flash would make otherwise simple
preliminary designs impossible.  Instead, every component is evaluated as a
pure gas and the explicitly documented mixing rules below are applied.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import CoolProp.CoolProp as CP


class FluidPropertyError(RuntimeError):
    """Raised when CoolProp cannot provide a required gas property."""


@dataclass(frozen=True)
class FluidState:
    """Properties of an ideal-gas mixture at one temperature and pressure.

    All values use SI units, irrespective of the eventual units used to
    dimensionalize the blade coordinates.

    :param float temperature: Mixture temperature, K.
    :param float pressure: Mixture absolute pressure, Pa.
    :param float density: Ideal-gas mixture density, kg/m³.
    :param float specific_heat_cp: Constant-pressure specific heat, J/(kg K).
    :param float specific_heat_cv: Ideal-gas constant-volume heat, J/(kg K).
    :param float gamma: Ratio of specific heats, dimensionless.
    :param float dynamic_viscosity: Mass-averaged viscosity, Pa s.
    :param float kinematic_viscosity: ``dynamic_viscosity / density``, m²/s.
    :param float thermal_conductivity: Mass-averaged conductivity, W/(m K).
    :param float prandtl_number: ``cp * dynamic_viscosity / conductivity``.
    :param float speed_of_sound: Ideal-gas speed of sound, m/s.
    :param tuple component_partial_pressures: Dalton partial pressures, Pa.
    :param tuple component_specific_heats_cp: Pure-component ideal-gas Cp.
    :param tuple component_dynamic_viscosities: Pure-component viscosities.
    :param tuple component_thermal_conductivities: Pure-component conductivities.
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
    """Represent an ideal-gas mixture built from CoolProp pure fluids.

    The composition is fixed at initialization.  State-dependent properties
    are obtained later with :meth:`properties`, because temperature and
    pressure belong to the turbomachinery operating point rather than to the
    chemical composition.

    The class is intended for gases.  A component that CoolProp identifies as
    liquid or two-phase at its Dalton partial pressure is rejected instead of
    silently applying gas equations to an invalid state.

    :param Sequence[str] coolprop_names: CoolProp names of mixture components.
    :param Sequence[float] mass_fractions: Component mass fractions; these
        must be positive and sum to one.
    """

    def __init__(self, coolprop_names: Sequence[str], mass_fractions: Sequence[float]) -> None:
        """Create a gas mixture with fixed chemical composition.

        Only composition is stored here. Temperature and pressure are supplied later to :meth:`properties`, which makes
        one ``Fluid`` object reusable at all stations of the rotor, stator, and boundary-layer calculations.

        :param Sequence[str] coolprop_names: CoolProp names of all pure-fluid components.
        :param Sequence[float] mass_fractions: Positive component mass fractions that sum to one, -.
        :raises ValueError: If names and fractions are empty, inconsistent,
            duplicated, non-positive, or do not sum to one.
        :raises FluidPropertyError: If CoolProp cannot provide molar mass or gas constant for a component.
        """

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
        molar_masses = tuple(self._trivial_property("MOLAR_MASS", name) for name in names)

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
        universal_gas_constant = self._trivial_property("GAS_CONSTANT", names[0])

        self.coolprop_names = names
        self.mass_fractions = fractions
        self.molar_masses = molar_masses
        self.mole_fractions = mole_fractions
        self.molar_mass = mixture_molar_mass
        self.universal_gas_constant = universal_gas_constant
        self.specific_gas_constant = universal_gas_constant / mixture_molar_mass

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
            raise FluidPropertyError(
                f"CoolProp could not evaluate {output!r} for "
                f"{fluid_name!r} at T={temperature:.6g} K and "
                f"P={pressure:.6g} Pa"
            ) from error
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
            raise FluidPropertyError(
                f"{fluid_name!r} is {phase!r} at its partial-pressure state; the Fluid model is restricted to gases"
            )

    def properties(self, temperature: float, pressure: float) -> FluidState:
        """Return mixture properties at a temperature and absolute pressure.

        Each pure component is evaluated at the mixture temperature and its
        Dalton partial pressure ``x_i * pressure``.  This is consistent with
        the ideal-gas density assumption and avoids asking CoolProp to solve a
        potentially unsupported mixture flash.

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

            # CP0MASS is CoolProp's ideal-gas Cp.  Using CPMASS here would
            # retain real-fluid residual effects while density and Cv below
            # deliberately use an ideal-gas model.
            component_cp.append(self._state_property("CP0MASS", temperature, partial_pressure, name))

            # The requested viscosity mixing law is a mass-fraction average.
            # It is simple and robust, although a Wilke-type rule is normally
            # more accurate for mixtures with very different molecular masses.
            component_viscosity.append(self._state_property("VISCOSITY", temperature, partial_pressure, name))

            # Thermal conductivity is the one additional transported property
            # required by the NASA TM X-2434 and NASA TM X-2343 boundary-layer models, because Prandtl
            # number is calculated as Cp*mu/k.  It uses the same explicit
            # mass-fraction average requested for Cp and viscosity.
            component_conductivity.append(self._state_property("CONDUCTIVITY", temperature, partial_pressure, name))

        specific_heat_cp = math.fsum(fraction * value for fraction, value in zip(self.mass_fractions, component_cp))
        dynamic_viscosity = math.fsum(
            fraction * value for fraction, value in zip(self.mass_fractions, component_viscosity)
        )
        thermal_conductivity = math.fsum(
            fraction * value for fraction, value in zip(self.mass_fractions, component_conductivity)
        )

        # The following properties do not require another CoolProp call.
        # They are consequences of the ideal-gas equation of state and the
        # calorically-perfect relation Cv = Cp - R_mix at this local state.
        density = pressure / (self.specific_gas_constant * temperature)
        specific_heat_cv = specific_heat_cp - self.specific_gas_constant
        if specific_heat_cv <= 0.0:
            raise FluidPropertyError("mixture Cp is not greater than its specific gas constant")
        gamma = specific_heat_cp / specific_heat_cv
        kinematic_viscosity = dynamic_viscosity / density
        prandtl_number = specific_heat_cp * dynamic_viscosity / thermal_conductivity
        speed_of_sound = math.sqrt(gamma * self.specific_gas_constant * temperature)

        return FluidState(
            temperature=temperature,
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
            component_thermal_conductivities=tuple(component_conductivity),
        )
