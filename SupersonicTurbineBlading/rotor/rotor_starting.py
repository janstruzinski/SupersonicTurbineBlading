"""Rotor-only supersonic-starting feasibility calculation."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from ..gas_dynamics import critical_velocity_ratio, mach_from_critical_velocity_ratio, prandtl_meyer_angle
from .rotor_results import StartingResult


def _bisect(function: Callable[[float], float], lower: float, upper: float, *, tolerance: float = 1.0e-10,
            iterations: int = 150) -> float:
    """Find a scalar root inside a known sign-changing bracket.

    Bisection is slower than Newton's method but cannot leave the physical bracket. That reliability is useful for the
    strongly nonlinear starting integrals, whose derivatives are not available in NASA TN D-4421.

    :param function function: Scalar residual function.
    :param float lower: Lower root bracket.
    :param float upper: Upper root bracket.
    :param float tolerance: Absolute residual and bracket-width tolerance.
    :param int iterations: Maximum bisection iterations.
    :return: Approximate root.
    :rtype: float
    :raises ValueError: If either endpoint is non-finite or the endpoints do not bracket a root.
    """

    f_lower = function(lower)
    f_upper = function(upper)
    if not (math.isfinite(f_lower) and math.isfinite(f_upper)):
        raise ValueError("non-finite value while bracketing a root")
    if abs(f_lower) <= tolerance:
        return lower
    if abs(f_upper) <= tolerance:
        return upper
    if f_lower * f_upper > 0.0:
        raise ValueError("root is not bracketed")
    for _ in range(iterations):
        middle = 0.5 * (lower + upper)
        f_middle = function(middle)
        if abs(f_middle) <= tolerance or abs(upper - lower) <= tolerance:
            return middle
        if f_lower * f_middle <= 0.0:
            upper = middle
        else:
            lower = middle
            f_lower = f_middle
    return 0.5 * (lower + upper)


def _simpson(function: Callable[[float], float], lower: float, upper: float) -> float:
    """Integrate a scalar function with deterministic composite Simpson refinement.

    :param function function: Scalar integrand.
    :param float lower: Lower integration limit.
    :param float upper: Upper integration limit.
    :return: Numerically integrated value, with sign reversed when limits are supplied in descending order.
    :rtype: float
    """

    if lower == upper:
        return 0.0
    sign = 1.0
    if upper < lower:
        lower, upper = upper, lower
        sign = -1.0
    # Double the even panel count on every pass. Comparing two consecutive results provides a transparent convergence
    # test and avoids adding a numerical-integration dependency to this small legacy calculation.
    previous = None
    for power in range(5, 15):
        count = 2**power
        x = np.linspace(lower, upper, count + 1)
        y = np.array([function(float(item)) for item in x], dtype=float)
        step = (upper - lower) / count
        result = step / 3.0 * (y[0] + y[-1] + 4.0 * y[1:-1:2].sum() + 2.0 * y[2:-2:2].sum())
        if previous is not None and abs(result - previous) <= 1.0e-9 * max(1.0, abs(result)):
            return sign * float(result)
        previous = result
    return sign * float(previous)


def calculate_starting_limit(
    ideal_inlet_relative_flow_mach: float,
    lower_surface_relative_flow_mach: float,
    upper_surface_relative_flow_mach: float,
    gamma: float) -> StartingResult:
    """Port the ``START`` calculation in NASA TN D-4421.

    The result is the largest relative inlet Mach number that can swallow the
    assumed passage-spanning normal shock for the selected vortex radii.

    :param float ideal_inlet_relative_flow_mach: Specified rotor-relative far-field inlet flow Mach, -.
    :param float lower_surface_relative_flow_mach: Constant pressure-side vortex flow Mach, -.
    :param float upper_surface_relative_flow_mach: Constant suction-side vortex flow Mach, -.
    :param float gamma: Frozen specific-heat ratio, -.
    :return: Starting limit and the intermediate NASA TN D-4421 diagnostics.
    :rtype: StartingResult
    :raises ValueError: If a required integral leaves its real domain or a starting root cannot be bracketed.
    """

    # NASA TN D-4421 performs most of the starting analysis in critical velocity
    # ratio M*=V/Vcr. The two surface Mach numbers therefore enter through
    # their corresponding vortex radii.
    gamma_minus = 0.5 * (gamma - 1.0)
    gamma_plus = 0.5 * (gamma + 1.0)
    exponent = 1.0 / (gamma - 1.0)
    limiting_ratio = math.sqrt(gamma_plus / gamma_minus)
    lower_mach_star = critical_velocity_ratio(lower_surface_relative_flow_mach, gamma)
    upper_mach_star = critical_velocity_ratio(upper_surface_relative_flow_mach, gamma)

    def alfunc(a: float, b: float, argument: float) -> float:
        """Evaluate the repeated algebraic factor in the starting integrals.

        :param float a: Constant term in the algebraic base.
        :param float b: Multiplier on the squared integration argument.
        :param float argument: Integration coordinate.
        :return: Algebraic factor raised to the NASA TN D-4421 exponent.
        :rtype: float
        """

        base = a - b * argument * argument
        if base < -1.0e-12:
            raise ValueError("starting integral left its real-valued domain")
        return max(base, 0.0) ** exponent / argument

    def ofact(argument: float) -> float:
        """Evaluate the NASA TN D-4421 ``OFACT`` integrand factor.

        :param float argument: Critical velocity ratio.
        :return: ``OFACT`` value.
        :rtype: float
        """

        return alfunc(gamma_plus, gamma_minus, argument)

    if abs(upper_mach_star - lower_mach_star) <= 1.0e-10:
        # Equal surface Mach numbers collapse the vortex interval. Use the
        # analytic limiting form instead of dividing by a vanishing width.
        k_max = 1.0 / limiting_ratio
        flow_reduction = 0.0
        q_value = upper_mach_star**2 * ofact(upper_mach_star)
        weight_integral = 0.0
    else:

        def fkmax(k_value: float, return_integral: bool = False) -> float:
            """Return the vortex-constant residual or integral for one trial.

            :param float k_value: Trial vortex constant.
            :param bool return_integral: Return the integral instead of its root residual.
            :return: Integral or ``FKMAX`` residual.
            :rtype: float
            """

            same = (k_value / lower_mach_star) ** 2

            def cfact(argument: float) -> float:
                """Evaluate ``CFACT`` at the current vortex constant.

                :param float argument: Critical velocity ratio.
                :return: ``CFACT`` integrand value.
                :rtype: float
                """

                return alfunc(1.0, same, argument)

            integral = _simpson(cfact, lower_mach_star, upper_mach_star)
            if return_integral:
                return integral
            return integral + upper_mach_star * cfact(upper_mach_star) - alfunc(1.0, k_value**2, 1.0)

        # Search near the NASA TN D-4421 estimate and locate a sign change on a fixed
        # grid before bisection. Keeping bracket discovery separate makes
        # failures easier to diagnose.
        estimate = (1.0 / limiting_ratio) * math.sqrt(lower_mach_star / upper_mach_star)
        lower = max(1.0e-10, estimate - 0.12)
        upper = min(lower_mach_star / upper_mach_star - 1.0e-10, estimate + 0.12)
        grid = np.linspace(lower, upper, 200)
        pairs = [(float(value), fkmax(float(value))) for value in grid]
        bracket = next(((pairs[index][0], pairs[index + 1][0])
                        for index in range(len(pairs) - 1)
                        if pairs[index][1] * pairs[index + 1][1] <= 0.0), None)
        if bracket is None:
            raise ValueError("could not bracket the NASA TN D-4421 starting root")
        k_max = _bisect(lambda value: fkmax(value), *bracket)
        weight_integral = fkmax(k_max, return_integral=True)
        flow_reduction = (1.0
            - limiting_ratio
            * gamma_plus**exponent
            * upper_mach_star
            / (upper_mach_star - lower_mach_star)
            * k_max
            * weight_integral)
        q_integral = _simpson(ofact, lower_mach_star, upper_mach_star)
        q_value = lower_mach_star * upper_mach_star / (upper_mach_star - lower_mach_star) * q_integral
        q_value /= 1.0 - flow_reduction

    def frat(argument: float) -> float:
        """Evaluate the mass-flow ratio used to recover maximum inlet Mach.

        :param float argument: Critical velocity ratio.
        :return: Final NASA TN D-4421 mass-flow ratio.
        :rtype: float
        """

        return argument ** (gamma / gamma_minus) * ofact(argument) / alfunc(-gamma_minus, -gamma_plus, argument)

    # The final root lies on the supersonic critical-velocity branch between
    # M*=1 and its infinite-Mach limit.
    grid = np.linspace(1.0 + 1.0e-9, limiting_ratio - 1.0e-9, 600)
    values = [(float(argument), frat(float(argument)) - q_value) for argument in grid]
    bracket = next(((values[index][0], values[index + 1][0])
                    for index in range(len(values) - 1)
                    if values[index][1] * values[index + 1][1] <= 0.0), None)
    if bracket is None:
        raise ValueError("could not bracket maximum starting inlet Mach")
    maximum_mach_star = _bisect(lambda value: frat(value) - q_value, *bracket)
    maximum_starting_ideal_inlet_relative_flow_mach = mach_from_critical_velocity_ratio(maximum_mach_star, gamma)
    maximum_starting_ideal_inlet_relative_prandtl_meyer_angle = math.degrees(
        prandtl_meyer_angle(maximum_starting_ideal_inlet_relative_flow_mach, gamma))
    weight_flow_parameter = (
        (1.0 / gamma_plus) ** (gamma_plus / (2.0 * gamma_minus)) * weight_integral if weight_integral else 0.0)
    return StartingResult(
        maximum_starting_ideal_inlet_relative_flow_mach=maximum_starting_ideal_inlet_relative_flow_mach,
        maximum_starting_ideal_inlet_relative_prandtl_meyer_angle=(
            maximum_starting_ideal_inlet_relative_prandtl_meyer_angle),
        specified_ideal_inlet_relative_flow_mach=ideal_inlet_relative_flow_mach,
        starts_supersonically=ideal_inlet_relative_flow_mach <= maximum_starting_ideal_inlet_relative_flow_mach,
        critical_vortex_constant=k_max,
        two_dimensional_flow_reduction=flow_reduction,
        weight_flow_parameter=weight_flow_parameter)
