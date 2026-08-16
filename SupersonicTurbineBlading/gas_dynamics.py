"""Perfect-gas relations shared by rotor and stator contouring."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import bisect, toms748


def critical_velocity_ratio(mach: float | np.ndarray, gamma: float) -> float | np.ndarray:
    """Calculate the critical velocity ratio ``M* = V/V_cr``.

    :param float or numpy.ndarray mach: Ordinary Mach number, -.
    :param float gamma: Frozen specific-heat ratio, -.
    :return: Critical velocity ratio with the same scalar or array form as ``mach``.
    :rtype: float or numpy.ndarray
    """

    mach_squared = np.asarray(mach) ** 2
    value = np.sqrt(((gamma + 1.0) * 0.5 * mach_squared) / (1.0 + (gamma - 1.0) * 0.5 * mach_squared))
    return float(value) if np.ndim(value) == 0 else value


def mach_from_critical_velocity_ratio(mach_star: float | np.ndarray, gamma: float) -> float | np.ndarray:
    """Calculate ordinary Mach number from critical velocity ratio.

    :param float or numpy.ndarray mach_star: Critical velocity ratio ``V/V_cr``, -.
    :param float gamma: Frozen specific-heat ratio, -.
    :return: Ordinary Mach number with the same scalar or array form as ``mach_star``.
    :rtype: float or numpy.ndarray
    :raises ValueError: If ``mach_star`` is outside the perfect-gas relation's real-valued domain.
    """

    mach_star_squared = np.asarray(mach_star) ** 2
    denominator = (gamma + 1.0) * 0.5 - (gamma - 1.0) * 0.5 * mach_star_squared
    if np.any(denominator <= 0.0):
        raise ValueError("critical velocity ratio is outside the perfect-gas domain")
    value = np.sqrt(mach_star_squared / denominator)
    return float(value) if np.ndim(value) == 0 else value


def prandtl_meyer_angle(mach: float | np.ndarray, gamma: float) -> float | np.ndarray:
    """Calculate the Prandtl--Meyer angle for supersonic flow.

    :param float or numpy.ndarray mach: Ordinary Mach number greater than or equal to one, -.
    :param float gamma: Frozen specific-heat ratio, -.
    :return: Prandtl--Meyer angle in radians.
    :rtype: float or numpy.ndarray
    :raises ValueError: If any supplied Mach number is subsonic.
    """

    mach = np.asarray(mach, dtype=float)
    if np.any(mach < 1.0):
        raise ValueError("the Prandtl-Meyer relation requires Mach >= 1")
    root = np.sqrt(np.maximum(mach * mach - 1.0, 0.0))
    value = (math.sqrt((gamma + 1.0) / (gamma - 1.0))
             * np.arctan(math.sqrt((gamma - 1.0) / (gamma + 1.0)) * root) - np.arctan(root))
    return float(value) if np.ndim(value) == 0 else value


def mach_from_prandtl_meyer(nu_rad: float, gamma: float) -> float:
    """Invert the Prandtl--Meyer relation on the supersonic branch.

    :param float nu_rad: Prandtl--Meyer angle, rad.
    :param float gamma: Frozen specific-heat ratio, -.
    :return: Ordinary supersonic Mach number, -.
    :rtype: float
    :raises ValueError: If the angle is negative or exceeds its infinite-Mach limit.
    :raises RuntimeError: If a numerical upper bound cannot be found or the inversion does not converge.
    """

    if nu_rad < -1.0e-12:
        raise ValueError("Prandtl-Meyer angle cannot be negative")
    if abs(nu_rad) <= 1.0e-12:
        return 1.0
    nu_limit = 0.5 * math.pi * (math.sqrt((gamma + 1.0) / (gamma - 1.0)) - 1.0)
    if nu_rad >= nu_limit:
        raise ValueError("Prandtl-Meyer angle is above its infinite-Mach limit")

    # Double the upper bound until it lies above the requested angle. This avoids imposing an arbitrary maximum design
    # Mach number while giving the TOMS Algorithm 748 iteration a guaranteed bracket.
    lower, upper = 1.0, 2.0
    while prandtl_meyer_angle(upper, gamma) < nu_rad:
        upper *= 2.0
        if upper > 1.0e8:
            raise RuntimeError("failed to bracket the Prandtl-Meyer inverse")
    return float(toms748(lambda mach: float(prandtl_meyer_angle(mach, gamma)) - nu_rad, lower, upper))


def mass_flow_parameter(mach: float | np.ndarray, gamma: float) -> float | np.ndarray:
    """Calculate the perfect-gas mass-flow function with constant factors omitted.

    :param float or numpy.ndarray mach: Ordinary Mach number, -.
    :param float gamma: Frozen specific-heat ratio, -.
    :return: Dimensionless mass-flow parameter in scalar or array form.
    :rtype: float or numpy.ndarray
    """

    mach = np.asarray(mach, dtype=float)
    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    value = mach / (1.0 + 0.5 * (gamma - 1.0) * mach * mach) ** exponent
    return float(value) if np.ndim(value) == 0 else value


def isentropic_area_ratio(mach: float | np.ndarray, gamma: float) -> float | np.ndarray:
    """Return the perfect-gas nozzle area ratio ``A/A*``.

    This is the area--Mach relation used for the conical stator
    option. Both its subsonic and supersonic branches have a minimum value
    of one at the sonic throat.

    :param float or numpy.ndarray mach: Positive ordinary Mach number, -.
    :param float gamma: Frozen specific-heat ratio greater than one, -.
    :return: Isentropic area ratio ``A/A*`` in scalar or array form.
    :rtype: float or numpy.ndarray
    :raises ValueError: If Mach or gamma is outside the perfect-gas domain.
    """

    mach = np.asarray(mach, dtype=float)
    if np.any(~np.isfinite(mach)) or np.any(mach <= 0.0):
        raise ValueError("Mach number must be positive and finite")
    if not math.isfinite(gamma) or gamma <= 1.0:
        raise ValueError("gamma must be greater than one")

    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    value = (2.0 / (gamma + 1.0) * (1.0 + 0.5 * (gamma - 1.0) * mach**2)) ** exponent / mach
    return float(value) if np.ndim(value) == 0 else value


def supersonic_mach_from_area_ratio(area_ratio: float, gamma: float) -> float:
    """Invert ``A/A*`` on the supersonic branch.

    :param float area_ratio: Isentropic area ratio ``A/A*`` greater than or equal to one, -.
    :param float gamma: Frozen specific-heat ratio greater than one, -.
    :return: Supersonic ordinary Mach number, -.
    :rtype: float
    :raises ValueError: If the area ratio or gamma is outside the perfect-gas domain.
    :raises RuntimeError: If the supersonic branch cannot be bracketed or the inversion does not converge.
    """

    if not math.isfinite(area_ratio) or area_ratio < 1.0:
        raise ValueError("area_ratio must be finite and at least one")
    if not math.isfinite(gamma) or gamma <= 1.0:
        raise ValueError("gamma must be greater than one")
    if abs(area_ratio - 1.0) <= 1.0e-14:
        return 1.0

    # The area-Mach relation is monotonic above Mach 1. Doubling the upper value gives SciPy's bisection iteration a
    # guaranteed bracket without requiring a user-supplied initial estimate.
    lower = 1.0
    upper = 2.0
    while isentropic_area_ratio(upper, gamma) < area_ratio:
        upper *= 2.0
        if upper > 1.0e8:
            raise RuntimeError("failed to bracket the supersonic area-ratio inverse")

    return float(bisect(lambda mach: float(isentropic_area_ratio(mach, gamma)) - area_ratio, lower, upper))
