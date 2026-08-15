"""Method-of-characteristics blade geometry from the NASA rotor programs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..common_results import SurfaceCoordinates
from ..gas_dynamics import (
    critical_velocity_ratio,
    mach_from_critical_velocity_ratio,
    mach_from_prandtl_meyer,
    prandtl_meyer_angle,
)
from .rotor_results import BladeShape


class GeometryError(ValueError):
    """Raised when the requested Mach/angle combination cannot make a blade."""


@dataclass(frozen=True)
class _Transition:
    """Store one unrotated MOC transition from uniform flow to a vortex arc.

    :ivar x: Axial coordinates divided by vortex sonic radius, -.
    :ivar y: Tangential coordinates divided by vortex sonic radius, -.
    :ivar mach_star: Critical velocity ratio ``V/V_cr`` at each station, -.
    :ivar eta: Positive surface-tangent angle magnitude, rad.
    :ivar nu: Prandtl--Meyer angle at each station, rad.
    """

    x: np.ndarray
    y: np.ndarray
    mach_star: np.ndarray
    eta: np.ndarray
    nu: np.ndarray


@dataclass(frozen=True)
class _RotatedTransition:
    """Store a transition after it has been rotated into the final blade frame.

    :ivar x: Rotated axial coordinates divided by vortex sonic radius, -.
    :ivar y: Rotated tangential coordinates divided by vortex sonic radius, -.
    :ivar mach_star: Critical velocity ratio ``V/V_cr`` at each station, -.
    :ivar eta: Positive surface-tangent angle magnitude, rad.
    """

    x: np.ndarray
    y: np.ndarray
    mach_star: np.ndarray
    eta: np.ndarray


@lru_cache(maxsize=20_000)
def _r_from_nu(nu: float, gamma: float) -> float:
    """Calculate vortex radius divided by sonic radius from Prandtl--Meyer angle.

    The cache is useful because inlet and outlet constructions repeatedly request identical radii for the same Mach
    levels during blade-angle iterations.

    :param float nu: Prandtl--Meyer angle, rad.
    :param float gamma: Frozen specific-heat ratio, -.
    :return: Nondimensional vortex radius ``r/r*``, -.
    :rtype: float
    """

    relative_flow_mach = mach_from_prandtl_meyer(max(0.0, nu), gamma)
    return 1.0 / critical_velocity_ratio(relative_flow_mach, gamma)


def _lower_unrotated(nu_uniform: float, nu_lower: float, number_of_nodes: int, gamma: float) -> _Transition:
    """Construct the unrotated pressure-side transition.

    The transition is integrated from the constant-Mach circular arc toward uniform flow, then reversed before return
    so every stored surface proceeds from uniform flow toward the circular arc.

    :param float nu_uniform: Prandtl--Meyer angle of the uniform inlet or outlet flow, rad.
    :param float nu_lower: Prandtl--Meyer angle on the constant-Mach pressure-side arc, rad.
    :param int number_of_nodes: Fixed number of nodes across the nonzero MOC transition.
    :param float gamma: Frozen specific-heat ratio, -.
    :return: Unrotated pressure-side transition normalized by vortex sonic radius.
    :rtype: _Transition
    :raises GeometryError: If a characteristic becomes parallel to its wall segment.
    """

    transition_turning = nu_uniform - nu_lower
    if transition_turning < -1.0e-12:
        raise GeometryError("transition turning cannot be negative")
    steps = number_of_nodes - 1
    local_increment = transition_turning / steps
    r_lower = _r_from_nu(nu_lower, gamma)
    gp = 0.5 * (gamma + 1.0)
    gm = 0.5 * (gamma - 1.0)

    # Start at the constant-Mach vortex circle. ``phi`` is the polar position of the current characteristic point and
    # ``mu`` is its local Mach angle.
    phi_previous = -(nu_uniform - nu_lower) + steps * local_increment
    mu_previous = math.asin(float(np.clip(math.sqrt(max(gp * r_lower * r_lower - gm, 0.0)), -1.0, 1.0)))
    x_wall = 0.0
    y_wall = r_lower
    circle_to_uniform = [(x_wall, y_wall, 1.0 / r_lower, 0.0, nu_lower)]

    if transition_turning <= 1.0e-12:
        data = np.asarray(circle_to_uniform, dtype=float)
        return _Transition(x=data[:, 0], y=data[:, 1], mach_star=data[:, 2], eta=data[:, 3], nu=data[:, 4])

    for k in range(steps, 0, -1):
        phi = phi_previous - local_increment
        local_nu = nu_uniform - (k - 1) * local_increment
        radius = _r_from_nu(local_nu, gamma)
        x_characteristic = radius * math.sin(phi)
        y_characteristic = radius * math.cos(phi)
        wall_slope = math.tan(-phi_previous)
        mu = math.asin(float(np.clip(math.sqrt(max(gp * radius * radius - gm, 0.0)), -1.0, 1.0)))
        characteristic_slope = -math.tan(0.5 * (phi + mu + phi_previous + mu_previous))
        # Intersect the current characteristic line with the straight wall segment propagated from the previous point.
        denominator = characteristic_slope - wall_slope
        if abs(denominator) < 1.0e-14:
            raise GeometryError("lower characteristic and wall segment are parallel")
        wall_intercept = y_wall - wall_slope * x_wall
        characteristic_intercept = y_characteristic - characteristic_slope * x_characteristic
        x_wall = (wall_intercept - characteristic_intercept) / denominator
        y_wall = (characteristic_slope * wall_intercept - wall_slope * characteristic_intercept) / denominator
        phi_previous = phi
        mu_previous = mu
        circle_to_uniform.append((x_wall, y_wall, 1.0 / radius, -phi, local_nu))

    data = np.asarray(circle_to_uniform[::-1], dtype=float)
    return _Transition(x=data[:, 0], y=data[:, 1], mach_star=data[:, 2], eta=data[:, 3], nu=data[:, 4])


def _upper_unrotated(nu_uniform: float, nu_upper: float, number_of_nodes: int, gamma: float) -> _Transition:
    """Construct the unrotated suction-side transition.

    :param float nu_uniform: Prandtl--Meyer angle of the uniform inlet or outlet flow, rad.
    :param float nu_upper: Prandtl--Meyer angle on the constant-Mach suction-side arc, rad.
    :param int number_of_nodes: Fixed number of nodes across the nonzero MOC transition.
    :param float gamma: Frozen specific-heat ratio, -.
    :return: Unrotated suction-side transition normalized by vortex sonic radius.
    :rtype: _Transition
    :raises GeometryError: If a characteristic becomes parallel to its wall segment.
    """

    # The suction-side equations use the opposite characteristic family but otherwise follow the same marching logic as
    # ``_lower_unrotated``. Keeping the two translations separate makes comparison with the legacy equations easier.
    transition_turning = nu_upper - nu_uniform
    if transition_turning < -1.0e-12:
        raise GeometryError("transition turning cannot be negative")
    steps = number_of_nodes - 1
    local_increment = transition_turning / steps
    r_upper = _r_from_nu(nu_upper, gamma)
    gp = 0.5 * (gamma + 1.0)
    gm = 0.5 * (gamma - 1.0)

    phi_previous = -(nu_upper - nu_uniform) + steps * local_increment
    mu_previous = math.asin(float(np.clip(math.sqrt(max(gp * r_upper * r_upper - gm, 0.0)), -1.0, 1.0)))
    x_wall = 0.0
    y_wall = r_upper
    circle_to_uniform = [(x_wall, y_wall, 1.0 / r_upper, 0.0, nu_upper)]

    if transition_turning <= 1.0e-12:
        data = np.asarray(circle_to_uniform, dtype=float)
        return _Transition(x=data[:, 0], y=data[:, 1], mach_star=data[:, 2], eta=data[:, 3], nu=data[:, 4])

    for j in range(steps, 0, -1):
        phi = phi_previous - local_increment
        local_nu = nu_uniform + (j - 1) * local_increment
        radius = _r_from_nu(local_nu, gamma)
        x_characteristic = radius * math.sin(phi)
        y_characteristic = radius * math.cos(phi)
        wall_slope = math.tan(-phi_previous)
        mu = math.asin(float(np.clip(math.sqrt(max(gp * radius * radius - gm, 0.0)), -1.0, 1.0)))
        characteristic_slope = math.tan(0.5 * (-phi + mu - phi_previous + mu_previous))
        denominator = characteristic_slope - wall_slope
        if abs(denominator) < 1.0e-14:
            raise GeometryError("upper characteristic and wall segment are parallel")
        wall_intercept = y_wall - wall_slope * x_wall
        characteristic_intercept = y_characteristic - characteristic_slope * x_characteristic
        x_wall = (wall_intercept - characteristic_intercept) / denominator
        y_wall = (characteristic_slope * wall_intercept - wall_slope * characteristic_intercept) / denominator
        phi_previous = phi
        mu_previous = mu
        circle_to_uniform.append((x_wall, y_wall, 1.0 / radius, -phi, local_nu))

    data = np.asarray(circle_to_uniform[::-1], dtype=float)
    return _Transition(x=data[:, 0], y=data[:, 1], mach_star=data[:, 2], eta=data[:, 3], nu=data[:, 4])


def _select_transition(transition: _Transition, uniform_nu: float) -> _Transition:
    """Select the transition part between ``uniform_nu`` and the circular arc.

    :param _Transition transition: Saved unrotated transition stations.
    :param float uniform_nu: Required uniform-flow Prandtl--Meyer angle, rad.
    :return: Independent transition arrays starting at the requested uniform-flow station.
    :rtype: _Transition
    :raises GeometryError: If numerical station saving omitted the required endpoint.
    """

    index = int(np.argmin(np.abs(transition.nu - uniform_nu)))
    if not math.isclose(float(transition.nu[index]), uniform_nu, rel_tol=0.0, abs_tol=2.0e-7):
        raise GeometryError("saved transition stations did not include the required endpoint")
    return _Transition(
        x=transition.x[index:].copy(),
        y=transition.y[index:].copy(),
        mach_star=transition.mach_star[index:].copy(),
        eta=transition.eta[index:].copy(),
        nu=transition.nu[index:].copy(),
    )


def _rotate_lower(transition: _Transition, alpha: float, *, outlet: bool) -> _RotatedTransition:
    """Rotate a pressure-side transition into the inlet or outlet blade frame.

    :param _Transition transition: Unrotated pressure-side transition.
    :param float alpha: Required transition rotation, rad.
    :param bool outlet: Select outlet rather than inlet coordinate signs.
    :return: Rotated pressure-side transition.
    :rtype: _RotatedTransition
    """

    sine, cosine = math.sin(alpha), math.cos(alpha)
    if outlet:
        x = transition.y * sine - transition.x * cosine
        y = transition.y * cosine + transition.x * sine
    else:
        x = transition.y * sine + transition.x * cosine
        y = transition.y * cosine - transition.x * sine
    return _RotatedTransition(x=x, y=y, mach_star=transition.mach_star.copy(), eta=transition.eta + abs(alpha))


def _rotate_upper(transition: _Transition, alpha: float, *, outlet: bool) -> _RotatedTransition:
    """Rotate a suction-side transition into the inlet or outlet blade frame.

    :param _Transition transition: Unrotated suction-side transition.
    :param float alpha: Required transition rotation, rad.
    :param bool outlet: Select outlet rather than inlet coordinate signs.
    :return: Rotated suction-side transition.
    :rtype: _RotatedTransition
    """

    sine, cosine = math.sin(alpha), math.cos(alpha)
    if outlet:
        x = transition.y * sine - transition.x * cosine
        y = transition.y * cosine + transition.x * sine
    else:
        x = transition.y * sine + transition.x * cosine
        y = transition.y * cosine - transition.x * sine
    return _RotatedTransition(x=x, y=y, mach_star=transition.mach_star.copy(), eta=transition.eta + abs(alpha))


def _inclusive_angles(start: float, end: float, number_of_nodes: int) -> np.ndarray:
    """Create an angle array that contains both requested endpoints.

    :param float start: First polar angle, rad.
    :param float end: Last polar angle, rad.
    :param int number_of_nodes: Fixed number of nodes across the nonzero circular arc.
    :return: Monotonic array including ``start`` and ``end`` exactly.
    :rtype: numpy.ndarray
    """

    if abs(end - start) <= 1.0e-12:
        return np.asarray([start], dtype=float)
    return np.linspace(start, end, number_of_nodes, dtype=float)


def design_ideal_geometry(
    *,
    real_inlet_relative_flow_mach: float,
    ideal_outlet_relative_flow_mach: float,
    lower_surface_relative_flow_mach: float,
    upper_surface_relative_flow_mach: float,
    real_inlet_relative_flow_angle: float,
    ideal_outlet_relative_flow_angle: float,
    inlet_metal_angle: float,
    outlet_metal_angle: float,
    number_of_nodes: int,
    gamma: float,
) -> BladeShape:
    """Port the ``ROTORU`` and ``ROTORR`` characteristic construction.

    Returned coordinates are divided by the vortex sonic radius ``r*``.
    Only the blade-frame surfaces are retained; the legacy alternate
    transition-coordinate arrays are discarded.

    :param float real_inlet_relative_flow_mach: Uniform rotor-relative passage-entry Mach number, -.
    :param float ideal_outlet_relative_flow_mach: Uniform rotor-relative ideal outlet Mach number, -.
    :param float lower_surface_relative_flow_mach: Constant pressure-side arc Mach number, -.
    :param float upper_surface_relative_flow_mach: Constant suction-side arc Mach number, -.
    :param float real_inlet_relative_flow_angle: Positive rotor-relative passage-entry angle measured from the axis, deg.
    :param float ideal_outlet_relative_flow_angle: Negative ideal rotor-relative outlet flow angle, deg.
    :param float inlet_metal_angle: Positive inlet metal angle measured from the machine axis, deg.
    :param float outlet_metal_angle: Negative outlet metal angle measured from the machine axis, deg.
    :param int number_of_nodes: Nodes used by every nonzero MOC transition and constant-Mach circular arc.
    :param float gamma: Frozen specific-heat ratio, -.
    :return: Ideal rotor passage normalized by vortex sonic radius.
    :rtype: BladeShape
    :raises GeometryError: If Mach ordering, turning limits, or the resulting geometry is not physical.
    """

    # Convert all four ordinary Mach numbers to the Prandtl--Meyer variables
    # used by the legacy characteristic equations.
    mach_values = (real_inlet_relative_flow_mach, ideal_outlet_relative_flow_mach, lower_surface_relative_flow_mach, upper_surface_relative_flow_mach)
    if any(value < 1.0 for value in mach_values):
        raise GeometryError("all design Mach numbers must be supersonic (>= 1)")
    if lower_surface_relative_flow_mach > min(real_inlet_relative_flow_mach, ideal_outlet_relative_flow_mach) + 1.0e-12:
        raise GeometryError("lower_surface_relative_flow_mach must not exceed inlet or outlet Mach")
    if upper_surface_relative_flow_mach < max(real_inlet_relative_flow_mach, ideal_outlet_relative_flow_mach) - 1.0e-12:
        raise GeometryError("upper_surface_relative_flow_mach must not be below inlet or outlet Mach")
    if not isinstance(number_of_nodes, int) or isinstance(number_of_nodes, bool) or number_of_nodes < 3:
        raise GeometryError("number_of_nodes must be an integer >= 3")

    nu_in, nu_out, nu_low, nu_up = (float(prandtl_meyer_angle(value, gamma)) for value in mach_values)
    beta_in = math.radians(real_inlet_relative_flow_angle)
    beta_out = math.radians(ideal_outlet_relative_flow_angle)
    inlet_metal_angle_rad = math.radians(inlet_metal_angle)
    outlet_metal_angle_rad = math.radians(outlet_metal_angle)

    # These four rotations place the generic lower and upper transition solutions at the specified inlet and outlet
    # directions. Their signs also determine whether the requested surface-Mach levels can be joined physically.
    alpha_lower_in = (nu_in - nu_low) - beta_in
    alpha_lower_out = -(nu_out - nu_low) - beta_out
    alpha_upper_in = (nu_up - nu_in) - beta_in
    alpha_upper_out = -(nu_up - nu_out) - beta_out
    tolerance = 1.0e-10
    if alpha_lower_in > tolerance or alpha_lower_out < -tolerance:
        raise GeometryError("lower-surface turning requires beta_in >= nu_in-nu_low and beta_out <= -(nu_out-nu_low)")
    if alpha_upper_in > tolerance or alpha_upper_out < -tolerance:
        raise GeometryError("upper-surface turning requires beta_in >= nu_up-nu_in and beta_out <= -(nu_up-nu_out)")

    # Build separate inlet and outlet transitions for both blade surfaces. This repetition mirrors ROTORU/ROTORR and
    # keeps the relationship between each NASA TN D-4421 equation and its Python counterpart visible.
    lower_in = _rotate_lower(
        _lower_unrotated(nu_in, nu_low, number_of_nodes, gamma), alpha_lower_in, outlet=False
    )
    lower_out = _rotate_lower(
        _lower_unrotated(nu_out, nu_low, number_of_nodes, gamma), alpha_lower_out, outlet=True
    )
    upper_in = _rotate_upper(
        _upper_unrotated(nu_in, nu_up, number_of_nodes, gamma), alpha_upper_in, outlet=False
    )
    upper_out = _rotate_upper(
        _upper_unrotated(nu_out, nu_up, number_of_nodes, gamma), alpha_upper_out, outlet=True
    )

    # Project between the two non-aligned transition endpoints along the uniform inlet/outlet flow directions. These are
    # open passage widths; finite leading- and trailing-edge metal is added later by ``SupersonicRotorBlade``.
    inlet_pitch = lower_in.y[0] - (
        upper_in.y[0] + math.tan(inlet_metal_angle_rad) * (lower_in.x[0] - upper_in.x[0])
    )
    outlet_pitch = lower_out.y[0] - (
        upper_out.y[0] + math.tan(outlet_metal_angle_rad) * (lower_out.x[0] - upper_out.x[0])
    )
    if inlet_pitch <= 0.0 or outlet_pitch <= 0.0:
        raise GeometryError("computed blade pitch is not positive")

    # Join the four transition arcs with fixed-node constant-Mach vortex circles.
    r_low = 1.0 / critical_velocity_ratio(lower_surface_relative_flow_mach, gamma)
    r_up = 1.0 / critical_velocity_ratio(upper_surface_relative_flow_mach, gamma)
    lower_angles = _inclusive_angles(alpha_lower_in, alpha_lower_out, number_of_nodes)
    upper_angles = _inclusive_angles(alpha_upper_in, alpha_upper_out, number_of_nodes)
    low_circle_x = r_low * np.sin(lower_angles)
    low_circle_y = r_low * np.cos(lower_angles)
    up_circle_x = r_up * np.sin(upper_angles)
    up_circle_y = r_up * np.cos(upper_angles)
    ms_low = critical_velocity_ratio(lower_surface_relative_flow_mach, gamma)
    ms_up = critical_velocity_ratio(upper_surface_relative_flow_mach, gamma)

    # The pressure surface contains inlet transition, constant-Mach arc, and reversed outlet transition.
    pressure_x = np.concatenate([lower_in.x, low_circle_x[1:-1], lower_out.x[::-1]])
    pressure_y = np.concatenate([lower_in.y, low_circle_y[1:-1], lower_out.y[::-1]])
    pressure_ms = np.concatenate(
        [lower_in.mach_star, np.full(max(0, len(lower_angles) - 2), ms_low), lower_out.mach_star[::-1]]
    )
    pressure_eta = np.concatenate([lower_in.eta, np.abs(lower_angles[1:-1]), lower_out.eta[::-1]])

    if lower_in.x[0] > upper_in.x[0] + 1.0e-12:
        raise GeometryError("upper inlet surface is longer than the lower inlet surface")
    if lower_out.x[0] < upper_out.x[0] - 1.0e-12:
        raise GeometryError("upper outlet surface is longer than the lower outlet surface")

    # Straight inlet and outlet extensions complete the suction surface where
    # the upper transition is shorter. Eleven points reproduce the legacy
    # construction without affecting the analytic endpoints.
    inlet_straight_x = np.linspace(lower_in.x[0], upper_in.x[0], 11)
    inlet_straight_y = upper_in.y[0] + math.tan(inlet_metal_angle_rad) * (inlet_straight_x - upper_in.x[0])
    outlet_straight_x = np.linspace(upper_out.x[0], lower_out.x[0], 11)
    outlet_straight_y = upper_out.y[0] + math.tan(outlet_metal_angle_rad) * (
        outlet_straight_x - upper_out.x[0]
    )

    suction_x = np.concatenate(
        [inlet_straight_x[:-1], upper_in.x, up_circle_x[1:-1], upper_out.x[::-1], outlet_straight_x[1:]]
    )
    suction_y = np.concatenate(
        [inlet_straight_y[:-1], upper_in.y, up_circle_y[1:-1], upper_out.y[::-1], outlet_straight_y[1:]]
    )
    suction_ms = np.concatenate(
        [
            np.full(10, critical_velocity_ratio(real_inlet_relative_flow_mach, gamma)),
            upper_in.mach_star,
            np.full(max(0, len(upper_angles) - 2), ms_up),
            upper_out.mach_star[::-1],
            np.full(10, critical_velocity_ratio(ideal_outlet_relative_flow_mach, gamma)),
        ]
    )
    suction_eta = np.concatenate(
        [
            np.full(10, abs(inlet_metal_angle_rad)),
            upper_in.eta,
            np.abs(upper_angles[1:-1]),
            upper_out.eta[::-1],
            np.full(10, abs(outlet_metal_angle_rad)),
        ]
    )

    chord = math.hypot(float(lower_out.x[0] - lower_in.x[0]), float(lower_out.y[0] - lower_in.y[0]))
    if chord <= 0.0:
        raise GeometryError("computed chord is zero")

    def make_surface(x: np.ndarray, y: np.ndarray, mach_star: np.ndarray, eta: np.ndarray) -> SurfaceCoordinates:
        """Convert assembled arrays into one public surface container.

        :param numpy.ndarray x: Axial coordinates divided by vortex sonic radius.
        :param numpy.ndarray y: Tangential coordinates divided by vortex sonic radius.
        :param numpy.ndarray mach_star: Critical velocity ratio at each station, -.
        :param numpy.ndarray eta: Surface-tangent angle magnitude at each station, rad.
        :return: Surface using ordinary Mach number and independent numeric arrays.
        :rtype: SurfaceCoordinates
        """

        # Remove only adjacent duplicates at analytic arc junctions.
        keep = np.ones(len(x), dtype=bool)
        if len(x) > 1:
            keep[1:] = np.hypot(np.diff(x), np.diff(y)) > 1.0e-12
        return SurfaceCoordinates(
            x=np.asarray(x[keep], dtype=float),
            y=np.asarray(y[keep], dtype=float),
            relative_flow_mach=np.asarray(
                mach_from_critical_velocity_ratio(mach_star[keep], gamma), dtype=float
            ),
            metal_angle=np.asarray(np.degrees(eta[keep]), dtype=float),
        )

    turning_spans = (
        nu_in - nu_low,
        nu_out - nu_low,
        nu_up - nu_in,
        nu_up - nu_out,
        abs(alpha_lower_out - alpha_lower_in),
        abs(alpha_upper_out - alpha_upper_in),
    )
    max_flow_turning_increment = math.degrees(max(turning_spans) / (number_of_nodes - 1))

    return BladeShape(
        pressure_surface=make_surface(pressure_x, pressure_y, pressure_ms, pressure_eta),
        suction_surface=make_surface(suction_x, suction_y, suction_ms, suction_eta),
        chord=chord,
        inlet_pitch=float(inlet_pitch),
        outlet_pitch=float(outlet_pitch),
        max_flow_turning_increment=max_flow_turning_increment,
        coordinate_scale="vortex sonic radius r*",
    )
