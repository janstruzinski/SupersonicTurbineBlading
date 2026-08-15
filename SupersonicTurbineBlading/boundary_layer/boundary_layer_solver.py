"""Compressible integral BL correction from NASA TM X-2434 and NASA TM X-2343.

The NACA Report 1294 Cohen--Reshotko laminar equations and the Sasman--Cresci
turbulent equations from AIAA Journal 4(1), 1966, doi:10.2514/3.3378, remain
nondimensional, as they were in the FORTRAN. Dimensional temperature and
pressure are nevertheless reconstructed at each surface station so that
viscosity, heat capacity and conductivity can come from the selected
:class:`~SupersonicTurbineBlading.fluid.Fluid` instead of the air-only
polynomial.

The wall temperature is set equal to total temperature, matching both the
NASA TM X-2434 rotor and NASA TM X-2343 stator drivers.
"""

from __future__ import annotations

import math
import warnings
from typing import Literal

import numpy as np

from ..common_results import BoundaryLayerResult, SurfaceCoordinates
from ..fluid import Fluid

BoundaryLayerMode = Literal["fully_turbulent", "laminar_then_turbulent"]


class BoundaryLayerError(RuntimeError):
    """Raised when a NACA Report 1294 or Sasman--Cresci correlation fails."""


def project_boundary_layer_result(
    *, result: BoundaryLayerResult, target_surface: SurfaceCoordinates, chord: float
) -> BoundaryLayerResult:
    """Interpolate a dense BL march onto the original geometry stations.

    The MOC routines choose the stations at which the ideal contour is
    defined.  Boundary-layer integration benefits from a finer arc-length
    grid, but that numerical grid must not silently replace the MOC geometry.
    This helper therefore transfers the predicted integral quantities from a
    dense marching surface back to the stations in ``target_surface``.

    Continuous quantities are interpolated in physical arc length divided by
    chord. Transition and separation are discrete events, so their first
    target station is the first MOC station at or downstream of the event.

    :param BoundaryLayerResult result: Boundary-layer result on the temporary dense marching grid.
    :param SurfaceCoordinates target_surface: Original MOC surface onto which the result is projected.
    :param float chord: Reference chord in the same length scale as ``target_surface``, - or m.
    :return: Boundary-layer result located exactly at the original MOC stations.
    :rtype: BoundaryLayerResult
    :raises ValueError: If chord is not positive and finite.
    :raises BoundaryLayerError: If either station coordinate is invalid or the two surfaces have different arc lengths.
    """

    if not math.isfinite(chord) or chord <= 0.0:
        raise ValueError("chord must be positive and finite")

    target_ds = np.hypot(
        np.diff(np.asarray(target_surface.x, dtype=float)), np.diff(np.asarray(target_surface.y, dtype=float))
    )
    if np.any(target_ds <= 0.0):
        raise BoundaryLayerError("target surface coordinates must advance without duplicates")
    target_s = np.concatenate(([0.0], np.cumsum(target_ds))) / chord
    source_s = np.asarray(result.s_over_chord, dtype=float)
    if len(source_s) < 2 or np.any(np.diff(source_s) <= 0.0):
        raise BoundaryLayerError("boundary-layer marching stations must strictly increase")
    length_tolerance = 1.0e-8 * max(1.0, abs(float(source_s[-1])))
    if abs(float(target_s[-1] - source_s[-1])) > length_tolerance:
        raise BoundaryLayerError("dense boundary-layer and target MOC surfaces have different arc lengths")

    def interpolate(values: np.ndarray) -> np.ndarray:
        """Interpolate one continuous dense-grid array onto the target arc coordinate.

        :param numpy.ndarray values: Dense-grid values to project.
        :return: Values at the target geometry stations.
        :rtype: numpy.ndarray
        """

        return np.asarray(np.interp(target_s, source_s, np.asarray(values, dtype=float)), dtype=float)

    transition_index: int | None = None
    if result.transition_index is not None:
        transition_s = float(source_s[result.transition_index])
        transition_index = min(int(np.searchsorted(target_s, transition_s, side="left")), len(target_s) - 1)

    separation_index: int | None = None
    if result.separation_index is not None:
        separation_s = float(source_s[result.separation_index])
        separation_index = min(int(np.searchsorted(target_s, separation_s, side="left")), len(target_s) - 1)

    regime = np.full(len(target_s), "laminar", dtype="<U12")
    if transition_index is not None:
        regime[transition_index:] = "turbulent"
    if separation_index is not None and separation_index + 1 < len(target_s):
        regime[separation_index + 1 :] = "separated"

    return BoundaryLayerResult(
        s_over_chord=np.asarray(target_s, dtype=float),
        # Retain the exact MOC Mach values rather than an interpolation of
        # the temporary dense surface representation.
        displacement_thickness_over_chord=interpolate(result.displacement_thickness_over_chord),
        momentum_thickness_over_chord=interpolate(result.momentum_thickness_over_chord),
        form_factor=interpolate(result.form_factor),
        regime=regime,
        transition_index=transition_index,
        separation_index=separation_index,
        freestream_absolute_flow_mach=(
            np.asarray(target_surface.absolute_flow_mach, dtype=float).copy()
            if target_surface.absolute_flow_mach is not None
            else None
        ),
        freestream_relative_flow_mach=(
            np.asarray(target_surface.relative_flow_mach, dtype=float).copy()
            if target_surface.relative_flow_mach is not None
            else None
        ),
    )


def _polyfit(coefficients: np.ndarray, x: float, y: float = 0.0) -> float:
    """Evaluate a FORTRAN ``CURVFT`` polynomial with ``x`` changing fastest.

    :param numpy.ndarray coefficients: Six one-dimensional or sixteen two-dimensional legacy coefficients.
    :param float x: First polynomial coordinate.
    :param float y: Optional second polynomial coordinate.
    :return: Evaluated curve-fit value.
    :rtype: float
    :raises ValueError: If the coefficient table has neither supported size.
    """

    if len(coefficients) == 6:
        return float(sum(value * x**power for power, value in enumerate(coefficients)))
    if len(coefficients) == 16:
        total = 0.0
        for y_power in range(4):
            for x_power in range(4):
                total += coefficients[y_power * 4 + x_power] * y**y_power * x**x_power
        return float(total)
    raise ValueError("unsupported legacy polynomial size")


def _transition_incompressible_form_factor(
    laminar_incompressible_form_factor: float, transition_reynolds_number: float
) -> float:
    """Convert the laminar form factor to the turbulent starting value.

    ``RTRAN`` is the transition momentum-thickness Reynolds number from the
    laminar calculation.  The legacy code substitutes 1000 when transition
    is triggered by imminent laminar separation and no positive ``RTRAN``
    has been calculated.

    :param float laminar_incompressible_form_factor: Transformed laminar form factor at transition, -.
    :param float transition_reynolds_number: Momentum-thickness Reynolds number at transition, -.
    :return: Turbulent transformed form factor used to initialize the Sasman--Cresci march, -.
    :rtype: float
    """

    rtran = float(transition_reynolds_number) if transition_reynolds_number > 0.0 else 1000.0
    log_rtran = math.log(rtran)
    return float(laminar_incompressible_form_factor) - 0.59389 - 0.06591 * log_rtran + 0.001272 * log_rtran**2


def _interp_extrap(x: np.ndarray, y: np.ndarray, argument: float) -> float:
    """Linearly interpolate or extrapolate a monotonic tabulated quantity.

    Unlike ``numpy.interp``, this helper deliberately continues the first or last line segment outside the table. The
    legacy BL correlations require this behavior after the generated ``CORLN`` table reaches its application limit.

    :param numpy.ndarray x: Monotonic independent coordinate.
    :param numpy.ndarray y: Tabulated dependent values.
    :param float argument: Coordinate at which the table is evaluated.
    :return: Interpolated or linearly extrapolated value.
    :rtype: float
    """

    if len(x) < 2:
        return float(y[0])
    index = int(np.searchsorted(x, argument, side="left"))
    index = min(max(index, 1), len(x) - 1)
    fraction = (argument - x[index - 1]) / (x[index] - x[index - 1])
    return float(y[index - 1] + fraction * (y[index] - y[index - 1]))


def _simpson_panel(function, lower: float, upper: float) -> float:
    """Integrate one interval with the three-point Simpson rule.

    :param function function: Scalar integrand.
    :param float lower: Lower interval boundary.
    :param float upper: Upper interval boundary.
    :return: Approximate integral over the panel.
    :rtype: float
    """

    if upper == lower:
        return 0.0
    middle = 0.5 * (lower + upper)
    return (upper - lower) / 6.0 * (function(lower) + 4.0 * function(middle) + function(upper))


def _surface_state(
    surface: SurfaceCoordinates,
    chord: float,
    inlet_edge_flow_mach: float,
    chord_reynolds_number: float,
    gamma: float,
    fluid: Fluid,
    inlet_total_temperature: float,
    inlet_total_pressure: float,
) -> dict[str, np.ndarray | float]:
    """Prepare the thermodynamic arrays used by both integral solvers.

    ``gamma`` is frozen at the calling geometry's reference static state:
    rotor inlet for :class:`SupersonicRotorBlade`, choked throat for
    :class:`SupersonicStatorNozzle`.  CoolProp still supplies
    temperature-dependent transport properties at every station.

    :param SurfaceCoordinates surface: Ideal surface, march start to exit.
    :param float chord: Nondimensional reference chord.
    :param float inlet_edge_flow_mach: Mach number at the first surface station.
    :param float chord_reynolds_number: Inlet-edge chord Reynolds number.
    :param float gamma: Frozen ratio of specific heats.
    :param Fluid fluid: Ideal-gas mixture property object.
    :param float inlet_total_temperature: Relative total temperature, K.
    :param float inlet_total_pressure: Relative total pressure, Pa.
    :return: Nondimensional geometry, thermodynamic, transport, and derivative arrays shared by both BL solvers.
    :rtype: dict
    :raises BoundaryLayerError: If the surface does not progress continuously from inlet to exit.
    """

    # The geometry coordinates are divided by r*.  Dividing both x and y by
    # the computed chord changes the boundary-layer marching coordinate to
    # s/c, which is the length scale used by NASA TM X-2434 and NASA TM X-2343.
    x = np.asarray(surface.x / chord, dtype=float)
    y = np.asarray(surface.y / chord, dtype=float)
    ds = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate(([0.0], np.cumsum(ds)))
    arcl = float(s[-1])
    if arcl <= 0.0 or np.any(ds <= 0.0):
        raise BoundaryLayerError("surface coordinates must advance without duplicates")
    sol = s / arcl
    edge_flow_mach = np.asarray(surface.flow_mach_values(), dtype=float)

    # The FORTRAN boundary-layer equations use total temperature and total
    # sonic speed as their reference state.  Static edge temperature and
    # pressure follow the constant-gamma isentropic relations.
    gm = gamma - 1.0
    gp = gamma + 1.0
    temperature = 1.0 / (1.0 + 0.5 * gm * edge_flow_mach**2)
    edge_temperature = inlet_total_temperature * temperature
    edge_pressure = inlet_total_pressure * temperature ** (gamma / gm)

    # Velocity remains normalized by the total-state sonic speed.  Therefore
    # U/a_t = M*sqrt(T/Tt), exactly as in the original nondimensional code.
    velocity = edge_flow_mach * np.sqrt(temperature)
    inlet_temperature_factor = 1.0 + 0.5 * gm * inlet_edge_flow_mach**2
    inlet_velocity = inlet_edge_flow_mach / math.sqrt(inlet_temperature_factor)
    inlet_static_temperature = inlet_total_temperature / inlet_temperature_factor
    inlet_static_pressure = inlet_total_pressure / inlet_temperature_factor ** (gamma / gm)

    total_state = fluid.properties(inlet_total_temperature, inlet_total_pressure)
    inlet_edge_state = fluid.properties(inlet_static_temperature, inlet_static_pressure)
    prandtl_number = inlet_edge_state.prandtl_number

    # chord_reynolds_number is defined here in the conventional inlet-edge
    # form Re_c = U_e,in*c/nu_e,in.  The physical chord is not yet known, but
    # it cancels when any physical kinematic viscosity is converted to the
    # nondimensional quantity nu/(a_t*c):
    #
    #   nu/(a_t*c) = (nu/nu_e,in)*(U_e,in/a_t)/Re_c.
    #
    viscosity_scale = inlet_velocity / chord_reynolds_number / inlet_edge_state.kinematic_viscosity
    total_kinematic_viscosity = total_state.kinematic_viscosity * viscosity_scale

    # NASA TM X-2434 and NASA TM X-2343 use Twall=Tt. Wall transport properties are evaluated at
    # total temperature and local edge pressure; density is still obtained
    # by the mixture object's ideal-gas equation of state.
    wall_states = tuple(fluid.properties(inlet_total_temperature, float(pressure)) for pressure in edge_pressure)
    local_kinematic_viscosity = np.asarray(
        [state.kinematic_viscosity * viscosity_scale for state in wall_states], dtype=float
    )

    # SW is the transformed wall-temperature parameter appearing in the
    # legacy correlations. It is zero for the NASA TM X-2434 and NASA TM X-2343 choice Twall=Tt.
    wall_temperature_ratio = np.zeros_like(edge_flow_mach)
    recovery_temperature = temperature * (1.0 + prandtl_number ** (1.0 / 3.0) * (1.0 / temperature - 1.0))

    # The empirical turbulent transformation evaluates viscosity at a
    # representative boundary-layer temperature Tbar, not solely at the edge
    # or wall.  This is where the old air viscosity curve is replaced by
    # component-wise CoolProp calls.
    mean_temperature = 0.5 * (1.0 + temperature) + 0.22 * prandtl_number ** (1.0 / 3.0) * (1.0 - temperature)
    mean_states = tuple(
        fluid.properties(float(inlet_total_temperature * temperature_ratio), float(pressure))
        for temperature_ratio, pressure in zip(mean_temperature, edge_pressure)
    )
    mean_dynamic_viscosity = np.asarray([state.dynamic_viscosity for state in mean_states], dtype=float)

    # ``AA``, ``BB``, and ``FF`` are the dimensionless correlation symbols in
    # NACA Report 1294 and the Sasman--Cresci AIAA Journal 4(1), 1966 paper.
    # factors. Their short keys are retained in the private state dictionary
    # so the subsequent laminar and turbulent equations can be checked term by
    # term against the FORTRAN; descriptive physical quantities use full names.
    bb = edge_flow_mach / total_kinematic_viscosity * temperature ** (gp / (2.0 * gm))
    aa = bb * temperature / mean_temperature * (mean_dynamic_viscosity / total_state.dynamic_viscosity) ** 0.268
    ff = 1.0 + 0.1599 * edge_flow_mach**2 + 0.0114 * edge_flow_mach**4
    duds = np.gradient(velocity, s, edge_order=1)
    dmds = np.gradient(edge_flow_mach, s, edge_order=1)
    dmdl = arcl * dmds

    return {
        "x": x,
        "y": y,
        "s": s,
        "sol": sol,
        "arcl": arcl,
        "edge_flow_mach": edge_flow_mach,
        "temperature": temperature,
        "velocity": velocity,
        "nu": local_kinematic_viscosity,
        "nu_total": total_kinematic_viscosity,
        "sw": wall_temperature_ratio,
        "tbar": mean_temperature,
        "t_recovery": recovery_temperature,
        "aa": aa,
        "bb": bb,
        "ff": ff,
        "duds": duds,
        "dmds": dmds,
        "dmdl": dmdl,
        "pr": float(prandtl_number),
        "gamma": float(gamma),
    }


def _laminar_solution(state: dict[str, np.ndarray | float], correlation_limit: float) -> dict[str, object]:
    """Evaluate the Cohen--Reshotko laminar correlation and transition test.

    :param dict state: Shared surface-state arrays produced by :func:`_surface_state`.
    :param float correlation_limit: Largest generated ``CORLN`` value before table extrapolation begins.
    :return: Laminar thicknesses, form factor, instability/transition indices, and transition Reynolds number.
    :rtype: dict
    """

    s = state["s"]
    sol = state["sol"]
    edge_flow_mach = state["edge_flow_mach"]
    arcl = float(state["arcl"])
    gamma = float(state["gamma"])
    pr = float(state["pr"])
    gm = gamma - 1.0
    nu_total = float(state["nu_total"])
    dmdl = state["dmdl"]
    sw = state["sw"]
    local_nu = state["nu"]
    velocity = state["velocity"]
    duds = state["duds"]
    ff = state["ff"]

    # These four coefficient arrays are copied from NACA Report 1294. Naming them by physical use makes the otherwise
    # opaque polynomial evaluations below easier to follow.
    c_shear = np.array(
        [
            0.224488,
            -1.91539,
            -9.894,
            -68.13488,
            -0.001512,
            -1.4768,
            -10.52925,
            -152.2781,
            -0.002406,
            -0.015629,
            -1.45743,
            -126.23395,
            0.000752,
            0.005385,
            0.917838,
            -39.40644,
        ]
    )
    c_dth = np.array(
        [
            8.02829,
            -4.30978,
            88.8244,
            36.4336,
            2.71101,
            -7.42259,
            242.293,
            -16.293,
            -0.16394,
            -7.61942,
            286.9795,
            64.11186,
            -0.16758,
            -3.70289,
            130.8107,
            111.3276,
        ]
    )
    c_rcrit = np.array([5.47073, 43.6053, 227.198, -2067.04, -27172.7, 13691.2])
    c_diff = np.array([903.785, 26365.0, 3.85695e5, 1.11044e6, -4.53853e7, -7.70276e7])

    # Integrate the two Cohen--Reshotko correlation variables on the same
    # 0.002*surface-length mesh used by the FORTRAN.
    step = 0.002 * arcl
    table_s = [0.0]
    corln_table = [0.0]
    corml_table = [0.0]
    previous_temperature_factor = 1.0 + 0.5 * gm * float(edge_flow_mach[0]) ** 2
    exponent = (3.0 * gamma - 1.0) / (2.0 * gm)
    current_s = 0.0
    while current_s < arcl - 1.0e-14:
        next_s = min(current_s + step, arcl)
        wall_ratio = _interp_extrap(s, sw, current_s)
        edge_flow_mach_here = _interp_extrap(s, edge_flow_mach, current_s)
        edge_flow_mach_next = _interp_extrap(s, edge_flow_mach, next_s)
        gradient_next = _interp_extrap(s, dmdl, next_s)
        corln = corln_table[-1]
        a1 = 0.43631 - 0.00367 * wall_ratio + 0.00481 * wall_ratio**2 + 0.00651 * wall_ratio**3
        a2 = 5.43220 + 2.25400 * wall_ratio - 0.06672 * wall_ratio**2 - 0.20637 * wall_ratio**3
        a3 = 4.51903 - 10.49775 * wall_ratio - 12.71732 * wall_ratio**2 - 2.95270 * wall_ratio**3
        a4 = 19.01831 + 62.76597 * wall_ratio + 115.00986 * wall_ratio**2 + 62.53113 * wall_ratio**3
        coefficient_a = a1 - a3 * corln**2 - 2.0 * a4 * corln**3
        coefficient_b = a2 + 2.0 * a3 * corln + 3.0 * a4 * corln**2
        if corln < -0.1:
            coefficient_a = 0.3953
            coefficient_b = 4.739

        def integrand(normalized_s: float) -> float:
            """Evaluate the local Cohen--Reshotko momentum integral source.

            :param float normalized_s: Surface distance divided by chord.
            :return: Local integral source term.
            :rtype: float
            """

            local_edge_flow_mach = _interp_extrap(sol, edge_flow_mach, normalized_s)
            return local_edge_flow_mach ** (coefficient_b - 1.0) / (
                1.0 + 0.5 * gm * local_edge_flow_mach**2
            ) ** exponent

        integral = _simpson_panel(integrand, current_s / arcl, next_s / arcl)
        next_temperature_factor = 1.0 + 0.5 * gm * edge_flow_mach_next**2
        next_scale = edge_flow_mach_next ** (-coefficient_b) * next_temperature_factor**exponent
        source = -coefficient_a * next_scale * integral
        convection = 0.0
        if len(table_s) > 1:
            previous_scale = edge_flow_mach_here ** (-coefficient_b) * previous_temperature_factor**exponent
            convection = next_scale / previous_scale * corml_table[-1]
        corml_next = source + convection
        corln_next = corml_next * gradient_next
        table_s.append(next_s)
        corml_table.append(corml_next)
        corln_table.append(corln_next)
        previous_temperature_factor = next_temperature_factor
        current_s = next_s
        if corln_next > correlation_limit:
            break

    # NASA TM X-2434 and NASA TM X-2343 stop generating the correlation table at an application-specific CORLN value.
    # Continue to requested
    # surface stations by explicit linear extrapolation and warn the designer instead of silently clipping the solution.
    table_s_array = np.asarray(table_s)
    if table_s_array[-1] < float(s[-1]) - 1.0e-12:
        warnings.warn(
            "laminar CORLN exceeded its "
            f"{correlation_limit:.2f} application limit; extrapolating "
            "the Cohen--Reshotko correlation table to the remaining "
            "surface stations",
            RuntimeWarning,
            stacklevel=3,
        )
    corln = np.array([_interp_extrap(table_s_array, np.asarray(corln_table), value) for value in s])
    corml = np.array([_interp_extrap(table_s_array, np.asarray(corml_table), value) for value in s])
    temperature_factor = 1.0 + 0.5 * gm * edge_flow_mach**2
    theta_argument = -corml * nu_total * arcl * temperature_factor ** ((3.0 - gamma) / (2.0 * gm))
    theta = np.sqrt(np.maximum(theta_argument, 0.0))
    form = (-1.1138 * corln + 2.38411) * (1.0 + (2.79 - 1.78 * math.sqrt(pr)) * (temperature_factor - 1.0)) + (
        4.65 * pr ** (1.0 / 3.0) - 3.65 * math.sqrt(pr)
    ) * math.sqrt(pr) * (temperature_factor - 1.0)
    displacement = theta * form
    dth = np.array([_polyfit(c_dth, value, 0.0) for value in corln])
    dth[corln < -0.1] = -22.222 * corln[corln < -0.1] + 7.1112
    delta = theta * (
        dth
        + (temperature_factor - 1.0) * ((form - math.sqrt(pr) * (temperature_factor - 1.0)) / temperature_factor + 1.0)
    )
    re_theta = np.divide(velocity * theta, local_nu, out=np.zeros_like(theta), where=local_nu > 0.0)
    shape_l = delta**2 / local_nu * duds
    shape_k = np.zeros_like(theta)
    nonzero_edge_flow_mach = edge_flow_mach > 1.0e-10
    shape_k[nonzero_edge_flow_mach] = (
        nu_total
        * re_theta[nonzero_edge_flow_mach] ** 2
        / ff[nonzero_edge_flow_mach]
        / arcl
        * dmdl[nonzero_edge_flow_mach]
        * temperature_factor[nonzero_edge_flow_mach] ** (1.0 / gm)
        / edge_flow_mach[nonzero_edge_flow_mach] ** 2
    )
    re_theta_i = re_theta / ff / np.sqrt(temperature_factor)

    # Transition is a two-stage test: first find the neutral-instability
    # Reynolds number, then integrate the amplification correlation until the
    # transition value is reached. Laminar separation triggers it immediately.
    instability_index = None
    transition_index = None
    separation_index = None
    transition_reynolds_number = 0.0
    r_instability = 0.0
    for index in range(len(s)):
        shear = _polyfit(c_shear, float(corln[index]), 0.0)
        if corln[index] < -0.1:
            shear = -1.2222 * corln[index] + 0.26
        if index > 0 and shear <= 0.0:
            separation_index = index
            transition_index = index
            break
        rcrit_log = 8.3163 if shape_k[index] > 0.07 else _polyfit(c_rcrit, float(shape_k[index]))
        rcrit = math.exp(min(rcrit_log, 50.0))
        if instability_index is None and re_theta_i[index] >= rcrit:
            instability_index = index
            r_instability = float(re_theta_i[index])
            continue
        if instability_index is not None and index > instability_index:
            x1 = float(sol[instability_index])
            x2 = float(sol[index])
            if x2 <= x1:
                continue
            samples = np.linspace(x1, x2, max(3, 2 * (index - instability_index) + 1))
            values = np.array([_interp_extrap(sol[: index + 1], shape_k[: index + 1], item) for item in samples])
            integral = np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(samples))
            kbar = float(integral / (x2 - x1))
            difference = 44000.0 * kbar + 700.0 if kbar > 0.03 else _polyfit(c_diff, kbar)
            r_transition = r_instability + difference
            if re_theta_i[index] >= r_transition:
                transition_index = index
                transition_reynolds_number = float(r_transition)
                break

    return {
        "theta": theta,
        "displacement": displacement,
        "form": form,
        "instability_index": instability_index,
        "transition_index": transition_index,
        "separation_index": separation_index,
        "transition_reynolds_number": transition_reynolds_number,
    }


def _turbulent_solution(
    state: dict[str, np.ndarray | float], start_index: int, initial_theta: float, initial_incompressible_form: float
) -> dict[str, np.ndarray | int | None]:
    """March the Sasman--Cresci turbulent equations from one surface station.

    :param dict state: Shared surface-state arrays produced by :func:`_surface_state`.
    :param int start_index: First turbulent surface station.
    :param float initial_theta: Compressible momentum thickness divided by chord at ``start_index``, -.
    :param float initial_incompressible_form: Transformed incompressible form factor at ``start_index``, -.
    :return: Turbulent momentum/displacement thickness, form factor, and optional separation index.
    :rtype: dict
    :raises BoundaryLayerError: If the turbulent integral equations diverge.
    """

    s = state["s"]
    edge_flow_mach = state["edge_flow_mach"]
    sw = state["sw"]
    aa = state["aa"]
    bb = state["bb"]
    dmds = state["dmds"]
    tbar = state["tbar"]
    temperature = state["temperature"]
    velocity = state["velocity"]
    local_nu = state["nu"]
    nu_total = float(state["nu_total"])
    gamma = float(state["gamma"])
    pr = float(state["pr"])
    arcl = float(state["arcl"])
    gm = gamma - 1.0
    gp = gamma + 1.0

    theta_transformed = initial_theta * temperature[start_index] ** (gp / (2.0 * gm))
    f_initial = max(edge_flow_mach[start_index] * theta_transformed / nu_total, 1.0e-15) ** 1.268
    state_vector = np.array([f_initial, max(initial_incompressible_form, 1.021)], dtype=float)
    table_s = [float(s[start_index])]
    table_f = [float(state_vector[0])]
    table_h = [float(state_vector[1])]
    step_nominal = 0.002 * arcl
    separation_index = None

    def derivative(location: float, values: np.ndarray) -> np.ndarray:
        """Evaluate the two coupled Sasman--Cresci ordinary differential equations.

        :param float location: Surface distance divided by chord.
        :param numpy.ndarray values: Current momentum thickness and transformed form factor.
        :return: Derivatives of both turbulent state variables.
        :rtype: numpy.ndarray
        """

        local_edge_flow_mach = _interp_extrap(s, edge_flow_mach, location)
        local_sw = _interp_extrap(s, sw, location)
        local_aa = _interp_extrap(s, aa, location)
        local_bb = _interp_extrap(s, bb, location)
        local_gradient = _interp_extrap(s, dmds, location)
        local_tbar = _interp_extrap(s, tbar, location)
        f_value = max(float(values[0]), 1.0e-15)
        h_value = max(float(values[1]), 1.001)
        temp1 = 1.0 + (1.0 + local_sw) * h_value
        skin_source = 0.123 * math.exp(-1.561 * h_value) * local_aa
        df = 1.268 * (-f_value / local_edge_flow_mach * local_gradient * temp1 + skin_source)
        temp3 = h_value * (h_value + 1.0) ** 2 * (h_value - 1.0)
        temp4 = 1.0 + local_sw * (h_value**2 + 4.0 * h_value - 1.0) / ((h_value + 1.0) * (h_value + 3.0))
        temp5 = (h_value**2 - 1.0) * h_value / f_value * skin_source
        temp6 = (
            (h_value**2 - 1.0)
            / f_value**0.7886
            * (0.011 * (h_value + 1.0) * (h_value - 1.0) ** 2 / h_value**2 / local_tbar)
            * local_bb
        )
        dh = -local_gradient * 0.5 / local_edge_flow_mach * temp3 * temp4 + temp5 - temp6
        return np.array([df, dh], dtype=float)

    # Use classical fourth-order Runge--Kutta with the fixed 0.002*surface-length step in NASA TM X-2434 and
    # NASA TM X-2343. Modern arrays store the
    # intermediate march, but the equations and step definition remain the legacy ones.
    location = float(s[start_index])
    end = float(s[-1])
    while location < end - 1.0e-14:
        step = min(step_nominal, end - location)
        k1 = derivative(location, state_vector)
        k2 = derivative(location + 0.5 * step, state_vector + 0.5 * step * k1)
        k3 = derivative(location + 0.5 * step, state_vector + 0.5 * step * k2)
        k4 = derivative(location + step, state_vector + step * k3)
        state_vector = state_vector + step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        location += step
        table_s.append(location)
        table_f.append(float(state_vector[0]))
        table_h.append(float(state_vector[1]))
        if not np.all(np.isfinite(state_vector)) or state_vector[0] <= 0.0:
            raise BoundaryLayerError("turbulent integral equations diverged")
        if state_vector[1] > 2.8:
            break

    table_s_array = np.asarray(table_s)
    table_f_array = np.asarray(table_f)
    table_h_array = np.asarray(table_h)
    theta = np.full(len(s), np.nan)
    displacement = np.full(len(s), np.nan)
    form = np.full(len(s), np.nan)
    for index in range(start_index, len(s)):
        if s[index] > table_s_array[-1] + 1.0e-12:
            separation_index = index - 1
            break
        f_value = _interp_extrap(table_s_array, table_f_array, float(s[index]))
        h_incompressible = _interp_extrap(table_s_array, table_h_array, float(s[index]))
        theta_transformed = nu_total * max(f_value, 1.0e-15) ** 0.7886 / edge_flow_mach[index]
        theta[index] = theta_transformed * temperature[index] ** (-gp / (2.0 * gm))
        temperature_factor = 1.0 / temperature[index]
        form[index] = h_incompressible * temperature_factor + pr ** (1.0 / 3.0) * (temperature_factor - 1.0)
        displacement[index] = theta[index] * form[index]

    # Preserve the supplied/transition integral values exactly at the initial
    # station.  The legacy exponents 1.268 and 0.7886 are rounded curve-fit
    # values and are not perfect mathematical reciprocals; allowing a round
    # trip through them would otherwise perturb the user's inlet thickness.
    initial_temperature_factor = 1.0 / temperature[start_index]
    theta[start_index] = initial_theta
    form[start_index] = initial_incompressible_form * initial_temperature_factor + pr ** (1.0 / 3.0) * (
        initial_temperature_factor - 1.0
    )
    displacement[start_index] = theta[start_index] * form[start_index]

    return {"theta": theta, "displacement": displacement, "form": form, "separation_index": separation_index}


def solve_boundary_layer(
    *,
    surface: SurfaceCoordinates,
    chord: float,
    inlet_edge_flow_mach: float,
    chord_reynolds_number: float,
    gamma: float,
    fluid: Fluid,
    inlet_total_temperature: float,
    inlet_total_pressure: float,
    mode: BoundaryLayerMode,
    initial_turbulent_displacement_thickness_over_chord: float | None,
    initial_turbulent_momentum_thickness_over_chord: float | None,
    laminar_correlation_limit: float,
) -> BoundaryLayerResult:
    """Solve from a blade/nozzle reference station along one ideal surface.

    :param SurfaceCoordinates surface: Ideal surface coordinates and Mach.
    :param float chord: Nondimensional blade chord.
    :param float inlet_edge_flow_mach: Mach number at the first surface station.
    :param float chord_reynolds_number: Inlet-edge chord Reynolds number.
    :param float gamma: Frozen ratio of specific heats.
    :param Fluid fluid: CoolProp-backed ideal-gas mixture.
    :param float inlet_total_temperature: Total temperature, K.
    :param float inlet_total_pressure: Total pressure, Pa.
    :param BoundaryLayerMode mode: Fully turbulent or natural transition.
    :param float | None initial_turbulent_displacement_thickness_over_chord:
        Inlet compressible displacement thickness divided by blade chord.
        Required only when ``mode="fully_turbulent"``.
    :param float | None initial_turbulent_momentum_thickness_over_chord:
        Inlet compressible momentum thickness divided by blade chord.
        Required only when ``mode="fully_turbulent"``.
    :param float laminar_correlation_limit: Maximum generated laminar
        ``CORLN`` value before subsequent surface stations use table
        extrapolation. NASA TM X-2434 uses 0.50 for the rotor, while NASA TM X-2343 uses 0.16 for the stator.
    :return: Boundary-layer integral quantities normalized by chord.
    :rtype: BoundaryLayerResult
    :raises ValueError: If mode, correlation limit, or fully turbulent inlet thicknesses are invalid.
    :raises BoundaryLayerError: If transformed inlet values are unphysical or either integral solution fails.
    """

    if mode not in ("fully_turbulent", "laminar_then_turbulent"):
        raise ValueError("boundary_layer_mode must be 'fully_turbulent' or 'laminar_then_turbulent'")
    if not math.isfinite(laminar_correlation_limit) or laminar_correlation_limit <= 0.0:
        raise ValueError("laminar_correlation_limit must be positive and finite")
    # Both solution branches use the same thermodynamic reconstruction and nondimensional surface coordinate.
    state = _surface_state(
        surface, chord, inlet_edge_flow_mach, chord_reynolds_number, gamma, fluid, inlet_total_temperature, inlet_total_pressure
    )
    s = np.asarray(state["s"])
    edge_flow_mach = np.asarray(state["edge_flow_mach"])
    count = len(s)
    regime = np.full(count, "laminar", dtype="<U12")
    transition_index = None
    separation_index = None

    if mode == "fully_turbulent":
        # A fully turbulent solution begins at station zero and therefore needs
        # both independent integral thicknesses from the user; one thickness
        # alone cannot define the initial form factor.
        transition_index = 0
        regime[:] = "turbulent"

        if (
            initial_turbulent_displacement_thickness_over_chord is None
            or initial_turbulent_momentum_thickness_over_chord is None
        ):
            raise ValueError("fully_turbulent mode requires both initial turbulent thicknesses")

        initial_displacement = float(initial_turbulent_displacement_thickness_over_chord)
        initial_theta = float(initial_turbulent_momentum_thickness_over_chord)

        # The two user inputs are the ordinary compressible integral
        # thicknesses delta* and theta.  Sasman--Cresci's differential
        # equations use the transformed incompressible form factor instead,
        # so reproduce the NASA TM X-2434 and NASA TM X-2343 transformation at the inlet:
        #
        #   H_i = [delta*/theta - Pr^(1/3)*(Tt/Te - 1)] / (Tt/Te)
        #
        # SW=0 because the wall temperature equals relative total
        # temperature in the NASA TM X-2434 and NASA TM X-2343 drivers.
        temperature_factor = 1.0 + 0.5 * (gamma - 1.0) * edge_flow_mach[0] ** 2
        compressible_form = initial_displacement / initial_theta
        initial_incompressible_form = (
            compressible_form - float(state["pr"]) ** (1.0 / 3.0) * (temperature_factor - 1.0)
        ) / temperature_factor
        if initial_incompressible_form <= 1.021:
            raise BoundaryLayerError(
                "initial turbulent thicknesses imply an incompressible "
                "form factor <= 1.021, outside the correlation domain"
            )

        turbulent = _turbulent_solution(state, 0, initial_theta, initial_incompressible_form)
        theta = np.asarray(turbulent["theta"])
        displacement = np.asarray(turbulent["displacement"])
        form = np.asarray(turbulent["form"])
        separation_index = turbulent["separation_index"]
    else:
        # Natural-transition mode first marches the laminar solution and
        # changes solver only when the correlation reports transition or
        # imminent laminar separation.
        laminar = _laminar_solution(state, laminar_correlation_limit)
        theta = np.asarray(laminar["theta"]).copy()
        displacement = np.asarray(laminar["displacement"]).copy()
        form = np.asarray(laminar["form"]).copy()
        transition_index = laminar["transition_index"]
        separation_index = laminar["separation_index"]
        if transition_index is not None and transition_index < count:
            # NASA TM X-2434 and NASA TM X-2343 recommend CTHET=1: transition at natural
            # onset or imminent laminar separation, conserving theta.
            initial_theta = max(float(theta[transition_index]), 1.0e-12)
            temperature_factor = 1.0 + 0.5 * (gamma - 1.0) * edge_flow_mach[transition_index] ** 2
            incompressible_form = (
                form[transition_index] - float(state["pr"]) ** (1.0 / 3.0) * (temperature_factor - 1.0)
            ) / temperature_factor
            turbulent_incompressible_form = _transition_incompressible_form_factor(
                float(incompressible_form), float(laminar["transition_reynolds_number"])
            )
            turbulent = _turbulent_solution(
                state, transition_index, initial_theta, max(turbulent_incompressible_form, 1.021)
            )
            for name, target in (("theta", theta), ("displacement", displacement), ("form", form)):
                source = np.asarray(turbulent[name])
                target[transition_index:] = source[transition_index:]
            regime[transition_index:] = "turbulent"
            # Laminar separation with CTHET=1 is treated as immediate
            # turbulent reattachment, following NASA TM X-2434 and NASA TM X-2343.
            separation_index = turbulent["separation_index"]

    if separation_index is not None and separation_index + 1 < count:
        regime[separation_index + 1 :] = "separated"
        # NASA TM X-2434 and NASA TM X-2343 terminate a case at turbulent separation. Keep arrays
        # usable for plotting, but do not invent further growth.
        theta[separation_index + 1 :] = theta[separation_index]
        displacement[separation_index + 1 :] = displacement[separation_index]
        form[separation_index + 1 :] = form[separation_index]
    if not np.all(np.isfinite(displacement)):
        raise BoundaryLayerError("boundary-layer solution contains non-finite thickness")

    return BoundaryLayerResult(
        s_over_chord=np.asarray(s, dtype=float),
        displacement_thickness_over_chord=np.asarray(displacement, dtype=float),
        momentum_thickness_over_chord=np.asarray(theta, dtype=float),
        form_factor=np.asarray(form, dtype=float),
        regime=regime,
        transition_index=transition_index,
        separation_index=separation_index,
        freestream_absolute_flow_mach=(
            np.asarray(edge_flow_mach, dtype=float) if surface.absolute_flow_mach is not None else None
        ),
        freestream_relative_flow_mach=(
            np.asarray(edge_flow_mach, dtype=float) if surface.relative_flow_mach is not None else None
        ),
    )
