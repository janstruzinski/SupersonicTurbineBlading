# SupersonicTurbineBlading

## Introduction

SupersonicTurbineBlading is a Python library for the preliminary aerodynamic design of two-dimensional supersonic
turbine rotor sections and supersonic stator nozzles. It combines method-of-characteristics (MOC)
design procedures with boundary-layer corrections. The library is largely based on old NASA Fortran codes.

The package is intended for aerospace and turbomachinery engineers who need a preliminary design tool.
It produces blade and nozzle coordinates, the principal quantities and checks needed
to assess a candidate geometry before higher-fidelity CFD and experimental work.

## General Overview

### Engineering basis

The rotor design is based primarily on:

- L. J. Goldman and V. J. Scullin, [*Analytical Investigation of Supersonic Turbomachinery Blading. I - Computer
  Program for Blading Design*](https://ntrs.nasa.gov/citations/19680009151), NASA TN D-4421 (1968);
- L. J. Goldman and V. J. Scullin, [*Computer Program for Design of Two-Dimensional Supersonic Turbine Rotor
  Blades with Boundary-Layer Correction*](https://ntrs.nasa.gov/citations/19720005326), NASA TM X-2434 (1971).

The ideal rotor implementation is also checked against the impulse-blade cases in L. J. Goldman's *Analytical
Investigation of Supersonic Turbomachinery Blading. II - Analysis of Impulse Turbine-Blade Sections*,
NASA TN D-4422 (1968).

The sharp-throat MOC stator design is based primarily on:

- M. R. Vanco and L. J. Goldman, [*Computer Program for Design of Two-Dimensional Supersonic Nozzle with
  Sharp-Edged Throat*](https://ntrs.nasa.gov/citations/19680005278), NASA TM X-1502 (1968);
- L. J. Goldman and M. R. Vanco, [*Computer Program for Design of Two-Dimensional Sharp-Edged-Throat Supersonic
  Nozzle with Boundary Layer Correction*](https://ntrs.nasa.gov/citations/19710023317), NASA TM X-2343 (1971).

The package also provides an axisymmetric, straight-wall conical de Laval nozzle. This option is sized from the
perfect-gas area-Mach relation rather than from the two-dimensional MOC construction.

All three design routes assume an isentropic, calorically perfect gas for the inviscid geometry. The ratio of specific
heats is evaluated from the selected mixture at a representative static state and then frozen for the gas-dynamic
construction. CoolProp transport properties may still vary along the surfaces during the boundary-layer calculation.

### Public classes

The three principal classes are imported directly from the package:

```python
from SupersonicTurbineBlading import Fluid, SupersonicRotorBlade, SupersonicStatorNozzle
```

| Class | Engineering purpose |
|---|---|
| `Fluid` | Defines a fixed gas composition and returns properties from pure-fluid CoolProp calls. |
| `SupersonicRotorBlade` | Designs and scales a vortex-flow rotor section from absolute or relative flow inputs. |
| `SupersonicStatorNozzle` | Sizes the throat and designs a planar MOC or axisymmetric conical nozzle. |

Construction of a rotor or stator object performs the complete selected design. Results are therefore available as
properties immediately after initialization.

The package also exposes result containers useful in engineering scripts:

| Result class | Contents                                                                                   |
|---|--------------------------------------------------------------------------------------------|
| `FluidState` | Thermodynamic and transport properties at one temperature and pressure.                    |
| `SurfaceCoordinates` | Coordinates, framed flow-Mach array and surface `metal_angle`.                              |
| `BladeShape`, `NozzleShape` | Pressure and suction surfaces plus passage dimensions in non-dimensional scale. |
| `DimensionalBladeShapes`, `DimensionalNozzleShapes` | Geometry in metres. |
| `BoundaryLayerResult` | Boundary-layer thicknesses, freestream flow Mach and transition or separation data.        |
| `FlowStateTable` | Printable comparison of the principal rotor flow states in both reference frames. |
| `StartingResult` | Rotor supersonic-starting limit and other design checks from NASA TN D-4421.               |

Unless stated otherwise, dimensional inputs and outputs use SI units: pressure in Pa, temperature in K, mass flow in
kg/s, length in m and rotational speed in rpm. Mach numbers in public properties are ordinary Mach numbers, not
critical velocity ratios. Flow and metal angles in public properties are in degrees. Row-level metal angles are
measured from the machine axial direction.

### Repository structure

```text
SupersonicTurbineBlading/
|-- boundary_layer/
|   `-- boundary_layer_solver.py
|-- rotor/
|   |-- rotor_blade.py
|   |-- rotor_geometry.py
|   |-- rotor_results.py
|   `-- rotor_starting.py
|-- stator/
|   |-- stator_geometry.py
|   |-- stator_results.py
|   `-- stator_nozzle.py
|-- common_results.py
|-- fluid.py
|-- gas_dynamics.py
`-- geometry_utils.py

rotor_example.py
stator_example.py
tests/
```

The `rotor` and `stator` folders contain the row-specific design procedures. The `boundary_layer` folder contains the
viscous integral solver shared by both rows. Fluid properties, perfect-gas relations, geometry-grid operations and
common result containers are kept in the package root so the rotor and stator use the same definitions.

## Installation

Install the latest version directly from the GitHub repository with:

```text
pip3 install git+https://github.com/janstruzinski/SupersonicTurbineBlading.git
```

This also installs the numerical dependencies declared by the package. To install a local copy instead, run the
following command from the project folder:

```text
python -m pip install .
```

For development, install the package in editable form and run the tests from the same folder:

```text
python -m pip install -e ".[test]"
python -m pytest
```

Python 3.10 or newer is required.

## Disclaimer

This library originated from legacy NASA FORTRAN IV programs. LLMs were used to assist with the
translation of that code to Python and with the building of additional features around it.

## Documentation

### Fluid class

`Fluid` represents the fixed composition of a gas. Temperature and pressure are supplied later to `properties()`, so
the same object can be reused at the stator throat, rotor inlet, blade surfaces and other states generated during a
design calculation.

#### Example and inputs

```python
from SupersonicTurbineBlading import Fluid

air = Fluid(
    coolprop_names=["Nitrogen", "Oxygen"],
    mass_fractions=[0.767, 0.233],
)

state = air.properties(temperature=900.0, pressure=1.0e6)

print(state.gamma)
print(state.speed_of_sound)
print(state.dynamic_viscosity)
```

`coolprop_names` contains the names recognized by CoolProp and `mass_fractions` contains the corresponding positive
mass fractions. Both lists must have the same length, names may not be repeated, and the mass fractions must sum to
one within a tolerance of $10^{-8}$.

Important composition properties stored by `Fluid` include:

| Property | Meaning |
|---|---|
| `coolprop_names`, `mass_fractions` | Validated fixed composition. |
| `mole_fractions` | Mole fractions calculated from the specified mass fractions. |
| `molar_masses`, `molar_mass` | Component and mixture molar masses in kg/mol. |
| `specific_gas_constant` | Mixture-specific gas constant in J/(kg K). |

`properties(temperature, pressure)` returns an immutable `FluidState`. Its most useful fields are `density`,
`specific_heat_cp`, `specific_heat_cv`, `gamma`, `dynamic_viscosity`, `kinematic_viscosity`,
`thermal_conductivity`, `prandtl_number` and `speed_of_sound`. All are in SI units.

#### CoolProp calls and mixing rules

CoolProp is not asked to perform a mixture flash. Instead, every component is evaluated separately at the mixture
temperature and its Dalton partial pressure. This avoids CoolProp raising any errors when mixtures are used.

Mass fractions $w_i$ are converted to mole fractions $x_i$ according to

$$x_i=\frac{w_i/M_i}{\sum_j w_j/M_j}, \qquad p_i=x_i p.$$

The mixture molar mass and gas constant are

$$\frac{1}{M_{\mathrm{mix}}}=\sum_i\frac{w_i}{M_i}, \qquad
R_{\mathrm{mix}}=\frac{R_u}{M_{\mathrm{mix}}}.$$

At each state, CoolProp supplies pure-component ideal-gas heat capacity (`CP0MASS`), viscosity and thermal
conductivity. The package applies the explicit mass-weighted rules

$$c_p=\sum_i w_i c_{p,i}^{0}, \qquad \mu=\sum_i w_i\mu_i, \qquad k=\sum_i w_i k_i.$$

The remaining mixture properties follow from

$$\rho=\frac{p}{R_{\mathrm{mix}}T}, \qquad c_v=c_p-R_{\mathrm{mix}}, \qquad \gamma=\frac{c_p}{c_v},$$

$$\mathrm{Pr}=\frac{c_p\mu}{k}, \qquad a=\sqrt{\gamma R_{\mathrm{mix}}T}, \qquad \nu=\frac{\mu}{\rho}.$$

The viscosity and conductivity averages are deliberately simple engineering approximations. The class also checks every
component phase at its partial-pressure state and rejects liquid or two-phase states.

### Rotor

#### Introduction to the `SupersonicRotorBlade`

`SupersonicRotorBlade` designs one two-dimensional section at the specified mean radius. The inlet and outlet flow
states can be supplied in either the absolute frame or the rotor-relative frame. The two input sets are mutually
exclusive. The class completes each velocity triangle and passes the rotor-relative Mach numbers and angles to the
blade-passage MOC tool that follows NASA TN D-4421.

The ideal passage consists of inlet transitions that convert uniform relative flow into a free-vortex distribution,
constant-Mach circular pressure- and suction-surface arcs, and outlet transitions that return the flow to a uniform
state. `lower_surface_relative_flow_mach` and `upper_surface_relative_flow_mach` are therefore surface-loading design
variables.

Positive flow and metal angles are measured from the machine axis toward the direction of rotation. The rotor design
domain uses a positive relative inlet flow angle and a negative relative outlet flow angle. Coordinates are initially
normalized by the vortex sonic radius $r^{\ast}$ and are subsequently scaled using mean radius and blade count.

#### Example of `SupersonicRotorBlade` with inputs and outputs

```python
from SupersonicTurbineBlading import Fluid, SupersonicRotorBlade

working_fluid = Fluid(["Nitrogen", "Oxygen"], [0.767, 0.233])

common_rotor_inputs = dict(
    lower_surface_relative_flow_mach=1.0,             # relative pressure-surface arc flow Mach
    upper_surface_relative_flow_mach=4.0,             # relative suction-surface arc flow Mach
    blade_count=80,
    mean_radius=0.15,                       # m
    rotational_speed_rpm=30000.0,
    fluid=working_fluid,
    inlet_total_temperature=1000.0,         # absolute total temperature, K
    inlet_total_pressure=5.0e6,             # absolute total pressure, Pa
    flow_turning_increment=0.1,              # optional MOC resolution; default 0.1 degrees
    leading_edge_thickness_over_total_pitch=0.07,
    use_leading_edge_entry_correction=True,
    calculate_starting=True,
    # The following three inputs are explained in "Boundary-layer correction - Rotor".
    boundary_layer_mode="fully_turbulent",
    initial_turbulent_displacement_thickness=2.0e-5,  # m
    initial_turbulent_momentum_thickness=5.0e-6,      # m
)

blade = SupersonicRotorBlade(
    ideal_inlet_absolute_flow_mach=2.80,              # far-field absolute inlet flow Mach
    ideal_inlet_absolute_flow_angle=70.0,             # far-field absolute inlet flow angle
    requested_outlet_absolute_flow_angle=-12.049849,  # ideal absolute outlet flow angle in this example
    requested_outlet_absolute_flow_mach=0.987512,     # optional ideal absolute outlet flow Mach
    **common_rotor_inputs,
)

# Alternative NASA TN D-4421 input convention: do not combine these with the four absolute-frame inputs above.
relative_blade = SupersonicRotorBlade(
    ideal_inlet_relative_flow_mach=2.50,
    ideal_inlet_relative_flow_angle=70.0,
    requested_outlet_relative_flow_angle=-65.0,
    # requested_outlet_relative_flow_mach=2.50,  # optional; omission selects an impulse blade
    **common_rotor_inputs,
)

# Separate flow states and metal angles
print(blade.ideal_inlet_absolute_flow_mach)
print(blade.ideal_inlet_absolute_flow_angle)
print(blade.ideal_inlet_relative_flow_mach)
print(blade.ideal_inlet_relative_flow_angle)
print(blade.real_inlet_relative_flow_mach)
print(blade.real_inlet_relative_flow_angle)
print(blade.inlet_metal_angle)
print(blade.ideal_outlet_relative_flow_mach)
print(blade.ideal_outlet_relative_flow_angle)
print(blade.outlet_metal_angle)
print(blade.flow_state_table)

# Ideal coordinates divided by r*
ideal_shape = blade.uncorrected_shape
pressure_x = ideal_shape.pressure.x
pressure_y = ideal_shape.pressure.y
pressure_relative_flow_mach = ideal_shape.pressure.relative_flow_mach

# Ideal coordinates in metres
ideal_shape_m = blade.dimensionalize().uncorrected

# CAD-ready ideal single-blade profile in millimetres
ideal_profile_x_mm = blade.uncorrected_blade_profile_x_CAD
ideal_profile_y_mm = blade.uncorrected_blade_profile_y_CAD
```

Exactly one complete flow-input family must be supplied:

| Input family | Required values |
|---|---|
| Absolute | `ideal_inlet_absolute_flow_mach`, `ideal_inlet_absolute_flow_angle`, and absolute outlet angle. |
| Relative | `ideal_inlet_relative_flow_mach`, `ideal_inlet_relative_flow_angle`, and relative outlet angle. |

The corresponding `requested_outlet_absolute_flow_mach` or `requested_outlet_relative_flow_mach` is optional.
Omitting it selects the impulse assumption $M_{\mathrm{rel,out}}=M_{\mathrm{rel,in}}$. Supplying any member of both
families raises `ValueError`; individual inlet or outlet quantities cannot be mixed across reference frames.
After validation, the corresponding requested outlet angle in the other frame is also available as a property. When
an outlet Mach is supplied, its transformed counterpart is stored as well; both requested Mach properties remain
`None` when the impulse assumption is selected.

The remaining required inputs describe the machine, thermodynamic operating point and two constant-Mach surface
arcs. The most important optional inputs used above are:

| Input | Meaning                                                                                       |
|---|-----------------------------------------------------------------------------------------------|
| `requested_outlet_*_flow_mach` | Optional outlet Mach in the selected input frame; `None` selects impulse flow. |
| `flow_turning_increment` | Maximum MOC turning step in $(0,1]$ degrees; default 0.1. |
| `leading_edge_thickness_over_total_pitch` | Ratio $t_{\mathrm{LE}}/G^{\ast}_{\mathrm{total}}$; default zero. |
| `use_leading_edge_entry_correction` | Corrects inlet Mach and flow angle for finite thickness; default `True`. |
| `calculate_starting` | Runs the NASA TN D-4421 supersonic-starting feasibility calculation; default `True`.          |

Useful properties available after construction include:

| Property | Engineering interpretation                                         |
|---|--------------------------------------------------------------------|
| `inlet_static_temperature`, `inlet_static_pressure`, `gamma` | Inlet static reference state and frozen $\gamma$. |
| `wheel_speed` | Blade speed $U$ at `mean_radius`.                                  |
| `ideal_inlet_absolute_flow_mach`, `ideal_inlet_absolute_flow_angle` | Far-field absolute inlet flow state. |
| `ideal_inlet_relative_flow_mach`, `ideal_inlet_relative_flow_angle` | Far-field relative inlet flow state. |
| `real_inlet_absolute_flow_mach`, `real_inlet_absolute_flow_angle` | Absolute finite-thickness passage-entry state. |
| `real_inlet_relative_flow_mach`, `real_inlet_relative_flow_angle` | Relative finite-thickness passage-entry state. |
| `inlet_metal_angle` | Inlet metal angle in the stationary machine frame. |
| `ideal_outlet_absolute_flow_mach`, `ideal_outlet_absolute_flow_angle` | Premixing absolute outlet flow state. |
| `ideal_outlet_relative_flow_mach`, `ideal_outlet_relative_flow_angle` | Premixing relative outlet flow state. |
| `outlet_metal_angle` | Outlet metal angle in the stationary machine frame. |
| `uncorrected_shape` | Ideal surfaces, chord and open pitches in one `BladeShape`.        |
| `physical_total_pitch`, `physical_passage_pitch` | Total and open inlet pitches in metres. |
| `sonic_radius_scale`, `physical_chord`, `chord_reynolds_number` | Dimensional scale and inlet-based Reynolds number. |
| `solidity` | Ideal axial chord divided by total blade pitch. |
| `leading_edge_thickness`, `physical_leading_edge_thickness` | Nondimensional and dimensional leading-edge thickness. |
| `starting_result` | `StartingResult` when `calculate_starting=True`, otherwise `None`. |
| `flow_state_table` | Printable inlet-to-outlet comparison of absolute and relative flow angles and Mach numbers. |

`flow_state_table.rows` retains the numerical values for further use, while `print(blade.flow_state_table)` produces
an aligned three-column engineering summary. The rows proceed from the ideal upstream inlet state through the real
passage-entry state, the ideal premixing outlet state and the real aftermixed outlet state. Every angle row precedes
the Mach-number row at the same station.

Each rotor `SurfaceCoordinates` object provides `x`, `y`, `relative_flow_mach` and `metal_angle` arrays at matching
stations. `absolute_flow_mach` is `None` for these surfaces. The local `metal_angle` array is stored in degrees.

#### Theory of `SupersonicRotorBlade`

##### Reference frames and thermodynamic state

The inlet API accepts either reference frame. For absolute inputs, the absolute velocity and wheel speed are

$$V_x=V\cos\alpha, \qquad V_{\theta}=V\sin\alpha, \qquad U=\frac{2\pi r_m N}{60}.$$

The rotor-relative velocity triangle is

$$W_x=V_x, \qquad W_{\theta}=V_{\theta}-U, \qquad
M_{\mathrm{rel}}=\frac{\sqrt{W_x^2+W_{\theta}^2}}{a}.$$

For relative inputs, the same triangle is evaluated in reverse:

$$V_x=W_x, \qquad V_{\theta}=W_{\theta}+U.$$

Because mixture heat capacity depends on temperature, the inlet static state and $\gamma$ are solved together:

$$T_{\mathrm{in}}=\frac{T_{t,\mathrm{abs}}}
{1+\frac{\gamma(T_{\mathrm{in}})-1}{2}M_{\mathrm{abs}}^2}, \qquad
p_{\mathrm{in}}=\frac{p_{t,\mathrm{abs}}}
{\left(1+\frac{\gamma-1}{2}M_{\mathrm{abs}}^2\right)^{\gamma/(\gamma-1)}}.$$

The converged inlet-static $\gamma$ is frozen for the ideal geometry. The relative total state then follows from the
same static state and the calculated relative Mach number.

When the relative inlet state is supplied, the code retains the absolute stagnation temperature. With
$q=\sqrt{T_{\mathrm{in}}}$, the temperature relation and velocity triangle give

$$Aq^2+Bq+C=0,$$

$$A=1+\frac{\gamma-1}{2}M_{\mathrm{rel}}^2, \qquad
B=\frac{(\gamma-1)M_{\mathrm{rel}}U\sin\beta}{\sqrt{\gamma R}}, \qquad
C=\frac{(\gamma-1)U^2}{2\gamma R}-T_{t,\mathrm{abs}}.$$

The positive root supplies the equivalent absolute Mach and static state. CoolProp then updates $\gamma$, and the
calculation repeats until the thermodynamic state and velocity triangle are consistent.

At the exit, the MOC construction requires a relative flow Mach and flow direction. If
`requested_outlet_absolute_flow_mach` is supplied, the class conserves relative total temperature at constant radius
and solves the velocity triangle for the corresponding relative state. The relative input family supplies that state
directly through `requested_outlet_relative_flow_angle` and optional `requested_outlet_relative_flow_mach`.

The current construction assumes zero incidence and zero deviation. Consequently, `inlet_metal_angle` is numerically
equal to `real_inlet_relative_flow_angle`, while `outlet_metal_angle` is numerically equal to
`ideal_outlet_relative_flow_angle`. They remain separate properties: the metal angles describe stationary geometry,
whereas the flow angles describe velocity directions in the explicitly named reference frame.

##### Vortex-flow blade construction

The NASA design uses the critical velocity ratio

$$M^{\ast}=\frac{V}{V_{\mathrm{cr}}}=
\sqrt{\frac{\frac{\gamma+1}{2}M^2}{1+\frac{\gamma-1}{2}M^2}}.$$

For free-vortex flow, the nondimensional radius is

$$\frac{r}{r^{\ast}}=\frac{1}{M^{\ast}}.$$

The pressure and suction constant-Mach arcs therefore have different radii. Their surface Mach numbers must bracket
the inlet and outlet relative Mach numbers:

$$1\le M_{\mathrm{lower}}\le\min(M_{\mathrm{rel,in}},M_{\mathrm{rel,out}}), \qquad
M_{\mathrm{upper}}\ge\max(M_{\mathrm{rel,in}},M_{\mathrm{rel,out}}).$$

The tighter angle-dependent limits are expressed conveniently with the Prandtl-Meyer angle $\nu$:

$$\nu(M)=\sqrt{\frac{\gamma+1}{\gamma-1}}
\tan^{-1}\sqrt{\frac{\gamma-1}{\gamma+1}(M^2-1)}-\tan^{-1}\sqrt{M^2-1}.$$

For $\beta_i>0$ and $\beta_o<0$,

$$\max(0,\nu_i-\beta_i,\nu_o-|\beta_o|)\le\nu_l\le\min(\nu_i,\nu_o),$$

$$\max(\nu_i,\nu_o)\le\nu_u\le\min(\nu_i+\beta_i,\nu_o+|\beta_o|).$$

The code checks these ranges before constructing the final section. Inlet and outlet transition arcs are generated by
the characteristic relations and rotated into the required relative flow directions. They are joined by the two
constant-Mach vortex arcs; short uniform-flow extensions complete the suction surface where required. The resulting
pressure and suction arrays bound one open periodic passage.

`lower_surface_relative_flow_mach` and `upper_surface_relative_flow_mach` strongly influence the loading distribution,
peak surface Mach, thickness, chord and solidity. They should be treated as preliminary blade-design variables.

##### NASA TN D-4422 impulse-blade validation

The ideal geometry is regression-tested against all 36 zero-edge-thickness impulse blades in NASA TN D-4422,
figures 8--13. The tests use the report's tabulated Prandtl-Meyer angles and specific-heat ratios rather than the
rounded Mach numbers printed below the figures. The inlet and outlet relative Mach numbers are equal, and each flow
angle is one half of the reported total turning angle.

For all cases, calculated solidity $C/G$ differs from the published two-decimal value by no more than 0.0051. Plots
of two adjacent Python-generated blades reproduce the corresponding published outlines and the reported trends in
blade thickness and curvature. NASA TN D-4422 does not provide numerical surface coordinates, so the shape comparison
is visual; solidity is the quantitative geometry check retained in the automated tests.

##### Finite leading-edge thickness and passage-entry state

The MOC solution supplies the open inlet passage pitch $G^{\ast}_{\mathrm{passage}}$. With
$\tau=t_{\mathrm{LE}}/G^{\ast}_{\mathrm{total}}$,

$$G^{\ast}_{\mathrm{total}}=\frac{G^{\ast}_{\mathrm{passage}}}{1-\tau}, \qquad
t^{\ast}_{\mathrm{LE}}=\frac{\tau}{1-\tau}G^{\ast}_{\mathrm{passage}}.$$

For positive thickness, `use_leading_edge_entry_correction=True` applies the correction of inlet flow conditions
described in [NACA RM L52B06](https://ntrs.nasa.gov/citations/19930087012). The far-field relative state
$(M_i,\beta_i)$ is converted to the open-passage entry state $(M_e,\beta_e)$ by simultaneously satisfying

$$\frac{A_e}{A_i}=(1-\tau)\frac{\cos\beta_e}{\cos\beta_i}
=\frac{(A/A^{\ast})_e}{(A/A^{\ast})_i},$$

$$\beta_e-\beta_i=\nu_e-\nu_i.$$

The physical weak-wave root nearest the far-field Mach is selected. Finite blockage therefore changes both the Mach
number and direction presented to the MOC passage. The `ideal_inlet_*_flow_*` properties store the far-field state;
the `real_inlet_*_flow_*` properties store the transformed entry state in both absolute and relative frames. Setting
the flag to `False` retains the finite metal thickness and pitch definitions but uses the far-field relative state
directly. The code warns when the correction is requested for supersonic relative axial inflow because the NACA
construction was derived for subsonic axial inflow.

##### Physical scale and Reynolds number

The machine circumference fixes the dimensional total pitch:

$$g_{\mathrm{total}}=\frac{2\pi r_m}{Z}, \qquad
r^{\ast}=\frac{g_{\mathrm{total}}}{G^{\ast}_{\mathrm{total}}}, \qquad
c=C^{\ast}r^{\ast}.$$

The passage-entry state defines the chord Reynolds number:

$$W_{\mathrm{entry}}=M_{\mathrm{entry}}a_{\mathrm{entry}}, \qquad
Re_c=\frac{W_{\mathrm{entry}}c}{\nu_{\mathrm{entry}}}.$$

`dimensionalize()` multiplies both stored shapes by the same final $r^{\ast}$ and returns coordinates in metres. The CAD
profile arrays are assembled as a single blade outline in millimetres, with the lower-surface leading edge at the
origin.

##### Supersonic-starting check

When `calculate_starting=True`, the `START` procedure from NASA TN D-4421 estimates the largest relative inlet Mach
number for which the passage can swallow the assumed passage-spanning normal shock. It searches for the vortex
constant that maximizes swallowed mass flow, equates that flow to the design supersonic flow and returns the limiting
Mach and Prandtl-Meyer angle.

The calculation is a feasibility screen and does not alter the coordinates. Its direct output is `starting_result`,
including `maximum_starting_ideal_inlet_relative_flow_mach` and `starts_supersonically`. Applicability should be judged
from the throughflow Mach number: subsonic axial velocity will not cause a normal shock across the axial inlet plane.

#### Code implementation of `SupersonicRotorBlade`

The ideal rotor design follows this sequence:

1. `rotor_blade.py` validates one complete flow-input frame and constructs the paired inlet state in the other frame.
2. The finite-thickness entry model supplies the Mach and angle at the open passage entrance.
3. `rotor_geometry.py` converts the four design Mach numbers to Prandtl-Meyer variables, builds the inlet/outlet
   transitions and vortex arcs, and returns a nondimensional `BladeShape`.
4. Mean radius and blade count establish pitch, $r^{\ast}$, physical chord and Reynolds number.
5. `rotor_starting.py` optionally evaluates the starting limit.
6. `rotor_results.py` and `common_results.py` store the resulting geometry and design checks.

The files in `rotor/` have the following roles:

| File | Role |
|---|---|
| `rotor_blade.py` | Public class, frame transformations, scaling and design orchestration. |
| `rotor_geometry.py` | NASA TN D-4421 MOC transition and vortex-arc geometry. |
| `rotor_results.py` | Geometry, starting-result and printable `FlowStateTable` containers. |
| `rotor_starting.py` | Rotor-only normal-shock swallowing and maximum-starting-Mach calculation. |

The principal internal calls are `design_ideal_geometry(...)` in `rotor_geometry.py` and
`calculate_starting_limit(...)` in `rotor_starting.py`. `SupersonicRotorBlade` supplies them with inputs,
then exposes their results through the public containers.
`BladeShape.scaled(...)` performs geometry-only scaling, while the public `dimensionalize()` method scales the
stored geometry to the final machine size.

Additional design steps performed by `rotor_blade.py` are documented in the dedicated sections below.

### Nozzle

#### Introduction to the `SupersonicStatorNozzle`

`SupersonicStatorNozzle` designs the supersonic part of a turbine stator passage. It first calculates the total choked
area from total mass flow and upstream total conditions, allocates that area among `nozzle_count` identical passages,
and then constructs one of two contours:

- `contour_method="moc"` creates a two-dimensional, sharp-throat slot nozzle that delivers uniform parallel flow
  at its ideal exit;
- `contour_method="conical"` creates a conical de Laval nozzle whose circular exit area produces
  the requested ideal exit Mach.

The MOC model requires an out-of-plane `throat_height`; each passage has a rectangular throat. The conical model uses a
circular throat and therefore derives `throat_diameter` directly from the nozzle throat area.

`outlet_metal_angle` is measured from the machine axis and stored in the stationary frame. The ideal outlet absolute
flow angle is stored separately, although the zero-deviation nozzle construction makes the two values numerically
equal. Coordinates remain in the unrotated nozzle-axis system; `outlet_metal_angle` is applied when plotting in
axial/tangential coordinates.

#### Examples of `SupersonicStatorNozzle` with inputs and outputs

The MOC nozzle is shown first, because it is the direct implementation of NASA TM X-1502.

##### MOC stator nozzle

```python
from SupersonicTurbineBlading import Fluid, SupersonicStatorNozzle

working_fluid = Fluid(["Nitrogen", "Oxygen"], [0.767, 0.233])

moc_stator = SupersonicStatorNozzle(
    requested_outlet_absolute_flow_mach=1.77,          # ideal absolute outlet flow Mach in this example
    requested_outlet_absolute_flow_angle=70.0,         # ideal absolute outlet flow angle in this example
    mass_flow_rate=5.0,                # complete stator row, kg/s
    nozzle_count=30,
    throat_height=0.05,                # out-of-plane span, m
    fluid=working_fluid,
    upstream_total_temperature=900.0,  # K
    upstream_total_pressure=1.0e6,     # Pa
    contour_method="moc",
    flow_turning_increment=0.5,         # required MOC resolution
)

print(moc_stator.total_throat_area)
print(moc_stator.throat_width)
print(moc_stator.actual_flow_turning_increment)
print(moc_stator.outlet_metal_angle)
print(moc_stator.ideal_outlet_absolute_flow_angle)
print(moc_stator.ideal_outlet_absolute_flow_mach)

# Ideal nozzle-axis coordinates divided by throat half-width
ideal_moc = moc_stator.uncorrected_shape

# Ideal coordinates in metres
ideal_moc_m = moc_stator.uncorrected_dimensional_shape
```

For the MOC route, `throat_height` is the physical out-of-plane blade span and `flow_turning_increment` must lie in
$(0,1]$ degrees. The algorithm adjusts the requested increment slightly so that half the exit Prandtl-Meyer angle is
divided into an integer number of characteristic regions; the value actually used is stored in
`actual_flow_turning_increment`.

##### Conical de Laval stator nozzle

```python
conical_stator = SupersonicStatorNozzle(
    requested_outlet_absolute_flow_mach=1.77,
    requested_outlet_absolute_flow_angle=70.0,
    mass_flow_rate=5.0,                # total flow through all nozzles, kg/s
    nozzle_count=30,
    fluid=working_fluid,
    upstream_total_temperature=900.0,  # K
    upstream_total_pressure=1.0e6,     # Pa
    contour_method="conical",
    half_cone_metal_angle=15.0,
)

print(conical_stator.single_nozzle_throat_area)
print(conical_stator.throat_diameter)
print(conical_stator.required_exit_area_ratio)
print(conical_stator.conical_divergent_length)

# Ideal meridional coordinates divided by throat diameter
ideal_conical = conical_stator.uncorrected_shape
ideal_conical_m = conical_stator.uncorrected_dimensional_shape
```

`half_cone_metal_angle` is the divergent-wall half angle from the nozzle axis and must lie between 0 and 90 degrees.
`throat_height` and `flow_turning_increment` must be omitted for this route because they belong only to the planar MOC
model. Conversely, `half_cone_metal_angle` must be omitted for an MOC nozzle.

Remaining input variables are documented later. Important base-design properties are:

| Property | Engineering interpretation                                           |
|---|----------------------------------------------------------------------|
| `gamma`, `throat_static_temperature`, `throat_static_pressure` | Fluid state at the throat. |
| `mass_flux_at_throat`, `total_throat_area` | Choked mass flux and total stator throat area.                       |
| `single_nozzle_throat_area` | Choked area assigned to one passage.                                 |
| `throat_width` | Rectangular opening of one MOC passage; `None` for a conical nozzle. |
| `throat_diameter`, `throat_radius` | Circular conical throat size; `None` for an MOC nozzle.              |
| `outlet_metal_angle` | Outlet metal angle in the stationary machine frame.                   |
| `ideal_outlet_absolute_flow_angle` | Uniform premixing absolute flow angle.                              |
| `ideal_outlet_absolute_flow_mach` | Uniform premixing absolute flow Mach used to construct the contour.  |
| `uncorrected_shape` | Ideal `NozzleShape` in throat-based nondimensional coordinates.      |
| `uncorrected_dimensional_shape` | Ideal `NozzleShape` in metres.                                       |
| `contour_point_count`, `pressure_number_of_stations` | MOC discretization diagnostics. |
| `required_exit_area_ratio`, `conical_divergent_length` | da Laval nozzle sizing results; `None` for an MOC nozzle. |
| `physical_chord`, `chord_reynolds_number` | Physical length and sonic-throat Reynolds scale.                     |

#### Theory of `SupersonicStatorNozzle`

##### Common throat sizing

The constant $\gamma$ used by either contour is evaluated at the actual sonic static state. Because `Fluid` heat
capacity varies with temperature, the class iterates

$$T^{\ast}=T_t\frac{2}{\gamma+1}, \qquad
p^{\ast}=p_t\left(\frac{2}{\gamma+1}\right)^{\gamma/(\gamma-1)}$$

until `fluid.properties(T*, p*)` returns a consistent $\gamma$. That value is then frozen for the contour and
choked-flow equations.

The total throat area follows from

$$\dot m=\frac{A^{\ast}p_t}{\sqrt{T_t}}\sqrt{\frac{\gamma}{R}}
\left(\frac{2}{\gamma+1}\right)^{\frac{\gamma+1}{2(\gamma-1)}}.$$

The area assigned to one of $N$ identical nozzles is

$$A^{\ast}_{\mathrm{one}}=\frac{A^{\ast}_{\mathrm{total}}}{N}.$$

##### MOC nozzle

The sharp edge at the throat initiates a centered Prandtl-Meyer expansion. One characteristic family travels from the
throat toward the centerline, reflects, intersects the opposite family and is cancelled by the shaped wall. At the end
of the characteristic region the flow is uniform and parallel to the nozzle axis.

For a sharp-throat minimum-length construction, the initial wall angle is half the exit Prandtl-Meyer angle:

$$\theta_{\mathrm{wall},\ast}=\frac{\nu_e}{2}.$$

The two-dimensional compatibility variables and Mach angle are

$$K_+=\theta+\nu, \qquad K_-=\theta-\nu, \qquad \mu=\sin^{-1}\left(\frac{1}{M}\right),$$

and the characteristic slopes are formed from $\tan(\theta+\mu)$ and $\tan(\theta-\mu)$. The code uses averages from
adjacent finite regions when intersecting characteristic lines, consistent with NASA TM X-1502.

The MOC coordinates use a throat half-width of one, so the full nondimensional opening is two. The physical opening is

$$w^{\ast}=\frac{A^{\ast}_{\mathrm{total}}}{Nh},$$

where $h$ is `throat_height`; every stored coordinate is multiplied by $w^{\ast}/2$ to scale them to machine size.

After the shaped divergent contour, the suction wall continues as a straight line in the nozzle-axis system. If
$(x_e,y_e)$ is the end of the nondimensional upper contour and $\alpha_N$ is `outlet_metal_angle`, measured from the
machine axis, the added straight length and periodic spacing are

$$L_s=2y_e\tan\alpha_N, \qquad S=\frac{2y_e}{\cos\alpha_N}.$$

The converging subsonic portion upstream of the sharp throat is not designed by this class.

##### Conical de Laval nozzle

The conical route is axisymmetric and uses the perfect-gas area-Mach relation:

$$\frac{A}{A^{\ast}}=\frac{1}{M}
\left[\frac{2}{\gamma+1}\left(1+\frac{\gamma-1}{2}M^2\right)\right]^{\frac{\gamma+1}{2(\gamma-1)}}.$$

Circular area scales with radius squared, giving

$$\frac{r_e}{r^{\ast}}=\sqrt{\frac{A_e}{A^{\ast}}}.$$

The throat diameter is obtained directly from the nozzle choked area:

$$D^{\ast}=\sqrt{\frac{4A^{\ast}_{\mathrm{one}}}{\pi}}.$$

Coordinates are normalized by $D^{\ast}$. The walls run from $y=\pm0.5$ at the throat to
$y=\pm0.5\sqrt{A_e/A^{\ast}}$ at the exit. For divergent half-angle $\theta_c$,

$$\frac{L_d}{D^{\ast}}=\frac{\sqrt{A_e/A^{\ast}}-1}{2\tan\theta_c}.$$

The suction wall has the same straight downstream line as used by the MOC contour.

#### Code implementation of `SupersonicStatorNozzle`

The base stator design follows this sequence:

1. `stator_nozzle.py` validates the contour-specific inputs and gets throat flow state.
2. The choked mass flux determines total and single nozzle throat areas.
3. For `"moc"`, `stator_geometry.py` builds the characteristic net and retains the final wall contour. For
   `"conical"`, it evaluates the area ratio, exit radius and straight-wall length.
4. The selected throat width or diameter establishes the dimensional coordinate scale.
5. `stator_results.py` stores the nondimensional and dimensional nozzle shapes.

The files in `stator/` have the following roles:

| File | Role |
|---|---|
| `stator_nozzle.py` | Public class, throat sizing, contour selection, physical scaling and engineering outputs. |
| `stator_geometry.py` | NASA TM X-1502 characteristic construction and the alternative conical de Laval geometry. |
| `stator_results.py` | `NozzleShape` and `DimensionalNozzleShapes` containers. |

`design_ideal_stator_nozzle(...)` builds the planar MOC contour and `design_conical_stator_nozzle(...)` builds the
axisymmetric alternative. Both return an `IdealNozzleConstruction`, which contains a `NozzleShape`.
`NozzleShape.scaled(...)` changes only length quantities. Each stator `SurfaceCoordinates` object stores
`absolute_flow_mach`, while `relative_flow_mach` is `None`. The `metal_angle` array remains in degrees.

Additional nozzle design features of `stator_nozzle.py` are documented below.

### Boundary-layer correction

#### Common theory and implementation

The rotor and stator use the same compressible integral boundary-layer solver in
`boundary_layer/boundary_layer_solver.py`. It follows the methods used by NASA TM X-2434 and NASA TM X-2343:

- the Cohen-Reshotko method from [NACA Report 1294](https://ntrs.nasa.gov/search.jsp?R=19930091005) for laminar flow;
- the Sasman-Cresci [compressible turbulent boundary-layer method](https://doi.org/10.2514/3.3378) for flow with
  pressure gradient and heat transfer.

The compressible displacement and momentum thicknesses represent the mass-flow and momentum deficits:

$$\delta^{\ast}=\int_0^\delta\left(1-\frac{\rho u}{\rho_e U_e}\right)dy,$$

$$\theta=\int_0^\delta\frac{\rho u}{\rho_e U_e}
\left(1-\frac{u}{U_e}\right)dy.$$

The solver reconstructs the local edge state from the surface Mach distribution using

$$\frac{T_e}{T_t}=\frac{1}{1+\frac{\gamma-1}{2}M_e^2}, \qquad
\frac{p_e}{p_t}=\left(\frac{T_e}{T_t}\right)^{\gamma/(\gamma-1)}.$$

The wall temperature is set equal to total temperature, matching the NASA rotor and stator drivers. The geometry uses
the frozen reference $\gamma$, while viscosity, heat capacity and conductivity are reevaluated through `Fluid` at the
local states.

The laminar method applies the Cohen-Reshotko transformation and correlation tables to the momentum integral
equation. It predicts neutral instability, transition and impending laminar separation. The turbulent method marches
the two coupled Sasman-Cresci integral equations for transformed momentum thickness and form factor with a fourth-order
Runge-Kutta scheme. At natural transition, or at laminar separation, the code follows the legacy `CTHET=1`
choice: momentum thickness is conserved and the turbulent march starts immediately.

Two inlet modes are available in both public classes:

| `boundary_layer_mode` | Required initialization |
|---|---|
| `"laminar_then_turbulent"` | Starts with zero thickness and predicts transition. |
| `"fully_turbulent"` | Requires positive inlet $\delta^{\ast}$ and $\theta$ in metres, with $\delta^{\ast}>\theta$. |

The initial thicknesses are in metres. Each geometry trial divides them by its own physical scale before marching.
Their ratio is transformed to the incompressible form factor required by the turbulent correlation.

`number_of_stations` is the minimum resolution of the temporary boundary-layer grid and must be at least 20. The grid
is densified within the original geometry segments determined by MOC. Results are projected back onto the
original MOC or conical stations for geometry correction, while the dense results remain available for convergence and
transition diagnostics.

In code, this shared workflow is `densify_surface(...)` -> `solve_boundary_layer(...)` ->
`project_boundary_layer_result(...)`. The first and third functions preserve the original design stations; only the
middle function performs the integral march.

The laminar correlation table is generated to the application limit used by the NASA code: `CORLN=0.50` for rotor
surfaces and `CORLN=0.16` for a stator nozzle. If the surface solution extends beyond the relevant limit, the code
issues a `RuntimeWarning` and linearly extrapolates the table rather than silently clipping the calculation.

Each `BoundaryLayerResult` contains:

- `s_over_chord` and explicitly framed `freestream_absolute_flow_mach` or `freestream_relative_flow_mach`;
- `displacement_thickness_over_chord` and `momentum_thickness_over_chord`;
- transformed `form_factor` and the local `regime`;
- `transition_index` and `separation_index` when the events occur.

#### Stator boundary-layer correction

The stator march starts at the sonic sharp throat. In natural-transition mode the throat thickness is therefore zero;
in fully turbulent mode the two specified thicknesses are throat values.

For the symmetric unrotated nozzle, the class performs one march along the upper wall. The pressure-side result is the
boundary layer ending at the divergent-contour exit; the suction-side result continues along the downstream straight.
This reproduces the station arrangement of NASA TM X-2343 and avoids two inconsistent solutions on what begins as the
same ideal contour.

The boundary layer correction is applied in the nozzle transverse direction:

$$y_{\mathrm{pressure,corr}}=y_{\mathrm{pressure}}-\delta^{\ast}, \qquad
y_{\mathrm{suction,corr}}=y_{\mathrm{suction}}+\delta^{\ast}.$$

It is a vertical offset in nozzle-axis coordinates. The suction-side straight wall segment is
then extended twice: first to restore the prescribed nozzle installation geometry after adding displacement thickness,
and second to account for continued thickness growth along the newly created segment. The corrected exit spacing and
the extrapolated exit thicknesses are retained for the final mixed-out calculation.

Stator controls and outputs are:

| Input or property | Meaning                                                      |
|---|--------------------------------------------------------------|
| `number_of_stations` | Minimum temporary marching resolution; default 101.          |
| `boundary_layer_mode` | Natural-transition or fully turbulent BL.                    |
| `initial_turbulent_displacement_thickness` | Input throat $\delta^{\ast}$ in fully turbulent mode, in metres.  |
| `initial_turbulent_momentum_thickness` | Input throat $\theta$ in fully turbulent mode, in metres. |
| `pressure_boundary_layer`, `suction_boundary_layer` | Results projected onto stored geometry stations.             |
| `pressure_boundary_layer_marching`, `suction_boundary_layer_marching` | Dense integration-grid results. |
| `corrected_shape`, `corrected_dimensional_shape` | Displacement-corrected geometry in throat units and metres.  |
| `corrected_exit_displacement_thickness` | Extrapolated final suction-side $\delta^{\ast}$ in metres.        |
| `corrected_exit_momentum_thickness` | Extrapolated final suction-side $\theta$ in metres.          |

For the conical contour, local Mach on the dense divergent-wall grid is found by inverting the circular area-Mach
relation. The same integral correction is then applied to its stored meridional walls.

`stator.plot(dimensional=True)` rotates both the ideal and corrected shapes by `outlet_metal_angle` and displays
coordinates in millimetres. Stored dimensional arrays remain in metres.

#### Rotor boundary-layer correction

The rotor performs independent marches along the pressure and suction surfaces, both beginning at the passage-entry
state.

The solver returns displacement thickness normal to the local flow. The NASA-style blade-coordinate correction uses
its vertical component:

$$\Delta y=\frac{\delta^{\ast}}{|\cos\eta|},$$

where $\eta$ is the local surface tangent angle. The pressure surface moves upward and the suction surface downward,
away from the open passage. Inlet and outlet open pitches are then recalculated from the displaced endpoints and their
uniform-flow directions.

Without legacy pitch closure, the finite trailing-edge metal remaining between the displaced surfaces is

$$t^{\ast}_{\mathrm{TE}}=\max\left[0,
t^{\ast}_{\mathrm{LE}}-\left(|\Delta y_{p,\mathrm{TE}}|+|\Delta y_{s,\mathrm{TE}}|\right)\right].$$

With legacy pitch closure, the model carries $t^{\ast}_{\mathrm{LE}}$ through to the trailing edge. It varies
`outlet_metal_angle`, so the requested outlet flow direction is only an initial estimate. The iterative scheme is
explained later in this documentation.
In both cases the resulting thickness is used in the rotor mixed-out blockage calculation.

Rotor controls and outputs are:

| Input or property | Meaning                                                            |
|---|--------------------------------------------------------------------|
| `number_of_stations` | Minimum temporary stations on each surface; default 101.           |
| `boundary_layer_mode` | Natural-transition or fully turbulent BL.                          |
| `initial_turbulent_displacement_thickness` | Inlet $\delta^{\ast}$ applied to both surfaces, in metres.              |
| `initial_turbulent_momentum_thickness` | Inlet $\theta$ applied to both surfaces, in metres.                |
| `pressure_boundary_layer`, `suction_boundary_layer` | Results projected onto the MOC stations. |
| `pressure_boundary_layer_marching`, `suction_boundary_layer_marching` | Dense integration-grid results. |
| `corrected_shape`, `dimensional_shapes.corrected` | Displacement-corrected passage in $r^{\ast}$ units and metres. |
| `trailing_edge_thickness`, `physical_trailing_edge_thickness` | Remaining trailing-edge metal. |
| `corrected_pitch_residual` | Corrected outlet pitch minus corrected inlet pitch in $r^{\ast}$ units. |

`blade.plot(dimensional=True, corrected=True)` plots the corrected geometry in millimetres;
`corrected=False` selects the ideal shape. `show_two_blades=True` completes the two blades surrounding the stored
passage. `blade_profile_x_CAD` and `blade_profile_y_CAD` contain the corrected single-blade outline in millimetres.

### Aftermixing

The `AFMIX` models in NASA TM X-2434 and NASA TM X-2343 replace the nonuniform boundary-layer wakes and finite
trailing-edge blockage by a uniform downstream state. They apply continuity, axial momentum, tangential momentum and
energy across the mixing plane. The result is an engineering estimate of mixed Mach number and flow angle.

For premixing Mach $M_1$ and direction $\alpha_1$, define the critical velocity ratio

$$q_1=M_1^{\ast}=\sqrt{\frac{\frac{\gamma+1}{2}M_1^2}{1+\frac{\gamma-1}{2}M_1^2}}$$

and the exit pitch projected onto the axial-normal plane

$$X_{X}=S\cos\alpha_1.$$

The displacement, momentum and metal blockage ratios are

$$D_{\delta}=\frac{\delta_{p}^{\ast}+\delta_{s}^{\ast}}{X_{X}}, \qquad
D_{\theta}=\frac{\theta_{p}+\theta_{s}}{X_{X}}, \qquad
D_{\mathrm{TE}}=\frac{t_{\mathrm{TE}}}{X_{X}}.$$

Following the legacy notation, the effective momentum and flow areas are

$$A=1-D_{\delta}-D_{\mathrm{TE}}-D_{\theta}, \qquad
A_1=1-D_{\delta}-D_{\mathrm{TE}}.$$

The conservation equations reduce to

$$C=\frac{(1-F_s)\frac{\gamma+1}{2\gamma}+
\cos^2\alpha_1\,Aq_1^2}{\cos\alpha_1\,A_1q_1}, \qquad
D=q_1\sin\alpha_1\frac{A}{A_1},$$

where

$$F_s=\frac{\gamma-1}{\gamma+1}q_1^2.$$

The downstream axial critical velocity ratio has two mathematical roots:

$$q_{x,2}=\frac{\gamma C}{\gamma+1}\pm
\sqrt{\left(\frac{\gamma C}{\gamma+1}\right)^2-1+
\frac{\gamma-1}{\gamma+1}D^2}.$$

The minus sign is the subsonic-axial root. The plus sign is the shockless supersonic-axial root and is available only
when the premixing axial Mach $M_1\cos\alpha_1$ is at least one. The total downstream critical velocity ratio, ordinary
Mach and direction are

$$q_2=\sqrt{q_{x,2}^2+D^2}, \qquad
M_2=\sqrt{\frac{\frac{2}{\gamma+1}q_2^2}
{1-\frac{\gamma-1}{\gamma+1}q_2^2}}, \qquad
\alpha_2=\tan^{-1}\left(\frac{D}{q_{x,2}}\right).$$

By default, `mixing_solution` is omitted and the root is selected automatically. Premixing axial Mach below one uses
the subsonic root. At axial Mach equal to or greater than one, the shockless supersonic root is used when available.
Set `mixing_solution="subsonic"` to force the subsonic root. No supersonic override is provided because automatic
selection already chooses that root whenever it is physical.

#### Rotor aftermixing

Rotor `AFMIX` is evaluated in the rotating frame using the corrected outlet pitch, the two surface thickness results
and the calculated trailing-edge thickness. Each available mixed relative state is then transformed to the stationary
frame by adding wheel speed to its tangential velocity.

`mixing_results["subsonic"]` and `mixing_results["supersonic"]` contain explicitly named fields for each available
root. They include `real_outlet_absolute_flow_mach`, `real_outlet_absolute_axial_flow_mach`,
`real_outlet_absolute_flow_angle`, `real_outlet_relative_flow_mach`, `real_outlet_relative_axial_flow_mach` and
`real_outlet_relative_flow_angle`. The selected-root properties are:

| Property | Meaning |
|---|---|
| `real_outlet_absolute_flow_mach`, `real_outlet_absolute_flow_angle` | Selected absolute mixed state. |
| `real_outlet_relative_flow_mach`, `real_outlet_relative_flow_angle` | Selected relative mixed state. |
| `ideal_outlet_relative_axial_flow_mach` | Relative axial Mach before mixing. |
| `supersonic_mixing_available` | Whether the shockless root is physically available. |
| `mixing_solution` | Root selected for the final design. |

#### Stator aftermixing

The stator is stationary, so the nozzle-axis velocity is already in the absolute frame. `trailing_edge_thickness` is a
physical metal thickness in metres and contributes to $D_{\mathrm{TE}}$. As in NASA TM X-2343, it affects the mixed-out
conservation calculation only.

`uncorrected_mixing_results` is a diagnostic calculation at the original exit stations and ideal spacing uncorrected by
the boundary layer. `mixing_results` uses the corrected spacing and nozzle trailing edge thicknesses. The selected flow
solution is stored under `mixing_solution` as `supersonic` or `subsonic`. The selected final values are available as
`real_outlet_absolute_flow_mach` and `real_outlet_absolute_flow_angle`, with
`ideal_outlet_absolute_axial_flow_mach` and `supersonic_mixing_available` providing additional information.

### Iterative schemes

By default, the requested outlet flow angle and Mach define the ideal premixing state. The optional matching schemes
instead interpret one or both requested quantities as real aftermixed targets and vary the metal angle or ideal flow
Mach needed to reach them. A separate rotor pitch-closure scheme varies the outlet metal angle to match inlet and
outlet pitch.

#### Rotor iterative schemes

Three rotor schemes are available.

Both mixed-flow matching schemes use the reference frame of the selected rotor input family. They therefore target
absolute aftermixed quantities for the absolute input set and relative aftermixed quantities for the relative input
set.

##### Mixed-flow angle matching

Set

```python
iterate_outlet_metal_angle=True
```

to vary `outlet_metal_angle`, stored in the stationary frame, until the selected real aftermixed flow angle matches
the requested outlet flow angle in the input reference frame:

$$\alpha_{\mathrm{mixed}}-\alpha_{\mathrm{requested}}=0 \quad \text{(absolute input set)},$$

$$\beta_{\mathrm{mixed}}-\beta_{\mathrm{requested}}=0 \quad \text{(relative input set)}.$$

The solver scans the admissible negative outlet-metal-angle range, brackets the residual and bisects it.

If a requested outlet Mach is supplied without Mach matching, it remains the specified ideal premixing Mach in the
input reference frame. For the absolute input set, `ideal_outlet_relative_flow_mach` is recovered at each trial from
the velocity triangle and constant-radius rothalpy relation. For the relative input set, the requested relative Mach
is used directly. If the optional requested outlet Mach is omitted, `ideal_outlet_relative_flow_mach` remains equal
to `ideal_inlet_relative_flow_mach`.

##### Coupled mixed-flow angle and Mach matching

Set all three controls

```python
iterate_outlet_metal_angle=True
match_real_outlet_mach=True
requested_outlet_absolute_flow_mach=1.20  # desired absolute mixed Mach
```

For the relative input set, use `requested_outlet_relative_flow_mach` instead; the same `match_real_outlet_mach` flag
then interprets it as a relative aftermixed target. A damped two-variable Newton solve varies
`ideal_outlet_relative_flow_mach` and `outlet_metal_angle` until the angle and Mach residuals vanish in the selected
input frame. Local derivative-free refinement is used if MOC station-count changes stall the Newton solve:

$$M_{\mathrm{mixed},f}-M_{\mathrm{target},f}=0, \qquad
\theta_{\mathrm{mixed},f}-\theta_{\mathrm{target},f}=0,$$

where $f$ is the selected absolute or relative input frame.

The final premixing values remain available as `ideal_outlet_relative_flow_mach` and
`ideal_outlet_relative_flow_angle`. The aftermixed results remain available in both frames. The Mach-target flag
requires `iterate_outlet_metal_angle=True` and the requested outlet Mach belonging to the selected input family.

##### Legacy pitch closure

Set

```python
iterate_pitch_closure=True
```

to reproduce the NASA TM X-2434 `BETAT` closure. It holds `ideal_outlet_relative_flow_mach` fixed and varies
`outlet_metal_angle` until

$$G^{\ast}_{\mathrm{out,corr}}-G^{\ast}_{\mathrm{in,ideal}}=0.$$

This allows to keep the same leading and trailing edge thickness for the rotor blade when boundary layer correction is
used. The first unbracketed update follows the legacy mass-continuity expression; once trial geometries exist on both
sides of equal pitch, arithmetic bisection is used. The tolerance corresponds to $10^{-6}$ m in the physical blade
scale.

In this mode, the requested outlet angle in the selected input frame is only the initial estimate, and the final outlet
direction will generally differ. Construction therefore emits a warning. Pitch closure is incompatible with
`iterate_outlet_metal_angle=True` and
`match_real_outlet_mach=True`. `pitch_closure_iteration_count`, `pitch_closure_outlet_metal_angle` and
`pitch_closure_residual` report the result.

#### Stator iterative schemes

The stator provides two mixed-out closure levels.

##### Mixed-flow angle matching

Set

```python
iterate_outlet_metal_angle=True
```

to vary `outlet_metal_angle` until `real_outlet_absolute_flow_angle` reaches the requested value:

$$\alpha_{\mathrm{mixed}}-\alpha_{\mathrm{requested}}=0.$$

The solver evaluates feasible metal angles around the target, locates a sign-changing bracket and uses bisection. The
ideal construction flow Mach remains `requested_outlet_absolute_flow_mach`; the converged metal angle is stored as
`outlet_metal_angle` and the separately stored zero-deviation flow direction is
`ideal_outlet_absolute_flow_angle`.

##### Coupled mixed-flow angle and Mach matching

Set

```python
iterate_outlet_metal_angle=True
match_real_outlet_absolute_flow_mach=True
```

to reinterpret `requested_outlet_absolute_flow_mach` as the requested real aftermixed flow Mach. A Newton solve varies
`outlet_metal_angle` and `ideal_outlet_absolute_flow_mach` until

$$M_{\mathrm{mixed}}-M_{\mathrm{requested}}=0, \qquad
\alpha_{\mathrm{mixed}}-\alpha_{\mathrm{requested}}=0.$$

For every MOC trial, the characteristic contour is rebuilt for the new ideal exit Mach. For every conical trial, the
area ratio, exit radius and divergent length are rebuilt. The converged premixing value is stored as
`ideal_outlet_absolute_flow_mach`, while `requested_outlet_absolute_flow_mach` retains the target. The corresponding
aftermixed value is `real_outlet_absolute_flow_mach`. `match_real_outlet_absolute_flow_mach=True` requires
`iterate_outlet_metal_angle=True`.

By default, each stator iteration trial selects its mixing root from that trial's
`ideal_outlet_absolute_axial_flow_mach`. The user can apply `mixing_solution="subsonic"` to every trial.

## License

SupersonicTurbineBlading is licensed under the GNU General Public License, version 3 only (`GPL-3.0-only`). See
[LICENSE](LICENSE) for the complete terms.
