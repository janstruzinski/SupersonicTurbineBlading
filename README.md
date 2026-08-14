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
| `SupersonicRotorBlade` | Designs and scales a vortex-flow rotor section from an absolute operating point. |
| `SupersonicStatorNozzle` | Sizes the throat and designs a planar MOC or axisymmetric conical nozzle. |

Construction of a rotor or stator object performs the complete selected design. Results are therefore available as
properties immediately after initialization.

The package also exposes result containers useful in engineering scripts:

| Result class | Contents                                                                                   |
|---|--------------------------------------------------------------------------------------------|
| `FluidState` | Thermodynamic and transport properties at one temperature and pressure.                    |
| `SurfaceCoordinates` | Surface coordinates, local Mach number and tangent angle.                                  |
| `BladeShape`, `NozzleShape` | Pressure and suction surfaces plus passage dimensions in non-dimensional scale. |
| `DimensionalBladeShapes`, `DimensionalNozzleShapes` | Geometry in metres. |
| `BoundaryLayerResult` | Boundary layer thicknesses, form factor, flow regime, transition and separation locations. |
| `StartingResult` | Rotor supersonic-starting limit and other design checks from NASA TN D-4421.               |

Unless stated otherwise, dimensional inputs and outputs use SI units: pressure in Pa, temperature in K, mass flow in
kg/s, length in m and rotational speed in rpm. Mach numbers in public properties are ordinary Mach numbers, not
critical velocity ratios. Angles in public properties are in degrees and are measured from the machine axial direction.

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

$$\frac{1}{M_\mathrm{mix}}=\sum_i\frac{w_i}{M_i}, \qquad R_\mathrm{mix}=\frac{R_u}{M_\mathrm{mix}}.$$

At each state, CoolProp supplies pure-component ideal-gas heat capacity (`CP0MASS`), viscosity and thermal
conductivity. The package applies the explicit mass-weighted rules

$$c_p=\sum_i w_i c_{p,i}^{0}, \qquad \mu=\sum_i w_i\mu_i, \qquad k=\sum_i w_i k_i.$$

The remaining mixture properties follow from

$$\rho=\frac{p}{R_\mathrm{mix}T}, \qquad c_v=c_p-R_\mathrm{mix}, \qquad \gamma=\frac{c_p}{c_v},$$

$$\mathrm{Pr}=\frac{c_p\mu}{k}, \qquad a=\sqrt{\gamma R_\mathrm{mix}T}, \qquad \nu=\frac{\mu}{\rho}.$$

The viscosity and conductivity averages are deliberately simple engineering approximations. The class also checks every
component phase at its partial-pressure state and rejects liquid or two-phase states.

### Rotor

#### Introduction to the `SupersonicRotorBlade`

`SupersonicRotorBlade` designs one two-dimensional section at the specified mean radius. The public
(that is, available to the user) inlet and outlet operating-point quantities use the stationary frame.
The class calculates the velocity triangles and passes the resulting rotor-relative Mach numbers and angles to the
blade passage MOC tool that follows NASA TN D-4421.

The ideal passage consists of inlet transitions that convert uniform relative flow into a free-vortex distribution,
constant-Mach circular pressure- and suction-surface arcs, and outlet transitions that return the flow to a uniform
state. `lower_surface_mach` and `upper_surface_mach` are therefore surface-loading design variables.

Positive inlet angle is measured from the machine axis toward the direction of rotation. The rotor design domain uses
a positive relative inlet angle and a negative relative outlet angle. Coordinates are initially normalized by the
vortex sonic radius $r^*$ and are subsequently scaled using mean radius and blade count.

#### Example of `SupersonicRotorBlade` with inputs and outputs

```python
from SupersonicTurbineBlading import Fluid, SupersonicRotorBlade

working_fluid = Fluid(["Nitrogen", "Oxygen"], [0.767, 0.233])

blade = SupersonicRotorBlade(
    inlet_mach=2.80,                         # absolute inlet Mach number
    inlet_flow_angle_deg=70.0,              # absolute angle from the machine axis
    outlet_flow_angle_deg=-12.0498490291,   # absolute ideal outlet direction
    outlet_mach=0.9875119383,               # optional absolute ideal outlet Mach
    lower_surface_mach=1.0,                 # relative pressure-surface arc Mach
    upper_surface_mach=4.0,                 # relative suction-surface arc Mach
    blade_count=80,
    mean_radius=0.15,                       # m
    rotational_speed_rpm=30000.0,
    fluid=working_fluid,
    inlet_total_temperature=1000.0,         # absolute total temperature, K
    inlet_total_pressure=5.0e6,             # absolute total pressure, Pa
    turning_increment_deg=0.1,              # optional MOC resolution; default 0.1 deg
    leading_edge_thickness_over_total_pitch=0.07,
    use_leading_edge_entry_correction=True,
    calculate_starting=True,
    # The following three inputs are explained in "Boundary-layer correction - Rotor".
    boundary_layer_mode="fully_turbulent",
    initial_turbulent_displacement_thickness=2.0e-5,  # m
    initial_turbulent_momentum_thickness=5.0e-6,      # m
)

# Rotor-relative states used by the ideal blade construction
print(blade.relative_inlet_mach)
print(blade.passage_inlet_mach)
print(blade.relative_outlet_mach)
print(blade.outlet_blade_angle_deg)

# Ideal coordinates divided by r*
ideal_shape = blade.uncorrected_shape
pressure_x = ideal_shape.pressure.x
pressure_y = ideal_shape.pressure.y
pressure_mach = ideal_shape.pressure.mach

# Ideal coordinates in metres
ideal_shape_m = blade.dimensionalize().uncorrected

# CAD-ready ideal single-blade profile in millimetres
ideal_profile_x_mm = blade.uncorrected_blade_profile_x_CAD
ideal_profile_y_mm = blade.uncorrected_blade_profile_y_CAD
```

The required inputs describe the operating point, the velocity triangles and the two constant-Mach surface arcs.
The most important optional inputs used above are:

| Input | Meaning                                                                                       |
|---|-----------------------------------------------------------------------------------------------|
| `outlet_mach` | Absolute ideal outlet Mach; `None` sets $M_{\mathrm{rel,out}}=M_{\mathrm{rel,in}}$.           |
| `turning_increment_deg` | Maximum MOC turning step in $(0,1]$ degrees; default 0.1. |
| `leading_edge_thickness_over_total_pitch` | Ratio $t_\mathrm{LE}/G^*_{\mathrm{total}}$; default zero. |
| `use_leading_edge_entry_correction` | Corrects inlet Mach and flow angle for finite thickness; default `True`. |
| `calculate_starting` | Runs the NASA TN D-4421 supersonic-starting feasibility calculation; default `True`.          |

Useful properties available after construction include:

| Property | Engineering interpretation                                         |
|---|--------------------------------------------------------------------|
| `inlet_static_temperature`, `inlet_static_pressure`, `gamma` | Inlet static reference state and frozen $\gamma$. |
| `wheel_speed` | Blade speed $U$ at `mean_radius`.                                  |
| `relative_inlet_mach`, `relative_inlet_flow_angle_deg` | Far-field relative inlet state. |
| `passage_inlet_mach`, `passage_inlet_flow_angle_deg` | Open-passage entry state. |
| `ideal_outlet_mach`, `ideal_outlet_flow_angle_deg` | Absolute ideal state before mixing at the passage outlet. |
| `relative_outlet_mach`, `outlet_blade_angle_deg` | Rotor-relative ideal outlet state used by the MOC construction. |
| `uncorrected_shape` | Ideal surfaces, chord and open pitches in one `BladeShape`.        |
| `physical_total_pitch`, `physical_passage_pitch` | Total and open inlet pitches in metres. |
| `sonic_radius_scale`, `physical_chord`, `chord_reynolds_number` | Dimensional scale and inlet-based Reynolds number. |
| `leading_edge_thickness`, `physical_leading_edge_thickness` | Nondimensional and dimensional leading-edge thickness. |
| `starting_result` | `StartingResult` when `calculate_starting=True`, otherwise `None`. |

Each `SurfaceCoordinates` object provides `x`, `y`, `mach` and `tangent_angle_rad` arrays at matching stations.

#### Theory of `SupersonicRotorBlade`

##### Reference frames and thermodynamic state

The inlet API uses the absolute frame. The absolute velocity and wheel speed are

$$V_x=V\cos\alpha, \qquad V_\theta=V\sin\alpha, \qquad U=\frac{2\pi r_m N}{60}.$$

The rotor-relative velocity triangle is

$$W_x=V_x, \qquad W_\theta=V_\theta-U, \qquad M_\mathrm{rel}=\frac{\sqrt{W_x^2+W_\theta^2}}{a}.$$

Because mixture heat capacity depends on temperature, the inlet static state and $γ$ are solved together:

$$T_\mathrm{in}=\frac{T_{t,\mathrm{abs}}}{1+\frac{\gamma(T_\mathrm{in})-1}{2}M_\mathrm{abs}^2}, \qquad
p_\mathrm{in}=\frac{p_{t,\mathrm{abs}}}{\left(1+\frac{\gamma-1}{2}M_\mathrm{abs}^2\right)^{\gamma/(\gamma-1)}}.$$

The converged inlet-static $γ$ is frozen for the ideal geometry. The relative total state then follows from the same
static state and the calculated relative Mach number.

At the exit, the MOC construction requires a relative Mach and relative flow direction. If an absolute `outlet_mach`
is supplied, the class conserves relative total temperature at constant radius and solves the velocity triangle for
the corresponding relative state. The ideal outlet metal angle equals the ideal relative-flow direction.

##### Vortex-flow blade construction

The NASA design uses the critical velocity ratio

$$M^*=\frac{V}{V_\mathrm{cr}}=
\sqrt{\frac{\frac{\gamma+1}{2}M^2}{1+\frac{\gamma-1}{2}M^2}}.$$

For free-vortex flow, the nondimensional radius is

$$\frac{r}{r^*}=\frac{1}{M^*}.$$

The pressure and suction constant-Mach arcs therefore have different radii. Their surface Mach numbers must bracket
the inlet and outlet relative Mach numbers:

$$1\le M_\mathrm{lower}\le\min(M_\mathrm{rel,in},M_\mathrm{rel,out}), \qquad
M_\mathrm{upper}\ge\max(M_\mathrm{rel,in},M_\mathrm{rel,out}).$$

The tighter angle-dependent limits are expressed conveniently with the Prandtl-Meyer angle $ν$:

$$\nu(M)=\sqrt{\frac{\gamma+1}{\gamma-1}}
\tan^{-1}\sqrt{\frac{\gamma-1}{\gamma+1}(M^2-1)}-\tan^{-1}\sqrt{M^2-1}.$$

For $β_i>0$ and $β_o<0$,

$$\max(0,\nu_i-\beta_i,\nu_o-|\beta_o|)\le\nu_l\le\min(\nu_i,\nu_o),$$

$$\max(\nu_i,\nu_o)\le\nu_u\le\min(\nu_i+\beta_i,\nu_o+|\beta_o|).$$

The code checks these ranges before constructing the final section. Inlet and outlet transition arcs are generated by
the characteristic relations and rotated into the required relative flow directions. They are joined by the two
constant-Mach vortex arcs; short uniform-flow extensions complete the suction surface where required. The resulting
pressure and suction arrays bound one open periodic passage.

`lower_surface_mach` and `upper_surface_mach` strongly influence loading distribution, peak surface Mach, thickness,
chord and solidity. They should be treated as preliminary design variables that allow to optimize the blade.

##### Finite leading-edge thickness and passage-entry state

The MOC solution supplies the open inlet passage pitch $G^*_{\mathrm{passage}}$. With
$\tau=t_\mathrm{LE}/G^*_{\mathrm{total}}$,

$$G^*_{\mathrm{total}}=\frac{G^*_{\mathrm{passage}}}{1-\tau}, \qquad
t^*_\mathrm{LE}=\frac{\tau}{1-\tau}G^*_{\mathrm{passage}}.$$

For positive thickness, `use_leading_edge_entry_correction=True` applies the correction of inlet flow conditions
described in [NACA RM L52B06](https://ntrs.nasa.gov/citations/19930087012). The far-field relative state
$(M_i,\beta_i)$ is converted to the open-passage entry state $(M_e,\beta_e)$ by simultaneously satisfying

$$\frac{A_e}{A_i}=(1-\tau)\frac{\cos\beta_e}{\cos\beta_i}
=\frac{(A/A^*)_e}{(A/A^*)_i},$$

$$\beta_e-\beta_i=\nu_e-\nu_i.$$

The physical weak-wave root nearest the far-field Mach is selected. Finite blockage therefore changes both the Mach
number and direction presented to the MOC passage; `relative_inlet_*` stores the far-field state and
`passage_inlet_*` stores the transformed state. Setting the flag to `False` retains the finite metal thickness and
pitch definitions but uses the far-field relative state directly. The code warns when the correction is requested for
supersonic relative axial inflow because the NACA construction was derived for subsonic axial inflow.

##### Physical scale and Reynolds number

The machine circumference fixes the dimensional total pitch:

$$g_\mathrm{total}=\frac{2\pi r_m}{Z}, \qquad
r^*=\frac{g_\mathrm{total}}{G^*_{\mathrm{total}}}, \qquad c=C^*r^*.$$

The passage-entry state defines the chord Reynolds number:

$$W_\mathrm{entry}=M_\mathrm{entry}a_\mathrm{entry}, \qquad
Re_c=\frac{W_\mathrm{entry}c}{\nu_\mathrm{entry}}.$$

`dimensionalize()` multiplies both stored shapes by the same final $r^*$ and returns coordinates in metres. The CAD
profile arrays are assembled as a single blade outline in millimetres, with the lower-surface leading edge at the
origin.

##### Supersonic-starting check

When `calculate_starting=True`, the `START` procedure from NASA TN D-4421 estimates the largest relative inlet Mach
number for which the passage can swallow the assumed passage-spanning normal shock. It searches for the vortex
constant that maximizes swallowed mass flow, equates that flow to the design supersonic flow and returns the limiting
Mach and Prandtl-Meyer angle.

The calculation is a feasibility screen and does not alter the coordinates. Its direct output is `starting_result`,
including `maximum_starting_inlet_mach` and `starts_supersonically`. Applicability should be judged from the
throughflow Mach number: subsonic axial velocity will not cause a normal shock across the axial inlet plane.

#### Code implementation of `SupersonicRotorBlade`

The ideal rotor design follows this sequence:

1. `rotor_blade.py` validates the operating point and constructs the absolute and relative inlet flow states.
2. The finite-thickness entry model supplies the Mach and angle at the open passage entrance.
3. `rotor_geometry.py` converts the four design Mach numbers to Prandtl-Meyer variables, builds the inlet/outlet
   transitions and vortex arcs, and returns a nondimensional `BladeShape`.
4. Mean radius and blade count establish pitch, $r^*$, physical chord and Reynolds number.
5. `rotor_starting.py` optionally evaluates the starting limit.
6. `rotor_results.py` and `common_results.py` store the resulting geometry and design checks.

The files in `rotor/` have the following roles:

| File | Role |
|---|---|
| `rotor_blade.py` | Public class, frame transformations, scaling and design orchestration. |
| `rotor_geometry.py` | NASA TN D-4421 MOC transition and vortex-arc geometry. |
| `rotor_results.py` | `BladeShape`, `DimensionalBladeShapes` and `StartingResult` containers. |
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

Angles are measured from the machine axis. Stored coordinates remain in the unrotated nozzle-axis system;
angle is applied when plotting in axial/tangential coordinates.

#### Examples of `SupersonicStatorNozzle` with inputs and outputs

The MOC nozzle is shown first, because it is the direct implementation of NASA TM X-1502.

##### MOC stator nozzle

```python
from SupersonicTurbineBlading import Fluid, SupersonicStatorNozzle

working_fluid = Fluid(["Nitrogen", "Oxygen"], [0.767, 0.233])

moc_stator = SupersonicStatorNozzle(
    exit_mach=1.77,                     # absolute ideal exit Mach
    outlet_flow_angle_deg=70.0,        # angle from the machine axis
    mass_flow_rate=5.0,                # complete stator row, kg/s
    nozzle_count=30,
    throat_height=0.05,                # out-of-plane span, m
    fluid=working_fluid,
    upstream_total_temperature=900.0,  # K
    upstream_total_pressure=1.0e6,     # Pa
    contour_method="moc",
    turning_increment_deg=0.5,         # required MOC resolution
)

print(moc_stator.total_throat_area)
print(moc_stator.throat_width)
print(moc_stator.actual_turning_increment_deg)

# Ideal nozzle-axis coordinates divided by throat half-width
ideal_moc = moc_stator.uncorrected_shape

# Ideal coordinates in metres
ideal_moc_m = moc_stator.uncorrected_dimensional_shape
```

For the MOC route, `throat_height` is the physical out-of-plane blade span and `turning_increment_deg` must lie in
$(0,1]$ degrees. The algorithm adjusts the requested increment slightly so that half the exit Prandtl-Meyer angle is
divided into an integer number of characteristic regions; the value actually used is stored in
`actual_turning_increment_deg`.

##### Conical de Laval stator nozzle

```python
conical_stator = SupersonicStatorNozzle(
    exit_mach=1.77,
    outlet_flow_angle_deg=70.0,
    mass_flow_rate=5.0,                # total flow through all nozzles, kg/s
    nozzle_count=30,
    fluid=working_fluid,
    upstream_total_temperature=900.0,  # K
    upstream_total_pressure=1.0e6,     # Pa
    contour_method="conical",
    half_cone_angle_deg=15.0,
)

print(conical_stator.single_nozzle_throat_area)
print(conical_stator.throat_diameter)
print(conical_stator.required_exit_area_ratio)
print(conical_stator.conical_divergent_length)

# Ideal meridional coordinates divided by throat diameter
ideal_conical = conical_stator.uncorrected_shape
ideal_conical_m = conical_stator.uncorrected_dimensional_shape
```

`half_cone_angle_deg` is the divergent-wall half angle from the nozzle axis and must lie between 0 and 90 degrees.
`throat_height` and `turning_increment_deg` must be omitted for this route because they belong only to the planar MOC
model. Conversely, `half_cone_angle_deg` must be omitted for an MOC nozzle.

Remaining input variables are documented later. Important base-design properties are:

| Property | Engineering interpretation                                           |
|---|----------------------------------------------------------------------|
| `gamma`, `throat_static_temperature`, `throat_static_pressure` | Fluid state at the throat. |
| `mass_flux_at_throat`, `total_throat_area` | Choked mass flux and total stator throat area.                       |
| `single_nozzle_throat_area` | Choked area assigned to one passage.                                 |
| `throat_width` | Rectangular opening of one MOC passage; `None` for a conical nozzle. |
| `throat_diameter`, `throat_radius` | Circular conical throat size; `None` for an MOC nozzle.              |
| `nozzle_angle_deg` | Final nozzle-axis angle from the machine axis.                       |
| `ideal_exit_mach` | Uniform Mach used to construct the ideal contour.                    |
| `uncorrected_shape` | Ideal `NozzleShape` in throat-based nondimensional coordinates.      |
| `uncorrected_dimensional_shape` | Ideal `NozzleShape` in metres.                                       |
| `contour_point_count`, `pressure_number_of_stations` | MOC discretization diagnostics. |
| `required_exit_area_ratio`, `conical_divergent_length` | da Laval nozzle sizing results; `None` for an MOC nozzle. |
| `physical_chord`, `chord_reynolds_number` | Physical length and sonic-throat Reynolds scale.                     |

#### Theory of `SupersonicStatorNozzle`

##### Common throat sizing

The constant $γ used by either contour is evaluated at the actual sonic static state. Because `Fluid` heat capacity
varies with temperature, the class iterates

$$T^*=T_t\frac{2}{\gamma+1}, \qquad
p^*=p_t\left(\frac{2}{\gamma+1}\right)^{\gamma/(\gamma-1)}$$

until `fluid.properties(T*, p*)` returns a consistent $γ. That value is then frozen for the contour and choked-flow
equations.

The total throat area follows from

$$\dot m=\frac{A^*p_t}{\sqrt{T_t}}\sqrt{\frac{\gamma}{R}}
\left(\frac{2}{\gamma+1}\right)^{\frac{\gamma+1}{2(\gamma-1)}}.$$

The area assigned to one of $N$ identical nozzles is

$$A^*_\mathrm{one}=\frac{A^*_\mathrm{total}}{N}.$$

##### MOC nozzle

The sharp edge at the throat initiates a centered Prandtl-Meyer expansion. One characteristic family travels from the
throat toward the centerline, reflects, intersects the opposite family and is cancelled by the shaped wall. At the end
of the characteristic region the flow is uniform and parallel to the nozzle axis.

For a sharp-throat minimum-length construction, the initial wall angle is half the exit Prandtl-Meyer angle:

$$\theta_\mathrm{wall,*}=\frac{\nu_e}{2}.$$

The two-dimensional compatibility variables and Mach angle are

$$K_+=\theta+\nu, \qquad K_-=\theta-\nu, \qquad \mu=\sin^{-1}\left(\frac{1}{M}\right),$$

and the characteristic slopes are formed from $\tan(\theta+\mu)$ and $\tan(\theta-\mu)$. The code uses averages from
adjacent finite regions when intersecting characteristic lines, consistent with NASA TM X-1502.

The MOC coordinates use a throat half-width of one, so the full nondimensional opening is two. The physical opening is

$$w^*=\frac{A^*_\mathrm{total}}{Nh},$$

where $h$ is `throat_height`; every stored coordinate is multiplied by $w^*/2$ to scale them to machine size.

After the shaped divergent contour, the suction wall continues as a straight line in the nozzle-axis system. If
$(x_e,y_e)$ is the end of the nondimensional upper contour and $\alpha_N$ is the nozzle angle from the machine axis,
the added straight length and periodic spacing are

$$L_s=2y_e\tan\alpha_N, \qquad S=\frac{2y_e}{\cos\alpha_N}.$$

The converging subsonic portion upstream of the sharp throat is not designed by this class.

##### Conical de Laval nozzle

The conical route is axisymmetric and uses the perfect-gas area-Mach relation:

$$\frac{A}{A^*}=\frac{1}{M}
\left[\frac{2}{\gamma+1}\left(1+\frac{\gamma-1}{2}M^2\right)\right]^{\frac{\gamma+1}{2(\gamma-1)}}.$$

Circular area scales with radius squared, giving

$$\frac{r_e}{r^*}=\sqrt{\frac{A_e}{A^*}}.$$

The throat diameter is obtained directly from the nozzle choked area:

$$D^*=\sqrt{\frac{4A^*_\mathrm{one}}{\pi}}.$$

Coordinates are normalized by $D^*$. The walls run from $y=\pm0.5$ at the throat to
$y=\pm0.5\sqrt{A_e/A^*}$ at the exit. For divergent half-angle $\theta_c$,

$$\frac{L_d}{D^*}=\frac{\sqrt{A_e/A^*}-1}{2\tan\theta_c}.$$

The suction wall has the same straigth downstream line as used by the MOC contour.

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
`NozzleShape.scaled(...)` changes only length quantities; Mach and tangent-angle
arrays remain dimensionless.

Additional nozzle design features of `stator_nozzle.py` are documented below.

### Boundary-layer correction

#### Common theory and implementation

The rotor and stator use the same compressible integral boundary-layer solver in
`boundary_layer/boundary_layer_solver.py`. It follows the methods used by NASA TM X-2434 and NASA TM X-2343:

- the Cohen-Reshotko method from [NACA Report 1294](https://ntrs.nasa.gov/search.jsp?R=19930091005) for laminar flow;
- the Sasman-Cresci [compressible turbulent boundary-layer method](https://doi.org/10.2514/3.3378) for flow with
  pressure gradient and heat transfer.

The compressible displacement and momentum thicknesses represent the mass-flow and momentum deficits:

$$\delta^*=\int_0^\delta\left(1-\frac{\rho u}{\rho_e U_e}\right)dy,$$

$$\theta=\int_0^\delta\frac{\rho u}{\rho_e U_e}
\left(1-\frac{u}{U_e}\right)dy.$$

The solver reconstructs the local edge state from the surface Mach distribution using

$$\frac{T_e}{T_t}=\frac{1}{1+\frac{\gamma-1}{2}M_e^2}, \qquad
\frac{p_e}{p_t}=\left(\frac{T_e}{T_t}\right)^{\gamma/(\gamma-1)}.$$

The wall temperature is set equal to total temperature, matching the NASA rotor and stator drivers. The geometry uses
the frozen reference $γ, while viscosity, heat capacity and conductivity are reevaluated through `Fluid` at the local
states.

The laminar method applies the Cohen-Reshotko transformation and correlation tables to the momentum integral
equation. It predicts neutral instability, transition and impending laminar separation. The turbulent method marches
lthe two coupled Sasman-Cresci integral equations for transformed momentum thickness and form factor with a fourth-order
Runge-Kutta scheme. At natural transition, or at laminar separation, the code follows the legacy `CTHET=1`
choice: momentum thickness is conserved and the turbulent march starts immediately.

Two inlet modes are available in both public classes:

| `boundary_layer_mode` | Required initialization |
|---|---|
| `"laminar_then_turbulent"` | Starts with zero thickness and predicts transition. |
| `"fully_turbulent"` | Requires positive inlet $\delta^*$ and $\theta$ in metres, with $\delta^*>\theta$. |

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

- `s_over_chord` and edge `mach` at every station;
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

$$y_\mathrm{pressure,corr}=y_\mathrm{pressure}-\delta^*, \qquad
y_\mathrm{suction,corr}=y_\mathrm{suction}+\delta^*.$$

It is a vertical offset in nozzle-axis coordinates. The suction-side straight wall segment is
then extended twice: first to restore the prescribed nozzle installation geometry after adding displacement thickness,
and second to account for continued thickness growth along the newly created segment. The corrected exit spacing and
the extrapolated exit thicknesses are retained for the final mixed-out calculation.

Stator controls and outputs are:

| Input or property | Meaning                                                      |
|---|--------------------------------------------------------------|
| `number_of_stations` | Minimum temporary marching resolution; default 101.          |
| `boundary_layer_mode` | Natural-transition or fully turbulent BL.                    |
| `initial_turbulent_displacement_thickness` | Input throat $\delta^*$ in fully turbulent mode, in metres.  |
| `initial_turbulent_momentum_thickness` | Input throat $\theta$ in fully turbulent mode, in metres. |
| `pressure_boundary_layer`, `suction_boundary_layer` | Results projected onto stored geometry stations.             |
| `pressure_boundary_layer_marching`, `suction_boundary_layer_marching` | Dense integration-grid results. |
| `corrected_shape`, `corrected_dimensional_shape` | Displacement-corrected geometry in throat units and metres.  |
| `corrected_exit_displacement_thickness` | Extrapolated final suction-side $\delta^*$ in metres.        |
| `corrected_exit_momentum_thickness` | Extrapolated final suction-side $\theta$ in metres.          |

For the conical contour, local Mach on the dense divergent-wall grid is found by inverting the circular area-Mach
relation. The same integral correction is then applied to its stored meridional walls.

`stator.plot(dimensional=True)` rotates both the ideal and corrected shapes by `nozzle_angle_deg` and displays
coordinates in millimetres. Stored dimensional arrays remain in metres.

#### Rotor boundary-layer correction

The rotor performs independent marches along the pressure and suction surfaces, both beginning at the passage-entry
state.

The solver returns displacement thickness normal to the local flow. The NASA-style blade-coordinate correction uses
its vertical component:

$$\Delta y=\frac{\delta^*}{|\cos\eta|},$$

where $η$ is the local surface tangent angle. The pressure surface moves upward and the suction surface downward,
away from the open passage. Inlet and outlet open pitches are then recalculated from the displaced endpoints and their
uniform-flow directions.

Without legacy pitch closure, the finite trailing-edge metal remaining between the displaced surfaces is

$$t^*_\mathrm{TE}=\max\left[0,
t^*_\mathrm{LE}-\left(|\Delta y_{p,\mathrm{TE}}|+|\Delta y_{s,\mathrm{TE}}|\right)\right].$$

With legacy pitch closure, the model carries $t^*_\mathrm{LE}$ through to the trailing edge, however the blade shape
does not maintain desired outlet angle. The iterative scheme used for this is explained later in this documentation.
In both cases the resulting thickness is used in the rotor mixed-out blockage calculation.

Rotor controls and outputs are:

| Input or property | Meaning                                                            |
|---|--------------------------------------------------------------------|
| `number_of_stations` | Minimum temporary stations on each surface; default 101.           |
| `boundary_layer_mode` | Natural-transition or fully turbulent BL.                          |
| `initial_turbulent_displacement_thickness` | Inlet $\delta^*$ applied to both surfaces, in metres.              |
| `initial_turbulent_momentum_thickness` | Inlet $\theta$ applied to both surfaces, in metres.                |
| `pressure_boundary_layer`, `suction_boundary_layer` | Results projected onto the MOC stations. |
| `pressure_boundary_layer_marching`, `suction_boundary_layer_marching` | Dense integration-grid results. |
| `corrected_shape`, `dimensional_shapes.corrected` | Displacement-corrected passage in $r^*$ units and metres. |
| `trailing_edge_thickness`, `physical_trailing_edge_thickness` | Remaining trailing-edge metal. |
| `corrected_pitch_residual` | Corrected outlet pitch minus corrected inlet pitch in $r^*$ units. |

`blade.plot(dimensional=True, corrected=True)` plots the corrected geometry in millimetres;
`corrected=False` selects the ideal shape. `show_two_blades=True` completes the two blades surrounding the stored
passage. `blade_profile_x_CAD` and `blade_profile_y_CAD` contain the corrected single-blade outline in millimetres.

### Aftermixing

The `AFMIX` models in NASA TM X-2434 and NASA TM X-2343 replace the nonuniform boundary-layer wakes and finite
trailing-edge blockage by a uniform downstream state. They apply continuity, axial momentum, tangential momentum and
energy across the mixing plane. The result is an engineering estimate of mixed Mach number and flow angle.

For premixing Mach $M_1$ and direction $α_1$, define the critical velocity ratio

$$q_1=M_1^*=\sqrt{\frac{\frac{\gamma+1}{2}M_1^2}{1+\frac{\gamma-1}{2}M_1^2}}$$

and the exit pitch projected onto the axial-normal plane

$$X_X=S\cos\alpha_1.$$

The displacement, momentum and metal blockage ratios are

$$D_\delta=\frac{\delta_p^*+\delta_s^*}{X_X}, \qquad
D_\theta=\frac{\theta_p+\theta_s}{X_X}, \qquad
D_\mathrm{TE}=\frac{t_\mathrm{TE}}{X_X}.$$

Following the legacy notation, the effective momentum and flow areas are

$$A=1-D_\delta-D_\mathrm{TE}-D_\theta, \qquad
A_1=1-D_\delta-D_\mathrm{TE}.$$

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

`mixing_results["subsonic"]` and `mixing_results["supersonic"]` contain both explicit relative and absolute fields.
For consistency with the public rotor API, the short keys `mach`, `axial_mach` and
`flow_angle_deg` refer to the absolute
mixed state. The properties are:

| Property | Meaning |
|---|---|
| `obtained_outlet_mach`, `obtained_outlet_flow_angle_deg` | Selected absolute mixed state. |
| `obtained_relative_outlet_mach`, `obtained_relative_outlet_flow_angle_deg` | Selected relative mixed state. |
| `premixing_axial_mach` | Relative axial Mach before mixing. |
| `supersonic_mixing_available` | Whether the shockless root is physically available. |
| `mixing_solution` | Root selected for the final design. |

#### Stator aftermixing

The stator is stationary, so the nozzle-axis velocity is already in the absolute frame. `trailing_edge_thickness` is a
physical metal thickness in metres and contributes to $D_\mathrm{TE}$. As in NASA TM X-2343, it affects the mixed-out
conservation calculation only.

`uncorrected_mixing_results` is a diagnostic calculation at the original exit stations and ideal spacing uncorrected by
the boundary layer. `mixing_results` uses the corrected spacing and nozzle trailing edge thicknesses. The selected flow
solution is stored under `mixing_solution` as `supersonic` or `subsonic`. The selected final values are available as
`obtained_outlet_mach` and `obtained_outlet_flow_angle_deg`, with `premixing_axial_mach` and
`supersonic_mixing_available` providing additional information.

### Iterative schemes

The default rotor and stator designs use the supplied ideal outlet angle and Mach number directly. However, in reality,
aftermixing will affect both. The following optional schemes allow to iterate the passage outlet angle
and Mach number, such that desired values for aftermixed angle and Mach number are reached. Furthermore,
alternative iterative scheme allows to obtain the same outlet pitch for the rotor blade as at the inlet,
thus keeping thickness of the leading and trailing edges the same.

#### Rotor iterative schemes

Three rotor schemes are available.

##### Mixed-flow angle matching

Set

```python
iterate_outlet_blade_angle=True
```

to vary the relative outlet metal angle until the selected mixed absolute angle matches `outlet_flow_angle_deg`:

$$\alpha_\mathrm{mixed,abs}-\alpha_\mathrm{requested,abs}=0.$$

The solver scans the admissible negative blade-angle range, brackets the residual and bisects it.

If `outlet_mach` is supplied in this mode, it remains the specified ideal absolute Mach at the outlet before mixing.
For each trial metal angle, the relative ideal Mach used for blade construction is recovered from the velocity triangle
and constant-radius rothalpy relation. If `outlet_mach=None`, the relative outlet Mach remains equal to the far-field
relative inlet Mach.

##### Coupled mixed-flow angle and Mach matching

Set all three controls

```python
iterate_outlet_blade_angle=True
match_outlet_mach_after_mixing=True
outlet_mach=1.20  # desired absolute mixed Mach
```

to treat `outlet_mach` as a mixed-state target as well. A damped two-variable Newton solve varies the ideal relative
outlet Mach and relative metal angle until

$$M_\mathrm{mixed,abs}-M_\mathrm{target,abs}=0, \qquad
\alpha_\mathrm{mixed,abs}-\alpha_\mathrm{target,abs}=0.$$

The final ideal construction value is stored in `relative_outlet_mach`; the requested and obtained values after mixing
remain available separately. The Mach target flag requires both a supplied `outlet_mach`, `outlet_flow_angle_deg` and
`iterate_outlet_blade_angle`.

##### Legacy pitch closure

Set

```python
iterate_pitch_closure=True
```

to reproduce the NASA TM X-2434 `BETAT` closure. It holds the ideal relative outlet Mach fixed and varies the relative
outlet angle until

$$G^*_{\mathrm{out,corr}}-G^*_{\mathrm{in,ideal}}=0.$$

This allows to keep the same leading and trailing edge thickness for the rotor blade when boundary layer correction is
used. The first unbracketed update follows the legacy mass-continuity expression; once trial geometries exist on both
sides of equal pitch, arithmetic bisection is used. The tolerance corresponds to $10^{-6}$ m in the physical blade
scale.

In this mode, `outlet_flow_angle_deg` is only the initial estimate and the final outlet direction will generally differ.
Construction therefore emits a warning. Pitch closure is incompatible with `iterate_outlet_blade_angle=True` and
`match_outlet_mach_after_mixing=True`. `pitch_closure_iteration_count`, `pitch_closure_outlet_angle_deg` and
`pitch_closure_residual` report the result.

#### Stator iterative schemes

The stator provides two mixed-out closure levels.

##### Mixed-flow angle matching

Set

```python
iterate_nozzle_angle=True
```

to vary the nozzle-axis metal angle until

$$\alpha_\mathrm{mixed}-\alpha_\mathrm{requested}=0.$$

The solver evaluates feasible angles around the target, locates a sign-changing bracket and uses bisection. The ideal
construction Mach remains `exit_mach`; the converged metal angle is stored as `nozzle_angle_deg`.

##### Coupled mixed-flow angle and Mach matching

Set

```python
iterate_nozzle_angle=True
match_exit_mach_after_mixing=True
```

to reinterpret `exit_mach` as the requested mixed Mach. A Newton solve varies `nozzle_angle_deg` and
`ideal_exit_mach` until

$$M_\mathrm{mixed}-M_\mathrm{requested}=0, \qquad
\alpha_\mathrm{mixed}-\alpha_\mathrm{requested}=0.$$

For every MOC trial, the characteristic contour is rebuilt for the new ideal exit Mach. For every conical trial, the
area ratio, exit radius and divergent length are rebuilt. The converged premixing value is stored as `ideal_exit_mach`,
while `requested_exit_mach` retains the target. `match_exit_mach_after_mixing=True` requires
`iterate_nozzle_angle=True`.

By default, each stator iteration trial geometry selects its mixing solution based on that trial's premixing axial
Mach. User can apply explicit `mixing_solution="subsonic"` override for every trial.

## License

SupersonicTurbineBlading is licensed under the GNU General Public License, version 3 only (`GPL-3.0-only`). See
[LICENSE](LICENSE) for the complete terms.
