# Supersonic turbine rotor and stator designers

This package is an object-oriented Python implementation of the two NASA
rotor programs in:

- L. J. Goldman and V. J. Scullin, *Analytical Investigation of Supersonic
  Turbomachinery Blading I — Computer Program for Blading Design*,
  NASA TN D-4421 (1968).
- L. J. Goldman and V. J. Scullin, *Computer Program for Design of
  Two-Dimensional Supersonic Turbine Rotor Blades with Boundary-Layer
  Correction*, NASA TM X-2434 (1971).

`SupersonicRotorBlade` performs the design during initialization and keeps all
results as properties. It stores only the final blade coordinates, not the
legacy unrotated coordinate arrays.

It also implements the sharp-edged-throat supersonic stator programs in:

- L. J. Goldman and W. R. Vanco, *Computer Program for Design of
  Two-Dimensional Supersonic Nozzle with Sharp-Edged Throat*,
  NASA TM X-1502 (1968).
- L. J. Goldman and W. R. Vanco, *Computer Program for Design of
  Two-Dimensional Sharp-Edged-Throat Supersonic Nozzle with Boundary-Layer
  Correction*, NASA TM X-2343 (1971).

`SupersonicStatorNozzle` uses the same `Fluid` and shared boundary-layer
solver. Its ideal and corrected nozzle passage geometries are stored both in
throat-normalized and dimensional form.

## Source layout

```text
SupersonicTurbineBlading/
├── boundary_layer/
│   └── boundary_layer_solver.py
├── rotor/
│   ├── rotor_blade.py
│   ├── rotor_geometry.py
│   ├── rotor_models.py
│   └── rotor_starting.py
├── stator/
│   ├── stator_geometry.py
│   ├── stator_models.py
│   └── stator_nozzle.py
├── common_models.py
├── fluid.py
├── gas_dynamics.py
└── geometry_utils.py
```

The subfolder `__init__.py` package markers are the only intentional
exceptions to the rotor/stator filename-prefix convention. Executable
examples, reports, tests, project metadata, and this README remain outside
the source package. The public imports are:

```python
from SupersonicTurbineBlading import Fluid, SupersonicRotorBlade, SupersonicStatorNozzle
```

## Installation

From the project folder, install the package and its numerical dependencies
with the conventional `setup.py` configuration:

```text
python -m pip install .
```

For development, install the package in editable form and run the tests from
the same folder:

```text
python -m pip install -e .
python -m pytest
```

## Supersonic stator nozzle

The stator API uses quantities that are directly available at a turbine
operating point:

```python
from SupersonicTurbineBlading import Fluid, SupersonicStatorNozzle

fluid = Fluid(["Nitrogen", "Oxygen"], [0.767, 0.233])

stator = SupersonicStatorNozzle(
    exit_mach=1.77,
    outlet_flow_angle_deg=70.0,  # from the machine axial direction
    mass_flow_rate=5.0,  # complete stator [kg/s]
    nozzle_count=30,
    throat_height=0.05,  # out-of-plane blade span [m]
    fluid=fluid,
    upstream_total_temperature=900.0,
    upstream_total_pressure=1.0e6,
    trailing_edge_thickness=1.0e-4,  # physical thickness [m]
    contour_method="moc",
    turning_increment_deg=0.1,  # required for MOC
    number_of_stations=101,
    iterate_nozzle_angle=True,
    match_exit_mach_after_mixing=True,
    boundary_layer_mode="fully_turbulent",
    initial_turbulent_displacement_thickness=2.0e-5,
    initial_turbulent_momentum_thickness=5.0e-6,
    mixing_solution="subsonic",
)

# Already stored when initialization finishes
ideal_nondimensional = stator.uncorrected_shape
corrected_nondimensional = stator.corrected_shape
ideal_dimensional = stator.uncorrected_dimensional_shape
corrected_dimensional = stator.corrected_dimensional_shape

stator.plot(dimensional=True)
```

Dimensional stator plots use millimetres. The dimensional shape properties
remain in metres for calculations.

By default, `exit_mach` is the ideal uniform Mach before mixing. With
`match_exit_mach_after_mixing=True`, it is instead the requested absolute
Mach after mixing and the resulting ideal construction value is stored as
`ideal_exit_mach`.

The old nozzle program asked for exit Mach, gamma, a Prandtl–Meyer increment,
physical throat half-height, gas constant, and numerous card/print controls.
The Python object retains exit Mach and the requested characteristic
resolution, but derives gamma, Prandtl–Meyer angles, gas constant, and physical
scale.

### Stator contour methods

`contour_method` selects one of two ideal supersonic contours:

- `"moc"` uses the Goldman–Vanco method-of-characteristics net and requires
  `turning_increment_deg`;
- `"conical"` uses an axisymmetric straight-wall de Laval contour and requires
  `half_cone_angle_deg`.

Inputs belonging to the other method are rejected instead of being silently
ignored. For example:

```python
conical_stator = SupersonicStatorNozzle(
    # Reuse the operating-point, fluid, BL, mixing, and iteration inputs from
    # the preceding MOC stator example.
    exit_mach=1.77,
    outlet_flow_angle_deg=70.0,
    mass_flow_rate=5.0,
    nozzle_count=30,
    fluid=fluid,
    upstream_total_temperature=900.0,
    upstream_total_pressure=1.0e6,
    contour_method="conical",
    half_cone_angle_deg=15.0,
)
```

The conical exit area follows the
[NASA Glenn area–Mach relation](https://www.grc.nasa.gov/www/k-12/airplane/rktthsum.html):

$$
\frac{A}{A^*}
=
\frac{1}{M}
\left[
\frac{2}{\gamma+1}
\left(1+\frac{\gamma-1}{2}M^2\right)
\right]^{\frac{\gamma+1}{2(\gamma-1)}}.
$$

The conical option is axisymmetric. Circular area scales with radius squared,
so

$$
\frac{r_e}{r^*}=\sqrt{\frac{A_e}{A^*}}.
$$

Its meridional coordinates are normalized by throat diameter $D^*$.
Consequently, the walls run linearly from $y=\pm0.5$ to
$y=\pm0.5\sqrt{A_e/A^*}$. The nondimensional divergent length is

$$
\frac{L_d}{D^*}
=
\frac{\sqrt{A_e/A^*}-1}{2\tan\theta_c},
$$

where `half_cone_angle_deg` is $\theta_c$. At the divergent exit, the suction
side receives the same one-sided straight section used by NASA TM X-1502
nozzle; after rotation it is parallel to `nozzle_angle_deg`. The local Mach
used by the dense BL march is obtained by inverting
$A/A^*=(2y)^2$ on the supersonic branch.

For a conical contour, `required_exit_area_ratio` and
`ideal_exit_area_ratio` store the area ratio used by the finished contour;
both are `None` for MOC. In coupled mixed-Mach iteration,
`ideal_exit_mach` is varied on every trial and the conical area ratio and
length are rebuilt from that trial Mach. Therefore the final area ratio
generally corresponds to `ideal_exit_mach`, not directly to the requested
mixed `exit_mach`.

### Throat gamma, area, and width

The constant gamma used by either supersonic contour is evaluated at the
actual choked static throat state, not at upstream total temperature.
Because the `Fluid` heat capacity changes with temperature, the object
iterates

$$
T^*=T_t\frac{2}{\gamma+1}, \qquad
p^*=p_t\left(\frac{2}{\gamma+1}\right)^{\gamma/(\gamma-1)}
$$

until the gamma returned by `fluid.properties(T*, p*)` is self-consistent.
That single value is then frozen for the calorically perfect contour and
mass-flow equations. CoolProp transport properties can still vary along the
boundary layer.

The total choked area follows the
[NASA Glenn choked mass-flow equation](https://www.grc.nasa.gov/www/k-12/airplane/rktthsum.html):

$$
\dot m =
\frac{A^*p_t}{\sqrt{T_t}}
\sqrt{\frac{\gamma}{R}}
\left(\frac{2}{\gamma+1}\right)^{
(\gamma+1)/(2(\gamma-1))}.
$$

For the rectangular MOC passage, the physical throat opening is

$$
A^*_\mathrm{total}=\frac{\dot m}{(\dot m/A^*)}, \qquad
w^*=\frac{A^*_\mathrm{total}}{N h}.
$$

Here `throat_height=h` is the out-of-plane span and `throat_width=w*` is the
opening of one passage. The MOC contour is normalized by throat half-width, so
every coordinate is multiplied by `w*/2`.

The conical option does not use `throat_height`. Its one-nozzle circular area
and throat diameter are

$$
A^*_\mathrm{one}=\frac{A^*_\mathrm{total}}{N},\qquad
D^*=\sqrt{\frac{4A^*_\mathrm{one}}{\pi}}.
$$

Because its coordinates are normalized directly by $D^*$, every conical
coordinate is multiplied by `throat_diameter`. The object stores
`single_nozzle_throat_area`, `throat_diameter`, `throat_radius`, and
`coordinate_scale_length`. It also stores the conical divergent length as
`conical_divergent_length_over_throat_diameter` and
`conical_divergent_length`. For MOC, `throat_diameter` is `None`; for
conical, `throat_width`, `throat_height`, and `throat_half_width_scale` are
`None`. Both modes use the resulting physical chord to calculate
`chord_reynolds_number`.

### Stator boundary layer and corrected straight

NASA TM X-2434 and NASA TM X-2343 use the same NACA Report 1294
Cohen–Reshotko laminar procedure and the Sasman–Cresci turbulent procedure
from *AIAA Journal* 4(1), 1966, doi:10.2514/3.3378. Both public classes call
the same boundary-layer solver. NASA TM X-2343 marches one symmetric nozzle
wall: the pressure-side exit is the end of the characteristic contour, while
the suction side continues through ten straight intervals. The conical mode
uses the same arrangement, with the pressure-side exit at the end of the
straight-wall divergent section. The Python class preserves that shared march
instead of calculating two unrelated wall solutions.

This class starts the boundary layer at the sharp throat. The converging
subsonic contour is not designed by NASA TM X-1502 or NASA TM X-2343. In
`"laminar_then_turbulent"` mode the throat layer begins at zero thickness. In
`"fully_turbulent"` mode the two dimensional throat thicknesses are required.
If a real design has known nonzero laminar throat thickness, it should be
obtained from a separate converging-section calculation before extending this
model.

Boundary-layer displacement is added in the nozzle-axis y direction, as in
the FORTRAN `NOZZLC` routine. The corrected suction straight is then extended
in the two `AFMIX` extrapolation steps so that the specified nozzle angle is
retained while displacement thickness continues to grow.

`trailing_edge_thickness` is the dimensional metal thickness in metres and
defaults to zero. It follows the NASA TM X-2343 input `TE` and participates in
both the corrected and uncorrected mixed-out calculations. With projected
exit pitch $X_X=SP\cos\alpha_1$, NASA TM X-2343 defines

$$
D_{TE}=\frac{TE}{X_X},\qquad
A=1-\delta^*/X_X-D_{TE}-\theta/X_X,\qquad
A_1=1-\delta^*/X_X-D_{TE}.
$$

The Python implementation uses these same `DTE`, `A`, and `A1` terms. The
physical input is divided by `coordinate_scale_length`: MOC uses throat
half-width, while conical uses throat diameter. The generic normalized value
is stored as `trailing_edge_thickness_over_coordinate_scale`; the
mode-specific property is
`trailing_edge_thickness_over_throat_half_width` or
`trailing_edge_thickness_over_throat_diameter`. Each mixing root reports its
actual `trailing_edge_blockage_ratio`.

Following NASA TM X-2343, `TE` is a mixing-loss/blockage input only: it does not
truncate or thicken the stored contour coordinates, and it does not alter the
choked throat-area calculation. Consequently, plotting still shows the sharp
pressure and suction contours. A geometrically resolved blunt trailing edge
would require a separate contour-construction choice.

### Stator mixing and outlet-angle iteration

Mixing is **not impossible** when the free-stream axial Mach before mixing is
subsonic. NASA TM X-2343 calculates the ordinary subsonic mixed solution in that
case. What is unavailable is the second, shockless supersonic root.

The condition used by the FORTRAN is

$$
M_{x,1}=M_e\cos(\alpha_1).
$$

When this value is below one, `mixing_results["subsonic"]` remains available
and `mixing_results["supersonic"]["available"]` is false. When it is at least
one, both mathematical roots can be requested; NASA TM X-2343 interprets the
subsonic root as mixing plus oblique-shock loss and the supersonic root as
shockless mixing.

With `iterate_nozzle_angle=False`, `nozzle_angle_deg` is set equal to the
requested `outlet_flow_angle_deg`. With `iterate_nozzle_angle=True`, every
trial rebuilds the straight section, recalculates its Reynolds number and
boundary layer, applies the corrected-pitch extension, and evaluates
aftermixing. The final nozzle-axis angle is the one whose selected mixed root
matches the requested flow direction.

With `match_exit_mach_after_mixing=True`, the iterative mode additionally
treats `exit_mach` as the desired absolute mixed Mach instead of the ideal
contour input. This flag requires `iterate_nozzle_angle=True`. A
coupled two-variable solve changes:

1. `ideal_exit_mach`, which is the supersonic Mach used to build the selected
   contour and evaluate the premixing state; and
2. `nozzle_angle_deg`.

Every coupled trial rebuilds the selected geometry, dimensional Reynolds scale,
dense BL march, corrected geometry, and selected aftermixing root. The two
residual equations are

$$
M_\mathrm{mixed}-M_\mathrm{requested}=0,\qquad
\alpha_\mathrm{mixed}-\alpha_\mathrm{requested}=0.
$$

When the constructor returns, `obtained_outlet_mach` and
`obtained_outlet_flow_angle_deg` meet those targets within the solver
tolerances. `ideal_exit_mach` records the generally different premixing Mach.
If the selected subsonic/supersonic mixing root cannot attain the requested
pair, `StatorDesignConvergenceError` is raised.

Stored coordinates remain in the unrotated nozzle-axis system. `plot()`
rotates both the corrected and uncorrected pressure/suction surfaces by
`nozzle_angle_deg` and labels the resulting axes as machine axial and
tangential directions.

## Geometry station counts

Both public designers accept `number_of_stations`, but it controls only the
minimum resolution of a temporary boundary-layer marching grid. The grid
builder retains every MOC vertex and inserts intermediate points along the
same piecewise-linear surface. For a conical contour, it retains the throat,
divergent exit, and ten straight-section stations while evaluating local Mach
from local area ratio at every inserted BL station. BL thicknesses predicted
on the dense grid are then interpolated back to the original stored stations
before the corrected geometry is created.

Consequently, changing `number_of_stations` does not change either stored
uncorrected geometry. Rotor corrected surfaces retain the corresponding MOC
stations. The stator corrected suction surface retains those stations plus
the two documented `AFMIX` exit-extrapolation stations. Projected results at
stored stations are available as `pressure_boundary_layer` and
`suction_boundary_layer`; the temporary high-resolution results are retained
separately as `pressure_boundary_layer_marching` and
`suction_boundary_layer_marching`.

The minimum input is 20 stations. If an MOC surface already has more stations,
all of them are used. Increasing `number_of_stations` improves only the BL
integration and correction prediction. Reduce `turning_increment_deg` when a
finer and more accurate stored characteristic contour is required.

## Important frame and scale conventions

NASA TN D-4421 and NASA TM X-2434 formulate the blade-to-blade problem in the **rotor-relative
frame**, but the rotor object's inlet API accepts stationary-frame quantities.
`inlet_mach`, `inlet_flow_angle_deg`, `inlet_total_temperature`, and
`inlet_total_pressure` are absolute-frame inputs. The angle is measured from
the positive axial direction toward the positive direction of rotation.

The object uses `rotational_speed_rpm` and `mean_radius` to calculate

$$
V_x=V\cos\alpha,\qquad V_\theta=V\sin\alpha,\qquad
U=\frac{2\pi r_m\,\mathrm{RPM}}{60},
$$

$$
W_x=V_x,\qquad W_\theta=V_\theta-U,\qquad
M_\mathrm{rel}=\frac{\sqrt{W_x^2+W_\theta^2}}{a}.
$$

The derived `relative_inlet_mach` and
`relative_inlet_flow_angle_deg` describe the far-field state. With zero
leading-edge thickness they are also the MOC passage-entry state. For a
positive `leading_edge_thickness_over_total_pitch`, the optional external-wave
correction described below produces `passage_inlet_mach` and
`passage_inlet_flow_angle_deg`; those passage values are supplied to the MOC
and BL calculations. Both relative states must remain supersonic and their
angles must lie in the NASA TN D-4421 0–90 degree design domain.

`outlet_flow_angle_deg` is also an **absolute-frame** input. NASA TN D-4421 still
requires a relative exit direction internally. In zero-deviation mode, the
object calculates the relative metal angle $\beta$ from the requested
absolute direction $\alpha$, relative exit speed $W$, and wheel speed
$U$:

$$
V_x=W\cos\beta,\qquad V_\theta=W\sin\beta+U.
$$

In iterative mode, every relative metal-angle trial is aftermixed in the
rotating frame and then transformed back to the absolute frame before its
residual is compared with `outlet_flow_angle_deg`. The optional
`outlet_mach` is an **absolute** Mach target. Normally it is the ideal value
before aftermixing. With `match_outlet_mach_after_mixing=True`, it is instead
the desired mixed value and the ideal relative Mach is included in the
iteration.

The nondimensional NASA TN D-4421 coordinates are divided by the vortex sonic radius
$r^*$; that is not the turbomachine mean radius. `BladeShape.inlet_pitch`
and `BladeShape.outlet_pitch` are open passage widths. Dimensionalization uses
the initialized blade count and treats the circumferential machine pitch as
the total pitch:

$$
g_\mathrm{total}=\frac{2\pi r_m}{Z},\qquad
G^*_\mathrm{total}=\frac{G^*_\mathrm{passage}}
{1-t_\mathrm{LE}/G^*_\mathrm{total}},\qquad
r^*=\frac{g_\mathrm{total}}{G^*_\mathrm{total}}.
$$

Boundary-layer thickness also needs a Reynolds scale. It is calculated from
the dimensional ideal chord and derived relative inlet speed. The code
reconstructs relative total temperature and pressure from the common static
state and relative Mach because those are required by the rotor BL equations.
The old air gas constant, air viscosity coefficients, user-entered Reynolds
number, unit-system, print, and card-format inputs are no longer exposed. The
wall temperature is set to relative total temperature, as in the NASA TM X-2434
rotor driver.

## Minimal use

The following operating point produces a symmetric uncorrected rotor blade at
50 bar inlet total pressure, 30,000 rpm, and 0.15 m mean radius. The absolute
outlet angle is the velocity-triangle result that gives equal-and-opposite
rotor-relative passage-entry and outlet angles. The explicit absolute outlet
Mach transforms to the passage-entry relative Mach after the finite-thickness
entry correction. Its premixing rotor-relative axial Mach number is
approximately 0.966, so the selected axial state is subsonic.

```python
from SupersonicTurbineBlading import Fluid, SupersonicRotorBlade

fluid = Fluid(coolprop_names=["Nitrogen", "Oxygen"], mass_fractions=[0.767, 0.233])

blade = SupersonicRotorBlade(
    inlet_mach=2.80,
    inlet_flow_angle_deg=70.0,
    outlet_flow_angle_deg=-12.049849029059851,
    lower_surface_mach=1.3,
    upper_surface_mach=2.8,
    blade_count=36,
    mean_radius=0.15,
    rotational_speed_rpm=30000.0,
    fluid=fluid,
    inlet_total_temperature=1000.0,
    inlet_total_pressure=5.0e6,
    number_of_stations=101,
    iterate_outlet_blade_angle=False,
    iterate_pitch_closure=False,
    leading_edge_thickness_over_total_pitch=0.07,
    use_leading_edge_entry_correction=True,
    calculate_starting=True,
    boundary_layer_mode="fully_turbulent",
    initial_turbulent_displacement_thickness=1.0e-5,
    initial_turbulent_momentum_thickness=2.5e-6,
    mixing_solution="subsonic",
    outlet_mach=0.9875119383104759,
)

# Coordinates normalized by r*
ideal = blade.uncorrected_shape
corrected = blade.corrected_shape

# Coordinates in metres
dimensional = blade.dimensionalize()

# BL-corrected single-blade profile in millimetres for CAD import
blade_x_mm = blade.blade_profile_x_CAD
blade_y_mm = blade.blade_profile_y_CAD

# Corresponding ideal, uncorrected profile in millimetres
ideal_blade_x_mm = blade.uncorrected_blade_profile_x_CAD
ideal_blade_y_mm = blade.uncorrected_blade_profile_y_CAD

# Two complete adjacent blades and their passage are plotted by default.
# With positive t_LE, only the two outer plotted surfaces are displaced.
blade.plot(dimensional=True, corrected=True)  # BL-corrected geometry
blade.plot(dimensional=True, corrected=False)  # ideal geometry

# Set this false for only the original passage boundaries.
blade.plot(dimensional=True, corrected=True, show_two_blades=False)
```

Dimensional rotor plots use millimetres; `dimensionalize()` and the stored
dimensional shapes continue to use metres.

## Finite leading- and trailing-edge thickness

`leading_edge_thickness_over_total_pitch` is
$t_\mathrm{LE}/G^*_\mathrm{total}$ and defaults to zero. The MOC geometry's
calculated inlet pitch is the open passage width
$G^*_\mathrm{passage}$, so the nondimensional metal thickness is

$$
G^*_\mathrm{total}=\frac{G^*_\mathrm{passage}}{1-\tau},\qquad
t^*_\mathrm{LE}=\frac{\tau}{1-\tau}G^*_\mathrm{passage},\qquad
\tau=\frac{t_\mathrm{LE}}{G^*_\mathrm{total}}.
$$

When `use_leading_edge_entry_correction=True` (the default) and
$\tau>0$, the NACA RM L52B06 external-wave method transforms the
far-field rotor-relative state $(M_i,\beta_i)$ into the passage-entry state
$(M_e,\beta_e)$. It solves the isentropic continuity equation

$$
\frac{A_e}{A_i}
=(1-\tau)\frac{\cos\beta_e}{\cos\beta_i}
=\frac{(A/A^*)_e}{(A/A^*)_i}
$$

together with the Prandtl–Meyer turning relation. The NACA RM L52B06 signed inlet
direction is opposite to this API's positive inlet-angle convention, so the
implemented magnitude relation is

$$
\beta_e-\beta_i=\nu_e-\nu_i.
$$

The weak-wave root nearest $M_i$ is used. The method can still be requested
when the far-field rotor-relative axial Mach is supersonic, but the code emits
a `RuntimeWarning` because NACA RM L52B06 derives the construction for subsonic
axial inflow. Setting `use_leading_edge_entry_correction=False` leaves the MOC
and BL inlet state equal to the far-field state while retaining the requested
metal thickness and pitch definitions.

The BL displacement correction itself is unchanged. With legacy pitch closure,
$t^*_\mathrm{TE}=t^*_\mathrm{LE}$. Without pitch closure, the trailing edge
is the leading-edge thickness minus the sum of the two **vertical** trailing-
edge BL displacement heights, limited to the physically meaningful minimum of
zero. This $t^*_\mathrm{TE}$ is included as `DTE=TE/XX` in the NASA TM X-2434
`AFMIX` equations. The values and scales are available as
`leading_edge_thickness`, `trailing_edge_thickness`,
`physical_leading_edge_thickness`, `physical_trailing_edge_thickness`,
`inlet_passage_pitch`, `inlet_total_pitch`, `physical_passage_pitch`, and
`physical_total_pitch`.

The `blade_profile_x_CAD` and `blade_profile_y_CAD` arrays contain the
BL-corrected profile of one blade in millimetres. The lower/pressure surface
runs from leading edge to trailing edge, followed by the translated
upper/suction surface from trailing edge back toward the leading edge. The
lower-surface leading edge is the first point at `(0, 0)` and is not appended
again for explicit closure. With zero thickness, the coincident upper-surface
leading-edge point is omitted so `(0, 0)` occurs only once. With finite leading-
edge thickness, the last point is the upper-surface leading edge displaced by
`physical_leading_edge_thickness` in the positive y direction before conversion
to millimetres. The `uncorrected_blade_profile_x_CAD` and
`uncorrected_blade_profile_y_CAD` properties provide the ideal profile before
BL displacement correction, using the same millimetre scale, point order, and
origin convention.

The central passage surfaces in `plot()` are unchanged. For two-blade plots,
the upper surface of the top blade is translated by $+t_\mathrm{LE}$ in
$y$, and the lower surface of the bottom blade by $-t_\mathrm{LE}$. A
straight line joins the upper and lower leading-edge endpoints of each blade,
closing the finite-thickness profile without crossing the flow passage. The
plot also joins the trailing-edge endpoints of each upper and lower blade with
separate lines, whether or not legacy pitch closure is enabled. Leading- and
trailing-edge closure lines are solid for BL-corrected geometry and dashed for
uncorrected geometry, matching their respective blade-surface styles.

## Physical chord and Reynolds number

Boundary-layer correction is performed only after the uncorrected geometry is
available. For every ideal-geometry trial, the code calculates

$$
g_\mathrm{total}=\frac{2\pi r_m}{Z},\qquad
r^*=\frac{g_\mathrm{total}}{G^*_\mathrm{total}},\qquad
c=C^*r^*.
$$

It then uses the passage-entry static mixture state to calculate

$$
W_\mathrm{entry}=M_\mathrm{entry}a_\mathrm{entry}, \qquad
Re_c=\frac{W_\mathrm{entry}c}{\nu_\mathrm{entry}}.
$$

The resulting values are stored as `physical_total_pitch` (`physical_pitch` is
a backward-compatible alias), `physical_passage_pitch`, `sonic_radius_scale`,
`physical_chord`, and `chord_reynolds_number`. In iterative outlet-angle mode,
physical chord and Reynolds number are recalculated for every trial geometry.
This is equivalent to NASA TM X-2434 supplying physical chord `XMAX`, but prevents
an independently entered Reynolds number from being inconsistent with the
eventual dimensionalized blade.

## Fluid model and mixture assumptions

`Fluid` is initialized only with composition. Its `properties(temperature,
pressure)` method returns an immutable `FluidState` in SI units. For every
requested state, each named component is queried separately in CoolProp at
the common temperature and its Dalton partial pressure. CoolProp is therefore
never asked to flash the mixture.

The following mixing rules are explicit:

- mixture molar mass:
  $1/M_\mathrm{mix}=\sum_i w_i/M_i$;
- specific gas constant:
  $R_\mathrm{mix}=R_u/M_\mathrm{mix}$;
- density:
  $\rho=p/(R_\mathrm{mix}T)$;
- ideal-gas constant-pressure heat capacity:
  $c_p=\sum_i w_i c_{p,i}^{0}$, using CoolProp `CP0MASS`;
- dynamic viscosity:
  $\mu=\sum_i w_i\mu_i$;
- thermal conductivity:
  $k=\sum_i w_i k_i$.

The conductivity average is the additional mixing rule needed by the rotor:
it permits $\mathrm{Pr}=c_p\mu/k$ to be calculated rather than supplied.
The code then derives $c_v=c_p-R_\mathrm{mix}$,
$\gamma=c_p/c_v$, $a=\sqrt{\gamma R_\mathrm{mix}T}$, and
$\nu=\mu/\rho$.

The mass-weighted viscosity and conductivity rules are intentionally simple
preliminary-design approximations. They are less accurate than kinetic-theory
mixture laws, especially for components with very different molecular
masses. The model also rejects a component if CoolProp identifies its
partial-pressure state as liquid or two-phase; this blade method is an
ideal-gas method, not an arbitrary-phase fluid solver.

The NASA TN D-4421 geometry/starting and NASA TM X-2434 aftermixing relations require a constant
$\gamma$. The implementation freezes it at the actual inlet static state,
not at total temperature. Because both static temperature and
$\gamma(T)$ appear in the total-to-static relation, the code iterates

$$
T_\mathrm{in} =
\frac{T_{t,\mathrm{abs}}}
{1+\frac{\gamma(T_\mathrm{in})-1}{2}M_\mathrm{abs}^2}
$$

until temperature and gamma are self-consistent. The relative total state is
then derived from this common static state and the calculated relative Mach.
CoolProp transport properties remain temperature-dependent along both
surfaces.

With `match_outlet_mach_after_mixing=False` (the default), `outlet_mach` is
the uniform **absolute** Mach at the ideal blade exit before aftermixing. At
fixed mean radius, the code conserves relative total temperature (the
constant-radius rothalpy relation) and uses the velocity triangle to calculate
the corresponding relative Mach required by NASA TN D-4421. It consequently
determines:

- how both surface transition arcs return to uniform outlet flow;
- the exit surface Mach distribution and part of the resulting chord/pitch;
- the relative Mach and direction used by the NASA TN D-4421 construction; and
- the premixing state supplied to the NASA TM X-2434 `AFMIX` equations.

In non-iterative zero-deviation mode, the specified absolute Mach and
`outlet_flow_angle_deg` define one complete ideal absolute exit velocity
triangle. In iterative mode, the absolute Mach remains fixed while each trial
relative metal angle produces a new relative Mach and ideal absolute flow
direction; aftermixing determines whether the requested absolute mixed-flow
angle has been reached.

The NASA TM X-2434 example is an impulse rotor. If `outlet_mach=None`, the internal
NASA TN D-4421 value defaults to
$M_{\mathrm{rel,out}}=M_{\mathrm{rel,in}}$, and the corresponding ideal
absolute outlet Mach is calculated and stored as `outlet_mach`. Thus the
omission is an explicitly documented relative impulse assumption, not a
change in the reference frame of a supplied input.

With `match_outlet_mach_after_mixing=True`, `outlet_mach` changes meaning from
an ideal-state specification to the requested absolute Mach of the selected
aftermixing root. This option requires both a supplied `outlet_mach` and
`iterate_outlet_blade_angle=True`. A coupled two-variable iteration then
changes:

1. relative ideal outlet Mach used by the NASA TN D-4421 construction; and
2. outlet metal/relative-flow angle.

Each trial rebuilds the ideal blade, physical scale, both dense BL marches,
corrected shape, and aftermixing result. A damped two-variable Newton solve
drives

$$
M_{\mathrm{mixed,abs}}-M_{\mathrm{target,abs}}=0,\qquad
\alpha_{\mathrm{mixed,abs}}-\alpha_{\mathrm{target,abs}}=0.
$$

If no physical pair exists within the selected surface-Mach interval, the
constructor raises `DesignConvergenceError` rather than modifying either
requested target.

`iterate_pitch_closure=True` selects the NASA TM X-2434 closure
iteration. It holds the ideal relative outlet Mach fixed and changes the
relative outlet angle until the BL-corrected outlet pitch equals the ideal
inlet pitch. The supplied `outlet_flow_angle_deg` is therefore only the
initial estimate, and construction emits a warning that the requested outlet
angle will change. This option is incompatible with
`iterate_outlet_blade_angle=True` or
`match_outlet_mach_after_mixing=True`; selecting either combination raises a
`ValueError`.

## Rotor surface Mach design inputs

`lower_surface_mach` and `upper_surface_mach` are rotor-relative surface
values, not alternative descriptions of the inlet or outlet meanline state.
They define the constant-Mach pressure- and suction-side arcs used by the
NASA TN D-4421 construction and thereby control the internal loading distribution,
surface acceleration/deceleration, thickness, and solidity of the resulting
blade. The first required ordering is

$$
1 \le M_\mathrm{lower}
\le \min(M_{\mathrm{rel,in}},M_{\mathrm{rel,out}}),\qquad
M_\mathrm{upper}
\ge \max(M_{\mathrm{rel,in}},M_{\mathrm{rel,out}}).
$$

These inequalities and the tighter angle-dependent bounds are obtained from
the NASA TN D-4421 blade description and equations (6a–b) and (7a–b). With
Prandtl–Meyer angles $\nu$, relative inlet angle $\beta_i>0$, and relative
outlet metal/flow angle $\beta_o<0$, the complete geometric intervals are

$$
\max(0,\nu_i-\beta_i,\nu_o-|\beta_o|)
\le \nu_l \le \min(\nu_i,\nu_o),
$$

$$
\max(\nu_i,\nu_o)
\le \nu_u \le
\min(\nu_i+\beta_i,\nu_o+|\beta_o|).
$$

Mach is monotonic in $\nu$, so the code converts these intervals to numeric
Mach limits using the frozen inlet-static gamma. It raises `ValueError` before
the final geometry and BL calculations when either supplied surface Mach is
outside its interval. Finite-value checks are also performed. During outlet-
angle iteration, inlet-side and broad physical limits are checked first, and
the complete interval is checked again for the converged relative metal angle.

For a preliminary design, choose the lower value from the acceptable
pressure-side deceleration/loading and choose the upper value from the
allowable suction-side peak Mach, shock strength, and boundary-layer risk.
The NASA TN D-4421 values are design variables and normally require a sweep rather
than a single universal rule.

Absolute inlet/outlet Mach numbers, velocities, and angles cannot uniquely
eliminate these two inputs. Together with RPM they determine the relative
meanline states at the two ends, but infinitely many surface velocity
distributions can connect the same end states. A future inverse-design API
could solve for the two surface Mach numbers from two independent blade
targets—for example target solidity plus allowable peak suction Mach, or
target loading plus minimum thickness—but those targets would replace, not
derive redundantly from, the surface Mach inputs. The current implementation
therefore leaves both values explicit.

## Outlet-angle flag

With `iterate_outlet_blade_angle=False`, the requested absolute direction is
first transformed through the exit velocity triangle. The resulting relative
direction is used as the ideal exit metal angle. This is the zero-relative-
deviation assumption:

```text
relative outlet blade metal angle = relative inviscid outlet flow angle
```

With `iterate_outlet_blade_angle=True`, the exit metal angle is varied until
the selected NASA TM X-2434 aftermixing solution has the specified **absolute**
flow angle.
`mixing_solution="subsonic"` (the default) selects the subsonic-axial root;
`"supersonic"` explicitly selects shockless supersonic-axial mixing.
The selected root is also the root used by angle and Mach matching. The
supersonic root is available only when the premixing axial Mach number is at
least one; selecting it for subsonic axial inflow raises
`DesignConvergenceError`. Use `mixing_solution="subsonic"` in that case.

The following diagnostics are kept:

- `relative_inlet_mach` and `relative_inlet_flow_angle_deg` (far field)
- `passage_inlet_mach` and `passage_inlet_flow_angle_deg` (MOC/BL entry)
- `outlet_blade_angle_deg` (relative metal/flow direction)
- `ideal_outlet_mach` (absolute, before aftermixing)
- `relative_outlet_mach` (ideal NASA TN D-4421 construction value)
- `obtained_outlet_flow_angle_deg` and `obtained_outlet_mach` (absolute)
- `obtained_relative_outlet_flow_angle_deg`
- `obtained_relative_outlet_mach`
- `mixing_results["supersonic"]` and `mixing_results["subsonic"]`
- `premixing_axial_mach` and `supersonic_mixing_available`
- `pitch_closure_outlet_angle_deg` and `pitch_closure_iteration_count`
- `corrected_pitch_residual` and `pitch_closure_residual`
- `pitch_residual`
- leading-/trailing-edge thickness and total-/passage-pitch properties listed
  in the finite-thickness section

Normally, `pitch_residual` is corrected outlet pitch minus corrected inlet
pitch in units of $r^*$, identical to `corrected_pitch_residual`. It should
be reviewed because a specified exit angle, a specified/impulse outlet Mach,
and exact periodic pitch are not generally three independent conditions.
With legacy pitch closure enabled, `pitch_residual` instead aliases
`pitch_closure_residual`: corrected outlet pitch minus ideal inlet pitch, the
quantity closed by NASA TM X-2434.

## Boundary-layer choices

Both modes start at the MOC passage inlet:

- `"laminar_then_turbulent"` ports the Cohen–Reshotko laminar correlation and
  its instability/transition test. Imminent laminar separation is treated as
  immediate turbulent reattachment with conserved momentum thickness
  (`CTHET=1` in NASA TM X-2434), avoiding another user input.
- `"fully_turbulent"` forces turbulence at the inlet and requires
  `initial_turbulent_displacement_thickness` and
  `initial_turbulent_momentum_thickness`. These are dimensional physical
  compressible integral thicknesses $\delta^*$ and $\theta$ in metres,
  and the same inlet values are applied to both blade surfaces. Each trial
  divides them by its calculated ideal chord before solving the boundary
  layer. The code also transforms their ratio to the incompressible form
  factor required by the Sasman–Cresci equations.

The laminar Cohen–Reshotko table is generated to the application-specific
`CORLN` limit: 0.50 for rotor surfaces and 0.16 for the stator nozzle. If a
surface march extends beyond that limit, the solver emits a `RuntimeWarning`
and continues by extrapolating the correlation table.

For example:

```python
blade = SupersonicRotorBlade(
    # ...geometry, fluid, and total-state inputs...
    boundary_layer_mode="fully_turbulent",
    initial_turbulent_displacement_thickness=2.4e-4,
    initial_turbulent_momentum_thickness=6.0e-5,
)
```

Both values must be omitted in `"laminar_then_turbulent"` mode. NASA TM X-2434's
rotor driver prohibited starting turbulence at station 1 precisely because it
did not have these initial values; the underlying boundary-layer formulation
does accept them.

Results are available separately as `pressure_boundary_layer` and
`suction_boundary_layer`. Each contains displacement and momentum thickness
normalized by chord, form factor, regime at every surface station, transition
index, and separation index.

The integral methods are preliminary-design correlations. Curvature,
shock/boundary-layer interaction, three-dimensional effects, tip clearance,
and real-gas effects are outside their model. A final rotor needs CFD and
experimental validation.

## Why the optional starting calculation exists

The NASA TN D-4421 starting calculation is a feasibility screen; it does not change
the blade coordinates. It assumes that a normal shock spans the cascade inlet
during startup. The code:

1. integrates the maximum mass flow that the vortex passage can pass after
   the normal-shock total-pressure loss;
2. finds the critical vortex constant that maximizes this swallowed flow;
3. equates it to the design supersonic mass flow; and
4. returns the maximum inlet Mach (and Prandtl–Meyer angle) that can start for
   the selected surface Mach numbers.

The result is stored in `starting_result`; set `calculate_starting=False` to
skip it.

This criterion is **not simply applicable to every rotor whose resultant
relative Mach exceeds one**. Starting is governed by the component normal to
the blade-row inlet plane. In an axial machine this is normally discussed as
the relative axial Mach number $M_x=W_x/a$. If $M_x>1$, the row cannot
communicate upstream and a passage-spanning shock/swallowing calculation is
relevant. If the resultant relative Mach is supersonic mainly because of
wheel speed while $M_x<1$, the NASA TN D-4421 normal-shock model is usually
over-conservative or inapplicable; oblique leading-edge shocks and the
unsteady rotor flow need a different starting analysis. Thus it is not
restricted by the machine label “axial,” but by the normal/throughflow Mach
and the NASA TN D-4421 shock model.

## Main inputs retained

| Input | Reason it remains |
|---|---|
| `inlet_mach` | Absolute inlet Mach used to reconstruct static state and velocity |
| `inlet_flow_angle_deg` | Absolute inlet direction measured from the axial direction |
| `rotational_speed_rpm` | Combines with mean radius to form the relative velocity triangle |
| `outlet_flow_angle_deg` | Requested absolute aftermixed exit direction |
| `outlet_mach` | Absolute ideal Mach by default, or absolute mixed-Mach target when `match_outlet_mach_after_mixing=True` |
| `lower_surface_mach` | Sets pressure-surface vortex radius and Mach distribution |
| `upper_surface_mach` | Sets suction-surface vortex radius and Mach distribution |
| `blade_count` | Relates mean radius to physical pitch |
| `mean_radius` | Sets wheel speed, physical pitch, chord, and Reynolds scale |
| `fluid` | Supplies composition, gas constant, heat capacity, and transport properties |
| `inlet_total_temperature`, `inlet_total_pressure` | Define the absolute total inlet state |
| Initial turbulent $\delta^*$, $\theta$ | Dimensional metres; required only for a fully turbulent inlet |
| `turning_increment_deg` | Characteristic-network resolution |
| `number_of_stations` | Minimum temporary BL marching resolution; stored stations remain MOC-defined |
| `iterate_pitch_closure` | Enables the legacy rotor pitch-closure iteration; incompatible with mixed-flow angle or Mach matching |
| `leading_edge_thickness_over_total_pitch` | Specifies $t_\mathrm{LE}/G^*_{\mathrm{total}}$; zero retains the sharp-edge design |
| `use_leading_edge_entry_correction` | Applies the finite-thickness external-wave transformation before MOC/BL construction |

Prandtl–Meyer angles, $M^*$, user-entered gamma and Prandtl number, gas
constant, air viscosity coefficients, unit switch, print switches, plot
switches, and rotated/unrotated-output switches are derived, assumed, or
removed.

## License

SupersonicTurbineBlading is licensed under the GNU General Public License,
version 3 only (`GPL-3.0-only`). See [LICENSE](LICENSE) for the complete license
terms.
