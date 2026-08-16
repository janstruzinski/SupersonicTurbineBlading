"""Executable example for :class:`SupersonicStatorNozzle`.

The object is given the operating point and discrete passage arrangement.
It calculates the choked throat area and either rectangular throat width or
circular throat diameter before dimensionalizing the selected supersonic
contour. Nothing in this example is an old FORTRAN print-control, unit-system,
Prandtl--Meyer-angle, or gas-property input; those quantities are derived
internally.
"""

from SupersonicTurbineBlading import Fluid, SupersonicStatorNozzle

# The same composition object can be reused by rotor and stator designs.
# Each pure component is queried separately, while mixture density follows
# the ideal-gas equation and transport/Cp properties are mass averaged.
working_fluid = Fluid(coolprop_names=["Nitrogen", "Oxygen"], mass_fractions=[0.767, 0.233])

# ---------------------------------------------------------------------------
# Method-of-characteristics stator nozzle
# ---------------------------------------------------------------------------
moc_stator = SupersonicStatorNozzle(requested_outlet_absolute_flow_mach=1.77,
    requested_outlet_absolute_flow_angle=70.0,  # requested absolute outlet flow angle
    mass_flow_rate=5.0,  # total flow through the complete stator [kg/s]
    nozzle_count=30,
    throat_height=0.05,  # out-of-plane blade span at the throat [m]
    fluid=working_fluid,
    upstream_total_temperature=900.0,  # [K]
    upstream_total_pressure=1.0e6,  # [Pa]
    # Physical metal thickness used by the NASA TM X-2343 AFMIX blockage terms.
    # It changes mixed-out conditions, not the stored sharp-profile contour.
    trailing_edge_thickness=1.0e-4,  # [m]; default: 0.0
    # Choose exactly one contour method. For a conical nozzle, use
    # contour_method="conical", omit throat_height, and provide
    # half_cone_metal_angle (for example 15).
    contour_method="moc",  # default: "moc"
    number_of_nodes=101,  # default: 101; nodes on each nozzle segment and BL mesh
    # False is the zero-deviation assumption. Set True to vary the outlet metal
    # angle until corrected aftermixing reaches the requested absolute flow angle.
    iterate_outlet_metal_angle=False,  # default: False
    # Default: False. True jointly iterates ideal flow Mach and outlet metal
    # angle until aftermixing reaches both requested absolute flow targets.
    match_real_outlet_absolute_flow_mach=False,
    # This example starts a known turbulent layer at the throat.  In
    # laminar_then_turbulent mode both thickness inputs must be omitted.
    boundary_layer_mode="fully_turbulent",  # default: laminar then turbulent
    initial_turbulent_displacement_thickness=2.0e-5,  # [m]
    initial_turbulent_momentum_thickness=5.0e-6,  # [m]
    # Axial Mach is subsonic in this highly turned example, so only the
    # ordinary subsonic mixed solution is available.
    mixing_solution="subsonic")  # force subsonic solution; omit for automatic selection

print("\nMOC stator nozzle")
print(f"Throat-static gamma: {moc_stator.gamma:.5f}")
print(f"Total required throat area: {moc_stator.total_throat_area:.6e} m^2")
print(f"One-passage throat width: {moc_stator.throat_width:.6e} m")
print(f"Contour method: {moc_stator.contour_method}")
if moc_stator.required_exit_area_ratio is not None:
    print(f"Ideal exit area ratio: {moc_stator.required_exit_area_ratio:.5f}")
print(f"Outlet metal angle: {moc_stator.outlet_metal_angle:.3f} deg")
print(f"Ideal absolute outlet flow Mach: {moc_stator.ideal_outlet_absolute_flow_mach:.3f}")
print(f"Ideal absolute outlet flow angle: {moc_stator.ideal_outlet_absolute_flow_angle:.3f} deg")
print(f"Real absolute outlet flow angle: {moc_stator.real_outlet_absolute_flow_angle:.3f} deg")
print(f"Ideal absolute outlet axial flow Mach: {moc_stator.ideal_outlet_absolute_axial_flow_mach:.3f}")
print(f"Shockless supersonic mixing solution available: {moc_stator.supersonic_mixing_available}")
print(f"Calculated chord Reynolds number: {moc_stator.chord_reynolds_number:.3e}")

# Rotation into axial/tangential turbine coordinates occurs only for plotting;
# stored geometry stays in the simpler nozzle-axis coordinate system.
moc_stator.plot(dimensional=True,  # default: False; dimensional plot axes are in mm
    ax=None,  # default: None; pass an existing Matplotlib axes if desired
    show=True)  # default: True


# ---------------------------------------------------------------------------
# Axisymmetric conical de Laval stator nozzle
# ---------------------------------------------------------------------------
# This case uses the same operating point and fluid so that its circular
# throat and contour can be compared directly with the rectangular MOC case.
# throat_height is intentionally absent because it is not part of the conical input set.
conical_stator = SupersonicStatorNozzle(requested_outlet_absolute_flow_mach=1.77,
    requested_outlet_absolute_flow_angle=70.0,  # requested absolute outlet flow angle
    mass_flow_rate=5.0,  # total flow through all circular nozzles [kg/s]
    nozzle_count=30,
    fluid=working_fluid,
    upstream_total_temperature=900.0,  # [K]
    upstream_total_pressure=1.0e6,  # [Pa]
    trailing_edge_thickness=1.0e-4,  # [m]; default: 0.0
    contour_method="conical",
    half_cone_metal_angle=15.0,  # required conical divergent half-angle
    number_of_nodes=101,  # nodes on each divergent/straight segment and BL mesh
    iterate_outlet_metal_angle=False,
    match_real_outlet_absolute_flow_mach=False,
    boundary_layer_mode="fully_turbulent",
    initial_turbulent_displacement_thickness=2.0e-5,  # [m]
    initial_turbulent_momentum_thickness=5.0e-6,  # [m]
    mixing_solution="subsonic")  # force subsonic solution; omit for automatic selection

print("\nConical de Laval stator nozzle")
print(f"Throat-static gamma: {conical_stator.gamma:.5f}")
print(f"Total required throat area: {conical_stator.total_throat_area:.6e} m^2")
print(f"One-nozzle throat area: {conical_stator.single_nozzle_throat_area:.6e} m^2")
print(f"One-nozzle throat diameter: {conical_stator.throat_diameter:.6e} m")
print(f"Contour method: {conical_stator.contour_method}")
print(f"Ideal exit area ratio: {conical_stator.required_exit_area_ratio:.5f}")
print(f"Conical divergent length: {conical_stator.conical_divergent_length:.6e} m")
print(f"Outlet metal angle: {conical_stator.outlet_metal_angle:.3f} deg")
print(f"Ideal absolute outlet flow Mach: {conical_stator.ideal_outlet_absolute_flow_mach:.3f}")
print(f"Ideal absolute outlet flow angle: {conical_stator.ideal_outlet_absolute_flow_angle:.3f} deg")
print(f"Real absolute outlet flow angle: {conical_stator.real_outlet_absolute_flow_angle:.3f} deg")
print(f"Ideal absolute outlet axial flow Mach: {conical_stator.ideal_outlet_absolute_axial_flow_mach:.3f}")
print(f"Shockless supersonic mixing solution available: {conical_stator.supersonic_mixing_available}")
print(f"Calculated chord Reynolds number: {conical_stator.chord_reynolds_number:.3e}")

# The same plot method rotates the meridional conical contour by the stator
# metal angle and overlays its corrected and uncorrected shapes.
conical_stator.plot(dimensional=True, ax=None, show=True)
