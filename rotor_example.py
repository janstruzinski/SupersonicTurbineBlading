"""Small executable example of the object-oriented rotor designer.

The fluid composition is independent of the operating point. Stationary-frame
total conditions, absolute inlet velocity, RPM, and mean radius define the
relative inlet state used by NASA TN D-4421 and NASA TM X-2434.
"""

from SupersonicTurbineBlading import Fluid, SupersonicRotorBlade

# Dry-air-like mass composition.  Each component is queried separately in
# CoolProp; the code never asks CoolProp to flash the binary mixture.
working_fluid = Fluid(coolprop_names=["Nitrogen", "Oxygen"], mass_fractions=[0.767, 0.233])

blade = SupersonicRotorBlade(
    inlet_mach=2.80,  # absolute Mach in the stationary frame
    inlet_flow_angle_deg=70.0,  # absolute angle from the machine axis
    # These ideal absolute outlet conditions transform to the same Mach and
    # opposite flow angle as the finite-thickness-corrected passage-entry state.
    # The resulting uncorrected NASA TN D-4421 blade is therefore symmetric.
    outlet_flow_angle_deg=-12.049849029059851,
    lower_surface_mach=1,
    upper_surface_mach=4,
    blade_count=80,
    mean_radius=0.15,  # [m], also sets physical chord and Reynolds number
    rotational_speed_rpm=30000.0,
    fluid=working_fluid,
    inlet_total_temperature=1000.0,  # absolute total temperature [K]
    inlet_total_pressure=5.0e6,  # absolute total pressure [Pa] = 50 bar
    # Optional controls are shown explicitly below. Their defaults are noted
    # so an example can also serve as a compact input reference.
    turning_increment_deg=0.1,  # default: 0.1; controls the MOC geometry
    number_of_stations=121,  # default: 101; temporary BL march only
    iterate_outlet_blade_angle=False,  # preserve the specified ideal outlet state
    iterate_pitch_closure=False,  # BL pitch closure would change the symmetric ideal angle
    # default: False. True makes outlet_mach the desired AFTERMIX result and
    # jointly iterates ideal relative outlet Mach and blade angle.
    match_outlet_mach_after_mixing=False,
    # Ratio t_LE/G*_total. Zero retains the original sharp-edge passage.
    leading_edge_thickness_over_total_pitch=0.07,  # default: 0.0
    # With positive thickness, transform far-field relative flow to the MOC
    # passage-entry Mach and angle using NACA RM L52B06.
    use_leading_edge_entry_correction=True,  # default: True
    calculate_starting=True,  # default: True
    # default: "laminar_then_turbulent", with no initial thickness inputs
    boundary_layer_mode="fully_turbulent",
    initial_turbulent_displacement_thickness=2.0e-5,  # [m]
    initial_turbulent_momentum_thickness=5e-6,  # [m]
    mixing_solution="subsonic",  # force subsonic root; omit for automatic selection
    # With match_outlet_mach_after_mixing=False, this is the ideal absolute
    # Mach before mixing. When that option is True, this is the desired
    # absolute Mach after mixing.
    outlet_mach=0.9875119383104759,
)

print(f"Outlet blade angle (relative-flow convention): {blade.outlet_blade_angle_deg:.3f} deg")
print(f"Relative inlet Mach: {blade.relative_inlet_mach:.3f}")
print(f"Passage-entry Mach: {blade.passage_inlet_mach:.3f}")
print(f"Ideal absolute outlet Mach: {blade.outlet_mach:.3f}")
print(f"Ideal relative outlet Mach: {blade.relative_outlet_mach:.3f}")
print(f"Premixing rotor-relative axial Mach: {blade.premixing_axial_mach:.3f}")
print(f"Relative inlet angle: {blade.relative_inlet_flow_angle_deg:.3f} deg")
print(f"Passage-entry angle: {blade.passage_inlet_flow_angle_deg:.3f} deg")
print(f"Absolute mixed outlet angle: {blade.obtained_outlet_flow_angle_deg:.3f} deg")
print(f"Relative mixed outlet angle: {blade.obtained_relative_outlet_flow_angle_deg:.3f} deg")
print(f"Frozen inlet-static gamma: {blade.gamma:.5f}")
print(f"Inlet static Prandtl number: {blade.prandtl_number:.5f}")
print(f"Calculated chord Reynolds number: {blade.chord_reynolds_number:.3e}")
print(f"Nondimensional chord C*: {blade.uncorrected_shape.chord:.5f}")
print(
    "Leading/ corrected trailing-edge thicknesses [m]: "
    f"{blade.physical_leading_edge_thickness:.6g}, "
    f"{blade.physical_trailing_edge_thickness:.6g}"
)
print(f"Corrected pitch residual: {blade.pitch_residual:.6f}")
if blade.starting_result is not None:
    print(f"Maximum starting inlet Mach: {blade.starting_result.maximum_starting_inlet_mach:.3f}")

blade.dimensionalize()
# CAD-ready BL-corrected single-blade profile [mm]; the origin occurs once.
blade_x_mm = blade.blade_profile_x_CAD
blade_y_mm = blade.blade_profile_y_CAD
# The corresponding ideal profile uses the same point order and units.
ideal_blade_x_mm = blade.uncorrected_blade_profile_x_CAD
ideal_blade_y_mm = blade.uncorrected_blade_profile_y_CAD
blade.plot(
    dimensional=True,  # default: False; dimensional plot axes are in mm
    corrected=False,  # default: True; False selects the uncorrected shape
    # True cross-pairs the passage boundaries into two complete blades and
    # shifts only their outer surfaces by the leading-edge thickness.
    show_two_blades=True,  # default: True
    ax=None,  # default: None; pass an existing Matplotlib axes if desired
    show=True,  # default: True
)
