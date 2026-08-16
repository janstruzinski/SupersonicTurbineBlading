"""Small executable example of the object-oriented rotor designer.

The fluid composition is independent of the operating point. Stationary-frame
total conditions, absolute inlet velocity, RPM, and mean radius define the
relative inlet state used by FORTRAN codes in NASA TN D-4421 and NASA TM X-2434.
The complete inlet/outlet flow input set can alternatively be supplied in the
rotor-relative frame used by the NASA reports.
"""

from SupersonicTurbineBlading import Fluid, SupersonicRotorBlade

# Dry-air-like mass composition.  Each component is queried separately in
# CoolProp; the code never asks CoolProp to flash the binary mixture.
working_fluid = Fluid(coolprop_names=["Water", "Oxygen"], mass_fractions=[0.54, 0.46])

blade = SupersonicRotorBlade(ideal_inlet_absolute_flow_mach=2.2,  # ideal absolute inlet flow Mach
    ideal_inlet_absolute_flow_angle=75.0,  # ideal absolute inlet flow angle
    requested_outlet_absolute_flow_angle=-25,  # requested absolute outlet flow angle
    lower_surface_relative_flow_mach=1.1,  # relative pressure-surface flow Mach
    upper_surface_relative_flow_mach=2.2,  # relative suction-surface flow Mach
    blade_count=80,
    mean_radius=0.15,  # [m], also sets physical chord and Reynolds number
    rotational_speed_rpm=30000.0,
    fluid=working_fluid,
    inlet_total_temperature=1000.0,  # absolute total temperature [K]
    inlet_total_pressure=5.0e6,  # absolute total pressure [Pa] = 50 bar
    # Some optional controls are shown explicitly below.
    number_of_nodes=121,  # default: 101; used by each MOC transition and circular arc, and by the BL march
    # Ratio t_LE/G*_total. Zero retains the original sharp-edge passage. Default: 0.0.
    leading_edge_thickness_over_total_pitch=0.07,
    # Transform ideal far-field relative flow to the real passage-entry state. Default: True.
    use_leading_edge_entry_correction=True,
    calculate_starting=False,  # default: True
    # Boundary layer calculations are "laminar_then_turbulent", with no initial thickness inputs
    boundary_layer_mode="laminar_then_turbulent",
    # Iterate pitch closure to maintain the same leading and trailing edge thickness
    iterate_pitch_closure=False)

print(f"Frozen inlet-static gamma: {blade.gamma:.5f}")
print(f"Solidity: {blade.solidity:.3f}")
print(blade.flow_state_table)
print(f"Inlet metal angle: {blade.inlet_metal_angle:.3f} deg")
print(f"Outlet metal angle: {blade.outlet_metal_angle:.3f} deg")
print(f"Solidity: {blade.solidity:.3f}")
if blade.starting_result is not None:
    print("Maximum starting ideal inlet relative flow Mach: "
          f"{blade.starting_result.maximum_starting_ideal_inlet_relative_flow_mach:.3f}")

blade.dimensionalize()
# CAD-ready BL-corrected single-blade profile [mm]; the origin occurs once.
blade_x_mm = blade.blade_profile_x_CAD
blade_y_mm = blade.blade_profile_y_CAD
# The corresponding ideal profile uses the same point order and units.
ideal_blade_x_mm = blade.uncorrected_blade_profile_x_CAD
ideal_blade_y_mm = blade.uncorrected_blade_profile_y_CAD

# Plot uncorrrected blades
blade.plot(dimensional=True,  # default: False; dimensional plot axes are in mm
    corrected=False,  # default: True; False selects the uncorrected shape
    show_two_blades=True,  # default: True
    ax=None,  # default: None; pass an existing Matplotlib axes if desired
    show=True)  # default: True

# Plot corrrected blades
blade.plot(dimensional=True,  # default: False; dimensional plot axes are in mm
    corrected=True,  # default: True; False selects the uncorrected shape
    show_two_blades=True,  # default: True
    ax=None,  # default: None; pass an existing Matplotlib axes if desired
    show=True)  # default: True
