"""
PPM1D Configuration Flags - Single Source of Truth

ALL configurable parameters are defined here. No other file should
contain hardcoded values that the user might want to change.

Usage:
    from ppm1d.flags import Flags

    # Use defaults
    flags = Flags()

    # Customize
    flags = Flags()
    flags.grid.nx = 1600
    flags.physics.gamma = 5.0/3.0
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PhysicsFlags:
    """Physical parameters."""
    gamma: float = 1.4              # Ratio of specific heats (adiabatic index)
    p_floor: float = 1e-6           # Minimum pressure (prevents division by zero)
    rho_floor: float = 1e-6         # Minimum density (prevents division by zero)


@dataclass
class NumericsFlags:
    """Numerical scheme parameters."""
    cfl: float = 0.9                # CFL number for timestep control
    n_ghost: int = 4                # Number of ghost cells on each boundary
    max_steps: int = 10000          # Maximum number of timesteps
    flux_limit_fraction: float = 0.98  # Max fraction of cell content to advect per step


@dataclass
class ShockFlags:
    """Shock detection and flattening parameters (Colella & Woodward 1984)."""
    # Shock detection thresholds
    shock_pressure_threshold: float = 0.33    # Pressure jump ratio for shock detection

    # Small delta for monotonicity (prevents division by zero)
    small_delta_fraction: float = 0.005       # Fraction of central difference
    small_delta_floor: float = 1e-7           # Minimum small delta value

    # Flattening parameters (PPM eq 4.1)
    omega0: float = 0.5             # Steepness threshold for flattening
    omega1: float = 5.0             # Flattening strength coefficient
    omega2: float = 0.1             # Pressure jump threshold for flattening
    eps_p: float = 0.005            # Pressure ratio threshold

    # Spread distances
    shock_spread_distance: int = 2     # Cells to spread shock unsmoothness
    flatten_spread_distance: int = 3   # Cells to spread flattening


@dataclass
class PPMFlags:
    """PPM interpolation parameters."""
    # Unsmoothness detection (intrf0 pass 2)
    unsmoothness_scale: float = 10.0   # Scale factor for unsmoothness
    unsmoothness_offset: float = 0.1   # Offset for unsmoothness threshold

    # Characteristic tracing constant
    forthd: float = 4.0 / 3.0          # 4/3 coefficient from PPMstar


@dataclass
class GridFlags:
    """Grid configuration."""
    nx: int = 800                   # Number of interior cells
    x_min: float = 0.0              # Domain left boundary
    x_max: float = 1.0              # Domain right boundary

    # Boundary condition types: 'reflective', 'outflow', 'periodic'
    bc_left: str = 'reflective'
    bc_right: str = 'reflective'


@dataclass
class OutputFlags:
    """Output configuration."""
    # Output directories
    output_dir: str = 'output'           # Base output directory
    history_subdir: str = 'history'      # Subdirectory for history files
    profiles_subdir: str = 'profiles'    # Subdirectory for profile files
    plots_subdir: str = 'plots'          # Subdirectory for PNG plots

    # History file
    history_filename: str = 'history.data'

    # Profile files
    profiles_index: str = 'profiles.index'
    profile_prefix: str = 'profile_'

    # Dump frequency (in sound crossing times)
    soundcrossings_per_dump: float = 0.005

    # Plot options
    save_plots: bool = True
    plot_dpi: int = 150

    # Console output
    print_every_n_steps: int = 100


@dataclass
class SimulationFlags:
    """Simulation control parameters."""
    t_final: float = 1.0            # Final simulation time

    # Problem type: 'sod', 'acoustic', 'igw', 'custom'
    problem: str = 'sod'

    # If True, get_initial_condition sets recommended BCs for the problem
    # If False, use whatever BCs are set in GridFlags
    use_default_bcs: bool = True


@dataclass
class Flags:
    """
    Master configuration container.

    All simulation parameters are organized into logical groups.
    This is the single source of truth - no parameters should be
    hardcoded elsewhere in the codebase.
    """
    physics: PhysicsFlags = field(default_factory=PhysicsFlags)
    numerics: NumericsFlags = field(default_factory=NumericsFlags)
    shock: ShockFlags = field(default_factory=ShockFlags)
    ppm: PPMFlags = field(default_factory=PPMFlags)
    grid: GridFlags = field(default_factory=GridFlags)
    output: OutputFlags = field(default_factory=OutputFlags)
    simulation: SimulationFlags = field(default_factory=SimulationFlags)

    def print_summary(self):
        """Print a summary of current configuration."""
        print("=" * 60)
        print("PPM1D Configuration Summary")
        print("=" * 60)
        print(f"\nPhysics:")
        print(f"  gamma = {self.physics.gamma}")
        print(f"  p_floor = {self.physics.p_floor}")
        print(f"  rho_floor = {self.physics.rho_floor}")
        print(f"\nGrid:")
        print(f"  nx = {self.grid.nx}")
        print(f"  domain = [{self.grid.x_min}, {self.grid.x_max}]")
        print(f"  bc_left = {self.grid.bc_left}, bc_right = {self.grid.bc_right}")
        print(f"\nNumerics:")
        print(f"  cfl = {self.numerics.cfl}")
        print(f"  n_ghost = {self.numerics.n_ghost}")
        print(f"\nSimulation:")
        print(f"  problem = {self.simulation.problem}")
        print(f"  t_final = {self.simulation.t_final}")
        print(f"\nOutput:")
        print(f"  output_dir = {self.output.output_dir}")
        print(f"  dump_interval = {self.output.soundcrossings_per_dump} sound crossings")
        print("=" * 60)


# Default flags instance for convenience
DEFAULT_FLAGS = Flags()


def load_flags_from_dict(config_dict: dict) -> Flags:
    """
    Load flags from a dictionary (e.g., from YAML/JSON).

    Args:
        config_dict: Dictionary with configuration values

    Returns:
        Flags object with values from dictionary
    """
    flags = Flags()

    # Map dictionary keys to flag groups
    group_map = {
        'physics': flags.physics,
        'numerics': flags.numerics,
        'shock': flags.shock,
        'ppm': flags.ppm,
        'grid': flags.grid,
        'output': flags.output,
        'simulation': flags.simulation,
    }

    for group_name, group_obj in group_map.items():
        if group_name in config_dict:
            for key, value in config_dict[group_name].items():
                if hasattr(group_obj, key):
                    setattr(group_obj, key, value)

    return flags
