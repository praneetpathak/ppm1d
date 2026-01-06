"""
Initial conditions module for PPM1D.

Contains functions to set up different test problems:
- Sod shock tube
- Acoustic wave
- Internal gravity wave (IGW)
"""

from .sod import sod_shock_tube
from .acoustic import acoustic_wave

__all__ = [
    'sod_shock_tube',
    'acoustic_wave',
    'get_initial_condition',
]


def get_initial_condition(problem: str, grid, flags, set_default_bcs: bool = True):
    """
    Factory function to get initial condition by name.

    Optionally sets appropriate boundary conditions for each problem type.

    Args:
        problem: Problem name ('sod', 'acoustic', 'igw')
        grid: Grid1D object
        flags: Flags object
        set_default_bcs: If True, set recommended BCs for the problem type.
                         If False, use whatever BCs are already in flags.

    Returns:
        State object with initial conditions
    """
    if problem == 'sod':
        if set_default_bcs:
            # Sod shock tube: use outflow BCs so waves exit cleanly
            flags.grid.bc_left = 'outflow'
            flags.grid.bc_right = 'outflow'
        return sod_shock_tube(grid, flags)
    elif problem == 'acoustic':
        if set_default_bcs:
            # Acoustic pulse: use periodic BCs for clean wave propagation
            flags.grid.bc_left = 'periodic'
            flags.grid.bc_right = 'periodic'
        return acoustic_wave(grid, flags)
    else:
        raise ValueError(f"Unknown problem type: {problem}. "
                         f"Available: 'sod', 'acoustic'")
