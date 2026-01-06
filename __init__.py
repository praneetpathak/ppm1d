"""
PPM1D - 1D Piecewise Parabolic Method Hydrodynamics Solver
Based on Prof. Paul Woodward's PPMstar 3D hydro code
A modular, parallelized implementation of the PPM hydrodynamics method.

Usage:
    from ppm1d import Flags, run_simulation

    # Run with defaults
    run_simulation()

    # Run with custom configuration
    flags = Flags()
    flags.grid.nx = 1600
    flags.simulation.problem = 'acoustic'
    run_simulation(flags)
"""

from .flags import Flags, DEFAULT_FLAGS, load_flags_from_dict
from .main import run_simulation

__version__ = '1.0.0'
__author__ = 'PPM1D Team'

__all__ = [
    'Flags',
    'DEFAULT_FLAGS',
    'load_flags_from_dict',
    'run_simulation',
]
