"""
Physics module for PPM1D.

Contains high-level physics routines:
- State container
- Interface state construction
- Time stepping
- Boundary conditions
"""

from .state import State
from .interface_states import construct_interface_states
from .timestep import time_step, compute_timestep
from .boundary import apply_boundary_conditions

__all__ = [
    'State',
    'construct_interface_states',
    'time_step',
    'compute_timestep',
    'apply_boundary_conditions',
]
