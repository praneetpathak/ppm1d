"""
I/O module for PPM1D.

Contains input/output routines:
- Profile column definitions
- Profile writer (MESA-style)
- History writer (MESA-style)
- Plotting functions
"""

from .profile_columns import (
    ProfileColumn,
    PRIMARY_COLUMNS,
    DERIVED_COLUMNS,
    get_profile_columns,
)
from .profile_writer import ProfileWriter
from .history_writer import HistoryWriter
from .plot import plot_solution

__all__ = [
    'ProfileColumn',
    'PRIMARY_COLUMNS',
    'DERIVED_COLUMNS',
    'get_profile_columns',
    'ProfileWriter',
    'HistoryWriter',
    'plot_solution',
]
