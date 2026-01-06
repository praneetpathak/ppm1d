"""
Profile Writer - MESA-style Profile Output

Writes profile_N.data files with spatial structure data at selected timesteps.
"""

import os
import numpy as np
from typing import List, TYPE_CHECKING

from .profile_columns import ProfileColumn, get_profile_columns

if TYPE_CHECKING:
    from ..physics.state import State
    from ..flags import Flags


class ProfileWriter:
    """
    Writes MESA-style profile files.

    Each profile contains:
    - Header with metadata (model number, time, step, etc.)
    - Column descriptions
    - Data rows (one per zone)
    """

    def __init__(self, output_dir: str, flags: 'Flags',
                 column_names: List[str] = None):
        """
        Initialize profile writer.

        Args:
            output_dir: Base output directory
            flags: Flags object
            column_names: List of column names to include, or None for all
        """
        self.flags = flags
        self.profiles_dir = os.path.join(output_dir, flags.output.profiles_subdir)
        self.index_path = os.path.join(self.profiles_dir, flags.output.profiles_index)
        self.profile_prefix = flags.output.profile_prefix
        self.columns = get_profile_columns(column_names)

        os.makedirs(self.profiles_dir, exist_ok=True)
        self._init_index()

    def _init_index(self):
        """Initialize profiles.index file."""
        with open(self.index_path, 'w') as f:
            f.write("# PPM1D Profile Index\n")
            f.write("# model_number  priority  profile_number  profile_filename\n")

    def write_profile(self, state: 'State', model_number: int,
                      step: int, priority: int = 1) -> str:
        """
        Write a single profile file.

        Args:
            state: State object
            model_number: Model/dump number
            step: Timestep count
            priority: Priority level (1=regular, 2=special event)

        Returns:
            Path to written profile file
        """
        profile_number = model_number
        filename = f"{self.profile_prefix}{model_number:04d}.data"
        filepath = os.path.join(self.profiles_dir, filename)

        # Compute all columns
        data = {}
        for col in self.columns:
            data[col.name] = col.compute(state, self.flags)

        # Write header and data
        with open(filepath, 'w') as f:
            # Header block 1: file info
            f.write(f"# PPM1D Profile\n")
            f.write(f"# model_number = {model_number}\n")
            f.write(f"# num_zones = {len(state.rho_interior)}\n")
            f.write(f"# time = {state.time:.15e}\n")
            f.write(f"# step = {step}\n")
            f.write(f"# gamma = {self.flags.physics.gamma}\n")
            f.write(f"# nx = {self.flags.grid.nx}\n")
            f.write(f"# dx = {state.dx:.15e}\n")
            f.write(f"#\n")

            # Header block 2: column info
            f.write(f"# {len(self.columns)} columns:\n")
            for i, col in enumerate(self.columns):
                f.write(f"# {i+1:3d}  {col.name:25s}  [{col.units:12s}]  {col.description}\n")
            f.write(f"#\n")

            # Column names row
            col_width = 22
            header_line = "  ".join(f"{col.name:>{col_width}s}" for col in self.columns)
            f.write(header_line + "\n")

            # Data rows
            n_zones = len(list(data.values())[0])
            for i in range(n_zones):
                row_values = []
                for col in self.columns:
                    val = data[col.name][i]
                    if isinstance(val, (int, np.integer)):
                        row_values.append(f"{val:>{col_width}d}")
                    else:
                        row_values.append(f"{val:>{col_width}.12e}")
                f.write("  ".join(row_values) + "\n")

        # Update index
        with open(self.index_path, 'a') as f:
            f.write(f"{model_number:12d}  {priority:8d}  {profile_number:14d}  {filename}\n")

        return filepath
