"""
History Writer - MESA-style History Output

Writes history.data file with one line per dump containing global quantities.
"""

import os
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..physics.state import State
    from ..flags import Flags


class HistoryWriter:
    """
    Writes MESA-style history file.

    The history file contains one line per dump with global quantities
    like total mass, energy, maximum/minimum values, etc.
    """

    # Define history columns: (name, format, description)
    COLUMNS = [
        ('model_number', '%12d', 'Model/dump number'),
        ('step', '%12d', 'Timestep count'),
        ('time', '%22.15e', 'Simulation time'),
        ('dt', '%22.15e', 'Current timestep'),
        ('total_mass', '%22.15e', 'Total mass in domain'),
        ('total_energy', '%22.15e', 'Total energy in domain'),
        ('total_kinetic', '%22.15e', 'Total kinetic energy'),
        ('total_internal', '%22.15e', 'Total internal energy'),
        ('total_momentum', '%22.15e', 'Total momentum'),
        ('max_density', '%22.15e', 'Maximum density'),
        ('min_density', '%22.15e', 'Minimum density'),
        ('max_velocity', '%22.15e', 'Maximum |velocity|'),
        ('max_pressure', '%22.15e', 'Maximum pressure'),
        ('min_pressure', '%22.15e', 'Minimum pressure'),
        ('max_mach', '%22.15e', 'Maximum Mach number'),
        ('max_sound_speed', '%22.15e', 'Maximum sound speed'),
    ]

    def __init__(self, output_dir: str, flags: 'Flags'):
        """
        Initialize history writer.

        Args:
            output_dir: Base output directory
            flags: Flags object
        """
        self.flags = flags
        self.history_dir = os.path.join(output_dir, flags.output.history_subdir)
        self.history_path = os.path.join(self.history_dir, flags.output.history_filename)

        os.makedirs(self.history_dir, exist_ok=True)
        self._write_header()

    def _write_header(self):
        """Write history file header."""
        with open(self.history_path, 'w') as f:
            f.write("# PPM1D History File\n")
            f.write(f"# gamma = {self.flags.physics.gamma}\n")
            f.write(f"# nx = {self.flags.grid.nx}\n")
            f.write(f"# cfl = {self.flags.numerics.cfl}\n")
            f.write(f"# problem = {self.flags.simulation.problem}\n")
            f.write("#\n")
            f.write(f"# {len(self.COLUMNS)} columns:\n")
            for i, (name, fmt, desc) in enumerate(self.COLUMNS):
                f.write(f"# {i+1:3d}  {name:20s}  {desc}\n")
            f.write("#\n")

            # Column header row
            col_width = 22
            header = "  ".join(f"{name:>{col_width}s}" for name, _, _ in self.COLUMNS)
            f.write(header + "\n")

    def write_entry(self, state: 'State', model_number: int,
                    step: int, dt: float) -> None:
        """
        Write one history entry.

        Args:
            state: State object
            model_number: Model/dump number
            step: Timestep count
            dt: Current timestep
        """
        gamma = self.flags.physics.gamma
        dx = state.dx
        n_ghost = state.n_ghost

        # Get interior values
        rho = state.rho[n_ghost:-n_ghost]
        u = state.u[n_ghost:-n_ghost]
        p = state.p[n_ghost:-n_ghost]

        # Compute global quantities
        total_mass = np.sum(rho) * dx
        kinetic = 0.5 * rho * u**2
        internal = p / (gamma - 1.0)
        total_energy = np.sum(kinetic + internal) * dx
        total_kinetic = np.sum(kinetic) * dx
        total_internal = np.sum(internal) * dx
        total_momentum = np.sum(rho * u) * dx

        c = np.sqrt(gamma * p / rho)
        mach = np.abs(u) / c

        # Build values list
        values = [
            model_number,
            step,
            state.time,
            dt,
            total_mass,
            total_energy,
            total_kinetic,
            total_internal,
            total_momentum,
            np.max(rho),
            np.min(rho),
            np.max(np.abs(u)),
            np.max(p),
            np.min(p),
            np.max(mach),
            np.max(c),
        ]

        # Write entry
        with open(self.history_path, 'a') as f:
            formatted = []
            for (_, fmt, _), val in zip(self.COLUMNS, values):
                formatted.append(fmt % val)
            f.write("  ".join(formatted) + "\n")
