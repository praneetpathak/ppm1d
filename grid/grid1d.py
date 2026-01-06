"""
1D Grid class for PPM1D.

Defines the computational grid including ghost cells.
"""

import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..flags import Flags


@dataclass
class Grid1D:
    """
    1D computational grid with ghost cells.

    Attributes:
        nx: Number of interior cells
        n_ghost: Number of ghost cells on each side
        x_min: Left boundary of physical domain
        x_max: Right boundary of physical domain
        dx: Cell width
        n_total: Total number of cells (interior + ghost)
        x: Cell center coordinates (including ghost cells)
    """
    nx: int
    n_ghost: int
    x_min: float
    x_max: float
    dx: float
    n_total: int
    x: np.ndarray

    @classmethod
    def from_flags(cls, flags: 'Flags') -> 'Grid1D':
        """
        Create a Grid1D from a Flags object.

        Args:
            flags: Configuration flags

        Returns:
            Grid1D object
        """
        nx = flags.grid.nx
        n_ghost = flags.numerics.n_ghost
        x_min = flags.grid.x_min
        x_max = flags.grid.x_max

        # Compute grid spacing
        dx = (x_max - x_min) / nx

        # Total number of cells
        n_total = nx + 2 * n_ghost

        # Create cell center coordinates
        # Interior cells: from x_min + 0.5*dx to x_max - 0.5*dx
        x_interior = np.linspace(x_min + 0.5 * dx, x_max - 0.5 * dx, nx)

        # Full array including ghost cells
        x = np.zeros(n_total)
        x[n_ghost:-n_ghost] = x_interior

        # Fill ghost cell coordinates
        for i in range(n_ghost):
            x[i] = x_interior[0] - (n_ghost - i) * dx
            x[nx + n_ghost + i] = x_interior[-1] + (i + 1) * dx

        return cls(
            nx=nx,
            n_ghost=n_ghost,
            x_min=x_min,
            x_max=x_max,
            dx=dx,
            n_total=n_total,
            x=x,
        )

    @property
    def interior_slice(self) -> slice:
        """Slice for interior cells only."""
        return slice(self.n_ghost, -self.n_ghost)

    @property
    def x_interior(self) -> np.ndarray:
        """Cell centers for interior cells only."""
        return self.x[self.interior_slice]

    def print_info(self):
        """Print grid information."""
        print(f"Grid1D:")
        print(f"  Interior cells: nx = {self.nx}")
        print(f"  Ghost cells: {self.n_ghost} per side")
        print(f"  Total cells: {self.n_total}")
        print(f"  Domain: [{self.x_min}, {self.x_max}]")
        print(f"  Cell width: dx = {self.dx:.6f}")
        print(f"  First interior cell center: x = {self.x[self.n_ghost]:.6f}")
        print(f"  Last interior cell center: x = {self.x[-self.n_ghost-1]:.6f}")
