"""
State container for PPM1D.

Holds all simulation state variables and pre-allocated work arrays.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from ..grid.grid1d import Grid1D
    from ..flags import Flags


@dataclass
class State:
    """
    Container for simulation state.

    Holds primitive variables (rho, u, p) and pre-allocated work arrays
    for efficient computation.

    Attributes:
        rho: Density array (n_total,)
        u: Velocity array (n_total,)
        p: Pressure array (n_total,)
        time: Current simulation time
        grid: Reference to Grid1D object
        flags: Reference to Flags object
        work: Pre-allocated work arrays for kernels
    """
    rho: np.ndarray
    u: np.ndarray
    p: np.ndarray
    time: float
    grid: 'Grid1D'
    flags: 'Flags'
    work: Dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def create(cls, grid: 'Grid1D', flags: 'Flags') -> 'State':
        """
        Create an empty State with pre-allocated work arrays.

        Args:
            grid: Grid1D object
            flags: Flags object

        Returns:
            State object with zero-initialized arrays
        """
        n_total = grid.n_total

        # Primitive variable arrays
        rho = np.zeros(n_total)
        u = np.zeros(n_total)
        p = np.zeros(n_total)

        # Pre-allocate work arrays for kernels
        work = {
            # intrf0 work arrays
            'duysppmzr': np.zeros(n_total),
            'duymnotzr': np.zeros(n_total),
            'uyrsmth': np.zeros(n_total),
            'uyrunsm': np.zeros(n_total),

            # Time step work arrays
            'rho_L_avg': np.zeros(n_total),
            'rho_R_avg': np.zeros(n_total),
            'u_L_avg': np.zeros(n_total),
            'u_R_avg': np.zeros(n_total),
            'p_L_avg': np.zeros(n_total),
            'p_R_avg': np.zeros(n_total),
            'F_rho': np.zeros(n_total),
            'F_rho_u': np.zeros(n_total),
            'F_E': np.zeros(n_total),
            'drho_dt': np.zeros(n_total),
            'drho_u_dt': np.zeros(n_total),
            'dE_dt': np.zeros(n_total),
            'rho_u': np.zeros(n_total),
            'E': np.zeros(n_total),
        }

        return cls(
            rho=rho,
            u=u,
            p=p,
            time=0.0,
            grid=grid,
            flags=flags,
            work=work,
        )

    @property
    def dx(self) -> float:
        """Grid spacing (convenience property)."""
        return self.grid.dx

    @property
    def x(self) -> np.ndarray:
        """Cell centers (convenience property)."""
        return self.grid.x

    @property
    def n_ghost(self) -> int:
        """Number of ghost cells (convenience property)."""
        return self.grid.n_ghost

    @property
    def n_total(self) -> int:
        """Total number of cells (convenience property)."""
        return self.grid.n_total

    @property
    def nx(self) -> int:
        """Number of interior cells (convenience property)."""
        return self.grid.nx

    @property
    def gamma(self) -> float:
        """Adiabatic index (convenience property)."""
        return self.flags.physics.gamma

    @property
    def interior_slice(self) -> slice:
        """Slice for interior cells only."""
        return self.grid.interior_slice

    @property
    def rho_interior(self) -> np.ndarray:
        """Density for interior cells only."""
        return self.rho[self.interior_slice]

    @property
    def u_interior(self) -> np.ndarray:
        """Velocity for interior cells only."""
        return self.u[self.interior_slice]

    @property
    def p_interior(self) -> np.ndarray:
        """Pressure for interior cells only."""
        return self.p[self.interior_slice]

    @property
    def x_interior(self) -> np.ndarray:
        """Cell centers for interior cells only."""
        return self.grid.x_interior

    def compute_sound_speed(self) -> np.ndarray:
        """Compute sound speed array."""
        gamma = self.flags.physics.gamma
        return np.sqrt(gamma * self.p / self.rho)

    def compute_total_energy(self) -> np.ndarray:
        """Compute total energy density: E = p/(gamma-1) + 0.5*rho*u^2"""
        gamma = self.flags.physics.gamma
        return self.p / (gamma - 1.0) + 0.5 * self.rho * self.u**2

    def compute_kinetic_energy(self) -> np.ndarray:
        """Compute kinetic energy density: KE = 0.5*rho*u^2"""
        return 0.5 * self.rho * self.u**2

    def compute_internal_energy(self) -> np.ndarray:
        """Compute internal energy density: e = p/(gamma-1)"""
        gamma = self.flags.physics.gamma
        return self.p / (gamma - 1.0)

    def print_info(self):
        """Print state information."""
        ng = self.n_ghost
        print(f"State at t = {self.time:.6f}:")
        print(f"  rho: [{self.rho[ng]:.6f}, ..., {self.rho[-ng-1]:.6f}]")
        print(f"       min = {np.min(self.rho[ng:-ng]):.6e}, "
              f"max = {np.max(self.rho[ng:-ng]):.6e}")
        print(f"  u:   [{self.u[ng]:.6f}, ..., {self.u[-ng-1]:.6f}]")
        print(f"       min = {np.min(self.u[ng:-ng]):.6e}, "
              f"max = {np.max(self.u[ng:-ng]):.6e}")
        print(f"  p:   [{self.p[ng]:.6f}, ..., {self.p[-ng-1]:.6f}]")
        print(f"       min = {np.min(self.p[ng:-ng]):.6e}, "
              f"max = {np.max(self.p[ng:-ng]):.6e}")
