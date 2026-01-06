"""
Sod Shock Tube Initial Condition

Classic test problem for compressible flow solvers.
A discontinuity at x=0.5 separates high-pressure/density
left state from low-pressure/density right state.
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..grid.grid1d import Grid1D
    from ..flags import Flags
    from ..physics.state import State


def sod_shock_tube(grid: 'Grid1D', flags: 'Flags') -> 'State':
    """
    Initialize Sod shock tube problem.

    Left state (x < 0.5):
        rho = 1.0
        p = 1.0
        u = 0.0

    Right state (x >= 0.5):
        rho = 0.125
        p = 0.1
        u = 0.0

    The interface is smoothed across cells that straddle x=0.5.

    Args:
        grid: Grid1D object
        flags: Flags object

    Returns:
        State object with Sod shock tube initial conditions
    """
    from ..physics.state import State

    # Create state object with pre-allocated arrays
    state = State.create(grid, flags)

    nx = grid.nx
    n_ghost = grid.n_ghost
    dx = grid.dx

    # Get interior cell centers
    x_interior = grid.x_interior

    # Initialize interior cells
    rho_interior = np.zeros(nx)
    p_interior = np.zeros(nx)
    u_interior = np.zeros(nx)

    # Discontinuity position
    x_disc = 0.5

    for i in range(nx):
        x_left = x_interior[i] - 0.5 * dx
        x_right = x_interior[i] + 0.5 * dx

        if x_right <= x_disc:
            # Entirely in left state
            rho_interior[i] = 1.0
            p_interior[i] = 1.0
        elif x_left >= x_disc:
            # Entirely in right state
            rho_interior[i] = 0.125
            p_interior[i] = 0.1
        else:
            # Cell straddles discontinuity - use volume average
            frac_left = (x_disc - x_left) / dx
            frac_right = (x_right - x_disc) / dx
            rho_interior[i] = frac_left * 1.0 + frac_right * 0.125
            p_interior[i] = frac_left * 1.0 + frac_right * 0.1

        u_interior[i] = 0.0

    # Copy to full arrays
    state.rho[n_ghost:-n_ghost] = rho_interior
    state.p[n_ghost:-n_ghost] = p_interior
    state.u[n_ghost:-n_ghost] = u_interior

    # Fill ghost cells with outflow BCs initially
    for i in range(n_ghost):
        state.rho[i] = rho_interior[0]
        state.p[i] = p_interior[0]
        state.u[i] = u_interior[0]

    for i in range(n_ghost):
        state.rho[-(i+1)] = rho_interior[-1]
        state.p[-(i+1)] = p_interior[-1]
        state.u[-(i+1)] = u_interior[-1]

    return state
