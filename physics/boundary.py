"""
Boundary Conditions for PPM1D.

Implements different boundary condition types:
- Reflective (wall)
- Outflow (zero gradient)
- Periodic
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import State


def apply_boundary_conditions(state: 'State') -> None:
    """
    Apply boundary conditions to ghost cells.

    Dispatches to appropriate BC handler based on flags.

    Args:
        state: State object (modified in-place)
    """
    flags = state.flags

    # Left boundary
    if flags.grid.bc_left == 'reflective':
        _apply_reflective_left(state)
    elif flags.grid.bc_left == 'outflow':
        _apply_outflow_left(state)
    elif flags.grid.bc_left == 'periodic':
        pass  # Handled together with right boundary
    else:
        raise ValueError(f"Unknown left BC type: {flags.grid.bc_left}")

    # Right boundary
    if flags.grid.bc_right == 'reflective':
        _apply_reflective_right(state)
    elif flags.grid.bc_right == 'outflow':
        _apply_outflow_right(state)
    elif flags.grid.bc_right == 'periodic':
        pass  # Handled below
    else:
        raise ValueError(f"Unknown right BC type: {flags.grid.bc_right}")

    # Handle periodic BCs (both sides must be periodic)
    if flags.grid.bc_left == 'periodic' and flags.grid.bc_right == 'periodic':
        _apply_periodic(state)


def _apply_reflective_left(state: 'State') -> None:
    """Apply reflective (wall) boundary condition on left.

    Ghost cells mirror interior cells across the boundary:
    - ghost[n_ghost-1] mirrors interior[n_ghost]
    - ghost[n_ghost-2] mirrors interior[n_ghost+1]
    - etc.
    """
    n_ghost = state.n_ghost

    for i in range(n_ghost):
        # Mirror index: ghost[i] mirrors interior[2*n_ghost - 1 - i]
        mirror_idx = 2 * n_ghost - 1 - i
        state.rho[i] = state.rho[mirror_idx]
        state.p[i] = state.p[mirror_idx]
        # Reflect velocity (change sign)
        state.u[i] = -state.u[mirror_idx]


def _apply_reflective_right(state: 'State') -> None:
    """Apply reflective (wall) boundary condition on right.

    Ghost cells mirror interior cells across the boundary:
    - ghost[nx+n_ghost] mirrors interior[nx+n_ghost-1]
    - ghost[nx+n_ghost+1] mirrors interior[nx+n_ghost-2]
    - etc.
    """
    n_ghost = state.n_ghost
    nx = state.nx

    # Last interior cell is at index (nx + n_ghost - 1)
    # First right ghost cell is at index (nx + n_ghost)
    for i in range(n_ghost):
        ghost_idx = nx + n_ghost + i
        # Mirror index: ghost[nx+n_ghost+i] mirrors interior[nx+n_ghost-1-i]
        mirror_idx = nx + n_ghost - 1 - i
        state.rho[ghost_idx] = state.rho[mirror_idx]
        state.p[ghost_idx] = state.p[mirror_idx]
        # Reflect velocity (change sign)
        state.u[ghost_idx] = -state.u[mirror_idx]


def _apply_outflow_left(state: 'State') -> None:
    """Apply outflow (zero gradient) boundary condition on left."""
    n_ghost = state.n_ghost

    for i in range(n_ghost):
        # Copy all values from first interior cell
        state.rho[i] = state.rho[n_ghost]
        state.u[i] = state.u[n_ghost]
        state.p[i] = state.p[n_ghost]


def _apply_outflow_right(state: 'State') -> None:
    """Apply outflow (zero gradient) boundary condition on right."""
    n_ghost = state.n_ghost
    nx = state.nx

    for i in range(nx + n_ghost, nx + 2 * n_ghost):
        # Copy all values from last interior cell
        state.rho[i] = state.rho[nx + n_ghost - 1]
        state.u[i] = state.u[nx + n_ghost - 1]
        state.p[i] = state.p[nx + n_ghost - 1]


def _apply_periodic(state: 'State') -> None:
    """Apply periodic boundary conditions."""
    n_ghost = state.n_ghost
    nx = state.nx

    # Left ghost cells get values from right interior cells
    for i in range(n_ghost):
        # Map ghost cell i to interior cell (nx - n_ghost + i)
        src = nx + i  # Interior index (with ghost offset)
        state.rho[i] = state.rho[src]
        state.u[i] = state.u[src]
        state.p[i] = state.p[src]

    # Right ghost cells get values from left interior cells
    for i in range(n_ghost):
        # Map ghost cell (nx + n_ghost + i) to interior cell (n_ghost + i)
        dst = nx + n_ghost + i
        src = n_ghost + i
        state.rho[dst] = state.rho[src]
        state.u[dst] = state.u[src]
        state.p[dst] = state.p[src]
