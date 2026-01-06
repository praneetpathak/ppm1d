"""
Acoustic Wave Initial Condition

A small-amplitude Gaussian pulse on a uniform background.
Good for testing wave propagation and dispersion properties.
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..grid.grid1d import Grid1D
    from ..flags import Flags
    from ..physics.state import State


def acoustic_wave(grid: 'Grid1D', flags: 'Flags',
                  amplitude: float = 0.01,
                  x0: float = 0.5,
                  sigma: float = 0.05,
                  rho0: float = 1.0,
                  p0: float = 1.0,
                  u0: float = 0.0) -> 'State':
    """
    Initialize acoustic wave problem with Gaussian pulse.

    A stationary Gaussian density/pressure perturbation with no initial
    velocity perturbation:
        rho = rho0 * (1 + amplitude * exp(-(x-x0)^2 / (2*sigma^2)))
        p = p0 * (1 + gamma * amplitude * exp(-(x-x0)^2 / (2*sigma^2)))
        u = u0  (no velocity perturbation)

    This stationary pulse will split into two counter-propagating
    acoustic waves (left-going and right-going), each with half
    the original amplitude.

    Args:
        grid: Grid1D object
        flags: Flags object
        amplitude: Pulse amplitude (relative to background)
        x0: Center of Gaussian pulse
        sigma: Width (standard deviation) of Gaussian pulse
        rho0: Background density
        p0: Background pressure
        u0: Background velocity

    Returns:
        State object with acoustic pulse initial conditions
    """
    from ..physics.state import State

    # Create state object with pre-allocated arrays
    state = State.create(grid, flags)

    gamma = flags.physics.gamma

    # Initialize all cells (including ghost)
    x = grid.x

    # Gaussian pulse perturbation
    perturbation = amplitude * np.exp(-((x - x0) ** 2) / (2.0 * sigma ** 2))

    # Set primitive variables
    # Stationary pulse: only density and pressure perturbed, velocity = 0
    # This will split into left and right going waves
    state.rho[:] = rho0 * (1.0 + perturbation)
    state.p[:] = p0 * (1.0 + gamma * perturbation)
    state.u[:] = u0  # No velocity perturbation - pulse will split

    return state
