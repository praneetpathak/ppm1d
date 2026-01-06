"""
Plotting Functions for PPM1D.

Creates visualization plots of the simulation state.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..physics.state import State
    from ..flags import Flags


def plot_solution(state: 'State', flags: 'Flags',
                  model_number: int, output_dir: str = None) -> str:
    """
    Plot the current solution (density, velocity, pressure).

    Args:
        state: State object
        flags: Flags object
        model_number: Model/dump number
        output_dir: Base output directory (uses flags.output.output_dir if None)

    Returns:
        Path to saved PNG file, or None if save_plots is False
    """
    if output_dir is None:
        output_dir = flags.output.output_dir

    # Get interior values
    x = state.x_interior
    rho = state.rho_interior
    u = state.u_interior
    p = state.p_interior

    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    # Density
    ax1.plot(x, rho, 'b-', linewidth=2, label='Density')
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Density (t = {state.time:.4f})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Velocity
    ax2.plot(x, u, 'r-', linewidth=2, label='Velocity')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Velocity')
    ax2.set_title(f'Velocity (t = {state.time:.4f})')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Pressure
    ax3.plot(x, p, 'g-', linewidth=2, label='Pressure')
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Pressure')
    ax3.set_title(f'Pressure (t = {state.time:.4f})')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()

    # Save if configured
    save_path = None
    if flags.output.save_plots:
        plots_dir = os.path.join(output_dir, flags.output.plots_subdir)
        os.makedirs(plots_dir, exist_ok=True)
        save_path = os.path.join(plots_dir, f'solution_{model_number:04d}.png')
        plt.savefig(save_path, dpi=flags.output.plot_dpi, bbox_inches='tight')

    plt.close()

    return save_path


def plot_comparison(state: 'State', flags: 'Flags',
                    exact_rho: np.ndarray = None,
                    exact_u: np.ndarray = None,
                    exact_p: np.ndarray = None,
                    model_number: int = 0,
                    output_dir: str = None) -> str:
    """
    Plot solution compared to exact solution.

    Args:
        state: State object
        flags: Flags object
        exact_rho: Exact density solution (optional)
        exact_u: Exact velocity solution (optional)
        exact_p: Exact pressure solution (optional)
        model_number: Model/dump number
        output_dir: Base output directory

    Returns:
        Path to saved PNG file
    """
    if output_dir is None:
        output_dir = flags.output.output_dir

    x = state.x_interior
    rho = state.rho_interior
    u = state.u_interior
    p = state.p_interior

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    # Density
    ax1.plot(x, rho, 'b-', linewidth=2, label='Numerical')
    if exact_rho is not None:
        ax1.plot(x, exact_rho, 'k--', linewidth=1, label='Exact')
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Density (t = {state.time:.4f})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Velocity
    ax2.plot(x, u, 'r-', linewidth=2, label='Numerical')
    if exact_u is not None:
        ax2.plot(x, exact_u, 'k--', linewidth=1, label='Exact')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Velocity')
    ax2.set_title(f'Velocity (t = {state.time:.4f})')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Pressure
    ax3.plot(x, p, 'g-', linewidth=2, label='Numerical')
    if exact_p is not None:
        ax3.plot(x, exact_p, 'k--', linewidth=1, label='Exact')
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Pressure')
    ax3.set_title(f'Pressure (t = {state.time:.4f})')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()

    plots_dir = os.path.join(output_dir, flags.output.plots_subdir)
    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, f'comparison_{model_number:04d}.png')
    plt.savefig(save_path, dpi=flags.output.plot_dpi, bbox_inches='tight')
    plt.close()

    return save_path
