"""
PPM1D Main Orchestration Script

This is the entry point for running PPM simulations.
All parameters are read from flags.py.

Usage:
    # Run as module
    python -m ppm1d.main

    # Or import and call
    from ppm1d import run_simulation, Flags
    flags = Flags()
    flags.grid.nx = 1600
    run_simulation(flags)
"""

import os
import numpy as np

from .flags import Flags
from .grid.grid1d import Grid1D
from .physics.state import State
from .physics.timestep import time_step, compute_timestep
from .initial_conditions import get_initial_condition
from .io.history_writer import HistoryWriter
from .io.profile_writer import ProfileWriter
from .io.plot import plot_solution


def run_simulation(flags: Flags = None) -> State:
    """
    Main simulation driver.

    Args:
        flags: Configuration flags (uses defaults if None)

    Returns:
        Final State object
    """
    if flags is None:
        flags = Flags()

    # Print configuration
    print("=" * 60)
    print("PPM1D - 1D Piecewise Parabolic Method Hydrodynamics")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Problem: {flags.simulation.problem}")
    print(f"  Grid: nx = {flags.grid.nx}")
    print(f"  Domain: [{flags.grid.x_min}, {flags.grid.x_max}]")
    print(f"  Ghost cells: {flags.numerics.n_ghost} per side")
    print(f"  gamma = {flags.physics.gamma}")
    print(f"  CFL = {flags.numerics.cfl}")
    print(f"  t_final = {flags.simulation.t_final}")
    print(f"  Output: {flags.output.output_dir}/")

    # Create output directory
    os.makedirs(flags.output.output_dir, exist_ok=True)

    # Create grid
    grid = Grid1D.from_flags(flags)
    print(f"\nGrid created:")
    print(f"  dx = {grid.dx:.6e}")
    print(f"  n_total = {grid.n_total}")

    # Create initial condition
    state = get_initial_condition(flags.simulation.problem, grid, flags,
                                   set_default_bcs=flags.simulation.use_default_bcs)
    print(f"\nInitial condition: {flags.simulation.problem}")

    # Create output writers
    history = HistoryWriter(flags.output.output_dir, flags)
    profiles = ProfileWriter(flags.output.output_dir, flags)
    print(f"\nOutput writers initialized:")
    print(f"  History: {flags.output.output_dir}/{flags.output.history_subdir}/")
    print(f"  Profiles: {flags.output.output_dir}/{flags.output.profiles_subdir}/")

    # Compute initial sound crossing time for dump scheduling
    gamma = flags.physics.gamma
    rho_avg = np.mean(state.rho_interior)
    p_avg = np.mean(state.p_interior)
    c_avg = np.sqrt(gamma * p_avg / rho_avg)
    sound_crossing = (flags.grid.x_max - flags.grid.x_min) / c_avg
    dump_interval = flags.output.soundcrossings_per_dump * sound_crossing

    print(f"\nTiming:")
    print(f"  Sound crossing time = {sound_crossing:.6e}")
    print(f"  Dump interval = {dump_interval:.6e} ({flags.output.soundcrossings_per_dump} sound crossings)")
    print("=" * 60)

    # Initial output (dump 0)
    model_number = 0
    step = 0
    history.write_entry(state, model_number, step, 0.0)
    profiles.write_profile(state, model_number, step)
    if flags.output.save_plots:
        plot_solution(state, flags, model_number, flags.output.output_dir)

    next_dump = dump_interval

    print(f"\nStarting simulation...")
    print(f"  Initial: t = {state.time:.6e}")

    # Main time loop
    while state.time < flags.simulation.t_final:
        # Compute timestep
        dt = compute_timestep(state)
        if state.time + dt > flags.simulation.t_final:
            dt = flags.simulation.t_final - state.time

        # Advance solution
        time_step(state, dt)
        step += 1

        # Check for dump
        if state.time >= next_dump or abs(state.time - flags.simulation.t_final) < 1e-12:
            model_number += 1
            history.write_entry(state, model_number, step, dt)
            profiles.write_profile(state, model_number, step)
            if flags.output.save_plots:
                plot_solution(state, flags, model_number, flags.output.output_dir)
            next_dump += dump_interval
            print(f"  Dump {model_number:4d}: t = {state.time:.6e}, step = {step}")

        # Progress report
        if step % flags.output.print_every_n_steps == 0:
            print(f"  Step {step:6d}: t = {state.time:.6e}, dt = {dt:.2e}")

        # Safety checks
        if step > flags.numerics.max_steps:
            print(f"Warning: Exceeded max_steps ({flags.numerics.max_steps})")
            break
        if dt < 1e-15:
            print(f"Warning: Timestep too small (dt = {dt:.2e})")
            break

    # Final output if not already dumped
    if abs(state.time - (next_dump - dump_interval)) > 1e-12:
        model_number += 1
        history.write_entry(state, model_number, step, dt)
        profiles.write_profile(state, model_number, step)
        if flags.output.save_plots:
            plot_solution(state, flags, model_number, flags.output.output_dir)

    print("=" * 60)
    print("Simulation complete!")
    print(f"  Final time: t = {state.time:.6e}")
    print(f"  Total steps: {step}")
    print(f"  Total dumps: {model_number + 1}")
    print(f"\nOutput files:")
    print(f"  History: {flags.output.output_dir}/{flags.output.history_subdir}/{flags.output.history_filename}")
    print(f"  Profiles: {flags.output.output_dir}/{flags.output.profiles_subdir}/")
    if flags.output.save_plots:
        print(f"  Plots: {flags.output.output_dir}/{flags.output.plots_subdir}/")
    print("=" * 60)

    return state


def main():
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='PPM1D - 1D Piecewise Parabolic Method Hydrodynamics Solver'
    )
    parser.add_argument('--nx', type=int, default=800,
                        help='Number of grid cells (default: 800)')
    parser.add_argument('--t_final', type=float, default=1.0,
                        help='Final simulation time (default: 1.0)')
    parser.add_argument('--problem', type=str, default='sod',
                        choices=['sod', 'acoustic'],
                        help='Problem type (default: sod)')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory (default: output)')
    parser.add_argument('--cfl', type=float, default=0.9,
                        help='CFL number (default: 0.9)')
    parser.add_argument('--gamma', type=float, default=1.4,
                        help='Adiabatic index (default: 1.4)')

    args = parser.parse_args()

    # Create flags from command-line arguments
    flags = Flags()
    flags.grid.nx = args.nx
    flags.simulation.t_final = args.t_final
    flags.simulation.problem = args.problem
    flags.output.output_dir = args.output_dir
    flags.numerics.cfl = args.cfl
    flags.physics.gamma = args.gamma

    # Run simulation
    run_simulation(flags)


if __name__ == "__main__":
    main()
