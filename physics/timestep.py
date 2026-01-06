"""
Time Stepping for PPM1D.

Implements the time integration algorithm including:
- CFL-based timestep computation
- Full timestep advancement
"""

import numpy as np
from typing import TYPE_CHECKING

from ..solvers.time_integration import (
    characteristic_trace,
    compute_flux_divergence,
    compute_flux_scale,
    conservative_update,
    primitive_recovery,
)
from ..solvers.riemann import riemann_solver_kernel
from .interface_states import construct_interface_states
from .boundary import apply_boundary_conditions

if TYPE_CHECKING:
    from .state import State
    from ..flags import Flags


def compute_timestep(state: 'State') -> float:
    """
    Compute maximum stable timestep using CFL condition.

    Uses the Eulerian CFL condition:
        dt = cfl * dx / max(|u| + c)

    where c is the sound speed.

    Args:
        state: State object

    Returns:
        Timestep dt
    """
    flags = state.flags
    gamma = flags.physics.gamma
    cfl = flags.numerics.cfl
    n_ghost = state.n_ghost

    # Get interior values
    rho_interior = state.rho[n_ghost:-n_ghost]
    u_interior = state.u[n_ghost:-n_ghost]
    p_interior = state.p[n_ghost:-n_ghost]

    # Compute sound speed
    c = np.sqrt(gamma * p_interior / rho_interior)

    # Maximum wave speed
    max_wave_speed = np.max(np.abs(u_interior)) + np.max(c)

    # Safety floor
    if max_wave_speed < 1e-12:
        max_wave_speed = 1e-12

    dt = cfl * state.dx / max_wave_speed
    return dt


def time_step(state: 'State', dt: float) -> None:
    """
    Advance solution by one timestep.

    Implements the full PPM algorithm:
    1. Reconstruct interface states using PPM
    2. Apply characteristic tracing for time-averaged states
    3. Solve Riemann problem at each interface
    4. Compute and apply flux divergences
    5. Apply flux limiting for stability
    6. Update conserved variables
    7. Recover primitive variables
    8. Update boundary conditions

    Args:
        state: State object (modified in-place)
        dt: Timestep
    """
    flags = state.flags
    gamma = flags.physics.gamma
    forthd = flags.ppm.forthd
    n_total = state.n_total
    n_ghost = state.n_ghost

    # 1. Reconstruct interface states using PPM
    (rho_L, rho_R, drho, rho6,
     u_L, u_R, du, u6,
     p_L, p_R, dp, p6,
     c_array) = construct_interface_states(state)

    # 2. Characteristic tracing (parallel)
    dtbydx = dt / state.dx

    # Get work arrays
    rho_L_avg = state.work['rho_L_avg']
    rho_R_avg = state.work['rho_R_avg']
    u_L_avg = state.work['u_L_avg']
    u_R_avg = state.work['u_R_avg']
    p_L_avg = state.work['p_L_avg']
    p_R_avg = state.work['p_R_avg']

    # Reset work arrays
    rho_L_avg.fill(0.0)
    rho_R_avg.fill(0.0)
    u_L_avg.fill(0.0)
    u_R_avg.fill(0.0)
    p_L_avg.fill(0.0)
    p_R_avg.fill(0.0)

    characteristic_trace(n_ghost, n_total, dtbydx, forthd, c_array,
                         rho_R, drho, rho6, u_R, du, u6, p_R, dp, p6,
                         rho_L, u_L, p_L,
                         rho_L_avg, rho_R_avg,
                         u_L_avg, u_R_avg,
                         p_L_avg, p_R_avg)

    # Enforce positivity on time-averaged states
    rho_floor = flags.physics.rho_floor
    p_floor = flags.physics.p_floor
    rho_L_avg[:] = np.maximum(rho_L_avg, rho_floor)
    rho_R_avg[:] = np.maximum(rho_R_avg, rho_floor)
    p_L_avg[:] = np.maximum(p_L_avg, p_floor)
    p_R_avg[:] = np.maximum(p_R_avg, p_floor)

    # 3. Riemann solver (parallel)
    F_rho = state.work['F_rho']
    F_rho_u = state.work['F_rho_u']
    F_E = state.work['F_E']

    F_rho.fill(0.0)
    F_rho_u.fill(0.0)
    F_E.fill(0.0)

    riemann_solver_kernel(n_ghost, n_total, gamma, p_floor,
                          rho_L_avg, rho_R_avg,
                          u_L_avg, u_R_avg,
                          p_L_avg, p_R_avg,
                          F_rho, F_rho_u, F_E)

    # 4. Convert to conserved variables
    rho_u = state.work['rho_u']
    E = state.work['E']
    rho_u[:] = state.rho * state.u
    E[:] = state.p / (gamma - 1.0) + 0.5 * state.rho * state.u**2

    # 5. Compute flux divergences (parallel)
    drho_dt = state.work['drho_dt']
    drho_u_dt = state.work['drho_u_dt']
    dE_dt = state.work['dE_dt']

    drho_dt.fill(0.0)
    drho_u_dt.fill(0.0)
    dE_dt.fill(0.0)

    compute_flux_divergence(n_ghost, n_total, state.dx,
                            F_rho, F_rho_u, F_E,
                            drho_dt, drho_u_dt, dE_dt)

    # 6. Compute flux scale (sequential - reduction)
    flux_scale = compute_flux_scale(n_ghost, n_total, dt,
                                    state.rho, rho_u, E,
                                    drho_dt, drho_u_dt, dE_dt,
                                    flags.numerics.flux_limit_fraction)

    # 7. Apply conservative updates (parallel)
    conservative_update(n_ghost, n_total, dt, flux_scale,
                        state.rho, rho_u, E,
                        drho_dt, drho_u_dt, dE_dt)

    # 8. Recover primitive variables (parallel)
    primitive_recovery(n_ghost, n_total, gamma,
                       flags.physics.rho_floor, flags.physics.p_floor,
                       state.rho, rho_u, E, state.u, state.p)

    # 9. Update boundary conditions
    apply_boundary_conditions(state)

    # 10. Update time
    state.time += dt

    if flux_scale < 0.9:
        print(f"Warning: Flux limited to {flux_scale:.3f} at t = {state.time:.6f}")
