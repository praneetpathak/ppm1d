"""
Time Integration Kernels

Pure NumPy implementation of time stepping:
- Characteristic tracing
- Flux divergence computation
- Flux limiting
- Conservative update
- Primitive variable recovery
"""

import numpy as np


def characteristic_trace(n_ghost, n_total, dtbydx, forthd, c_array,
                         rho_R, drho, rho6, u_R, du, u6, p_R, dp, p6,
                         rho_L, u_L, p_L,
                         rho_L_avg, rho_R_avg, u_L_avg, u_R_avg, p_L_avg, p_R_avg):
    """
    Apply characteristic tracing for time-averaged interface states.

    Traces characteristics back in time to compute time-averaged
    interface states for the Riemann solver.

    Args:
        n_ghost: Number of ghost cells
        n_total: Total number of cells
        dtbydx: dt / dx ratio
        forthd: 4/3 coefficient
        c_array: Sound speed array
        rho_R, drho, rho6: Density PPM reconstruction
        u_R, du, u6: Velocity PPM reconstruction
        p_R, dp, p6: Pressure PPM reconstruction
        rho_L, u_L, p_L: Left interface values
        rho_L_avg, ..., p_R_avg: Output time-averaged states
    """
    for i in range(n_ghost, n_total - n_ghost + 1):
        # Left-going characteristic from cell i-1 (for left state at interface)
        xfrp = 0.5 * dtbydx * c_array[i-1]
        xfrp1 = 1.0 - forthd * xfrp

        # Right-going characteristic from cell i (for right state at interface)
        xfrm = 0.5 * dtbydx * c_array[i]
        xfrm1 = 1.0 - forthd * xfrm

        # Left state: from right interface of cell i-1, traced back
        rho_L_avg[i] = rho_R[i-1] - xfrp * (drho[i-1] - xfrp1 * rho6[i-1])
        u_L_avg[i] = u_R[i-1] - xfrp * (du[i-1] - xfrp1 * u6[i-1])
        p_L_avg[i] = p_R[i-1] - xfrp * (dp[i-1] - xfrp1 * p6[i-1])

        # Right state: from left interface of cell i, traced back
        rho_R_avg[i] = rho_L[i] + xfrm * (drho[i] + xfrm1 * rho6[i])
        u_R_avg[i] = u_L[i] + xfrm * (du[i] + xfrm1 * u6[i])
        p_R_avg[i] = p_L[i] + xfrm * (dp[i] + xfrm1 * p6[i])


def compute_flux_divergence(n_ghost, n_total, dx, F_rho, F_rho_u, F_E,
                            drho_dt, drho_u_dt, dE_dt):
    """
    Compute flux divergences for all interior cells.

    Computes: dQ/dt = -(F[i+1] - F[i]) / dx

    Args:
        n_ghost: Number of ghost cells
        n_total: Total number of cells
        dx: Grid spacing
        F_rho, F_rho_u, F_E: Fluxes at interfaces
        drho_dt, drho_u_dt, dE_dt: Output time derivatives
    """
    for i in range(n_ghost, n_total - n_ghost):
        drho_dt[i] = -(F_rho[i+1] - F_rho[i]) / dx
        drho_u_dt[i] = -(F_rho_u[i+1] - F_rho_u[i]) / dx
        dE_dt[i] = -(F_E[i+1] - F_E[i]) / dx


def compute_flux_scale(n_ghost, n_total, dt, rho, rho_u, E, drho_dt, drho_u_dt, dE_dt,
                       flux_limit_fraction=0.98):
    """
    Compute flux limiting scale factor.

    Ensures that no more than flux_limit_fraction of any conserved
    quantity is removed from a cell in a single timestep.

    Args:
        n_ghost: Number of ghost cells
        n_total: Total number of cells
        dt: Timestep
        rho, rho_u, E: Conserved variables
        drho_dt, drho_u_dt, dE_dt: Time derivatives
        flux_limit_fraction: Maximum fraction of cell content to advect

    Returns:
        Scale factor (0 < scale <= 1)
    """
    flux_scale = 1.0

    for i in range(n_ghost, n_total - n_ghost):
        # Check mass constraint
        if drho_dt[i] < 0:
            max_allowed_loss = flux_limit_fraction * rho[i]
            actual_loss = -dt * drho_dt[i]
            if actual_loss > max_allowed_loss:
                required_scale = max_allowed_loss / actual_loss
                if required_scale < flux_scale:
                    flux_scale = required_scale

        # Note: Unlike mass and energy (which must be positive), momentum can be
        # positive, negative, or zero. We don't apply momentum-based flux limiting
        # because there's no physical constraint on momentum sign.
        # The mass and energy constraints are sufficient to ensure stability.

        # Check energy constraint
        if dE_dt[i] < 0:
            max_allowed_loss = flux_limit_fraction * E[i]
            actual_loss = -dt * dE_dt[i]
            if actual_loss > max_allowed_loss:
                required_scale = max_allowed_loss / actual_loss
                if required_scale < flux_scale:
                    flux_scale = required_scale

    return flux_scale


def conservative_update(n_ghost, n_total, dt, flux_scale,
                        rho, rho_u, E, drho_dt, drho_u_dt, dE_dt):
    """
    Apply conservative updates to all cells.

    Updates: Q_new = Q_old + dt * flux_scale * dQ/dt

    Args:
        n_ghost: Number of ghost cells
        n_total: Total number of cells
        dt: Timestep
        flux_scale: Scale factor from flux limiting
        rho, rho_u, E: Conserved variables (modified in-place)
        drho_dt, drho_u_dt, dE_dt: Time derivatives
    """
    for i in range(n_ghost, n_total - n_ghost):
        rho[i] += dt * flux_scale * drho_dt[i]
        rho_u[i] += dt * flux_scale * drho_u_dt[i]
        E[i] += dt * flux_scale * dE_dt[i]


def primitive_recovery(n_ghost, n_total, gamma, rho_floor, p_floor,
                       rho, rho_u, E, u, p):
    """
    Recover primitive variables from conserved variables.

    Computes:
        rho = max(rho, rho_floor)
        u = rho_u / rho
        p = (gamma - 1) * (E - 0.5 * rho * u^2)
        p = max(p, p_floor)

    Args:
        n_ghost: Number of ghost cells
        n_total: Total number of cells
        gamma: Adiabatic index
        rho_floor: Minimum density
        p_floor: Minimum pressure
        rho, rho_u, E: Conserved variables
        u, p: Primitive variables (modified in-place)
    """
    for i in range(n_ghost, n_total - n_ghost):
        if rho[i] < rho_floor:
            rho[i] = rho_floor

        u[i] = rho_u[i] / rho[i]

        kinetic = 0.5 * rho[i] * u[i]**2
        p[i] = (gamma - 1.0) * (E[i] - kinetic)

        if p[i] < p_floor:
            p[i] = p_floor
