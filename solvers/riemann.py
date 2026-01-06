"""
Riemann Solver Kernel

Pure NumPy implementation of linearized acoustic Riemann solver.
Based on PPMstar methodology.
"""

import numpy as np


def riemann_solver_kernel(n_ghost, n_total, gamma, p_floor,
                          rho_L_avg, rho_R_avg, u_L_avg, u_R_avg, p_L_avg, p_R_avg,
                          F_rho, F_rho_u, F_E):
    """
    Linearized acoustic Riemann solver.

    Computes fluxes at each interface using a linearized acoustic
    approximation. The star state is computed analytically.

    Args:
        n_ghost: Number of ghost cells
        n_total: Total number of cells
        gamma: Adiabatic index
        p_floor: Minimum pressure
        rho_L_avg: Time-averaged left state density
        rho_R_avg: Time-averaged right state density
        u_L_avg: Time-averaged left state velocity
        u_R_avg: Time-averaged right state velocity
        p_L_avg: Time-averaged left state pressure
        p_R_avg: Time-averaged right state pressure
        F_rho: Output mass flux
        F_rho_u: Output momentum flux
        F_E: Output energy flux
    """
    for i in range(n_ghost, n_total - n_ghost + 1):
        # Left and right states
        rho_l = rho_L_avg[i]
        u_l = u_L_avg[i]
        p_l = p_L_avg[i]
        rho_r = rho_R_avg[i]
        u_r = u_R_avg[i]
        p_r = p_R_avg[i]

        # Sound speeds
        c_l = np.sqrt(gamma * p_l / rho_l)
        c_r = np.sqrt(gamma * p_r / rho_r)
        c_avg = 0.5 * (c_l + c_r)
        rho_avg = 0.5 * (rho_l + rho_r)

        # Acoustic impedance
        Z = rho_avg * c_avg

        # Star state (linearized Riemann solution)
        u_star = 0.5 * (u_l + u_r) + (p_l - p_r) / (2.0 * Z)
        p_star = 0.5 * (p_l + p_r) + Z * (u_l - u_r) / 2.0

        # Apply pressure floor
        if p_star < p_floor:
            p_star = p_floor

        # Upwind density
        if u_star > 0:
            rho_star = rho_l
        else:
            rho_star = rho_r

        # Total energy in star state
        E_star = p_star / (gamma - 1.0) + 0.5 * rho_star * u_star**2

        # Compute fluxes
        F_rho[i] = rho_star * u_star
        F_rho_u[i] = rho_star * u_star**2 + p_star
        F_E[i] = (E_star + p_star) * u_star
