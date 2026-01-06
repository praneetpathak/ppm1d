"""
Shock Detection and Flattening Kernels

Pure NumPy implementation of shock detection and PPM flattening.
Based on Colella & Woodward (1984).
"""

import numpy as np


def shock_detection_kernel(n, p, u, is_shock, shock_pressure_threshold=0.33):
    """
    Pass 1: Detect which cells are shocks.

    A cell is marked as a shock if:
    1. Pressure jump across it exceeds threshold
    2. Velocity is converging (du < 0)

    Args:
        n: Number of cells
        p: Pressure array
        u: Velocity array
        is_shock: Output array (int8), 1 if shock, 0 otherwise
        shock_pressure_threshold: Minimum pressure jump ratio for shock
    """
    for i in range(2, n - 2):
        dp = p[i+1] - p[i-1]
        p_min = min(p[i-1], p[i+1])
        if p_min > 0:
            pressure_jump = abs(dp) / p_min
        else:
            pressure_jump = 0.0

        du = u[i+1] - u[i-1]

        if pressure_jump > shock_pressure_threshold and du < 0:
            is_shock[i] = 1


def shock_spread_kernel(n, is_shock, unsmrho, unsmp, spread_distance=2):
    """
    Pass 2: For each cell, check if any neighbor (within spread_distance) is a shock.

    Args:
        n: Number of cells
        is_shock: Shock detection array from pass 1
        unsmrho: Output unsmoothness array for density
        unsmp: Output unsmoothness array for pressure
        spread_distance: Number of cells to spread shock marking
    """
    for i in range(0, n):
        for j in range(max(0, i - spread_distance), min(n, i + spread_distance + 1)):
            if is_shock[j] == 1:
                unsmrho[i] = 1.0
                unsmp[i] = 1.0
                break


def flattening_omega_kernel(n, p, u, omega,
                            omega0=0.5, omega1=5.0, omega2=0.1, eps_p=0.005):
    """
    Compute flattening coefficient omega at shocks.

    Based on Colella & Woodward (1984) equation 4.1.

    Args:
        n: Number of cells
        p: Pressure array
        u: Velocity array
        omega: Output flattening coefficient array
        omega0: Steepness threshold
        omega1: Flattening strength coefficient
        omega2: Pressure jump threshold
        eps_p: Pressure ratio threshold
    """
    for i in range(2, n - 2):
        dp_local = p[i+1] - p[i-1]
        du_local = u[i+1] - u[i-1]

        # Only apply flattening at converging flows
        if du_local >= 0:
            continue

        p_min = min(p[i-1], p[i+1])
        if p_min <= 0:
            continue

        if abs(dp_local) / p_min > eps_p:
            dp_wide = p[i+2] - p[i-2]
            if abs(dp_wide) > 1e-30:
                zeta = abs(dp_local) / abs(dp_wide)
            else:
                zeta = 1.0

            if zeta >= omega0:
                omega_tilde = omega1 * (zeta - omega0)
                omega_tilde = min(1.0, omega_tilde)

                p_jump = abs(dp_local) / p_min
                if p_jump > omega2:
                    omega_tilde = max(omega_tilde, min(1.0, (p_jump - omega2) / (3.0 * omega2)))

                omega[i] = omega_tilde


def flattening_spread_kernel(n, omega, omega_spread, spread_distance=3):
    """
    Spread flattening to neighboring cells.

    The flattening coefficient decays with distance from the shock.

    Args:
        n: Number of cells
        omega: Input flattening coefficients
        omega_spread: Output spread flattening coefficients
        spread_distance: Number of cells to spread flattening
    """
    for i in range(0, n):
        max_val = omega_spread[i]
        for j in range(max(0, i - spread_distance), min(n, i + spread_distance + 1)):
            if omega[j] > 0:
                dist = abs(j - i)
                decay = 1.0 / (1.0 + dist)
                contribution = decay * omega[j]
                if contribution > max_val:
                    max_val = contribution
        omega_spread[i] = max_val
