"""
Interface State Construction for PPM1D.

Implements the PPM reconstruction algorithm to compute
interface states for the Riemann solver.
"""

import numpy as np
from typing import TYPE_CHECKING, Tuple

from ..solvers.interpolation import (
    intrf0_pass1_slopes,
    intrf0_pass1_interfaces,
    intrf0_pass2,
)
from ..solvers.shock import (
    shock_detection_kernel,
    shock_spread_kernel,
    flattening_omega_kernel,
    flattening_spread_kernel,
)

if TYPE_CHECKING:
    from .state import State
    from ..flags import Flags


def intrf0(state: 'State', uy: np.ndarray, smaldu: np.ndarray, unsmuy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    PPM interpolation to compute left and right interface states.

    Args:
        state: State object (for work arrays)
        uy: Input field array
        smaldu: Small delta values for division safety
        unsmuy: Unsmoothness array (modified in-place)

    Returns:
        Tuple of (uyl, uyr, duy, uy6):
            uyl: Left interface values
            uyr: Right interface values
            duy: Delta (uyr - uyl)
            uy6: Parabola curvature
    """
    n_total = len(uy)
    flags = state.flags

    # Output arrays
    uyl = np.zeros(n_total)
    uyr = np.zeros(n_total)
    duy = np.zeros(n_total)
    uy6 = np.zeros(n_total)

    # Get work arrays
    duysppmzr = state.work['duysppmzr']
    duymnotzr = state.work['duymnotzr']
    uyrsmth = state.work['uyrsmth']
    uyrunsm = state.work['uyrunsm']

    # Reset work arrays
    duysppmzr.fill(0.0)
    duymnotzr.fill(0.0)
    uyrsmth.fill(0.0)
    uyrunsm.fill(0.0)

    # Pass 1a: Compute slopes (parallel)
    intrf0_pass1_slopes(uy, duysppmzr, duymnotzr)

    # Pass 1b: Compute interfaces (parallel, depends on 1a)
    intrf0_pass1_interfaces(uy, duysppmzr, duymnotzr, uyrsmth, uyrunsm)

    # Pass 2: Monotonicity constraints (parallel)
    intrf0_pass2(uy, smaldu, unsmuy, uyrsmth, uyrunsm, uyl, uyr, duy, uy6,
                 flags.ppm.unsmoothness_scale, flags.ppm.unsmoothness_offset)

    # Handle boundary cells (sequential - only 4 cells each side)
    uyl[0] = uyl[2]
    uyr[0] = uyr[2]
    duy[0] = duy[2]
    uy6[0] = uy6[2]
    uyl[1] = uyl[2]
    uyr[1] = uyr[2]
    duy[1] = duy[2]
    uy6[1] = uy6[2]

    uyl[n_total-2] = uyl[n_total-3]
    uyr[n_total-2] = uyr[n_total-3]
    duy[n_total-2] = duy[n_total-3]
    uy6[n_total-2] = uy6[n_total-3]
    uyl[n_total-1] = uyl[n_total-3]
    uyr[n_total-1] = uyr[n_total-3]
    duy[n_total-1] = duy[n_total-3]
    uy6[n_total-1] = uy6[n_total-3]

    return uyl, uyr, duy, uy6


def construct_interface_states(state: 'State') -> Tuple:
    """
    Construct interface states according to PPM algorithm.

    This is the main PPM reconstruction routine. It:
    1. Detects shocks and marks cells for unsmoothness
    2. Computes small delta values for monotonicity
    3. Applies PPM interpolation to all primitive variables
    4. Computes and applies flattening at shocks
    5. Enforces positivity constraints

    Args:
        state: State object

    Returns:
        Tuple of 13 arrays:
            (rho_L, rho_R, drho, rho6,
             u_L, u_R, du, u6,
             p_L, p_R, dp, p6,
             c_array)
    """
    flags = state.flags
    gamma = flags.physics.gamma

    # Compute sound speed
    c_array = np.sqrt(gamma * state.p / state.rho)

    # Create unsmoothness arrays
    n = len(state.rho)
    unsmrho = np.zeros(n)
    unsmp = np.zeros(n)

    # Shock flattener (parallel two-pass)
    is_shock = np.zeros(n, dtype=np.int8)
    shock_detection_kernel(n, state.p, state.u, is_shock,
                           flags.shock.shock_pressure_threshold)
    shock_spread_kernel(n, is_shock, unsmrho, unsmp,
                        flags.shock.shock_spread_distance)

    # Compute small delta values
    small_frac = flags.shock.small_delta_fraction
    small_floor = flags.shock.small_delta_floor

    smalldrho = np.zeros(n)
    smalldrho[1:-1] = small_frac * np.abs(state.rho[2:] - state.rho[:-2])
    smalldrho[0] = smalldrho[1]
    smalldrho[-1] = smalldrho[-2]
    smalldrho = np.maximum(smalldrho, small_floor * 2.0 * state.rho)

    smalldp = np.zeros(n)
    smalldp[1:-1] = small_frac * np.abs(state.p[2:] - state.p[:-2])
    smalldp[0] = smalldp[1]
    smalldp[-1] = smalldp[-2]
    smalldp = np.maximum(smalldp, small_floor * 2.0 * state.p)

    smaldu = small_frac * c_array

    # Apply PPM interpolation
    rho_L, rho_R, drho, rho6 = intrf0(state, state.rho, smalldrho, unsmrho)
    p_L, p_R, dp, p6 = intrf0(state, state.p, smalldp, unsmp)
    u_L, u_R, du, u6 = intrf0(state, state.u, smaldu, unsmp)

    # PPM flattening coefficient (JIT compiled)
    omega = np.zeros(n)
    flattening_omega_kernel(n, state.p, state.u, omega,
                            flags.shock.omega0, flags.shock.omega1,
                            flags.shock.omega2, flags.shock.eps_p)

    # Spread flattening (JIT compiled)
    omega_spread = omega.copy()
    flattening_spread_kernel(n, omega, omega_spread,
                             flags.shock.flatten_spread_distance)
    omega = np.maximum(omega, omega_spread)

    # Apply flattening
    one_minus_omega = 1.0 - omega

    rho_L = one_minus_omega * rho_L + omega * state.rho
    rho_R = one_minus_omega * rho_R + omega * state.rho
    drho = rho_R - rho_L
    rho6 = 6.0 * (state.rho - 0.5 * (rho_L + rho_R))

    p_L = one_minus_omega * p_L + omega * state.p
    p_R = one_minus_omega * p_R + omega * state.p
    dp = p_R - p_L
    p6 = 6.0 * (state.p - 0.5 * (p_L + p_R))

    u_L = one_minus_omega * u_L + omega * state.u
    u_R = one_minus_omega * u_R + omega * state.u
    du = u_R - u_L
    u6 = 6.0 * (state.u - 0.5 * (u_L + u_R))

    # Enforce positivity
    rho_floor = flags.physics.rho_floor
    p_floor = flags.physics.p_floor

    rho_L = np.maximum(rho_L, rho_floor)
    rho_R = np.maximum(rho_R, rho_floor)
    p_L = np.maximum(p_L, p_floor)
    p_R = np.maximum(p_R, p_floor)

    return (rho_L, rho_R, drho, rho6,
            u_L, u_R, du, u6,
            p_L, p_R, dp, p6,
            c_array)
