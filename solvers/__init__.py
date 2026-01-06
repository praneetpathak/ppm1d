"""
Solvers module for PPM1D.

Contains pure NumPy implementations for:
- PPM interpolation (intrf0)
- Shock detection and flattening
- Riemann solver
- Time integration (characteristic tracing, flux updates)
"""

from .interpolation import (
    intrf0_pass1_slopes,
    intrf0_pass1_interfaces,
    intrf0_pass2,
)

from .shock import (
    shock_detection_kernel,
    shock_spread_kernel,
    flattening_omega_kernel,
    flattening_spread_kernel,
)

from .riemann import riemann_solver_kernel

from .time_integration import (
    characteristic_trace,
    compute_flux_divergence,
    compute_flux_scale,
    conservative_update,
    primitive_recovery,
)

__all__ = [
    # Interpolation
    'intrf0_pass1_slopes',
    'intrf0_pass1_interfaces',
    'intrf0_pass2',
    # Shock
    'shock_detection_kernel',
    'shock_spread_kernel',
    'flattening_omega_kernel',
    'flattening_spread_kernel',
    # Riemann
    'riemann_solver_kernel',
    # Time integration
    'characteristic_trace',
    'compute_flux_divergence',
    'compute_flux_scale',
    'conservative_update',
    'primitive_recovery',
]
