"""
Profile Column Definitions

Defines which quantities to save in profile files and how to compute them.
Users can modify this file to add/remove profile columns.

Each column is defined by:
- name: Column name (used in header)
- units: Units string
- description: Human-readable description
- compute: Function that takes (state, flags) and returns array
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..physics.state import State
    from ..flags import Flags


@dataclass
class ProfileColumn:
    """Definition of a single profile column."""
    name: str
    units: str
    description: str
    compute: Callable[['State', 'Flags'], np.ndarray]


# ============================================================================
# COMPUTED QUANTITIES
# ============================================================================

def _compute_zone(state: 'State', flags: 'Flags') -> np.ndarray:
    """Zone index (0-based)."""
    return np.arange(len(state.rho_interior))


def _compute_x(state: 'State', flags: 'Flags') -> np.ndarray:
    """Cell center position."""
    return state.x_interior


def _compute_density(state: 'State', flags: 'Flags') -> np.ndarray:
    """Mass density."""
    return state.rho_interior


def _compute_velocity(state: 'State', flags: 'Flags') -> np.ndarray:
    """Velocity."""
    return state.u_interior


def _compute_pressure(state: 'State', flags: 'Flags') -> np.ndarray:
    """Pressure."""
    return state.p_interior


def _compute_total_energy(state: 'State', flags: 'Flags') -> np.ndarray:
    """Total energy density: E = p/(gamma-1) + 0.5*rho*u^2"""
    gamma = flags.physics.gamma
    rho = state.rho_interior
    u = state.u_interior
    p = state.p_interior
    return p / (gamma - 1.0) + 0.5 * rho * u**2


def _compute_kinetic_energy(state: 'State', flags: 'Flags') -> np.ndarray:
    """Kinetic energy density: KE = 0.5*rho*u^2"""
    rho = state.rho_interior
    u = state.u_interior
    return 0.5 * rho * u**2


def _compute_internal_energy(state: 'State', flags: 'Flags') -> np.ndarray:
    """Internal energy density: e = p/(gamma-1)"""
    gamma = flags.physics.gamma
    p = state.p_interior
    return p / (gamma - 1.0)


def _compute_sound_speed(state: 'State', flags: 'Flags') -> np.ndarray:
    """Local sound speed: c = sqrt(gamma*p/rho)"""
    gamma = flags.physics.gamma
    rho = state.rho_interior
    p = state.p_interior
    return np.sqrt(gamma * p / rho)


def _compute_mach_number(state: 'State', flags: 'Flags') -> np.ndarray:
    """Local Mach number: M = |u|/c"""
    c = _compute_sound_speed(state, flags)
    u = state.u_interior
    return np.abs(u) / c


def _compute_entropy(state: 'State', flags: 'Flags') -> np.ndarray:
    """Entropy proxy: s = p/rho^gamma"""
    gamma = flags.physics.gamma
    rho = state.rho_interior
    p = state.p_interior
    return p / (rho ** gamma)


def _compute_temperature(state: 'State', flags: 'Flags') -> np.ndarray:
    """Temperature (ideal gas, mu=1): T = p/rho"""
    rho = state.rho_interior
    p = state.p_interior
    return p / rho


def _compute_specific_internal_energy(state: 'State', flags: 'Flags') -> np.ndarray:
    """Specific internal energy: eps = p/((gamma-1)*rho)"""
    gamma = flags.physics.gamma
    rho = state.rho_interior
    p = state.p_interior
    return p / ((gamma - 1.0) * rho)


# ============================================================================
# PROFILE COLUMN REGISTRY
# ============================================================================

# Primary quantities (directly from state)
PRIMARY_COLUMNS = [
    ProfileColumn('zone', '1', 'Zone index', _compute_zone),
    ProfileColumn('x', 'cm', 'Cell center position', _compute_x),
    ProfileColumn('density', 'g/cm^3', 'Mass density', _compute_density),
    ProfileColumn('velocity', 'cm/s', 'Velocity', _compute_velocity),
    ProfileColumn('pressure', 'dyn/cm^2', 'Pressure', _compute_pressure),
]

# Derived quantities (computed from state)
DERIVED_COLUMNS = [
    ProfileColumn('total_energy', 'erg/cm^3', 'Total energy density E=e+KE', _compute_total_energy),
    ProfileColumn('kinetic_energy', 'erg/cm^3', 'Kinetic energy density 0.5*rho*u^2', _compute_kinetic_energy),
    ProfileColumn('internal_energy', 'erg/cm^3', 'Internal energy density p/(gamma-1)', _compute_internal_energy),
    ProfileColumn('sound_speed', 'cm/s', 'Local sound speed sqrt(gamma*p/rho)', _compute_sound_speed),
    ProfileColumn('mach_number', '1', 'Local Mach number |u|/c', _compute_mach_number),
    ProfileColumn('entropy', 'cgs', 'Entropy proxy p/rho^gamma', _compute_entropy),
    ProfileColumn('temperature', 'K', 'Temperature (ideal gas) p/rho', _compute_temperature),
    ProfileColumn('specific_internal_energy', 'erg/g', 'Specific internal energy p/((gamma-1)*rho)', _compute_specific_internal_energy),
]

# Default columns to include in profiles
DEFAULT_PROFILE_COLUMNS = PRIMARY_COLUMNS + DERIVED_COLUMNS


def get_profile_columns(column_names: List[str] = None) -> List[ProfileColumn]:
    """
    Get list of ProfileColumn objects by name, or all defaults.

    Args:
        column_names: List of column names to include, or None for all defaults

    Returns:
        List of ProfileColumn objects
    """
    all_columns = {c.name: c for c in PRIMARY_COLUMNS + DERIVED_COLUMNS}

    if column_names is None:
        return DEFAULT_PROFILE_COLUMNS

    result = []
    for name in column_names:
        if name in all_columns:
            result.append(all_columns[name])
        else:
            raise ValueError(f"Unknown profile column: {name}. "
                             f"Available: {list(all_columns.keys())}")
    return result
