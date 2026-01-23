# PPM1D: 1D Piecewise Parabolic Method Hydrodynamics Solver

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-GPL--v3-blue.svg)](LICENSE)

A modular, high-performance Python implementation of the Piecewise Parabolic Method (PPM) for solving 1D compressible hydrodynamics problems. **This code is based on Prof. Paul R. Woodward's PPMstar 3D hydrodynamics code**, reimplemented in Python for educational purposes and 1D applications.

## Based on PPMstar

**This implementation is directly inspired by and follows the methodology of Prof. Paul R. Woodward's PPMstar**, a state-of-the-art 3D hydrodynamics code developed at the Laboratory for Computational Science and Engineering (LCSE) at the University of Minnesota. PPMstar is a production-level code used for large-scale simulations of astrophysical and geophysical flows, turbulent convection, and compressible fluid dynamics.

### Key PPMstar Features Implemented

This 1D code adopts the following core algorithms and design principles from PPMstar:

- **PPM Interpolation Scheme**: The `intrf0` subroutine for computing interface states with unsmoothness detection
- **Characteristic Tracing**: Time-averaged interface states using characteristic wave propagation
- **Linearized Acoustic Riemann Solver**: The `Riemann0` subroutine for solving interface states
- **Shock Detection and Flattening**: Pressure-jump based shock detection with contact steepening
- **Monotonicity Constraints**: Van Leer limiters and PPM-specific constraints to prevent spurious oscillations
- **Cell-Averaged Representation**: Finite-volume formulation with conservative updates

### PPMstar References

- **PPMstar Homepage**: [http://www.lcse.umn.edu](http://www.lcse.umn.edu)
- **Original PPM Paper**: Colella, P., & Woodward, P. R. (1984). "The Piecewise Parabolic Method (PPM) for Gas-Dynamical Simulations", *Journal of Computational Physics*, 54(1), 174-201.
- **Shock Simulation Paper**: Woodward, P. R., & Colella, P. (1984). "The Numerical Simulation of Two-Dimensional Fluid Flow with Strong Shocks", *Journal of Computational Physics*, 54(1), 115-173.

## Table of Contents

- [Based on PPMstar](#based-on-ppmstar)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Features

### Core Capabilities
- **1D PPM Hydrodynamics**: Full implementation of the Piecewise Parabolic Method for compressible Euler equations
- **Shock Capturing**: Advanced shock detection and flattening algorithms (Colella & Woodward 1984)
- **Multiple Riemann Solvers**: Support for various approximate Riemann solvers
- **Adaptive Time Stepping**: CFL-based timestep control with stability guarantees

### Problem Types
- **Sod Shock Tube**: Classic 1D shock tube problem for code verification
- **Acoustic Waves**: Linear wave propagation for convergence testing
- **Custom Initial Conditions**: Extensible framework for user-defined problems

### Output and Visualization
- **History Files**: Time-series data of global quantities (mass, energy, momentum conservation)
- **Profile Data**: Spatial profiles at specified timesteps
- **Interactive Plots**: Built-in matplotlib visualization with customizable styling
- **Jupyter Notebook Explorer**: Interactive parameter exploration and visualization

### Software Design
- **Modular Architecture**: Clean separation of physics, numerics, and I/O
- **Configuration Management**: Single source of truth for all parameters via `Flags` class
- **Command-Line Interface**: Easy execution from terminal with argument parsing
- **Python API**: Full programmatic access for integration and automation

## Installation

### Prerequisites
- Python 3.7 or higher
- NumPy
- Matplotlib (for plotting)
- Jupyter (for interactive notebooks)

### Install from Source
```bash
# Clone the repository
git clone https://github.com/yourusername/ppm1d.git
cd ppm1d

# Install in development mode
pip install -e .
```

### Basic Usage
```python
from ppm1d import run_simulation

# Run with default settings (Sod shock tube)
state = run_simulation()
```

## Quick Start

### Command Line
```bash
# Run Sod shock tube with 800 cells
python -m ppm1d.main --nx 800 --problem sod

# Run acoustic wave test
python -m ppm1d.main --nx 1600 --problem acoustic --t_final 2.0
```

### Python API
```python
from ppm1d import Flags, run_simulation

# Create custom configuration
flags = Flags()
flags.grid.nx = 1600
flags.simulation.problem = 'acoustic'
flags.physics.gamma = 5.0/3.0  # Monatomic gas

# Run simulation
final_state = run_simulation(flags)
```

### Interactive Exploration
Launch the Jupyter notebook explorer:
```bash
jupyter notebook PPM1D_Explorer.ipynb
```

## Usage

### Basic Simulation Workflow

1. **Configure**: Set up simulation parameters using the `Flags` class
2. **Initialize**: Create grid and initial conditions
3. **Evolve**: Time-step the solution using PPM algorithm
4. **Output**: Write history, profiles, and visualization data

### Configuration System

All parameters are managed through the `Flags` class with sensible defaults:

```python
from ppm1d.flags import Flags

flags = Flags()

# Grid parameters
flags.grid.nx = 800          # Number of cells
flags.grid.x_min = 0.0       # Left boundary
flags.grid.x_max = 1.0       # Right boundary

# Physics parameters
flags.physics.gamma = 1.4    # Adiabatic index
flags.physics.p_floor = 1e-6 # Pressure floor

# Numerical parameters
flags.numerics.cfl = 0.9     # CFL number
flags.numerics.n_ghost = 4   # Ghost cells per side

# Simulation control
flags.simulation.t_final = 1.0    # Final time
flags.simulation.problem = 'sod'  # Initial condition

# Output control
flags.output.output_dir = 'output'
flags.output.save_plots = True
```

### Available Problems

- **`'sod'`**: Sod shock tube problem (left: high pressure, right: low pressure)
- **`'acoustic'`**: Acoustic wave propagation test

### Output Files

Simulations generate several types of output:

- **History files** (`history.txt`): Time evolution of conserved quantities
- **Profile files** (`profiles/`): Snapshots of spatial profiles at dump times
- **Plot files** (`plots/`): PNG images of solution profiles (if enabled)

## Configuration

### Physics Flags
| Parameter | Default | Description |
|-----------|---------|-------------|
| `gamma` | 1.4 | Ratio of specific heats |
| `p_floor` | 1e-6 | Minimum pressure floor |
| `rho_floor` | 1e-6 | Minimum density floor |

### Numerical Flags
| Parameter | Default | Description |
|-----------|---------|-------------|
| `cfl` | 0.9 | CFL stability parameter |
| `n_ghost` | 4 | Ghost cells per boundary |
| `max_steps` | 10000 | Maximum timesteps |
| `flux_limit_fraction` | 0.98 | Maximum advection fraction |

### Shock Detection Flags
| Parameter | Default | Description |
|-----------|---------|-------------|
| `shock_pressure_threshold` | 0.33 | Pressure jump for shock detection |
| `small_delta_fraction` | 0.005 | Monotonicity parameter |
| `flattening_coefficient` | 0.0 | Shock flattening strength |

## Project Structure

```
ppm1d/
├── __init__.py          # Package initialization and exports
├── main.py              # Main simulation driver
├── flags.py             # Configuration management
├── PPM1D_Explorer.ipynb # Interactive notebook
├── grid/                # Grid management
│   ├── __init__.py
│   └── grid1d.py
├── physics/             # Physical equations and state
│   ├── __init__.py
│   ├── state.py
│   └── timestep.py
├── solvers/             # Numerical algorithms
│   ├── __init__.py
│   ├── interpolation.py
│   ├── riemann.py
│   ├── shock.py
│   └── time_integration.py
├── initial_conditions/  # Problem setup
│   ├── __init__.py
│   ├── sod.py
│   └── acoustic.py
├── io/                  # Input/Output
│   ├── __init__.py
│   ├── history_writer.py
│   ├── profile_writer.py
│   └── plot.py
└── output_*/            # Generated output directories
```

## Examples

### Sod Shock Tube
```python
from ppm1d import Flags, run_simulation

flags = Flags()
flags.grid.nx = 800
flags.simulation.problem = 'sod'
flags.simulation.t_final = 0.2

state = run_simulation(flags)
```

### High-Resolution Acoustic Wave
```python
from ppm1d import Flags, run_simulation

flags = Flags()
flags.grid.nx = 1600
flags.simulation.problem = 'acoustic'
flags.simulation.t_final = 2.0
flags.output.save_plots = True

state = run_simulation(flags)
```

### Custom Configuration
```python
from ppm1d import Flags, run_simulation

flags = Flags()
# High-resolution grid
flags.grid.nx = 3200
# Different equation of state
flags.physics.gamma = 5.0/3.0
# Stricter CFL
flags.numerics.cfl = 0.8
# More frequent output
flags.output.soundcrossings_per_dump = 0.1

state = run_simulation(flags)
```

## Dependencies

### Core Dependencies
- **NumPy**: Numerical computations and array operations
- **Matplotlib**: Plotting and visualization

### Optional Dependencies
- **Jupyter**: Interactive notebook support
- **IPython**: Enhanced interactive Python
- **IPyWidgets**: Interactive controls in notebooks

### Development Dependencies
- **pytest**: Unit testing framework
- **black**: Code formatting
- **flake8**: Linting and style checking

## Acknowledgments

### Primary Acknowledgment

**This code is based on Prof. Paul R. Woodward's PPMstar 3D hydrodynamics code.** We are deeply grateful to Prof. Woodward for developing PPMstar and making the methodology available to the scientific community. The algorithms, numerical techniques, and design principles implemented in this 1D version are directly derived from PPMstar's Fortran implementation.

### Contributors

- **Praneet Pathak** (University of Victoria, Canada)

---
