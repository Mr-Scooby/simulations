# Radiation Pattern Simulation Project

## Purpose
This project simulates radiation patterns from collections of atoms or emitters.
It includes tools for building angular grids, generating atom positions and velocities,
computing array factors, converting fields to intensity, running Monte Carlo averages,
and visualizing the resulting patterns and atom configurations.

## Main workflow
The codebase follows this general pipeline:

1. Define the geometry and observation grid.
2. Generate atom positions and, when needed, atom velocities.
3. Compute the array factor for the chosen configuration.
4. Convert the field-like quantity into intensity.
5. Average over Monte Carlo realizations for time-dependent simulations.
6. Plot the pattern, planar cuts, or atom geometry.

## Active files
- `helpers.py` — support utilities for grids, sampling, weights, and intensity-related helpers.
- `rpattern.py` — static array-factor and radiation-pattern calculations.
- `mcpattern.py` — time-dependent Monte Carlo simulation for moving atoms.
- `rplotting.py` — plotting utilities for patterns, cuts, and atom clouds.

## File responsibilities
### `helpers.py`
Contains reusable utilities used by the simulation and pattern modules, such as:
- angle-grid construction
- atom position sampling
- thermal velocity sampling
- Gaussian beam weights
- field-to-intensity conversion helpers

### `rpattern.py`
Contains the core pattern and array-factor calculations for a fixed configuration.
This is the main physics/math module for the static pattern calculation.

### `mcpattern.py`
Contains the time-dependent Monte Carlo workflow.
It samples realizations, evolves atom positions in time, computes the time-dependent
array factor, converts it to intensity, and averages over realizations.

### `rplotting.py`
Contains plotting functions only.
It is responsible for visualizing:
- 3D radiation patterns
- planar angular cuts
- atom positions, weights, dipole direction, incident direction, and velocities

## Suggested usage pattern
For a static calculation:

1. Build the angular grid.
2. Generate atom positions.
3. Compute the array factor with `rpattern.py`.
4. Convert to intensity.
5. Plot with `rplotting.py`.

For a time-dependent Monte Carlo calculation:

1. Build the angular grid.
2. Set the number of atoms, realizations, and time points.
3. Sample positions and velocities.
4. Compute the time-dependent array factor for each realization.
5. Convert to intensity and average across realizations.
6. Plot one or more time slices.

## Data flow
Typical data flow in the project is:

`parameters -> geometry/sampling -> array factor -> intensity -> average -> plots`

For the Monte Carlo case:

`parameters -> sample atoms -> evolve atoms in time -> array factor -> intensity -> Monte Carlo average -> plots`

## Documentation style recommendation
A simple and consistent style will make the project easier to maintain:

- Use a short module docstring at the top of each file.
- Give every important function a docstring stating what it computes, its inputs, and its outputs.
- Use inline comments only for non-obvious logic or physics assumptions.
- Keep plotting, physics, and Monte Carlo logic separated by file responsibility.

## Current structure goal
The intended separation of responsibilities is:

- reusable support code in `helpers.py`
- static physics calculations in `rpattern.py`
- Monte Carlo and time dependence in `mcpattern.py`
- visualization in `rplotting.py`

This separation keeps the project easier to navigate and reduces confusion about where new code should go.
