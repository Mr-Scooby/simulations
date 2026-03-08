# File Map

This document is a navigation guide for the active modules in the project.
Each module section starts with a quick function index, followed by short entries
with purpose, inputs, and outputs.

## Files: 
1. helpers.py
3. rpattern.py
4. mcpattern.py
5. rplotting.py


---

# `helpers.py`

## Module purpose
Reusable support utilities for angular grids, dipole-field factors, atom sampling,
and atom weights.

## Functions in this module
1. `get_logger(name="helpers", level=logging.INFO)`
2. `make_angle_grid(n_theta=241, n_phi=481)`
3. `single_dipole_E(nx, ny, nz, p_hat=PZ_HAT)`
4. `intensity_from_field(AF, nx, ny, nz, p_hat)`
5. `atom_grid(Nx, Ny, Nz=1, dx=1.0, dy=1.0, dz=1.0, z0=0.0, plane_restricted=True)`
6. `random_position(N, box_size=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), seed=0, plane_restricted=True)`
7. `random_velocity_thermal(r_xyz, v_std=0.01, seed=0, plane_restricted=True)`
8. `gaussian_weights(r_xyz, w0, k_in_hat, k_in=1.0)`

---

## `get_logger(name="helpers", level=logging.INFO)`
**What it does**  
Creates and returns a module logger.

**Inputs**
- `name`: logger name
- `level`: logging level

**Output**
- configured `logging.Logger`

## `make_angle_grid(n_theta=241, n_phi=481)`
**What it does**  
Builds the angular observation grid and the corresponding direction cosines.

**Inputs**
- `n_theta`: number of polar-angle samples
- `n_phi`: number of azimuth-angle samples

**Outputs**
- `theta`: polar-angle array, shape `(n_theta,)`
- `phi`: azimuth-angle array, shape `(n_phi,)`
- `nx`, `ny`, `nz`: direction-cosine arrays, shape `(n_theta, n_phi)`

## `single_dipole_E(nx, ny, nz, p_hat=PZ_HAT)`
**What it does**  
Computes the far-field angular dependence of a single dipole,
using the form `E ∝ p - n (n·p)`.

**Inputs**
- `nx`, `ny`, `nz`: direction-cosine arrays
- `p_hat`: dipole orientation vector, shape `(3,)`

**Outputs**
- `Ex`, `Ey`, `Ez`: dipole-field components with the same shape as `nx`

## `intensity_from_field(AF, nx, ny, nz, p_hat)`
**What it does**  
Converts an array factor into intensity by applying the single-dipole pattern and computing `|E|^2`.

**Inputs**
- `AF`: complex array factor, shape `(n_theta, n_phi)`
- `nx`, `ny`, `nz`: direction-cosine arrays
- `p_hat`: dipole orientation vector

**Output**
- `I`: intensity array, shape `(n_theta, n_phi)`

## `atom_grid(Nx, Ny, Nz=1, dx=1.0, dy=1.0, dz=1.0, z0=0.0, plane_restricted=True)`
**What it does**  
Builds a centered atom grid in either the `xy` plane or full 3D.

**Inputs**
- `Nx`, `Ny`, `Nz`: number of atoms along each axis
- `dx`, `dy`, `dz`: lattice spacings
- `z0`: z offset of the grid center
- `plane_restricted`: if `True`, keep atoms in a plane

**Output**
- `r_xyz`: atom positions, shape `(N, 3)`

## `random_position(N, box_size=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), seed=0, plane_restricted=True)`
**What it does**  
Samples atom positions uniformly inside a rectangular box.

**Inputs**
- `N`: number of atoms
- `box_size`: box side lengths `(Lx, Ly, Lz)`
- `center`: box center `(cx, cy, cz)`
- `seed`: random seed
- `plane_restricted`: if `True`, set `z = cz`

**Output**
- `r_xyz`: sampled positions, shape `(N, 3)`

## `random_velocity_thermal(r_xyz, v_std=0.01, seed=0, plane_restricted=True)`
**What it does**  
Samples thermal velocities from a Gaussian distribution.

**Inputs**
- `r_xyz`: atom positions, used to determine the number of atoms
- `v_std`: standard deviation of the velocity components
- `seed`: random seed
- `plane_restricted`: if `True`, set `v_z = 0`

**Output**
- `v_xyz`: sampled velocities, shape `(N, 3)`

## `gaussian_weights(r_xyz, w0, k_in_hat, k_in=1.0)`
**What it does**  
Computes complex atom weights from a Gaussian transverse envelope and an incident plane-wave phase.

**Inputs**
- `r_xyz`: atom positions, shape `(N, 3)`
- `w0`: Gaussian waist
- `k_in_hat`: incident propagation direction, shape `(3,)`
- `k_in`: incident wave number magnitude

**Output**
- `w`: complex atom weights, shape `(N,)`

---

# `rpattern.py`

## Module purpose
Static array-factor calculations and related pattern utilities.

## Functions in this module
1. `get_logger(name="rpattern", level=logging.INFO)`
2. `array_factor_general(nx, ny, nz, k_out, r_xyz, w=None, chunk_atoms=20000)`
3. `centered_indices(N)`
4. `array_factor_separable(nx, ny, nz, k, dx, dy, dz, Nx, Ny, Nz)`
5. `get_I_at(th0, ph0)`
6. `sanity_printing()`

---

## `get_logger(name="rpattern", level=logging.INFO)`
**What it does**  
Creates and returns a module logger.

**Inputs**
- `name`: logger name
- `level`: logging level

**Output**
- configured `logging.Logger`

## `array_factor_general(nx, ny, nz, k_out, r_xyz, w=None, chunk_atoms=20000)`
**What it does**  
Computes the general array factor
`AF(n̂) = Σ_j w_j exp(i k_out n̂·r_j)`
for arbitrary atom positions.

**Inputs**
- `nx`, `ny`, `nz`: direction-cosine arrays, shape `(n_theta, n_phi)`
- `k_out`: output wave number
- `r_xyz`: atom positions, shape `(N, 3)`
- `w`: optional complex atom weights, shape `(N,)`
- `chunk_atoms`: chunk size for memory control

**Output**
- `AF`: complex array factor, shape `(n_theta, n_phi)`

## `centered_indices(N)`
**What it does**  
Returns centered integer-like lattice indices.

**Input**
- `N`: number of sites

**Output**
- centered index array, shape `(N,)`

## `array_factor_separable(nx, ny, nz, k, dx, dy, dz, Nx, Ny, Nz)`
**What it does**  
Computes the array factor for a separable rectangular lattice using 1D sums along `x`, `y`, and `z`.

**Inputs**
- `nx`, `ny`, `nz`: direction-cosine arrays
- `k`: wave number
- `dx`, `dy`, `dz`: lattice spacings
- `Nx`, `Ny`, `Nz`: number of lattice sites along each axis

**Output**
- `AF`: complex array factor, shape `(n_theta, n_phi)`

## `get_I_at(th0, ph0)`
**What it does**  
Reads off the intensity near a chosen angle from the global arrays used in the file's sanity checks.

**Inputs**
- `th0`: target polar angle
- `ph0`: target azimuth angle

**Output**
- scalar intensity value near that angle

## `sanity_printing()`
**What it does**  
Prints a few intensity values along selected directions for quick manual checks.

**Inputs**
- none

**Output**
- no returned value; prints to console

---

# `mcpattern.py`

## Module purpose
Time-dependent and Monte Carlo pattern calculations.

## Functions in this module
1. `get_logger(name="mcpattern", level=logging.INFO)`
2. `positions_at_time(r0_xyz, v_xyz, t)`
3. `array_factor_general_time(nx, ny, nz, k_out, r0_xyz, v_xyz, t, w_fn=None, chunk_atoms=20000)`
4. `make_weight_fn_gaussian_beam(w0, k_in_hat, k_in)`
5. `mc_intensity_time_series(theta, phi, nx, ny, nz, k_out, p_hat, times, n_mc, n_atoms, w_fn_factory=None, v_std=0.01, plane_restricted=True, chunk_atoms=200, seed=0, normalize_each_time=False)`
6. `main()`

---

## `get_logger(name="mcpattern", level=logging.INFO)`
**What it does**  
Creates and returns a module logger.

**Inputs**
- `name`: logger name
- `level`: logging level

**Output**
- configured `logging.Logger`

## `positions_at_time(r0_xyz, v_xyz, t)`
**What it does**  
Advances atom positions assuming straight-line motion.

**Inputs**
- `r0_xyz`: initial positions, shape `(N, 3)`
- `v_xyz`: velocities, shape `(N, 3)`
- `t`: time

**Output**
- `r_t`: positions at time `t`, shape `(N, 3)`

## `array_factor_general_time(nx, ny, nz, k_out, r0_xyz, v_xyz, t, w_fn=None, chunk_atoms=20000)`
**What it does**  
Computes the time-dependent array factor for moving atoms.
It first evolves atom positions to time `t`, then evaluates the general array-factor formula.

**Inputs**
- `nx`, `ny`, `nz`: direction-cosine arrays
- `k_out`: output wave number
- `r0_xyz`: initial positions, shape `(N, 3)`
- `v_xyz`: atom velocities, shape `(N, 3)`
- `t`: time
- `w_fn`: optional callable returning weights from positions
- `chunk_atoms`: chunk size for memory control

**Output**
- `AF_t`: complex array factor at time `t`, shape `(n_theta, n_phi)`

## `make_weight_fn_gaussian_beam(w0, k_in_hat, k_in)`
**What it does**  
Builds a weight function that applies Gaussian beam weights to a given position array.

**Inputs**
- `w0`: Gaussian waist
- `k_in_hat`: incident propagation direction
- `k_in`: incident wave number

**Output**
- `w_fn`: callable with signature `w_fn(r_xyz) -> w`

## `mc_intensity_time_series(theta, phi, nx, ny, nz, k_out, p_hat, times, n_mc, n_atoms, w_fn_factory=None, v_std=0.01, plane_restricted=True, chunk_atoms=200, seed=0, normalize_each_time=False)`
**What it does**  
Runs a Monte Carlo average of the time-dependent intensity.
For each realization, it samples atoms and velocities, computes the time-dependent array factor,
converts it to intensity, and averages over realizations.

**Inputs**
- `theta`, `phi`: angular grids
- `nx`, `ny`, `nz`: direction-cosine arrays
- `k_out`: output wave number
- `p_hat`: dipole orientation vector
- `times`: time samples
- `n_mc`: number of Monte Carlo realizations
- `n_atoms`: number of atoms in each realization
- `w_fn_factory`: optional factory that returns a weight function per realization
- `v_std`: thermal velocity scale
- `plane_restricted`: if `True`, restrict motion to the plane
- `chunk_atoms`: chunk size for AF computation
- `seed`: master random seed
- `normalize_each_time`: whether to normalize each time slice before averaging

**Output**
- Monte Carlo averaged intensity time series, typically shape `(n_times, n_theta, n_phi)`

## `main()`
**What it does**  
Example script entry point that sets parameters, runs the simulation, and generates plots.

**Inputs**
- none

**Output**
- no returned value; produces plots and console output

---

# `rplotting.py`

## Module purpose
Visualization functions for intensity patterns and atom configurations.

## Functions in this module
1. `plot_pattern_3d(nx, ny, nz, I, title="Radiation pattern", alpha=1.0, stride=3, cmap="inferno")`
2. `_wrap_to_pi(angle_rad)`
3. `_nearest_index_periodic(angle_grid, target)`
4. `_nearest_index(angle_grid, target)`
5. `plot_planar_cuts(theta, phi, I, title_prefix="")`
6. `plot_atoms(r_xyz, title="Atoms", s=3, alpha=0.6, equal_axes=True, w=None, show_colorbar=True, p_hat=None, k_in_hat=None, arrow_scale=0.25, v_xyz=None, v_subsample=20, v_arrow_scale=0.08, seed=0, r_subsample=10000)`

---

## `plot_pattern_3d(nx, ny, nz, I, title="Radiation pattern", alpha=1.0, stride=3, cmap="inferno")`
**What it does**  
Plots the 3D radiation pattern by using the normalized intensity as a radius.

**Inputs**
- `nx`, `ny`, `nz`: direction-cosine arrays
- `I`: intensity array, shape `(n_theta, n_phi)`
- `title`: plot title
- `alpha`: global radial scale factor
- `stride`: surface plotting stride
- `cmap`: colormap name

**Outputs**
- `fig`, `ax`: Matplotlib figure and 3D axes

## `_wrap_to_pi(angle_rad)`
**What it does**  
Internal helper that maps angles to `(-π, π]`.

**Input**
- `angle_rad`: angle array or scalar

**Output**
- wrapped angle with the same shape

## `_nearest_index_periodic(angle_grid, target)`
**What it does**  
Internal helper that finds the nearest index on a periodic angular grid.

**Inputs**
- `angle_grid`: periodic angle array
- `target`: target angle

**Output**
- nearest grid index

## `_nearest_index(angle_grid, target)`
**What it does**  
Internal helper that finds the nearest index on a non-periodic grid.

**Inputs**
- `angle_grid`: angle array
- `target`: target angle

**Output**
- nearest grid index

## `plot_planar_cuts(theta, phi, I, title_prefix="")`
**What it does**  
Plots intensity cuts in the `xy`, `xz`, and `yz` planes.

**Inputs**
- `theta`: polar-angle grid
- `phi`: azimuth-angle grid
- `I`: intensity array, shape `(n_theta, n_phi)`
- `title_prefix`: optional title prefix

**Outputs**
- `fig`, `axes`: Matplotlib figure and axes array

## `plot_atoms(r_xyz, title="Atoms", s=3, alpha=0.6, equal_axes=True, w=None, show_colorbar=True, p_hat=None, k_in_hat=None, arrow_scale=0.25, v_xyz=None, v_subsample=20, v_arrow_scale=0.08, seed=0, r_subsample=10000)`
**What it does**  
Plots atom positions in 3D, optionally colored by weights and annotated with dipole,
incident-wave, and velocity arrows.

**Inputs**
- `r_xyz`: atom positions, shape `(N, 3)`
- `title`: plot title
- `s`: marker size
- `alpha`: marker transparency
- `equal_axes`: whether to use equal axis scaling
- `w`: optional atom weights, shape `(N,)`
- `show_colorbar`: whether to show the weight colorbar
- `p_hat`: optional dipole direction vector
- `k_in_hat`: optional incident direction vector
- `arrow_scale`: scale for dipole and incident arrows
- `v_xyz`: optional velocities, shape `(N, 3)`
- `v_subsample`: maximum number of velocity arrows to draw
- `v_arrow_scale`: scale for velocity arrows
- `seed`: random seed for subsampling
- `r_subsample`: maximum number of atoms to scatter

**Outputs**
- `fig`, `ax`: Matplotlib figure and 3D axes
