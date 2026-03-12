# Project birds-eye view

## Goal of the project

The current codebase simulates **far-field radiation from a cloud of moving dipoles**, where each atom contributes with a phase factor set by its position and with a complex **weight** set by the incident beam / pulse.

At the moment, the code is best interpreted as a **classical / semiclassical forward-scattering model**:

- atoms are point scatterers / dipoles,
- atomic motion is ballistic: `r_j(t) = r_j(0) + v_j t`,
- the incoming field is encoded through a complex weight function `w_j(t)`,
- the emitted field is built by coherent summation over all atoms,
- the observable is the angular intensity pattern `I(theta, phi, t)`.

So the main object being simulated is:

- **Array factor** from all atoms,
- multiplied by a **single-dipole radiation pattern**,
- then squared to obtain intensity.

---

## Core physics idea

For each observation direction `n_hat`, the emitted field is modeled as

`E(n_hat, t) ~ dipole(n_hat) * sum_j [ w_j(t) * exp(i k_out n_hat · r_j(t)) ]`

where:

- `r_j(t)` is the atom position,
- `w_j(t)` is the incident-beam-induced complex excitation weight,
- `k_out` is the output wave number,
- `dipole(n_hat)` is the angular dipole emission factor.

Then

`I(n_hat, t) = |E(n_hat, t)|^2`.

### What controls directionality

A directional lobe appears when many atoms contribute with **phase correlations** in a preferred direction. In this code, that can come from:

1. a large-enough cloud in optical units,
2. a structured phase in the weights `w_j`,
3. coherent summation in the array factor,
4. low enough motional dephasing during the sampled time window.

### What the current model does well

- coherent interference from many atoms,
- motion-induced dephasing,
- finite beam waist,
- finite pulse length,
- Monte Carlo averaging over random clouds.

### What is still not explicitly modeled

For a true **stored collective excitation readout**, the code still does **not** include a separate storage stage that writes a spin-wave and later converts it into light. Right now the model is closer to:

- **driven cloud illuminated by a moving pulse**,
- not yet a full memory sequence of **write/store/read**.

That is the main physics limitation to keep in mind in the meeting.

---

## High-level simulation pipeline

`run_sim.py`
-> build physical regime and numerical parameters  
-> build angular grid  
-> build pulse weight function `w_fn`  
-> call Monte Carlo simulation in `mcpattern.py`  
-> save intensity + atom motion + beam motion  
-> plot 3D pattern, planar cuts, atom cloud, animation

More explicitly:

1. **Choose regime**
   - optical size `L/lambda`
   - spacing `a/lambda`
   - illumination ratio `w0/L`
   - filling factor `sigma_long/L`
   - pulse transit `v t / L`

2. **Convert regime to physical scales** in `setup_params.py`
   - wavelength, `k0`, box size, density, beam waist, pulse length, `t_char`, motional dephasing.

3. **Create observation directions** in `helpers.make_angle_grid()`.

4. **Build incoming pulse weight function** in `beam.make_weight_fn_gaussian_pulse()`.

5. **For each Monte Carlo realization** in `mcpattern.py`
   - sample atom positions,
   - sample thermal velocities,
   - propagate atoms ballistically,
   - evaluate weights `w_fn(r(t), t)`,
   - compute coherent array factor,
   - convert field to intensity.

6. **Average over realizations**.

7. **Store and visualize** the result.

---

## File map

## `run_sim.py`
**Role:** main entry point for a production run.

### Important functions
- `get_logger(name="run_sim", level=logging.INFO)`  
  Creates a stream logger.

- `main()`  
  Main orchestration function.

### What `main()` currently does
- defines `PhysicalRegime`, `PhysicalParams`, `SimParams`,
- creates angular grid,
- builds the pulse weight function,
- runs `mcpattern.mc_sim(...)`,
- saves output with `helpers.save_simulation_npz(...)`,
- makes plots and animation.

### Important dependencies
- `helpers.make_angle_grid(...)`
- `beam.make_weight_fn_gaussian_pulse(...)`
- `mcpattern.mc_sim(...)`
- `helpers.save_simulation_npz(...)`
- `rplotting.plot_pattern_3d(...)`
- `rplotting.plot_planar_cuts(...)`
- `rplotting.plot_atoms(...)`
- `rplotting.animation_atoms_with_pulse(...)`

### Meeting note
This is the best file to open first when explaining the whole project, because it shows the **full workflow** from parameters to saved output.

---

## `setup_params.py`
**Role:** translate intuitive regime parameters into actual simulation scales.

### Important classes and functions
- `PhysicalRegime`  
  Dimensionless regime controls.

- `PhysicalParams`  
  Derived physical scales.

- `PhysicalParams.__post_init__(self)`  
  Computes:
  - `k0 = 2 pi / wavelength`
  - cloud size `L`
  - `box_size`
  - spacing and density
  - beam waist
  - pulse length `sigma_long`
  - characteristic time `t_char`
  - motional dephasing `k0 v_thermal t_char`

- `SimParams`  
  Numerical controls: number of atoms, MC runs, time sampling, angle grid, chunk size, seed.

- `SimParams.__post_init__(self)`  
  Computes `t_max` and `grid_shape`.

- `SimParams.times(self) -> np.ndarray`  
  Returns the time grid.

- `_k_tag(k_hat) -> str`  
  Compact string tag for the incident direction.

- `SetupParams`  
  Bundles `regime`, `phys`, `sim`.

- `SetupParams.run_name(self) -> str`  
  Builds a hashed run name for saving files.

- `log_main_params(log, main) -> None`  
  Logging helper, but appears to be from an older parameter interface.

### Meeting note
This file defines the **language of the project**: optical size, optical spacing, illumination ratio, filling factor, pulse transit.

### Important caveat
`log_main_params(...)` refers to fields like `pulse_duration`, `pulse_speed`, `pulse_waist`, `thermal_velocity`, `beam_cloud_overlap`, which do **not** match the current dataclasses. It looks outdated.

---

## `beam.py`
**Role:** define how the incident pulse weights the atoms.

### Important functions
- `upstream_front_position(center, box_size, k_in_hat, margin=1.0) -> np.array`  
  Places the initial pulse front upstream of the cloud.

- `make_weight_fn_gaussian_pulse(w0, sigma_long, k_in_hat, k_in=1.0, v_front=1.0, box_size=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), margin=0.0, pulse_center_t0=0.0, pcenter_at_origin=False)`  
  Returns a callable `w_fn(r_xyz, t, return_pulse_center=False)`.

### Physics encoded here
The weight is

`w(r, t) = env_perp * env_long * exp(-i k_in k_in_hat · r)`

with

- transverse Gaussian envelope,
- longitudinal Gaussian pulse envelope,
- moving pulse center,
- optical phase of the incoming beam.

### Meeting note
This file is where the **illumination model** lives. If someone asks “how is the pulse entering the cloud?”, this is the file to show.

---

## `mcpattern.py`
**Role:** core Monte Carlo time-dependent simulation.

### Important functions
- `positions_at_time(r0_xyz, v_xyz, t) -> np.ndarray`  
  Ballistic motion update.

- `sample_realization(n_atoms, rng, **kwargs)`  
  Samples one cloud realization:
  - random positions,
  - random thermal velocities.

- `compute_realization_af_series_ballistic(n_hat_flat, grid_shape, k_out, times, r0_xyz, v_xyz, w_fn, chunk_atoms=20000, **kwargs) -> np.ndarray`  
  Computes the complex array factor time series for one realization.

- `compute_realization_intensity_series(n_hat_flat, grid_shape, dipole, k_out, p_hat, times, n_atoms, rng, w_fn, chunk_atoms=20000, normalize_each_time=False, **kwargs) -> np.ndarray`  
  Samples atoms, computes field, converts to intensity.

- `mc_sim(nx, ny, nz, grid_shape, k_out, p_hat, times, n_mc, n_atoms, w_fn, chunk_atoms=20000, seed=0, normalize_each_time=False, **kwargs)`  
  Main Monte Carlo loop. Returns:
  - mean intensity series,
  - atom positions versus time,
  - atom weights versus time,
  - pulse-center trajectory.

### Internal data flow
`mc_sim(...)`  
-> flatten observation directions  
-> build dipole factor  
-> for each MC realization call `compute_realization_intensity_series(...)`  
-> inside it call `sample_realization(...)`  
-> then call `compute_realization_af_series_ballistic(...)`  
-> then `helpers.intensity_from_field(...)`

### Physics encoded here
This is the file that actually performs

- ballistic motion,
- phase accumulation,
- coherent summation,
- Monte Carlo averaging.

### Important caveats
1. `normalize_each_time` is passed around, but in `compute_realization_intensity_series(...)` it is not actually used to normalize snapshots.
2. `p_hat` is accepted by `compute_realization_intensity_series(...)`, but intensity is computed using the already prepared `dipole`; `p_hat` is not directly used there.
3. The extra call at the end of `mc_sim(...)` that resamples atoms for `atom_weights_sim(...)` means the returned atom/pulse visualization data is **not necessarily the same realization** as the one contributing to `I_mean` when `n_mc > 1`.
4. The example `main()` at the bottom appears stale and refers to functions / names that do not match the current interface.

---

## `helpers.py`
**Role:** shared numerical and utility functions.

### Important functions
- `make_angle_grid(n_theta=241, n_phi=481)`  
  Builds `theta`, `phi`, and direction cosine grids `nx, ny, nz`.

- `single_dipole_E(nx, ny, nz, p_hat=PZ_HAT)`  
  Computes the single-dipole angular radiation factor.

- `intensity_from_field(AF, dipole)`  
  Converts complex field amplitude into intensity.

- `atom_grid(Nx, Ny, Nz=1, dx=1.0, dy=1.0, dz=1.0, z0=0.0, plane_restricted=False)`  
  Creates a regular atom lattice.

- `random_position(N, box_size=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), seed=0, plane_restricted=False)`  
  Random atom positions in a box.

- `random_velocity_thermal(r_xyz, v_std=0.01, seed=0, plane_restricted=False)`  
  Samples thermal velocities.

- `gaussian_weights(r_xyz, w0, k_in_hat, k_in=1.0)`  
  Static Gaussian beam weights.

- `filter_kwargs(func, kwargs)`  
  Utility for forwarding relevant keyword arguments.

- `save_simulation_npz(path, **data)`  
  Saves simulation outputs.

- `atom_weights_sim(times, r_xyz, v_xyz, w_fn)`  
  Produces atom positions, weights, and pulse center versus time for visualization.

### Meeting note
This file contains the **basic building blocks** reused everywhere else.

---

## `rpattern.py`
**Role:** array-factor calculations.

### Important functions
- `array_factor_general(n_hat_flat, grid_shape, k_out, r_xyz, w=None, chunk_atoms=20000)`  
  General array factor on an arbitrary cloud.

- `centered_indices(N)`  
  Helper for symmetric indexing.

- `array_factor_separable(nx, ny, nz, k, dx, dy, dz, Nx, Ny, Nz)`  
  Separable array factor for regular lattices.

- `get_I_at(th0, ph0)`  
  Small helper / test-style accessor.

- `sanity_printing()`  
  Debug helper.

### Meeting note
This file holds the **core coherent sum** logic in the simplest possible form.

---

## `rplotting.py`
**Role:** visualization of patterns, cuts, atom clouds, and animations.

### Important functions
- `plot_pattern_3d(nx, ny, nz, I, title="", alpha=1.0, stride=2, cmap="viridis")`
- `animate_pattern_3d(nx, ny, nz, I_series, title="", alpha=1.0, stride=2, cmap="viridis", interval=50)`
- `plot_planar_cuts(theta, phi, I, title_prefix="")`
- `plot_atoms(...)`
- `animation_atoms_with_pulse(r_pos, T, weights=None, pulse_center=None)`

### Meeting note
Use this file to explain **what the simulation output looks like**.

### Important caveat
`plot_pattern_3d(...)` assumes `nx`, `ny`, `nz`, and `I` all have matching shapes. The earlier grid mismatch error you saw comes from violating that assumption.

---

## `animation.py`
**Role:** tiny motion helper.

### Important functions
- `updated_pos(r_xyz, v_xyz, t=1)`  
  Position update helper.

### Meeting note
This is currently minor compared with `mcpattern.py` and `rplotting.py`.

---

## `test.py`
**Role:** ad hoc test / plotting script.

### Meeting note
Treat this as a sandbox file, not as core project logic.

---

## `radiation_pattern_plot.py`
**Role:** older standalone demonstration script for building a radiation pattern from a dipole plus a simple array factor.

### Meeting note
Useful pedagogically, but not part of the main simulation pipeline.

---

## Most important functions to explain in the meeting

If you need only a short list, use these:

1. `run_sim.main()`  
   Top-level orchestration.

2. `setup_params.PhysicalRegime` / `PhysicalParams` / `SimParams`  
   Defines the physical and numerical regime.

3. `beam.make_weight_fn_gaussian_pulse(...)`  
   Defines how the incoming pulse excites the cloud.

4. `mcpattern.sample_realization(...)`  
   Builds one random atomic cloud with thermal motion.

5. `mcpattern.compute_realization_af_series_ballistic(...)`  
   Computes the coherent field sum in time.

6. `helpers.single_dipole_E(...)`  
   Encodes dipole angular emission.

7. `helpers.intensity_from_field(...)`  
   Converts field to intensity.

8. `mcpattern.mc_sim(...)`  
   Produces the final Monte Carlo-averaged time-dependent pattern.

9. `rplotting.plot_pattern_3d(...)` and `rplotting.plot_planar_cuts(...)`  
   Show the result in a readable way.

---

## Minimal story to tell in the meeting

A good one-minute explanation is:

> We model a cloud of randomly distributed moving atoms as classical dipole emitters. A traveling Gaussian pulse gives each atom a complex excitation weight that depends on position and time. For every observation direction, we sum the phase contributions of all atoms coherently to build the far-field array factor, multiply by the dipole emission pattern, and square to get the intensity. We repeat this for many random realizations and average the result. This lets us study how cloud size, density, beam geometry, and thermal motion affect directionality and dephasing.

---

## Current parameter language

These are the key regime knobs already built into the project:

- `optical_size = L / lambda`  
  Cloud size in optical units.

- `optical_spacing = a / lambda`  
  Mean interparticle spacing in optical units.

- `illumination_ratio = w0 / L`  
  Transverse beam coverage.

- `filling_factor = sigma_long / L`  
  Longitudinal pulse coverage.

- `pulse_transit = v_front * t_char / L`  
  How far the pulse travels relative to cloud length.

- `mot_dephase = k0 * v_thermal * t_char`  
  Thermal phase scrambling over the characteristic time.

These are exactly the quantities to mention when discussing which regime should give a forward lobe.

---

## Important conceptual limitations to mention honestly

1. **No explicit storage stage yet**  
   The code models driven emission, not yet a write-store-read memory protocol.

2. **No dipole-dipole interaction / multiple scattering kernel**  
   The current model is an independent-emitter coherent sum, not a full coupled-dipole model.

3. **Density is inferred from a spacing proxy**  
   `density = 1 / spacing^3` is a simple estimate, not a full many-body packing model.

4. **Monte Carlo atom count is a numerical sample size**  
   `n_atoms` here is mainly the number of emitters in the simulation realization, not automatically the physical atom number of an experimental cloud.

5. **Some example / helper code is stale**  
   A few functions and comments still reflect older interfaces.

---

## Suggested order of slides / whiteboard explanation

1. **Physics goal**: directional readout from a cloud.
2. **Model ingredients**: atoms, velocities, pulse, coherent sum.
3. **Equation for the field**.
4. **Meaning of the regime parameters**.
5. **Code pipeline**: `run_sim -> beam/setup -> mcpattern -> helpers/rpattern -> rplotting`.
6. **What is already captured** vs **what is still missing**.
7. **Most important current outputs**: 3D lobe, planar cuts, atom/pulse animation.

---

## Short takeaway

The project is already a solid **time-dependent coherent scattering simulator** for a moving random cloud under pulsed illumination. The central physics object is the **Monte Carlo averaged far-field intensity pattern**. For meeting purposes, the clearest message is:

- **input pulse builds complex atomic weights**,
- **atomic motion changes phases in time**,
- **coherent summation produces directionality**,
- **Monte Carlo averaging tells you how robust the lobe is**.
