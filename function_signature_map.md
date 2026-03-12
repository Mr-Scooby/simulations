# Function and call-signature map
Format used:
- `function_name(full current signature)`
- For internal/project calls inside that function:
  - `-> called_function(current call)` => `required full signature`

## helpers.py

### `make_angle_grid(n_theta=241, n_phi=481)`
Calls to project/local functions: _none_

### `single_dipole_E(nx, ny, nz, p_hat=PZ_HAT)`
Calls to project/local functions: _none_

### `intensity_from_field(AF, dipole)`
Calls to project/local functions: _none_

### `atom_grid(Nx, Ny, Nz=1, dx=1.0, dy=1.0, dz=1.0, z0=0.0, plane_restricted=False)`
Calls to project/local functions: _none_

### `random_position(N, box_size=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), seed=0, plane_restricted=False)`
Calls to project/local functions: _none_

### `random_velocity_thermal(r_xyz, v_std=0.01, seed=0, plane_restricted=False)`
Calls to project/local functions: _none_

### `gaussian_weights(r_xyz, w0, k_in_hat, k_in=1.0)`
Calls to project/local functions: _none_

### `filter_kwargs(func, kwargs)`
Calls to project/local functions: _none_

### `save_simulation_npz(path, **data)`
Calls to project/local functions: _none_

### `atom_weights_sim(times, r_xyz, v_xyz, w_fn)`
Calls to project/local functions: _none_

## beam.py

### `upstream_front_position(center, box_size, k_in_hat, margin=1.0) -> np.array`
Calls to project/local functions: _none_

### `make_weight_fn_gaussian_pulse(w0, sigma_long, k_in_hat, k_in=1.0, v_front=1.0, box_size=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), margin=0.0, pulse_center_t0=0.0)`
Nested/local functions:
- `w_fn(r_xyz, t, return_pulse_center=False)`
Calls to project/local functions:
- `-> upstream_front_position(center=center, box_size=box_size, k_in_hat=k_in_hat, margin=margin)` => `upstream_front_position(center, box_size, k_in_hat, margin=1.0) -> np.array`

## animation.py

### `updated_pos(r_xyz, v_xyz, t=1)`
Calls to project/local functions: _none_

## run_sim.py

### `get_logger(name='run_sim', level=logging.INFO)`
Calls to project/local functions: _none_

### `main()`
Calls to project/local functions:
- `-> make_angle_grid(n_theta=sim.n_theta, n_phi=sim.n_phi)` => `make_angle_grid(n_theta=241, n_phi=481)`
- `-> make_weight_fn_gaussian_pulse(phys.beam_waist, phys.sigma_long, phys.k_in_hat, phys.k0, v_front=phys.v_front, box_size=phys.box_size, pulse_center_t0=phys.beam_r0)` => `make_weight_fn_gaussian_pulse(w0, sigma_long, k_in_hat, k_in=1.0, v_front=1.0, box_size=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), margin=0.0, pulse_center_t0=0.0)`
- `-> mc_sim(nx=nx, ny=ny, nz=nz, grid_shape=sim.grid_shape, k_out=phys.k0, p_hat=phys.p_hat, times=sim.times, n_mc=sim.n_mc, n_atoms=sim.n_atoms, plane_restricted=False, w_fn=w_fn, box_size=phys.box_size, v_std=phys.v_thermal)` => `mc_sim(nx, ny, nz, grid_shape, k_out: float, p_hat: np.ndarray, times: np.ndarray, n_mc: int, n_atoms: int, w_fn, chunk_atoms: int=20000, seed: int=0, normalize_each_time: bool=False, **kwargs) -> np.ndarray`
- `-> save_simulation_npz(setup.run_name, metadata=asdict(setup), intensity=I, atom_pos=position, w=weights, pcenter=pulse_center, times=setup.sim.times)` => `save_simulation_npz(path, **data)`

## test.py
_No top-level functions._

### Script-level calls
- `plot_pattern_3d(nx, ny, nz, I[20])` => `plot_pattern_3d(nx, ny, nz, I, title='', alpha=1.0, stride=2, cmap='viridis')`

## setup_params.py

### `_k_tag(k_hat) -> str`
Calls to project/local functions: _none_

### `log_main_params(log, main) -> None`
Calls to project/local functions: _none_

## rpattern.py

### `array_factor_general(n_hat_flat, grid_shape, k_out, r_xyz, w=None, chunk_atoms=20000)`
Calls to project/local functions: _none_

### `centered_indices(N)`
Calls to project/local functions: _none_

### `array_factor_separable(nx, ny, nz, k, dx, dy, dz, Nx, Ny, Nz)`
Calls to project/local functions:
- `-> centered_indices(Nx)` => `centered_indices(N)`
- `-> centered_indices(Ny)` => `centered_indices(N)`
- `-> centered_indices(Nz)` => `centered_indices(N)`

### `get_I_at(th0, ph0)`
Calls to project/local functions: _none_

### `sanity_printing()`
Calls to project/local functions:
- `-> get_I_at(np.pi/2, 0.0)` => `get_I_at(th0, ph0)`
- `-> get_I_at(np.pi/2, np.pi/2)` => `get_I_at(th0, ph0)`
- `-> get_I_at(0.0, 0.0)` => `get_I_at(th0, ph0)`

## rplotting.py

### `plot_pattern_3d(nx, ny, nz, I, title='', alpha=1.0, stride=2, cmap='viridis')`
Calls to project/local functions: _none_

### `animate_pattern_3d(nx, ny, nz, I_series, title='', alpha=1.0, stride=2, cmap='viridis', interval=50)`
Nested/local functions:
- `update(frame)`
Calls to project/local functions: _none_

### `_wrap_to_pi(angle_rad: np.ndarray) -> np.ndarray`
Calls to project/local functions: _none_

### `_nearest_index_periodic(angle_grid: np.ndarray, target: float) -> int`
Calls to project/local functions:
- `-> _wrap_to_pi(angle_grid - target)` => `_wrap_to_pi(angle_rad: np.ndarray) -> np.ndarray`

### `_nearest_index(angle_grid: np.ndarray, target: float) -> int`
Calls to project/local functions: _none_

### `plot_planar_cuts(theta, phi, I, title_prefix='')`
Nested/local functions:
- `build_plane_cut(phi_targets)`
Calls to project/local functions:
- `-> _nearest_index(theta, np.pi / 2)` => `_nearest_index(angle_grid: np.ndarray, target: float) -> int`
- `-> build_plane_cut((0.0, np.pi))` => `build_plane_cut(phi_targets)`
- `-> build_plane_cut((np.pi / 2, 3 * np.pi / 2))` => `build_plane_cut(phi_targets)`
- `-> _nearest_index_periodic(phi, target)` => `_nearest_index_periodic(angle_grid: np.ndarray, target: float) -> int`
- `-> _wrap_to_pi(phi[-1] - phi[0])` => `_wrap_to_pi(angle_rad: np.ndarray) -> np.ndarray`
- `-> _wrap_to_pi(target - 0.0)` => `_wrap_to_pi(angle_rad: np.ndarray) -> np.ndarray`
- `-> _wrap_to_pi(target - np.pi)` => `_wrap_to_pi(angle_rad: np.ndarray) -> np.ndarray`

### `_validate_plot_atoms_inputs(r_xyz, w=None, v_xyz=None)`
Calls to project/local functions: _none_

### `_compute_cloud_geometry(r_xyz)`
Calls to project/local functions: _none_

### `plot_atoms(r_xyz, title='Atoms', s=3, alpha=0.6, equal_axes=True, w=None, show_colorbar=True, p_hat=None, k_in_hat=None, arrow_scale=0.25, v_xyz=None, v_subsample=20, v_arrow_scale=0.08, seed=0, r_subsample=10000)`
Nested/local functions:
- `_norm(v)`
Calls to project/local functions:
- `-> _validate_plot_atoms_inputs(r_xyz, w=w, v_xyz=v_xyz)` => `_validate_plot_atoms_inputs(r_xyz, w=None, v_xyz=None)`
- `-> _compute_cloud_geometry(r_xyz)` => `_compute_cloud_geometry(r_xyz)`
- `-> _norm(p_hat)` => `_norm(v)`
- `-> _norm(k_in_hat)` => `_norm(v)`

### `animation_atoms_with_pulse(r_pos, T, weights: np.array=None, pulse_center: np.array=None)`
Nested/local functions:
- `update(frame)`
Calls to project/local functions:
- `-> plot_atoms(r_pos[0], w=weights)` => `plot_atoms(r_xyz, title='Atoms', s=3, alpha=0.6, equal_axes=True, w=None, show_colorbar=True, p_hat=None, k_in_hat=None, arrow_scale=0.25, v_xyz=None, v_subsample=20, v_arrow_scale=0.08, seed=0, r_subsample=10000)`
- `-> plot_atoms(r_pos[0], w=weights[0])` => `plot_atoms(r_xyz, title='Atoms', s=3, alpha=0.6, equal_axes=True, w=None, show_colorbar=True, p_hat=None, k_in_hat=None, arrow_scale=0.25, v_xyz=None, v_subsample=20, v_arrow_scale=0.08, seed=0, r_subsample=10000)`

## radiation_pattern_plot.py

### `single_dipole(theta, phi)`
Calls to project/local functions: _none_

### `AF(theta, phi, spacing=0.2, atoms=4)`
Calls to project/local functions: _none_

### `E_theta_phi(theta, phi)`
Calls to project/local functions:
- `-> single_dipole(theta, phi)` => `single_dipole(theta, phi)`
- `-> AF(theta, phi, atoms=2)` => `AF(theta, phi, spacing=0.2, atoms=4)`

## mcpattern.py

### `positions_at_time(r0_xyz: np.ndarray, v_xyz: np.ndarray, t: float) -> np.ndarray`
Calls to project/local functions: _none_

### `sample_realization(n_atoms, rng, **kwargs)`
Calls to project/local functions:
- `-> filter_kwargs(random_position, kwargs)` => `filter_kwargs(func, kwargs)`
- `-> filter_kwargs(random_velocity_thermal, kwargs)` => `filter_kwargs(func, kwargs)`
- `-> random_position(n_atoms, seed=int(rng.integers(0, 2**32)), **pos_kwargs)` => `random_position(N, box_size=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), seed=0, plane_restricted=False)`
- `-> random_velocity_thermal(r0_xyz, seed=int(rng.integers(2**32)), **vel_kwargs)` => `random_velocity_thermal(r_xyz, v_std=0.01, seed=0, plane_restricted=False)`

### `compute_realization_af_series_ballistic(n_hat_flat: np.ndarray, grid_shape, k_out: float, times: np.ndarray, r0_xyz: np.ndarray, v_xyz: np.ndarray, w_fn, chunk_atoms: int=20000, **kwargs) -> np.ndarray`
Calls to project/local functions: _none_

### `compute_realization_intensity_series(n_hat_flat, grid_shape, dipole: np.ndarray, k_out: float, p_hat: np.ndarray, times: np.ndarray, n_atoms: int, rng: np.random, w_fn, chunk_atoms: int=20000, normalize_each_time: bool=False, **kwargs) -> np.ndarray`
Calls to project/local functions:
- `-> sample_realization(n_atoms, rng, **kwargs)` => `sample_realization(n_atoms, rng, **kwargs)`
- `-> compute_realization_af_series_ballistic(n_hat_flat=n_hat_flat, grid_shape=grid_shape, k_out=k_out, times=times, r0_xyz=r0_xyz, v_xyz=v_xyz, w_fn=w_fn, chunk_atoms=chunk_atoms, **kwargs)` => `compute_realization_af_series_ballistic(n_hat_flat: np.ndarray, grid_shape, k_out: float, times: np.ndarray, r0_xyz: np.ndarray, v_xyz: np.ndarray, w_fn, chunk_atoms: int=20000, **kwargs) -> np.ndarray`
- `-> intensity_from_field(AF_series[it], dipole=dipole)` => `intensity_from_field(AF, dipole)`

### `mc_sim(nx, ny, nz, grid_shape, k_out: float, p_hat: np.ndarray, times: np.ndarray, n_mc: int, n_atoms: int, w_fn, chunk_atoms: int=20000, seed: int=0, normalize_each_time: bool=False, **kwargs) -> np.ndarray`
Calls to project/local functions:
- `-> single_dipole_E(nx, ny, nz, p_hat)` => `single_dipole_E(nx, ny, nz, p_hat=PZ_HAT)`
- `-> sample_realization(n_atoms, rng, **kwargs)` => `sample_realization(n_atoms, rng, **kwargs)`
- `-> atom_weights_sim(times, r_xyz, v_xyz, w_fn)` => `atom_weights_sim(times, r_xyz, v_xyz, w_fn)`
- `-> compute_realization_intensity_series(n_hat_flat=n_hat_flat, grid_shape=grid_shape, dipole=dipole, k_out=k_out, p_hat=p_hat, times=times, n_atoms=n_atoms, rng=rng, chunk_atoms=chunk_atoms, normalize_each_time=normalize_each_time, w_fn=w_fn, **kwargs)` => `compute_realization_intensity_series(n_hat_flat, grid_shape, dipole: np.ndarray, k_out: float, p_hat: np.ndarray, times: np.ndarray, n_atoms: int, rng: np.random, w_fn, chunk_atoms: int=20000, normalize_each_time: bool=False, **kwargs) -> np.ndarray`

### `main()`
Nested/local functions:
- `w_fn_factory(rng)`
Calls to project/local functions:
- `-> make_angle_grid(n_theta=241, n_phi=481)` => `make_angle_grid(n_theta=241, n_phi=481)`
- `-> random_position(n_atoms)` => `random_position(N, box_size=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), seed=0, plane_restricted=False)`
- `-> random_velocity_thermal(r0_xyz, v_std=0.01, seed=0, plane_restricted=True)` => `random_velocity_thermal(r_xyz, v_std=0.01, seed=0, plane_restricted=False)`
- `-> gaussian_weights(np.asarray(r0_xyz), w0, k_in_hat)` => `gaussian_weights(r_xyz, w0, k_in_hat, k_in=1.0)`
- `-> plot_atoms(r0_xyz, w=w0_once, p_hat=p_hat, k_in_hat=k_in_hat, v_xyz=v_xyz, r_subsample=200, v_subsample=30)` => `plot_atoms(r_xyz, title='Atoms', s=3, alpha=0.6, equal_axes=True, w=None, show_colorbar=True, p_hat=None, k_in_hat=None, arrow_scale=0.25, v_xyz=None, v_subsample=20, v_arrow_scale=0.08, seed=0, r_subsample=10000)`
- `-> plot_planar_cuts(theta, phi, I_plot0, title_prefix=f"MC mean, t={times[0]:.3g}")` => `plot_planar_cuts(theta, phi, I, title_prefix='')`
- `-> plot_pattern_3d(nx, ny, nz, I_plot0, title=f"MC mean pattern, t={times[0]:.3g}", alpha=1.0, stride=2)` => `plot_pattern_3d(nx, ny, nz, I, title='', alpha=1.0, stride=2, cmap='viridis')`
- `-> plot_planar_cuts(theta, phi, I_plotT, title_prefix=f"MC mean, t={times[it]:.3g}")` => `plot_planar_cuts(theta, phi, I, title_prefix='')`
- `-> plot_pattern_3d(nx, ny, nz, I_plotT, title=f"MC mean pattern, t={times[it]:.3g}", alpha=1.0, stride=2)` => `plot_pattern_3d(nx, ny, nz, I, title='', alpha=1.0, stride=2, cmap='viridis')`

## Notes
- This map only tracks top-level project functions and nested local functions defined inside them.
- External library calls (`numpy`, `matplotlib`, logging, dataclasses, etc.) are not expanded.
- `setup_params.py` contains dataclasses (`PhysicalRegime`, `PhysicalParams`, `SimParams`, `SetupParams`) in addition to the functions listed here.
