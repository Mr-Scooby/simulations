#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

# ---------------------------------------------------------------------
# Motion + time-dependent weights
# ---------------------------------------------------------------------

def positions_at_time(r0_xyz, v_xyz, t):
    """
    Inputs
    - r0_xyz: (N,3) initial positions
    - v_xyz:  (N,3) velocities
    - t:      scalar time

    Returns
    - r_xyz(t): (N,3) positions at time t (ballistic motion)
    """
    r0_xyz = np.asarray(r0_xyz, dtype=float)
    v_xyz = np.asarray(v_xyz, dtype=float)
    return r0_xyz + v_xyz * float(t)


def gaussian_weights_time(r_xyz_t, t, w0, k_in_hat, k_in=1.0, omega_in=0.0):
    """
    Gaussian envelope in x-y times incident plane-wave phase.

    Inputs
    - r_xyz_t: (N,3) positions at time t
    - t: scalar time
    - w0: Gaussian waist
    - k_in_hat: (3,) incident direction (will be normalized)
    - k_in: incident wavenumber magnitude
    - omega_in: optional carrier frequency term for exp(-i omega t) (set 0 if not needed)

    Returns
    - w: (N,) complex weights at time t
    """
    r_xyz_t = np.asarray(r_xyz_t, dtype=float)
    x, y, z = r_xyz_t[:, 0], r_xyz_t[:, 1], r_xyz_t[:, 2]

    k_in_hat = np.asarray(k_in_hat, dtype=float)
    k_in_hat = k_in_hat / (np.linalg.norm(k_in_hat) + 1e-15)

    env = np.exp(-(x * x + y * y) / (w0 * w0))
    phase_space = k_in * (k_in_hat[0] * x + k_in_hat[1] * y + k_in_hat[2] * z)
    phase_time = -omega_in * float(t)

    return (env * np.exp(1j * (phase_space + phase_time))).astype(np.complex128)


# ---------------------------------------------------------------------
# Time-dependent array factor (general sum with chunking)
# ---------------------------------------------------------------------

def array_factor_general_time(
    nx, ny, nz,
    k_out,
    r0_xyz,
    v_xyz,
    t,
    weight_fn=None,
    chunk_atoms=20000,
):
    """
    General time-dependent array factor:
      AF(n_hat, t) = sum_j w_j(t) * exp(i k_out * n_hat · r_j(t))

    Inputs
    - nx, ny, nz: (nt,np) direction cosines
    - k_out: scalar wavenumber magnitude
    - r0_xyz: (N,3) initial positions
    - v_xyz:  (N,3) velocities (ballistic)
    - t: scalar time
    - weight_fn: None or callable weight_fn(r_xyz_t, t) -> (N,) complex
    - chunk_atoms: atoms per chunk

    Returns
    - AF: (nt,np) complex array factor at time t
    """
    nt, np_ = nx.shape
    r0_xyz = np.asarray(r0_xyz, dtype=float)
    v_xyz = np.asarray(v_xyz, dtype=float)

    if r0_xyz.shape != v_xyz.shape:
        raise ValueError(f"v_xyz must match r0_xyz shape {r0_xyz.shape}, got {v_xyz.shape}")

    N = r0_xyz.shape[0]

    # Flatten directions
    n_hat_flat = np.stack([nx, ny, nz], axis=-1).reshape(-1, 3)  # (M,3)
    M = n_hat_flat.shape[0]

    AF_flat = np.zeros(M, dtype=np.complex128)

    # Compute positions at time t once
    r_t = positions_at_time(r0_xyz, v_xyz, t)

    # Compute weights at time t
    if weight_fn is None:
        w_t = np.ones(N, dtype=np.complex128)
    else:
        w_t = np.asarray(weight_fn(r_t, t), dtype=np.complex128)
        if w_t.shape != (N,):
            raise ValueError(f"weight_fn must return shape (N,), got {w_t.shape}")

    # Chunk over atoms
    for a0 in range(0, N, chunk_atoms):
        r_chunk = r_t[a0 : a0 + chunk_atoms]      # (C,3)
        w_chunk = w_t[a0 : a0 + chunk_atoms]      # (C,)

        # dots: (M,C) = (k_out*n_hat_flat) @ r_chunk.T
        dots = (k_out * n_hat_flat) @ r_chunk.T
        AF_flat += np.exp(1j * dots) @ w_chunk

    return AF_flat.reshape(nt, np_)


def intensity_time(
    nx, ny, nz,
    k_out,
    r0_xyz,
    v_xyz,
    t,
    p_hat,
    intensity_from_field_fn,
    weight_fn=None,
    chunk_atoms=20000,
    normalize=False,
):
    """
    Compute intensity pattern I(theta,phi,t) from AF(t).

    Inputs
    - p_hat: (3,) dipole direction (passed to intensity_from_field_fn)
    - intensity_from_field_fn: your existing intensity_from_field(AF, nx, ny, nz, p_hat)
    - normalize: if True, normalize by max at this time

    Returns
    - I_t: (nt,np) real intensity
    """
    AF_t = array_factor_general_time(
        nx, ny, nz, k_out,
        r0_xyz, v_xyz, t,
        weight_fn=weight_fn,
        chunk_atoms=chunk_atoms,
    )
    I_t = intensity_from_field_fn(AF_t, nx, ny, nz, p_hat)

    if normalize:
        I_t = I_t / (I_t.max() + 1e-15)

    return I_t


# ---------------------------------------------------------------------
# Monte Carlo simulation (average over random realizations)
# ---------------------------------------------------------------------

def monte_carlo_intensity_time_series(
    nx, ny, nz,
    k_out,
    times,
    n_mc,
    pos_sampler_fn,
    vel_sampler_fn,
    p_hat,
    intensity_from_field_fn,
    weight_fn_factory=None,
    chunk_atoms=20000,
    normalize_each_time=False,
    seed=0,
):
    """
    Monte Carlo average of I(theta,phi,t) over many random realizations.

    Inputs
    - times: (T,) array of times
    - n_mc: number of MC realizations
    - pos_sampler_fn: callable rng -> r0_xyz (N,3)
    - vel_sampler_fn: callable (r0_xyz, rng) -> v_xyz (N,3)
    - weight_fn_factory: None or callable that returns a weight_fn for one realization:
          weight_fn = weight_fn_factory(r0_xyz, v_xyz, rng)
      where weight_fn must have signature weight_fn(r_xyz_t, t) -> (N,) complex
      If None, weights are 1.
    - normalize_each_time: if True, normalize I(t) by its own max before averaging
      (often you want False for physical decay comparisons)

    Returns
    - I_mean: (T, nt, np) averaged intensity
    """
    times = np.asarray(times, dtype=float)
    T = times.size
    nt, np_ = nx.shape

    I_accum = np.zeros((T, nt, np_), dtype=float)

    rng_master = np.random.default_rng(seed)

    for mc in range(n_mc):
        # Independent rng per realization (reproducible)
        rng = np.random.default_rng(rng_master.integers(0, 2**63 - 1))

        r0_xyz = np.asarray(pos_sampler_fn(rng), dtype=float)
        v_xyz = np.asarray(vel_sampler_fn(r0_xyz, rng), dtype=float)

        if weight_fn_factory is None:
            weight_fn = None
        else:
            weight_fn = weight_fn_factory(r0_xyz, v_xyz, rng)

        for it, t in enumerate(times):
            I_t = intensity_time(
                nx, ny, nz, k_out,
                r0_xyz, v_xyz, t,
                p_hat,
                intensity_from_field_fn,
                weight_fn=weight_fn,
                chunk_atoms=chunk_atoms,
                normalize=normalize_each_time,
            )
            I_accum[it] += np.asarray(I_t, dtype=float)

    I_mean = I_accum / float(n_mc)
    return I_mean


# ---------------------------------------------------------------------
# Example: plug into your script
# ---------------------------------------------------------------------
#
# 1) Define samplers using your existing helpers:
#
 def pos_sampler(rng):
     # Example: your existing function might not accept rng; adapt as needed
     # return random_position(400)  # if deterministic inside, consider adding rng to it
     return sample_positions_gaussian_xyz(400, sx=2.0, sy=2.0, sz=0.0, rng=rng)

 def vel_sampler(r0_xyz, rng):
     # Thermal velocities (use your function; if it takes seed, pass rng.integers)
     # return random_velocity_thermal(r0_xyz, v_std=0.01, seed=int(rng.integers(1e9)), plane_restricted=True)
     N = r0_xyz.shape[0]
     v = rng.normal(0.0, 0.01, size=(N,3))
     v[:,2] = 0.0
     return v
#
# 2) Define weight_fn_factory that captures beam parameters:
#
# w0 = 10.0
# k_in = 2*np.pi/lam  # if positions are in same length units as lam
# k_in_hat = np.array([1.0, 0.0, 0.0])  # make sure length 3
# omega_in = 0.0
#
# def weight_factory(r0_xyz, v_xyz, rng):
#     def w_fn(r_xyz_t, t):
#         return gaussian_weights_time(r_xyz_t, t, w0=w0, k_in_hat=k_in_hat, k_in=k_in, omega_in=omega_in)
#     return w_fn
#
# 3) Run MC:
#
# times = np.linspace(0.0, 50.0, 21)  # choose your time units consistently with v
# I_mean = monte_carlo_intensity_time_series(
#     nx, ny, nz, k_out,
#     times=times,
#     n_mc=50,
#     pos_sampler_fn=pos_sampler,
#     vel_sampler_fn=vel_sampler,
#     p_hat=p_hat,
#     intensity_from_field_fn=intensity_from_field,
#     weight_fn_factory=weight_factory,
#     chunk_atoms=20000,
#     normalize_each_time=False,
#     seed=123,
# )
#
# Now I_mean[it] is the averaged intensity pattern at times[it]
# Example: plot planar cuts for a specific time index:
# it = 0
# plot_planar_cuts(theta, phi, I_mean[it] / (I_mean[it].max() + 1e-15), title_prefix=f"t={times[it]:.3g}")
