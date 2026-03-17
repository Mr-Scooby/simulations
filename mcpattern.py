#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Monte Carlo simulation for a time-dependent array factor with moving atoms.
"""

import time
import logging
import numpy as np

from helpers import (
    make_angle_grid,
    gaussian_weights,
    intensity_from_field,
    random_position,
    random_velocity_thermal,
    single_dipole_E,
    filter_kwargs,
    atom_weights_sim
)

from rpattern import (
    array_factor_general
    )

from rplotting import (
    plot_pattern_3d,
    plot_planar_cuts,
    plot_atoms,
)


# ---------------------------------------------------------------------
# Logging
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Core physics

def positions_at_time(r0_xyz: np.ndarray, v_xyz: np.ndarray, t: float) -> np.ndarray:
    """Ballistic motion: r(t) = r0 + v*t."""
    r0_xyz = np.asarray(r0_xyz, dtype=float)
    v_xyz = np.asarray(v_xyz, dtype=float)

    log.debug("Update position. time: %f", t)
    return r0_xyz + v_xyz * float(t)

def sample_realization(n_atoms, rng, **kwargs):
    """Sample random initial positions and velocities for one MC realization."""
    # Sample one realization of initial positions and velocities.
    log.info("Generating random sample.")
    r0_xyz = np.asarray(random_position(n_atoms, seed=int(rng.integers(0, 2**32)),
                                        box_size = kwargs.get("box_size"), center = kwargs.get("center"),
                                        plane_restricted = kwargs.get("plane_restricted")
                                        ) , dtype=float)
    v_xyz = np.asarray(
        random_velocity_thermal(r0_xyz,seed=int(rng.integers(2**32)),
                                v_std = kwargs.get("v_std"), plane_restricted = kwargs.get("plane_restricted")),dtype=float)

    return r0_xyz, v_xyz

def compute_realization_af_series_ballistic(
    n_hat_flat: np.ndarray,
    grid_shape,
    k_out: float,
    times: np.ndarray,
    r0_xyz: np.ndarray,
    v_xyz: np.ndarray,
    w_fn,
    chunk_atoms: int = 20000,
    **kwargs
) -> np.ndarray:
    """
    Compute AF(t, theta, phi) for one realization using ballistic motion:
        r_j(t) = r0_j + v_j t

    We precompute, per chunk:
        phi_r0 = k_out * (n_hat · r0)
        phi_v  = k_out * (n_hat · v)

    Then at each time:
        phase(t) = phi_r0 + t * phi_v

    Notes
    -----
    - w_fn must be atom-local, since it is called chunk by chunk.
    - Returns complex AF series with shape (T, nt, np_).
    """
    times = np.asarray(times, dtype=float)
    nt, np_ = grid_shape
    T = times.size
    n_atoms = r0_xyz.shape[0]
    
    log.info("computing AF ... ") 
    AF_series = np.zeros((T, nt * np_), dtype=np.complex128)

    for a0 in range(0, n_atoms, chunk_atoms):
        a1 = min(a0 + chunk_atoms, n_atoms)

        r0_chunk = np.asarray(r0_xyz[a0:a1], dtype=float)   # (C, 3)
        v_chunk  = np.asarray(v_xyz[a0:a1], dtype=float)    # (C, 3)

        # Precompute ballistic phase ingredients for this chunk
        phi_r0 = k_out * (n_hat_flat @ r0_chunk.T)          # (M, C)
        phi_v  = k_out * (n_hat_flat @ v_chunk.T)           # (M, C)

        for it, t in enumerate(times):
            rt_chunk = r0_chunk + t * v_chunk              # (C, 3)
            wt_chunk = np.asarray(w_fn(rt_chunk, t, return_pulse_center = kwargs.get('return_pulse_center') ), dtype=np.complex128)  # (C,)

            AF_series[it] += np.exp(1j * (phi_r0 + t * phi_v)) @ wt_chunk

    
    return AF_series.reshape(T, nt, np_)


def compute_realization_intensity_series(n_hat_flat, grid_shape, dipole:np.ndarray,  k_out: float,
                                        p_hat: np.ndarray, times: np.ndarray,
                                         n_atoms:int,rng:np.random, w_fn,
                                         chunk_atoms: int = 20000,
                                        normalize_each_time: bool = False,
                                         **kwargs
                                        ) -> np.ndarray:
    """
    Compute the intensity time series for one realization of moving atoms.
    basically a single run simulation. deterministic physics. 

    Parameters
    ----------
    n_hat_flat : np.ndarray
        stack array of the irection cosine grids for the observation directions.
    grid_shape: (nt, np_) 
        for shape later 
    k_out : float
        Output wave number.
    p_hat : np.ndarray
        Dipole polarization / orientation unit vector.

    n_atoms: int 
        number of atoms
    times : np.ndarray
        Times at which the intensity is evaluated.
    rng: np.random 
        Radom generator object
    w_fn : callable or None, optional
        Weight function w_fn(r_t, t) for this realization. If None, unit weights are used.
    chunk_atoms : int, optional
        Number of atoms per chunk in the array factor calculation.
    normalize_each_time : bool, optional
        If True, normalize each time snapshot by its own maximum.

    Returns
    -------
    np.ndarray
        Intensity array with shape (T, n_theta, n_phi).
    """

    # time array
    times = np.asarray(times, dtype=float)
    T = times.size
    nt, np_ = grid_shape

    # Sample radom atoms and velocity
    r0_xyz, v_xyz = sample_realization(n_atoms,rng, **kwargs)
    
    # memory assignation
    I_series = np.zeros((T, nt, np_), dtype=float)

    log.info(
        "compute_realization_intensity_series : N=%d, T=%d, grid=(%d,%d), chunk_atoms=%d, normalize_each_time=%s",
        len(r0_xyz), T, nt, np_, chunk_atoms, normalize_each_time
    )

    AF_series = compute_realization_af_series_ballistic(
        n_hat_flat=n_hat_flat,
        grid_shape = grid_shape,
        k_out=k_out,
        times=times,
        r0_xyz=r0_xyz,
        v_xyz=v_xyz,
        w_fn=w_fn,
        chunk_atoms=chunk_atoms,
        **kwargs
    )

    log.info("Computing intensities...") 
    for it in range(T):
        I_series[it] = intensity_from_field(
            AF_series[it],
            dipole=dipole,
        )


    return I_series



def mc_sim( nx,ny,nz, grid_shape,
    k_out: float, p_hat: np.ndarray,
    times: np.ndarray,
    n_mc: int,
    n_atoms: int, 
    w_fn,
    chunk_atoms: int = 20000, seed: int = 0,
    normalize_each_time: bool = False,
    **kwargs
) -> np.ndarray:
    """
    Monte Carlo average of the intensity time series for moving atoms.

    Each realization samples a new atomic configuration internally and
    computes its intensity time series. The result returned here is the
    average over all realizations.

    Parameters
    ----------
    nx, ny, nz : np.ndarray
        Direction cosine grids for the observation directions.
    k_out : float
        Output wave number.
    p_hat : np.ndarray
        Dipole orientation / polarization vector.
    times : np.ndarray
        Times at which the intensity is evaluated.
    n_mc : int
        Number of Monte Carlo realizations.
    n_atoms : int
        Number of atoms in each realization.
    chunk_atoms : int, optional
        Chunk size used in the array factor calculation.
    seed : int, optional
        Master seed for reproducible Monte Carlo sampling.
    normalize_each_time : bool, optional
        If True, normalize each time snapshot by its own maximum before averaging.
    w_fn_factory : callable or None, optional
        Callable with signature w_fn_factory(rng) -> w_fn(r_t, t).
        If None, unit weights are used.

    Returns
    -------
    np.ndarray
        Monte Carlo averaged intensity series with shape (T, n_theta, n_phi).
    """

    times = np.asarray(times, dtype=float)
    T = times.size
    nt, np_ = grid_shape 

    log.info(
        "mc_sim: n_mc=%d, n_atoms=%d, T=%d, grid=(%d,%d), "
        "chunk_atoms=%d,  normalize_each_time=%s",
        n_mc, n_atoms, T, nt, np_, chunk_atoms, 
         normalize_each_time,
    )

    # Flatten directions: n_hat_flat (M,3), M=nt*np
    n_hat_flat = np.stack([nx, ny, nz], axis=-1).reshape(-1, 3)
    dipole = single_dipole_E(nx,ny,nz, p_hat) 

    I_accum = np.zeros((T, nt, np_), dtype=float)
    # Random generator object
    # helps to reproduce runs. 
    rng_master = np.random.default_rng(seed)
    # time to measure how long it takes. 
    t_start = time.time()

    # Runs of the simulation
    log.info("Starting mc runs ... 1 /%d", n_mc)
    for mc in range(n_mc):
        # each simulation time keeper
        t_mc = time.time()

        # Independent RNG for this realization.
        rng = np.random.default_rng(rng_master.integers(0, 2**32))

        # Simulation
        I_mc = compute_realization_intensity_series(n_hat_flat= n_hat_flat,
                grid_shape = grid_shape, 
                dipole= dipole, 
                k_out=k_out,
                p_hat=p_hat,
                times=times,
                n_atoms = n_atoms,
                rng=rng,
                chunk_atoms=chunk_atoms,
                normalize_each_time=normalize_each_time,
                w_fn=w_fn, **kwargs
        )

        # Intensity acumalation. 
        I_accum += I_mc

        log.info(
            "mc run %d/%d done in %.2fs",
            mc + 1, n_mc, time.time() - t_mc
        )

    I_mean = I_accum / float(n_mc)

    log.info(
        "mc_intensity_time_series end: total=%.2fs, avg_per_mc=%.2fs",
        time.time() - t_start,
        (time.time() - t_start) / max(float(n_mc), 1.0),
    )

    log.info("Simulating atom and beam movement")
    r_xyz, v_xyz = sample_realization(n_atoms, rng, **kwargs) 
    pos, weight, pulscenter = atom_weights_sim(times, r_xyz, v_xyz, w_fn)

    return I_mean, pos, weight, pulscenter
