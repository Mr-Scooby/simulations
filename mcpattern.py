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
    filter_kwargs
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
def get_logger(name="mcpattern", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False 

    if logger.handlers:
        return logger

    h = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    h.setFormatter(fmt)
    logger.addHandler(h)
    return logger


log = get_logger()

# ---------------------------------------------------------------------
# Core physics

def positions_at_time(r0_xyz: np.ndarray, v_xyz: np.ndarray, t: float) -> np.ndarray:
    """Ballistic motion: r(t) = r0 + v*t."""
    r0_xyz = np.asarray(r0_xyz, dtype=float)
    v_xyz = np.asarray(v_xyz, dtype=float)

    log.info("Update position. time: %f", t)
    return r0_xyz + v_xyz * float(t)

def sample_realization(n_atoms, rng, **kwargs):
    """Sample random initial positions and velocities for one MC realization."""
    # filter kwargs
    pos_kwargs = filter_kwargs(random_position, kwargs)
    vel_kwargs = filter_kwargs(random_velocity_thermal, kwargs)

    # Sample one realization of initial positions and velocities.
    log.info("Generating random sample.")
    r0_xyz = np.asarray(random_position(n_atoms, seed=int(rng.integers(0, 2**32)),**pos_kwargs) , dtype=float)
    v_xyz = np.asarray(
        random_velocity_thermal(r0_xyz,seed=int(rng.integers(2**32)),
                                **vel_kwargs),dtype=float)

    return r0_xyz, v_xyz


def compute_realization_intensity_series(n_hat_flat, nx, dipole:np.ndarray,  k_out: float,
                                        p_hat: np.ndarray, times: np.ndarray,
                                         n_atoms:int,rng:np.random, w_fn,
                                         chunk_atoms: int = 20000,
                                        normalize_each_time: bool = False,**kwargs
                                        ) -> np.ndarray:
    """
    Compute the intensity time series for one realization of moving atoms.
    basically a single run simulation. deterministic physics. 

    Parameters
    ----------
    n_hat_flat : np.ndarray
        stack array of the irection cosine grids for the observation directions.
    nx: array 
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
    nt, np_ = nx.shape

    # Sample radom atoms and velocity
    r0_xyz, v_xyz = sample_realization(n_atoms,rng, **kwargs)
    
    # memory assignation
    I_series = np.zeros((T, nt, np_), dtype=float)

    log.info(
        "compute_realization_intensity_series : N=%d, T=%d, grid=(%d,%d), chunk_atoms=%d, normalize_each_time=%s",
        len(r0_xyz), T, nt, np_, chunk_atoms, normalize_each_time
    )

    # Time evolution simulation. 
    for it, t in enumerate(times):
        
        # evolve positions.
        rt_xyz = positions_at_time(r0_xyz, v_xyz, t)
        # evolve weights
        wt = w_fn(rt_xyz,t)

        # compute array factor
        AF_t = array_factor_general(
            n_hat_flat, nx, k_out,
            rt_xyz,
            w = wt,
            chunk_atoms=chunk_atoms,
        )

        # compute intensity.
        I_t = intensity_from_field(AF_t,dipole)
        I_t = np.asarray(I_t, dtype=float)

        if normalize_each_time:
            I_t = I_t / (I_t.max() + 1e-15)

        I_series[it] = I_t

    return I_series



def mc_sim( nx, ny, nz,
    k_out: float, p_hat: np.ndarray,
    times: np.ndarray,
    n_mc: int,
    n_atoms: int, 
    w_fn, chunk_atoms: int = 20000, seed: int = 0,
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
    nt, np_ = nx.shape

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
    for mc in range(n_mc):
        # each simulation time keeper
        t_mc = time.time()

        # Independent RNG for this realization.
        rng = np.random.default_rng(rng_master.integers(0, 2**32))

        # Simulation
        I_mc = compute_realization_intensity_series(n_hat_flat= n_hat_flat,
                nx = nx,
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

    return I_mean

# ---------------------------------------------------------------------
# Example main

def main():
    lam = 1.0
    k_out = 2 * np.pi / lam

    # Dipole orientation
    p_hat = np.array([1.0, 0.0, 0.0], dtype=float)
    p_hat /= (np.linalg.norm(p_hat) + 1e-15)

    # Angle grid
    theta, phi, nx, ny, nz = make_angle_grid(n_theta=241, n_phi=481)

    # MC settings
    n_atoms = 400
    n_mc = 30
    times = np.linspace(0.0, 30.0, 16)

    # Incident beam parameters for weights
    w0 = 10.0
    k_in_hat = np.array([0.0, 0.0, 1.0], dtype=float)
    k_in = 1.0  # keep consistent with how gaussian_weights was designed

    log.info("""==== Paramaters =====
             lam=%0.3f,
             Atom number = %d,
             Dipole vector = %s,
             Beam: w0 = %0.3f, k_in = %0.3f, wavevector = %s.
             =====================""", 
             lam, n_atoms, p_hat, w0, k_in, k_in_hat)
    


    def w_fn_factory(rng):
        _ = rng
        # Uses gaussian_weights(r_xyz, w0, k_in_hat)
        # If you want explicit time dependence later, modify make_weight_fn_gaussian_beam.
        return make_weight_fn_gaussian_beam(w0=w0, k_in_hat=k_in_hat, k_in=k_in)

    I_mean = mc_intensity_time_series(
        theta, phi, nx, ny, nz,
        k_out=k_out,
        p_hat=p_hat,
        times=times,
        n_mc=n_mc,
        n_atoms=n_atoms,
        w_fn_factory=w_fn_factory,
        v_std=0.01,
        plane_restricted=True,
        chunk_atoms=20000,
        seed=123,
        normalize_each_time=False,
    )

    # Normalize once for plotting (global normalization across time)
    I_plot0 = I_mean[0] / (np.max(I_mean[0]) + 1e-15)

    # Optional: show one realization geometry
    r0_xyz = random_position(n_atoms)
    v_xyz = random_velocity_thermal(r0_xyz, v_std=0.01, seed=0, plane_restricted=True)
    w0_once = gaussian_weights(np.asarray(r0_xyz), w0, k_in_hat)
    plot_atoms(r0_xyz, w=w0_once, p_hat=p_hat, k_in_hat=k_in_hat, v_xyz=v_xyz, r_subsample=200, v_subsample=30)

    # Plots at t = times[0]
    plot_planar_cuts(theta, phi, I_plot0, title_prefix=f"MC mean, t={times[0]:.3g}")
    plot_pattern_3d(nx, ny, nz, I_plot0, title=f"MC mean pattern, t={times[0]:.3g}", alpha=1.0, stride=2)

    # Example: plot another time index
    it = -1
    I_plotT = I_mean[it] / (np.max(I_mean[it]) + 1e-15)
    plot_planar_cuts(theta, phi, I_plotT, title_prefix=f"MC mean, t={times[it]:.3g}")
    plot_pattern_3d(nx, ny, nz, I_plotT, title=f"MC mean pattern, t={times[it]:.3g}", alpha=1.0, stride=2)

    plt = __import__("matplotlib.pyplot", fromlist=["show"])
    plt.show()


if __name__ == "__main__":

    log.info("Starting mcpattern. TIME DEPENDENC MC SIMULATION")
    main()
