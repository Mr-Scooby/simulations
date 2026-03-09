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
    single_dipole_E
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


def make_weight_fn_gaussian_beam(w0: float, k_in_hat: np.ndarray, k_in: float = 1.0):
    """
    Returns a w_fn(r_t, t) that uses your existing gaussian_weights(r_xyz, w0, k_in_hat).
    If your gaussian_weights has extra args (like k_in), adapt inside this function.
    """
    log.info("Time dependent weight")
    k_in_hat = np.asarray(k_in_hat, dtype=float)
    k_in_hat = k_in_hat / (np.linalg.norm(k_in_hat) + 1e-15)

    def w_fn(r_t: np.ndarray, t: float):
        _ = t  # time available if you later extend weights to include explicit time terms
        return gaussian_weights(r_t, w0, k_in_hat)  # <-- adjust if your signature differs

    return w_fn

def sample_realization(n_atoms, v_std, plane_restricted, rng):
    """Sample initial positions and velocities for one MC realization."""
    # Sample one realization of initial positions and velocities.
    log.info("Generating random sample")
    r0_xyz = np.asarray(random_position(n_atoms, seed=int(rng.integers(0, 2**32))) , dtype=float)
    v_xyz = np.asarray(
        random_velocity_thermal(r0_xyz,v_std=v_std,seed=int(rng.integers(2**32)),
                                plane_restricted=plane_restricted),
        dtype=float)

    return r0_xyz, v_xyz


def weight_evolution(r_xyz, t) -> np.ndarray: 
    """computes the weights factors for time t
    
    iputs: 
    r_xyz: np.ndarry (N,3). with the atoms position.
    t: float, time step. 

    output:
    w: np.ndarray (N,3) weights updated"""

    return np.ones(r_xyz.shape[0], dtype=complex)

def compute_realization_intensity_series(n_hat_flat, nx, dipole:np.ndarray,  k_out: float,
                                        p_hat: np.ndarray, times: np.ndarray,
                                         n_atoms:int, v_std: float, rng:np.random, plane_restricted: bool, 
                                        w_fn = weight_evolution , chunk_atoms: int = 20000,
                                        normalize_each_time: bool = False,
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
    v_std: float
        velocity standard deviation
    plane_restricted: Bool
        Dimension restricted to 2D. 
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

    
    # Sample atoms and velocity
    r0_xyz, v_xyz = sample_realization(n_atoms, v_std, plane_restricted, rng)
    
    # memory assignation
    I_series = np.zeros((T, nt, np_), dtype=float)

    log.info(
        "compute_realization_intensity_series : N=%d, T=%d, grid=(%d,%d), chunk_atoms=%d, normalize_each_time=%s",
        len(r0_xyz), T, nt, np_, chunk_atoms, normalize_each_time
    )
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
    n_atoms: int, v_std: float = 0.01,
    plane_restricted: bool = True,
    chunk_atoms: int = 20000, seed: int = 0,
    normalize_each_time: bool = False,
    w_fn=weight_evolution,
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
    v_std : float, optional
        Thermal velocity spread used when sampling each realization.
    plane_restricted : bool, optional
        Whether sampled velocities are restricted to a plane.
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
        "chunk_atoms=%d, v_std=%.3g, plane_restricted=%s, normalize_each_time=%s",
        n_mc, n_atoms, T, nt, np_, chunk_atoms, v_std,
        plane_restricted, normalize_each_time,
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

        I_mc = compute_realization_intensity_series(n_hat_flat= n_hat_flat,
                nx = nx,
                dipole= dipole, 
                k_out=k_out,
                p_hat=p_hat,
                times=times,
                n_atoms = n_atoms,
                rng=rng,
                v_std=v_std,
                plane_restricted=plane_restricted,
                chunk_atoms=chunk_atoms,
                normalize_each_time=normalize_each_time,
                w_fn=w_fn,
        )

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

def mc_intensity_time_series(
    theta: np.ndarray,
    phi: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
    nz: np.ndarray,
    k_out: float,
    p_hat: np.ndarray,
    times: np.ndarray,
    n_mc: int,
    n_atoms: int,
    w_fn_factory=None,
    v_std: float = 0.01,
    plane_restricted: bool = True,
    chunk_atoms: int = 20000,
    seed: int = 0,
    normalize_each_time: bool = False,
):
    """
    Monte Carlo average of intensity patterns with moving atoms.

    Inputs
    - times: (T,) times
    - n_mc: number of realizations
    - n_atoms: atoms per realization
    - w_fn_factory: callable (rng) -> w_fn(r_t, t). If None, weights=1
    - v_std, plane_restricted: passed to random_velocity_thermal
    - normalize_each_time: normalize each I(t) by its own max before averaging

    Returns
    - I_mean: (T, n_theta, n_phi)
    """
    # Logging time it takes to run. 
    t_start = time.time()

    times = np.asarray(times, dtype=float)
    T = times.size
    nt, np_ = nx.shape

    log.info(
        "mc_intensity_moving_atoms start: n_mc=%d, n_atoms=%d, T=%d, grid=(%d,%d), chunk_atoms=%d, v_std=%.3g, plane=%s, normalize_each_time=%s",
        n_mc, n_atoms, T, nt, np_, chunk_atoms, v_std, plane_restricted, normalize_each_time
    )

    I_accum = np.zeros((T, nt, np_), dtype=float) # Acumulated intensity array 
    rng_master = np.random.default_rng(seed) # random number generator object


    # Montecarlo realization:
    for mc in range(n_mc):
        # New random generator for each MC simulation.
        # each mcsimulation is independent but reproducible. 
        rng = np.random.default_rng(rng_master.integers(0, 2**32)) 

        # Sample one realization of initial positions and velocities.
        r0_xyz = np.asarray(random_position(n_atoms, seed=int(rng.integers(0, 2**32))), dtype=float)
        v_xyz = np.asarray(
            random_velocity_thermal(
                r0_xyz,
                v_std=v_std,
                seed=int(rng.integers(2**32)),
                plane_restricted=plane_restricted,
            ),
            dtype=float,
        )

        # Build a per-realization weight function if a factory is provided.
        w_fn = None if (w_fn_factory is None) else w_fn_factory(rng)

        t_mc = time.time()
        for it, t in enumerate(times):
            # Compute the time-dependent array factor for all observation directions.
            AF_t = array_factor_general_time(
                nx, ny, nz, k_out,
                r0_xyz, v_xyz, t,
                w_fn=w_fn,
                chunk_atoms=chunk_atoms,
            )

            # Convert field-like quantity into intensity including dipole/polarization effects.
            I_t = intensity_from_field(AF_t, nx, ny, nz, p_hat)
            I_t = np.asarray(I_t, dtype=float)

            # Normalize each snapshot by its own maximum if requested.
            if normalize_each_time:
                I_t = I_t / (I_t.max() + 1e-15)

            # Accumulate intensity for later averaging across realizations.
            I_accum[it] += I_t

        log.info(
            "mc %d/%d done in %.2fs (N=%d, T=%d)",
            mc + 1, n_mc, time.time() - t_mc, n_atoms, T
        )

    I_mean = I_accum / float(n_mc)

    log.info(
        "mc_intensity_moving_atoms end: total=%.2fs, avg_per_mc=%.2fs",
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
