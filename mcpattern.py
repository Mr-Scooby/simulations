#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
)

from rplotting import (
    plot_pattern_3d,
    plot_planar_cuts,
    plot_atoms,
)


# ---------------------------------------------------------------------
# Logging
def get_logger(name="mc_af", level=logging.INFO):
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
    return r0_xyz + v_xyz * float(t)


def array_factor_general_time(
    nx: np.ndarray,
    ny: np.ndarray,
    nz: np.ndarray,
    k_out: float,
    r0_xyz: np.ndarray,
    v_xyz: np.ndarray,
    t: float,
    w_fn=None,
    chunk_atoms: int = 20000,
) -> np.ndarray:
    """
    Time-dependent general array factor:
      AF(n_hat,t) = sum_j w_j(t) * exp(i k_out * n_hat · r_j(t))

    Inputs
    - nx,ny,nz: (nt,np) direction cosines
    - k_out: scalar wavenumber
    - r0_xyz: (N,3) initial positions
    - v_xyz:  (N,3) velocities
    - t: scalar time
    - w_fn: callable w_fn(r_t, t) -> (N,) complex, or None for all ones
    - chunk_atoms: chunk size for atom summation

    Returns
    - AF: (nt,np) complex
    """
    nt, np_ = nx.shape
    r0_xyz = np.asarray(r0_xyz, dtype=float)
    v_xyz = np.asarray(v_xyz, dtype=float)

    if v_xyz.shape != r0_xyz.shape:
        raise ValueError(f"v_xyz must have shape {r0_xyz.shape}, got {v_xyz.shape}")

    N = r0_xyz.shape[0]

    n_hat_flat = np.stack([nx, ny, nz], axis=-1).reshape(-1, 3)  # (M,3)
    M = n_hat_flat.shape[0]
    AF_flat = np.zeros(M, dtype=np.complex128)

    r_t = positions_at_time(r0_xyz, v_xyz, t)

    if w_fn is None:
        w_t = np.ones(N, dtype=np.complex128)
    else:
        w_t = np.asarray(w_fn(r_t, t), dtype=np.complex128)
        if w_t.shape != (N,):
            raise ValueError(f"w_fn must return shape (N,), got {w_t.shape}")

    for a0 in range(0, N, chunk_atoms):
        r = r_t[a0:a0 + chunk_atoms]         # (C,3)
        ww = w_t[a0:a0 + chunk_atoms]        # (C,)
        dots = (k_out * n_hat_flat) @ r.T    # (M,C)
        AF_flat += np.exp(1j * dots) @ ww

    return AF_flat.reshape(nt, np_)


# ---------------------------------------------------------------------
# Monte Carlo driver
# ---------------------------------------------------------------------

def make_weight_fn_gaussian_beam(w0: float, k_in_hat: np.ndarray, k_in: float = 1.0):
    """
    Returns a w_fn(r_t, t) that uses your existing gaussian_weights(r_xyz, w0, k_in_hat).
    If your gaussian_weights has extra args (like k_in), adapt inside this function.
    """
    k_in_hat = np.asarray(k_in_hat, dtype=float)
    k_in_hat = k_in_hat / (np.linalg.norm(k_in_hat) + 1e-15)

    def w_fn(r_t: np.ndarray, t: float):
        _ = t  # time available if you later extend weights to include explicit time terms
        return gaussian_weights(r_t, w0, k_in_hat)  # <-- adjust if your signature differs

    return w_fn


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
    times = np.asarray(times, dtype=float)
    T = times.size
    nt, np_ = nx.shape

    I_accum = np.zeros((T, nt, np_), dtype=float)

    rng_master = np.random.default_rng(seed)

    for mc in range(n_mc):
        rng = np.random.default_rng(rng_master.integers(0, 2**63 - 1))

        # Your helpers create one realization
        r0_xyz = np.asarray(random_position(n_atoms), dtype=float)
        v_xyz = np.asarray(
            random_velocity_thermal(r0_xyz, v_std=v_std, seed=int(rng.integers(1_000_000_000)), plane_restricted=plane_restricted),
            dtype=float
        )

        w_fn = None if (w_fn_factory is None) else w_fn_factory(rng)

        t0 = time.time()
        for it, t in enumerate(times):
            AF_t = array_factor_general_time(
                nx, ny, nz, k_out,
                r0_xyz, v_xyz, t,
                w_fn=w_fn,
                chunk_atoms=chunk_atoms
            )
            I_t = intensity_from_field(AF_t, nx, ny, nz, p_hat)
            I_t = np.asarray(I_t, dtype=float)

            if normalize_each_time:
                I_t = I_t / (I_t.max() + 1e-15)

            I_accum[it] += I_t

        dt = time.time() - t0
        log.info("mc %d/%d done (N=%d, T=%d) in %.2fs", mc + 1, n_mc, n_atoms, T, dt)

    I_mean = I_accum / float(n_mc)
    return I_mean


# ---------------------------------------------------------------------
# Example main
# ---------------------------------------------------------------------

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
    main()
