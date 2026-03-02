#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from helpers import (
    sample_positions_gaussian_xyz,
    make_angle_grid,
    atom_grid_2d,
    gaussian_weights_from_XY,
    intensity_from_field,
    plot_pattern_3d,
    plot_planar_cuts,
)

import logging

def get_logger(name="mc_generalAF", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent messages from being passed to the root logger
    logger.propagate = False

    # Add handler only once (avoid duplicate lines in notebooks / reruns)
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                                datefmt="%H:%M:%S")
        h.setFormatter(fmt)
        logger.addHandler(h)

    return logger

log = get_logger()

log.info("starting")


def array_factor_general_rt(
    nx, ny, nz,
    k_out: float,
    r0_xyz: np.ndarray,
    t: float,
    v_xyz: np.ndarray | None = None,
    weight_fn=None,
    extra_phase_fn=None,
    chunk_atoms: int = 20000,
    chunk_dirs: int | None = None,
    log=None,
    log_every: int = 10,
):
    """
    General AF with motion and time-dependent weights.

    AF(n, t) = Σ_j  w(r_j(t), t) * exp(i * extra_phase(r_j(t), t)) * exp(i * k_out * n·r_j(t))

    Inputs
    ------
    nx,ny,nz : (n_theta, n_phi) floats
        Direction cosines grid defining n_hat.
    k_out : float
        Emitted wave number (2π/λ in your units).
    r0_xyz : (N,3) floats
        Initial atom positions at t=0.
    t : float
        Time.
    v_xyz : (N,3) floats or None
        Velocities. If None -> atoms fixed: r(t)=r0.
    weight_fn : callable or None
        weight_fn(r_t, t) -> (N,) complex weights.
        If None -> all ones.
    extra_phase_fn : callable or None
        extra_phase_fn(r_t, t) -> (N,) real phase (radians).
        If None -> zeros.
    chunk_atoms : int
        Chunk size over atoms (controls memory).
    chunk_dirs : int or None
        If None: process all directions at once (fastest but can use lots of RAM).
        If set: process directions in blocks of this size (gives progress even for small N).
    log : logging.Logger or None
        Logger for progress.
    log_every : int
        Log every this many blocks.

    Returns
    -------
    AF : (n_theta, n_phi) complex
    """
    r0_xyz = np.asarray(r0_xyz, dtype=float)
    if r0_xyz.ndim != 2 or r0_xyz.shape[1] != 3:
        raise ValueError("r0_xyz must have shape (N,3)")

    N = r0_xyz.shape[0]

    if v_xyz is None:
        # no motion
        r_t = r0_xyz
    else:
        v_xyz = np.asarray(v_xyz, dtype=float)
        if v_xyz.shape != r0_xyz.shape:
            raise ValueError("v_xyz must have same shape as r0_xyz")
        r_t = r0_xyz + v_xyz * t

    # weights w(r(t),t)
    if weight_fn is None:
        w_all = np.ones(N, dtype=np.complex128)
    else:
        w_all = np.asarray(weight_fn(r_t, t), dtype=np.complex128)
        if w_all.shape != (N,):
            raise ValueError(f"weight_fn must return shape (N,), got {w_all.shape}")

    # extra phase φ(r(t),t)
    if extra_phase_fn is None:
        phi_all = np.zeros(N, dtype=float)
    else:
        phi_all = np.asarray(extra_phase_fn(r_t, t), dtype=float)
        if phi_all.shape != (N,):
            raise ValueError(f"extra_phase_fn must return shape (N,), got {phi_all.shape}")

    # flatten directions
    nt, np_ = nx.shape
    n_hat_flat = np.stack([nx, ny, nz], axis=-1).reshape(-1, 3)   # (Mdir,3)
    Mdir = n_hat_flat.shape[0]

    # choose direction chunking
    if chunk_dirs is None:
        # --------- chunk atoms (like your current AF) ----------
        AF_flat = np.zeros(Mdir, dtype=np.complex128)

        n_chunks = (N + chunk_atoms - 1) // chunk_atoms
        for ci, a0 in enumerate(range(0, N, chunk_atoms), start=1):
            r = r_t[a0:a0+chunk_atoms]          # (C,3)
            ww = w_all[a0:a0+chunk_atoms]       # (C,)
            ph = phi_all[a0:a0+chunk_atoms]     # (C,)

            dots = (k_out * n_hat_flat) @ r.T   # (Mdir,C)
            # exp(i*(k n·r + phi))
            AF_flat += np.exp(1j * (dots + ph[None, :])) @ ww

            if log is not None and (ci == 1 or ci == n_chunks or ci % log_every == 0):
                log.info("AF(t=%.3g) atom chunk %d/%d", t, ci, n_chunks)

        return AF_flat.reshape(nt, np_)

    else:
        # --------- chunk directions (better for progress when N is small, controls RAM) ----------
        AF_flat = np.zeros(Mdir, dtype=np.complex128)
        rT = r_t.T  # (3,N)

        n_blocks = (Mdir + chunk_dirs - 1) // chunk_dirs
        for bi, d0 in enumerate(range(0, Mdir, chunk_dirs), start=1):
            d1 = min(d0 + chunk_dirs, Mdir)
            nh = n_hat_flat[d0:d1]                  # (B,3)
            dots = (k_out * nh) @ rT                # (B,N)
            # multiply per atom: w * exp(i phi)
            atom_factor = w_all * np.exp(1j * phi_all)  # (N,)
            AF_flat[d0:d1] = np.exp(1j * dots) @ atom_factor

            if log is not None and (bi == 1 or bi == n_blocks or bi % log_every == 0):
                log.info("AF(t=%.3g) dir block %d/%d", t, bi, n_blocks)

        return AF_flat.reshape(nt, np_)


def make_weight_pulse_beam(w0, z0, sigma_z, v_g):
    """
    w(r,t) = exp(-(x^2+y^2)/w0^2) * exp(-( (z - z0 - v_g t)^2 )/(2 sigma_z^2))
    """
    def weight_fn(r, t):
        x, y, z = r[:,0], r[:,1], r[:,2]
        transverse = np.exp(-(x*x + y*y)/(w0*w0))
        pulse = np.exp(-0.5 * ((z - z0 - v_g*t)/sigma_z)**2)
        return (transverse * pulse).astype(np.complex128)
    return weight_fn

def make_extra_phase_kin(k_in_vec):
    k_in_vec = np.asarray(k_in_vec, dtype=float).reshape(3,)
    def extra_phase_fn(r, t):
        return r @ k_in_vec  # (N,)
    return extra_phase_fn


# Example: grid positions (Nx,Ny) at z=0
r0_xyz, Xg, Yg = atom_grid_2d(Nx, Ny, dx, dy, z0=0.0)

# Example: random velocities (optional)
rng = np.random.default_rng(0)
v_xyz = np.zeros_like(r0_xyz)
v_xyz[:,0] = rng.normal(0.0, 0.02, size=r0_xyz.shape[0])
v_xyz[:,1] = rng.normal(0.0, 0.02, size=r0_xyz.shape[0])
v_xyz[:,2] = rng.normal(0.0, 0.02, size=r0_xyz.shape[0])

weight_fn = make_weight_pulse_beam(w0=4.0, z0=-10.0, sigma_z=3.0, v_g=1.0)
extra_phase_fn = make_extra_phase_kin(k_in_vec=np.array([0.0, 0.0, k_out]))

t = 0.0
AF = array_factor_general_rt(
    nx, ny, nz,
    k_out=k_out,
    r0_xyz=r0_xyz,
    v_xyz=v_xyz,
    t=t,
    weight_fn=weight_fn,
    extra_phase_fn=extra_phase_fn,
    # choose ONE of these chunking strategies:
    chunk_atoms=20000,
    # chunk_dirs=20000,   # <- enable this if you prefer direction-block progress + lower RAM
    log=log,
    log_every=5,
)
