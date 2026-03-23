#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from radpattern.helpers.helpers import (
    make_angle_grid,
    atom_grid,
    gaussian_weights,
    intensity_from_field,
    random_position, 
    random_velocity_thermal,
    single_dipole_E
    )

import logging

log = logging.getLogger(__name__)

#################################################################################
# Array factor
# Flatten directions: n_hat_flat (M,3), M=nt*np
#n_hat_flat = np.stack([nx, ny, nz], axis=-1).reshape(-1, 3)

def array_factor_general(n_hat_flat, grid_shape, k_out, r_xyz, w=None, chunk_atoms=2000):
    """
    General array factor:
        AF(n_hat) = sum_j w_j * exp(i k_out n_hat · r_j)

    Inputs:
      n_hat:_flat: (M,3) array: shpuld be np.stack[nx,ny,nz]: (nt,np) direction cosines on the sphere  #n_hat_flat = np.stack([nx, ny, nz], axis=-1).reshape(-1, 3)
      grid_shape: (nt, np_) direction cosine. LAter for reshape. 
      k_out        : scalar wave number
      r_xyz    : (N,3) atom positions
      w        : (N,) complex weights (amplitude*exp(i phase)). If None -> all ones.
      chunk    : atoms per chunk (tune for speed/memory)

    Returns:
      AF : (nt,np) complex array factor
    """
    nt, np_ = grid_shape
    # atom number
    N = r_xyz.shape[0]
    # grid size flat
    M = n_hat_flat.shape[0]
    
    # Assign memory 
    AF_flat = np.zeros(M, dtype=np.complex128)

    if w is None:
        w = np.ones(N, dtype=np.complex128)
        log.info("AF: Weights None") 
    else:
        w = np.asarray(w, dtype=np.complex128)
        log.info("AF: Weights provided")
        if w.shape != (N,):
            raise ValueError(f"w must have shape (N,), got {w.shape}")

    
    n_chunks = (N + chunk_atoms - 1) // chunk_atoms
    # Chunk over atoms to control memory
    for a0 in range(0, N, chunk_atoms):
        ci = a0 // chunk_atoms + 1
        log.info("AF: chunk %d/%d", ci, n_chunks)
        a1 = min(a0 + chunk_atoms, N)
        r = r_xyz[a0:a1]
        ww = w[a0:a1]

        phase = k_out * (n_hat_flat @ r.T)
        AF_flat +=np.exp(1j * phase) @ ww

    log.info("AF: ended")
    return AF_flat.reshape(*grid_shape)


# Array factor (separable lattice)
# ---------------------------
def centered_indices(N):
    return np.arange(N) - (N - 1)/2

def array_factor_separable(nx, ny, nz, k, dx, dy, dz, Nx, Ny, Nz):
    log.info("AF separable")
    mx = centered_indices(Nx)[:, None, None]
    my = centered_indices(Ny)[:, None, None]
    mz = centered_indices(Nz)[:, None, None]

    ux = k * dx * nx[None, :, :]
    uy = k * dy * ny[None, :, :]
    uz = k * dz * nz[None, :, :]

    Sx = np.sum(np.exp(1j * mx * ux), axis=0)
    Sy = np.sum(np.exp(1j * my * uy), axis=0)
    Sz = np.sum(np.exp(1j * mz * uz), axis=0)
    return Sx * Sy * Sz


# ---- sanity checks for x-dipole with 1 atom ----
def get_I_at(th0, ph0):
    i = np.argmin(np.abs(theta - th0))
    dphi = np.angle(np.exp(1j*(phi - ph0)))
    j = np.argmin(np.abs(dphi))
    return float(I[i, j])

def sanity_printing():
    print("Sanity :")
    print("  I(+x):", get_I_at(np.pi/2, 0.0))
    print("  I(+y):", get_I_at(np.pi/2, np.pi/2))
    print("  I(+z):", get_I_at(0.0, 0.0))
