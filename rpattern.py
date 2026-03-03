#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from helpers import (
    make_angle_grid,
    atom_grid,
    gaussian_weights,
    intensity_from_field,
    random_position, 
    random_velocity_thermal
    )

from rplotting import ( 
    plot_pattern_3d,
    plot_planar_cuts,
    plot_atoms,
)

import logging

def get_logger(name="rpattern", level=logging.INFO):
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


#################################################################################
# Array factor
def array_factor_general(nx, ny, nz, k_out, r_xyz, w=None, chunk_atoms=20000):
    """
    General array factor:
        AF(n_hat) = sum_j w_j * exp(i k_out n_hat · r_j)

    Inputs:
      nx,ny,nz : (nt,np) direction cosines on the sphere
      k_out        : scalar wave number
      r_xyz    : (N,3) atom positions
      w        : (N,) complex weights (amplitude*exp(i phase)). If None -> all ones.
      chunk    : atoms per chunk (tune for speed/memory)

    Returns:
      AF : (nt,np) complex array factor
    """
    nt, np_ = nx.shape
    N = r_xyz.shape[0]

    # Flatten directions: n_hat_flat (M,3), M=nt*np
    n_hat_flat = np.stack([nx, ny, nz], axis=-1).reshape(-1, 3)
    M = n_hat_flat.shape[0]
    
    # Assign memory 
    AF_flat = np.zeros(M, dtype=np.complex128)

    M = nx.size      # n_theta * n_phi
    N = r_xyz.shape[0]
    log.info("AF: M directions = %d, N atoms = %d, M*N = %d", M, N, M*N)

    if w is None:
        w = np.ones(N, dtype=np.complex128)
        log.i-nfo("AF: Weights None") 
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
        r = r_xyz[a0 : a0 + chunk_atoms]      # (C,3)
        ww = w[a0 : a0 + chunk_atoms]         # (C,)

        # dots: (M,C) = (k_out*n_hat_flat) @ r.T
        dots = (k_out * n_hat_flat) @ r.T
        AF_flat += np.exp(1j * dots) @ ww

    log.info("AF: ended")
    return AF_flat.reshape(nt, np_)


# Array factor (separable lattice)
# ---------------------------
def centered_indices(N):
    return np.arange(N) - (N - 1)/2

def array_factor_separable(nx, ny, nz, k, dx, dy, dz, Nx, Ny, Nz):
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

################################################################################
################################################################################
################################################################################

# Parameters
lam = 1.0
k_out = 2 * np.pi / lam

# dipole orientation
p_hat = np.array([1.0, 0.0, 0.0], dtype=float)

# atoms: 2D lattice / Number of atoms.
Nx = 10  
Ny = 10
Nz = 10

N = Nx*Ny*Nz

## Atomic Interdistance
dx = dy = dz= 0.5 * lam

# Array factor weights: Gaussian beam envelope
w0 = 10.0 # waist
k_in= 1   # wavevector magnitude
k_in_dir = np.array([0.0,3.0, 1.0]) # wavevector direction

alpha = 1.0  # radius scaling for 3D plot

log.info("""==== Paramaters =====
         lam=%0.3f,
         Atom number = %d,
         Dipole vector = %s,
         Beam: w0 = %0.3f, k_in = %0.3f, wavevector = %s.
         =====================""", 
         lam, N, p_hat, w0, k_in, k_in_dir)

# Normalization of vectors
p_hat /= (np.linalg.norm(p_hat) + 1e-15) # Dipole vector
k_in_hat = k_in_dir / (np.linalg.norm(k_in_dir) + 1e-15) # Incident wave wavevector

# Construction of vectors arrays
## Position vectors
r_xyz = random_position(N, plane_restricted= False)
#r_xyz = atom_grid(Nx, Ny, Nz, dx, dy,dz, plane= False)

## Velocity vectors
v_xyz = None #  random_velocity_thermal(r_xyz)

## Array factor Weights. 
w = gaussian_weights(r_xyz, w0, k_in_hat)

# angle grid
theta, phi, nx, ny, nz = make_angle_grid(n_theta=241, n_phi=481) # Grid resolution

# Compute
# ----------------------------
AF = array_factor_general(nx, ny, nz, k_out, r_xyz, w=w, chunk_atoms=20000)
I = intensity_from_field(AF, nx, ny, nz, p_hat)
I /= (I.max() + 1e-15)

sanity_printing()

# Plot
# ----------------------------
plot_atoms(r_xyz,w=w,p_hat = p_hat, k_in_hat =k_in_hat, v_xyz = v_xyz )
title = f"Radiation pattern: {Nx}x{Ny}, d={dx/lam:.2f}λ, w0={w0}"
plot_pattern_3d(nx, ny, nz, I, title=title, alpha=1.0, stride=2)

# Optional: a theta cut at phi=0
plot_planar_cuts(theta, phi, I, title_prefix="Theta cut (phi=0)")

plt.show()
