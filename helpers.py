#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# helpers.py

import numpy as np
import matplotlib.pyplot as plt
import logging

def get_logger(name="helpers", level=logging.INFO):
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

# Angle grids / directions
def make_angle_grid(n_theta=241, n_phi=481):
    """
    Returns:
      theta (n_theta,1)
      phi   (n_phi,1)
      nx, ny, nz  (n_theta, n_phi) each
    """
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")

    nx = np.sin(TH) * np.cos(PH)
    ny = np.sin(TH) * np.sin(PH)
    nz = np.cos(TH)

    log.info("Building angle grid. Resolution %d x %d", n_theta, n_phi) 
    return theta, phi, nx, ny, nz

# Dipole pattern
PZ_HAT = np.array([0.0, 0.0, 1.0], dtype=float) # +z vector 
def single_dipole_E(nx, ny, nz, p_hat = PZ_HAT):
    """
    Far-field dipole angular dependence:
      E propto n X (n X p) = p - n (n·p)

    p_hat: (3,) unit vector (or will work if not perfectly unit) as array
    Returns Ex,Ey,Ez complex arrays with same shape as nx.
    """

    ndotp = nx * p_hat[0] + ny * p_hat[1] + nz * p_hat[2]
    Ex = p_hat[0] - nx * ndotp
    Ey = p_hat[1] - ny * ndotp
    Ez = p_hat[2] - nz * ndotp
    
    log.info("Construction of single dipole pattern. Dipole vector %s", p_hat)
    return (np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2)


def intensity_from_field(AF, dipole):
    """
    I = |E|^2 where E = AF * E_single_dipole
    """
    log.info("Computing intensity from field: E shape=%s.", AF.shape)
    return np.abs(AF)**2 * dipole 


# Atom layouts + weights
def atom_grid(Nx, Ny, Nz=1, dx=1.0, dy=1.0, dz=1.0, z0=0.0, plane_restricted=True):
    """
    Centered grid positions as an (N,3) array.

    plane=True  -> XY grid at fixed z=z0 (Nz ignored)
    plane=False -> full 3D grid centered around z0
    """
    x = (np.arange(Nx) - (Nx - 1) / 2.0) * dx
    y = (np.arange(Ny) - (Ny - 1) / 2.0) * dy

    if plane_restricted:
        # (Nx*Ny, 3)
        X, Y = np.meshgrid(x, y, indexing="ij")
        r_xyz = np.column_stack((X.ravel(), Y.ravel(), np.full(X.size, float(z0))))
        return r_xyz

    z = (np.arange(Nz) - (Nz - 1) / 2.0) * dz + z0
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    r_xyz = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

    log.info("Construction position vector: GRID: Nx=%d, Ny=%d, Nz=%d, spacing=%.3f lambda, N=%d, r_xyz shape=%s, Plane_restricted  = %s", Nx, Ny, Nz, dx, r_xyz.shape[0], r_xyz.shape, plane_restricted)
    return r_xyz


def random_position(N, box_size=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), seed=0, plane_restricted=True):
    """
    Generate random positions uniformly inside a rectangular box.

    Inputs
    - N: number of particles
    - box_size: (Lx, Ly, Lz) side lengths of the box
    - center: (cx, cy, cz) center of the box
    - seed: random seed for reproducibility
    - plane_restricted: if True, set z = cz (positions only in x-y plane)

    Returns
    - r_xyz: (N, 3) positions
    """
    rng = np.random.default_rng(seed)

    Lx, Ly, Lz = box_size
    cx, cy, cz = center

    # Uniform in [-L/2, +L/2] for each axis, then shift to the chosen center
    x = rng.uniform(-Lx / 2, Lx / 2, size=N) + cx
    y = rng.uniform(-Ly / 2, Ly / 2, size=N) + cy

    if plane_restricted:
        z = np.full(N, cz)
    else:
        z = rng.uniform(-Lz / 2, Lz / 2, size=N) + cz

    r_xyz = np.column_stack([x, y, z])

    log.info("Construction vector position: RANDOM. N =%d, box_size = %s, plane_restricted = %s.", N, box_size, plane_restricted)
    return r_xyz

def random_velocity_thermal(r_xyz, v_std=0.01, seed=0, plane_restricted=True):
    """
    Generate random velocities for N particles with a thermal (Maxwell-Boltzmann) model:
    each velocity component is drawn from a normal distribution.

    Inputs
    - r_xyz: (N, 3) positions (only used to get N)
    - v_std: standard deviation of each velocity component
    - seed: random seed for reproducibility
    - plane_restricted: if True, set vz = 0 (motion only in x-y plane)

    Returns
    - v_xyz: (N, 3) velocities
    """
    r_xyz = np.asarray(r_xyz)
    N = r_xyz.shape[0]

    rng = np.random.default_rng(seed)
    v_xyz = rng.normal(loc=0.0, scale=v_std, size=(N, 3))

    if plane_restricted:
        v_xyz[:, 2] = 0.0
    log.info("Cosntruction velocity vectors: Thermal distribution. v_std = %0.3f", v_std)
    return v_xyz


def gaussian_weights(r_xyz, w0, k_in_hat, k_in=1.0):
    """
    Gaussian envelope in x-y times incident plane-wave phase.

    Inputs
    - r_xyz: (N,3) atom positions
    - w0: Gaussian waist
    - k_in_hat: (3,) incident propagation direction (will be normalized)
    - k_in: scalar wavenumber magnitude

    Returns
    - w: (N,) complex weights
    """
    x, y, z = r_xyz[:, 0], r_xyz[:, 1], r_xyz[:, 2]
    k_in_hat = k_in_hat / (np.linalg.norm(k_in_hat) + 1e-15)

    env = np.exp(-(x**2 + y**2) / (w0**2))
    phase = np.exp(1j * k_in * (k_in_hat[0]*x + k_in_hat[1]*y + k_in_hat[2]*z))

    log.info("Computing gaussian weights: waist=%d, wavevector = %0.3f %s",w0,k_in, k_in_hat) 

    return (env * phase).astype(np.complex128)

