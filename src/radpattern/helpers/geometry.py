#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import logging


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

# Atom layouts + weights
def atom_grid(Nx, Ny, Nz=1, dx=1.0, dy=1.0, dz=1.0, z0=0.0, plane_restricted=False):
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

def random_position(N, box_size=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), seed=0, plane_restricted=False):
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

    log.info("Construction vector position: RANDOM. n_atoms =%d, box_size = %s, center=%s, plane_restricted = %s, seed=%s.", N, box_size,center, plane_restricted, seed)
    return r_xyz

def random_velocity_thermal(r_xyz, v_std=0.01, seed=0, plane_restricted=False):
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
    log.info("Cosntruction velocity vectors: Thermal distribution. v_std = %0.3f, seed =%s, plane_restricted =%s ", v_std, seed, plane_restricted)
    return v_xyz


