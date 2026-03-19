#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from dataclasses import asdict

import numpy as np
import matplotlib.pyplot as plt

import helpers
import rpattern
import rplotting
import beam
import setup_params as stp


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


regime = stp.PhysicalRegime(
    optical_size=10.0,
    optical_spacing=0.10,
    illumination_ratio=1.0,
    filling_factor=1.0,
    pulse_transit=1.0,
)

phys = stp.PhysicalParams(
    regime=regime,
    wavelength=1.0,
    v_front=1.0,
    v_thermal=0.0,
    k_in_hat=[0.0, 0.0, 1.0],
    p_hat=[1.0, 0.0, 0.0],
    beam_r0=0.0,
    pcenter_atOrigin=True,
)

sim = stp.SimParams(
    n_atoms=30**3,
    n_mc=1,
    t_max_factor=1.0,
    t_char=1.0,
    n_times=1,
    n_theta=181,
    n_phi=361,
    seed=1,
    chunk_atoms = 1000
)

setup = stp.SetupParams(regime, phys, sim)

# make grid. 
theta, phi, nx, ny, nz = helpers.make_angle_grid(
    n_theta=sim.n_theta,
    n_phi=sim.n_phi,
)

n_hat_flat = np.stack([nx, ny, nz], axis=-1).reshape(-1, 3)


# Static regular cube of atoms
n_side = int(round(phys.L / phys.spacing)) + 1
r_xyz = helpers.atom_grid(
    Nx=n_side,
    Ny=n_side,
    Nz=n_side,
    dx=phys.spacing,
    dy=phys.spacing,
    dz=phys.spacing,
)
sim.n_atoms = r_xyz.shape[0]
log.info("Static cube: n_side=%d, n_atoms=%d, L=%.3f", n_side, r_xyz.shape[0], phys.L)

# Plane-wave weights
w_fn = beam.make_weight_fn_plane_wave(
    k_in_hat=phys.k_in_hat,
    k_in=phys.k0,
)

w = w_fn(r_xyz, t=0.0)

# Array factor + intensity
AF = rpattern.array_factor_general(
    n_hat_flat=n_hat_flat,
    grid_shape=sim.grid_shape,
    k_out=phys.k0,
    r_xyz=r_xyz,
    w=w,
    chunk_atoms=sim.chunk_atoms,
)

dipole = helpers.single_dipole_E(nx, ny, nz, phys.p_hat)
I = helpers.intensity_from_field(AF, dipole=dipole)
I_plot = I / (np.max(I) + 1e-15)

# Save
helpers.save_simulation_npz(
    "sims_runs/" + setup.run_name,
    metadata=asdict(setup),
    intensity=I,
    atom_pos=r_xyz,
    w=w,
    theta=theta,
    phi=phi,
)

# Plot
rplotting.plot_planar_cuts(
    theta,
    phi,
    I_plot,
    title_prefix="Static cube, plane-wave weights",
)

rplotting.plot_pattern_3d(
    nx,
    ny,
    nz,
    I_plot,
    title="Static cube, plane-wave weights",
    alpha=1.0,
    stride=2,
)

plt.show()
