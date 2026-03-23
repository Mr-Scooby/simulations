#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from dataclasses import asdict

import numpy as np
import matplotlib.pyplot as plt

from radpattern.helpers import helpers
from radpattern.physics import rpattern
from radpattern.physics import beam
from radpattern.physics import setup_params as stp


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

regime = stp.PhysicalRegime(
    optical_size=200.0,
    optical_spacing=1.5,
    illumination_ratio=0.0205,   # sqrt(0.084 / 200)
    filling_factor=0.10,
    pulse_transit=1.2,
)

phys = stp.PhysicalParams(
    regime=regime,
    wavelength=1.0,               # dimensionless units
    v_front=1.0,
    v_thermal=0.0,
    k_in_hat=[0.0, 0.0, 1.0],     # along cell axis
    p_hat=[1.0, 0.0, 0.0],
    beam_r0=-10.0,
    pcenter_atOrigin=False,
)

sim = stp.SimParams(
    n_atoms=1000,
    n_mc=1,
    t_max_factor=1.0,
    t_char=phys.t_char,
    n_times=20,
    n_theta=91,
    n_phi=181,
    seed=1,
    chunk_atoms=2000,
    normalize_each_time=False,
    plane_restricted=False,
)

setup = stp.SetupParams(regime, phys, sim)

# angular grid
theta, phi, nx, ny, nz = helpers.make_angle_grid(
    n_theta=sim.n_theta,
    n_phi=sim.n_phi,
)

n_hat_flat = np.stack([nx, ny, nz], axis=-1).reshape(-1, 3)


# Static regular cube of atoms
r_xyz = helpers.atom_grid(
    Nx=100,
    Ny=100,
    Nz=1000,
    dx=0.01,
    dy=0.01,
    dz=0.01,
)
sim.n_atoms = r_xyz.shape[0]
log.info("Static cube: ")

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
    "../data/results_sims/test2Dslab" + setup.run_name,
    metadata=asdict(setup),
    intensity=I,
    atom_pos=r_xyz,
    w=w,
    theta=theta,
    phi=phi,
)
