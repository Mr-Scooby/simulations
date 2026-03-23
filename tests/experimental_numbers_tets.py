#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This is the setup to test whether emission is limited by the narrow signal mode."""



from radpattern.physics import setup_params as stp
from radpattern.physics import beam, mcpattern
from radpattern.helpers import helpers
from dataclasses import asdict


import numpy as np
import logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)



# ---------- 895 nm normalization ----------
lambda0 = 895e-9          # m
L_cell = 75e-3            # m
D_signal = 150e-6         # m
D_control = 300e-6        # m

# radius-like waist scale to match Gaussian code
w_signal = D_signal / 2   # 75 um
w_control = D_control / 2 # 150 um



#### SETTING UP 

#regime = stp.PhysicalRegime(
#    optical_size=L_cell / lambda0,          # ~8.38e4
#    optical_spacing=5.0,                    # keep dilute enough numerically
#    illumination_ratio=w_signal / L_cell,   # 1.0e-3
#    filling_factor=0.02,                    # choose pulse length ~1.5 mm as example
#    pulse_transit=1.2,
#)
#
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

# Gaussian beam + pulse weights
w_fn = beam.make_weight_fn_gaussian_pulse(
    w0=phys.beam_waist,
    sigma_long=phys.sigma_long,
    k_in_hat=phys.k_in_hat,
    k_in=phys.k0,
    v_front=phys.v_front,
    box_size=phys.box_size,
    pulse_center_t0=phys.beam_r0,
    pcenter_at_origin=phys.pcenter_atOrigin,
    margin=0.0,
)

I, position, weights, pulse_center = mcpattern.mc_sim(
    nx=nx,
    ny=ny,
    nz=nz,
    grid_shape=sim.grid_shape,
    k_out=phys.k0,
    p_hat=phys.p_hat,
    times=sim.times,
    n_mc=sim.n_mc,
    n_atoms=sim.n_atoms,
    w_fn=w_fn,
    chunk_atoms=sim.chunk_atoms,
    seed=sim.seed,
    normalize_each_time=sim.normalize_each_time,
    plane_restricted=sim.plane_restricted,
    box_size=phys.box_size,
    v_std=phys.v_thermal,
    center=[0, 0, 0],
    pcenter_at_origin=phys.pcenter_atOrigin,
)

helpers.save_simulation_npz(
    "../data/results_sims/exp_numbers_" + setup.run_name,
    metadata=asdict(setup),
    intensity=I,
    atom_pos=position,
    w=weights,
    pcenter=pulse_center,
    times=sim.times,
)
