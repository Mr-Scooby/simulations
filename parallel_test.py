#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import logging
import numpy as np
import matplotlib.pyplot as plt

import setup_params as stp
import helpers
import AF_parallel
import rplotting
import beam 
import os

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
    v_front=0.0,
    v_thermal=0.0,
    k_in_hat=[0.0, 0.0, 1.0],
    p_hat=[1.0, 0.0, 0.0],
    beam_r0=0.0,
    pcenter_atOrigin=True,
)

sim = stp.SimParams(
    n_atoms=phys.atoms,
    n_mc=1,
    t_max_factor=1.0,
    t_char=1.0,
    n_times=1,
    n_theta=181,
    n_phi=361,
    seed=1,
    chunk_atoms=2000,
)

setup = stp.SetupParams(regime, phys, sim )

n_workers= 4 
log.info(" CPU count %s", n_workers +2 )

if __name__ == "__main__":

    theta, phi, nx, ny, nz = helpers.make_angle_grid(
        n_theta=sim.n_theta,
        n_phi=sim.n_phi,
    )

    n_hat_flat = np.stack([nx, ny, nz], axis=-1).reshape(-1, 3)

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

    w_fn = beam.make_weight_fn_plane_wave(
        k_in_hat=phys.k_in_hat,
        k_in=phys.k0,
    )
    w = w_fn(r_xyz, t=0.0)

    dipole = helpers.single_dipole_E(nx, ny, nz, np.asarray(phys.p_hat, dtype=float))

    t2 = time.perf_counter()
    AF_par = AF_parallel.array_factor_general_parallel(
        n_hat_flat=n_hat_flat,
        grid_shape=sim.grid_shape,
        k_out=phys.k0,
        r_xyz=r_xyz,
        w=w,
        chunk_atoms=sim.chunk_atoms,
        n_workers=n_workers,
    )
    t3 = time.perf_counter()

    I_parallel = helpers.intensity_from_field(AF_par, dipole)
    I_parallel /= I_parallel.max() + 1e-15

    helpers.save_simulation_npz("sims_runs/" + setup.run_name, intensity=I_parallel)

    print("parallel time =", t3 - t2)

    rplotting.plot_pattern_3d(
        nx, ny, nz, I_parallel,
        title="Parallel static AF",
        alpha=1.0,
        stride=3,
    )

    plt.show()
