#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from dataclasses import asdict

import numpy as np

import helpers
import rpattern
import beam
import setup_params as stp

import rplotting as rplt
import matplotlib.pyplot as plt


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    """
    Run a batch of static cube + plane-wave simulations.
    No MC, no time dependence, no Gaussian profile.
    """

    cases = [
        {
            "name": "baseline_z",
            "optical_size": 30.0,
            "optical_spacing": 0.25,
            "k_in_hat": [0.0, 0.0, 1.0],
            "p_hat": [1.0, 0.0, 0.0],
        }
           ]

    for i_case, case in enumerate(cases, start=1):
        log.info("===================================================")
        log.info("Running case %d / %d : %s", i_case, len(cases), case["name"])

        regime = stp.PhysicalRegime(
            optical_size=case["optical_size"],
            optical_spacing=case["optical_spacing"],
            illumination_ratio=1.0,
            filling_factor=1.0,
            pulse_transit=1.0,
        )

        phys = stp.PhysicalParams(
            regime=regime,
            wavelength=1.0,
            v_front=0.0,
            v_thermal=0.0,
            k_in_hat=case["k_in_hat"],
            p_hat=case["p_hat"],
            beam_r0=0.0,
            pcenter_atOrigin=True,
        )

        sim = stp.SimParams(
            n_atoms=1,          # placeholder, corrected after grid is built
            n_mc=1,
            t_max_factor=1.0,
            t_char=1.0,
            n_times=1,
            n_theta=181,
            n_phi=361,
            seed=1,
            chunk_atoms=4000,
        )

        # angle grid
        theta, phi, nx, ny, nz = helpers.make_angle_grid(
            n_theta=sim.n_theta,
            n_phi=sim.n_phi,
        )
        n_hat_flat = np.stack([nx, ny, nz], axis=-1).reshape(-1, 3)

        # static regular cube
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

        # create setup AFTER correcting sim.n_atoms
        setup = stp.SetupParams(regime, phys, sim)

        L_eff = (n_side - 1) * phys.spacing

        log.info(
            "Cube built: n_side=%d, n_atoms=%d, L=%.6f, L_eff=%.6f, spacing=%.6f",
            n_side, sim.n_atoms, phys.L, L_eff, phys.spacing
        )
        log.info("k_in_hat = %s", phys.k_in_hat)
        log.info("p_hat    = %s", phys.p_hat)

        # plane-wave weights
        w_fn = beam.make_weight_fn_plane_wave(
            k_in_hat=phys.k_in_hat,
            k_in=phys.k0,
        )
        w = w_fn(r_xyz, t=0.0)

        rplt.plot_atoms(r_xyz, w=w )
        plt.show()

        ## array factor
        #AF = rpattern.array_factor_general(
        #    n_hat_flat=n_hat_flat,
        #    grid_shape=sim.grid_shape,
        #    k_out=phys.k0,
        #    r_xyz=r_xyz,
        #    w=w,
        #    chunk_atoms=sim.chunk_atoms,
        #)

        ## dipole factor + intensity
        #dipole = helpers.single_dipole_E(nx, ny, nz, phys.p_hat)
        #I = helpers.intensity_from_field(AF, dipole=dipole)

        ## save exactly in the existing style
        #helpers.save_simulation_npz(
        #    "sims_runs/" + setup.run_name,
        #    metadata=asdict(setup),
        #    intensity=I,
        #    atom_pos=r_xyz,
        #    w=w,
        #    theta=theta,
        #    phi=phi,
        #)

        #log.info("Saved: sims_runs/%s.npz", setup.run_name)

    log.info("All static batch runs finished.")


if __name__ == "__main__":
    log.info("Starting static batch run: cube + plane wave only")
    main()
