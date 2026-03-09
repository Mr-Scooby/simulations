#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import helpers 
import mcpattern
import rpattern
import rplotting
import time
import logging
import numpy as np
import matplotlib.pyplot as plt 


# ---------------------------------------------------------------------
# Logging
def get_logger(name="run_sim", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False 

    if logger.handlers:
        return logger

    h = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    h.setFormatter(fmt)
    logger.addHandler(h)
    return logger


log = get_logger()

def main():
    """ run simualtion"""

    # Parameters
    lam = 1.0
    k_out = 2 * np.pi / lam

    # Dipole orientation
    p_hat = np.array([1.0, 0.0, 0.0], dtype=float)
    p_hat /= (np.linalg.norm(p_hat) + 1e-15)

    # MC settings
    n_atoms = 1000
    n_mc = 1 # number of runs
    times = np.linspace(0.0, 30.0, 16)

    # Incident beam parameters for weights
    w0 = 10.0
    k_in_hat = np.array([0.0, 0.0, 1.0], dtype=float)
    k_in = 1.0  # keep consistent with how gaussian_weights was designed

    log.info("""==== Paramaters =====
             lam=%0.3f,
             Atom number = %d,
             Dipole vector = %s,
             Beam: w0 = %0.3f, k_in = %0.3f, wavevector = %s.
             =====================""", 
             lam, n_atoms, p_hat, w0, k_in, k_in_hat)
    
    # Angle grid
    theta, phi, nx, ny, nz = helpers.make_angle_grid(n_theta=241, n_phi=481)
    

    I = mcpattern.mc_sim(
            nx = nx, ny = ny, nz = nz,
            k_out = k_out, p_hat = p_hat,
            times = times, 
            n_mc = n_mc,
            n_atoms = n_atoms, 
            plane_restricted = False, 
            w_fn = mcpattern.weight_evolution
            ) 

    I_plot0 = I[0] / (np.max(I[0]) + 1e-15)

    # Plots. 
    log.info("plotting") 

    rplotting.plot_pattern_3d(nx, ny, nz, I_plot0, title=f"MC mean pattern, t={times[0]:.3g}", alpha=1.0, stride=2)


if __name__ == "__main__":

    log.info("Starting run_sim. TIME DEPENDENC MC SIMULATION")
    main()
    plt.show()
    



