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
import beam


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


def atom_weights_sim(n_atoms, rng, w_fn, T , **kwargs): 
    """ produce a small simulation of the atoms and the weights movement to check"""
   
    r_xyz, v_xyz = mcpattern.sample_realization(n_atoms,rng,**kwargs)

    atoms, dim = r_xyz.shape 
    positions = np.zeros((T,atoms,dim ))
    weights = np.zeros((T,atoms))
    pulse_center = np.zeros((T,3))
    for t in range(0,T): 
        r_upd = mcpattern.positions_at_time(r_xyz, v_xyz, t ) 
        positions[t] = r_upd
        weights[t], pulse_center[t] = w_fn(r_upd, t,return_pulse_center =True )

    return positions, weights, pulse_center



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
    T = 20
    times = np.linspace(0.0, 30.0, T)

    # Incident beam parameters for weights
    w0 = 10.0
    k_in_hat = np.array([0.0, 0.0, 1.0], dtype=float)
    k_in = k_out  # keep consistent with how gaussian_weights was designed
    v_front = 1.0
    r_t0 = 0

    log.info("""==== Paramaters =====
             lam=%0.3f,
             Atom number = %d,
             Dipole vector = %s,
             Beam: w0 = %0.3f, k_in = %0.3f, wavevector = %s.
             =====================""", 
             lam, n_atoms, p_hat, w0, k_in, k_in_hat)
    
    # sim box size 
    sim_box = (10,10,10) 

    # Angle grid
    theta, phi, nx, ny, nz = helpers.make_angle_grid(n_theta=241, n_phi=481)
    
    # weights 
    w_fn = beam.make_weight_fn_gaussian_pulse(5,5,k_in_hat,v_front = v_front, box_size= sim_box, pulse_center_t0= r_t0)

    rng =  np.random.default_rng(0)
    position, weights, pulse_center = atom_weights_sim(n_atoms,rng, w_fn,T, k_in_hat= k_in_hat, box_size= sim_box,v_front = v_front)


    I = mcpattern.mc_sim(
            nx = nx, ny = ny, nz = nz,
            k_out = k_out, p_hat = p_hat,
            times = times, 
            n_mc = n_mc,
            n_atoms = n_atoms, 
            plane_restricted = False, 
            w_fn = w_fn,
            box_size = sim_box
            ) 
    helpers.save_simulation_npz("./sim_test7", intensity = I, atom_pos = position, w = weights, pcenter = pulse_center, times = times) 

    I_plot0 = I[0] / (np.max(I[0]) + 1e-15)

    # Plots. 
    #fig,ax, ani = rplotting.animate_pattern_3d(nx,ny,nz, I, "MC animation pattern")
    #plt.show()
    log.info("plotting") 

    #rplotting.plot_pattern_3d(nx, ny, nz, I_plot0, title=f"MC mean pattern, t={times[0]:.3g}", alpha=1.0, stride=2)


if __name__ == "__main__":

    log.info("Starting run_sim. TIME DEPENDENC MC SIMULATION")
    main()
    plt.show()
    
    



