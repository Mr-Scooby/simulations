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
import setup_params as stp 
from dataclasses import asdict 

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

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

#log = get_logger()



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
    regime = stp.PhysicalRegime(
        optical_size=100.0,
        optical_spacing=1.5,
        illumination_ratio=0.7,
        filling_factor=0.1,
        pulse_transit=1.5,
    )
    
    phys = stp.PhysicalParams(
        regime=regime,
        wavelength=1.0,
        v_front=1.0,
        v_thermal=1e-3,
        k_in_hat = [0,1,1],
        p_hat = [1,0,0]
    )
    
    sim = stp.SimParams(
        n_atoms=500,
        n_mc= 5,
        t_max_factor = 2,
        t_char= phys.t_char,
        n_times=100,
        n_theta=91,#241,#91
        n_phi=181,#481,#181
        seed=1 ,
    )

    setup = stp.SetupParams(regime, phys, sim) 
   
   # Angle grid
    theta, phi, nx, ny, nz = helpers.make_angle_grid(n_theta= sim.n_theta, n_phi=sim.n_phi)
    
    # weights 
    w_fn = beam.make_weight_fn_gaussian_pulse(phys.beam_waist,
                                              phys.sigma_long,
                                              phys.k_in_hat,
                                              phys.k0,
                                              v_front = phys.v_front, 
                                              box_size= phys.box_size, 
                                              )
    rng =  np.random.default_rng(0)
    position, weights, pulse_center = atom_weights_sim(sim.n_atoms,
                                                       rng,
                                                       w_fn,
                                                       sim.n_times,
                                                       k_in_hat= phys.k_in_hat, 
                                                       box_size= phys.box_size,
                                                       v_front = phys.v_front)


    I = mcpattern.mc_sim(
            nx = nx, ny = ny, nz = nz,
            grid_shape = sim.grid_shape,
            k_out = phys.k0, p_hat = phys.p_hat ,
            times = sim.times, 
            n_mc = sim.n_mc,
            n_atoms = sim.n_atoms, 
            plane_restricted = False, 
            w_fn = w_fn,
            box_size = phys.box_size
            ) 

    # Saving
    helpers.save_simulation_npz(setup.run_name,
    metadata=asdict(setup), intensity = I, atom_pos = position, w = weights, pcenter = pulse_center,times=setup.sim.times) 

#    #plt.show()
    log.info("plotting") 
#
#    #rplotting.plot_pattern_3d(nx, ny, nz, I_plot0, title=f"MC mean pattern, t={times[0]:.3g}", alpha=1.0, stride=2)
#

if __name__ == "__main__":

    log.info("Starting run_sim. TIME DEPENDENC MC SIMULATION")
    main()
    



