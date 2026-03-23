#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from rplotting import (plot_atoms, animation_atoms_with_pulse)
import helpers as hs
import beam
import logging 

log = logging.getLogger(__name__)

""" Testing animation functions """


# -----------------------------
# Example data
# positions.shape = (T, N, 3)
# T = number of frames
# N = number of points
# -----------------------------
T = 100
N = 10000

r_xyz = hs.random_position(N, plane_restricted = False, box_size=[10,10,10])
#r_xyz = hs.atom_grid(20,20,20, plane_restricted = False) 
v_xyz = hs.random_velocity_thermal(r_xyz, plane_restricted = False)

def updated_pos(r_xyz, v_xyz, t =1): 
    return r_xyz + 0.0*v_xyz* t


# update weights. 
w= hs.gaussian_weights(r_xyz,10, [0,0,1])

w_f = beam.make_weight_fn_gaussian_pulse(5,5,[0,0,-1], v_front= 0.1, margin = 0, pulse_center_t0=+0.5 , box_size=(10,10,10), k_in = 10)


atoms, dim = r_xyz.shape
T = 100
positions = np.zeros((T,atoms, dim))
weights = np.zeros((T,atoms))
pulse_center = np.zeros((T,3))

for t in range(0,T): 
    r_upd = updated_pos(r_xyz, v_xyz, t ) 
    positions[t] = r_upd
    weights[t], pulse_center[t] = w_f(r_upd, t,return_pulse_center =True )

print("pulse center =", pulse_center[0])

ani =  animation_atoms_with_pulse(positions, T, weights = weights, pulse_center = pulse_center)
plt.show()





