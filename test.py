#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rplotting 
import matplotlib.pyplot as plt 
import numpy as np
import helpers as hs




theta, phi, nx, ny, nz = hs.make_angle_grid(n_theta=241, n_phi=481)
npz = np.load('./sim_test_paramssetup.npz')

#['intensity', 'atom_pos', 'w', 'pcenter']

pos  = npz['atom_pos']
w = npz['w']
pulsecenter = npz['pcenter']
I = npz['intensity']
times = npz['times']
T = times.shape[0]


rplotting.plot_pattern_3d(nx,ny,nz, I[0])

#ani = rplotting.animation_atoms_with_pulse(pos,times.shape[0],w, pulsecenter)

# normalize relative to the global max over all entries
#I_max = np.max(I)
#if I_max > 0:
#    I = I / I_max


# normalize each frame by its own max
frame_max = np.max(I, axis=tuple(range(1, I.ndim)), keepdims=True)
frame_max[frame_max == 0] = 1.0
I = I / frame_max

#ani1 = rplotting.animate_pattern_3d(nx,ny,nz, I)


plt.show()
