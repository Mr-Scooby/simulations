#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rplotting 
import matplotlib.pyplot as plt 
import numpy as np
import helpers as hs




theta, phi, nx, ny, nz = hs.make_angle_grid(n_theta=241, n_phi=481)
npz = np.load('./sim_test6.npz')

#['intensity', 'atom_pos', 'w', 'pcenter']

pos  = npz['atom_pos']
w = npz['w']
pulsecenter = npz['pcenter']
I = npz['intensity']
times = npz['times']
T = times.shape[0]

#ani = rplotting.animation_atoms_with_pulse(pos,times.shape[0],w, pulsecenter)

ani1 = rplotting.animate_pattern_3d(nx,ny,nz, I)


plt.show()
