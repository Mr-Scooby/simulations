#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rplotting 
import matplotlib.pyplot as plt 
import numpy as np
import helpers as hs


FILE = "sims_runs/N1000_mc1_nt100_k011_6d6f26bb"
K = [0,1,1]

print(f"showing file = {FILE}")

npz = np.load(FILE+'.npz')

#['intensity', 'atom_pos', 'w', 'pcenter']

pos  = npz['atom_pos']
w = npz['w']
pulsecenter = npz['pcenter']
I = npz['intensity']
times = npz['times']
T = times.shape[0]

nt, np_ = I.shape[1], I.shape[2]
theta, phi, nx, ny, nz = hs.make_angle_grid(n_theta=nt, n_phi=np_)

#rplotting.plot_atoms(pos[1],w=w[1], k_in_hat=[0,1,0])
ani = rplotting.animation_atoms_with_pulse(pos,times.shape[0],w, pulsecenter)
plt.show()

# normalize relative to the global max over all entries
I_max = np.max(I)
print( f" Max intensity {I_max}") 
#if I_max > 0:
#    I = I / I_max
#

# normalize each frame by its own max
frame_max = np.max(I, axis=tuple(range(1, I.ndim)), keepdims=True)
frame_max[frame_max == 0] = 1.0
I = I / frame_max

#ani1 = rplotting.animate_pattern_3d(nx,ny,nz, I)
rplotting.plot_atoms(pos[15],w=w[15], k_in_hat=K, title="frame 15")
rplotting.plot_atoms(pos[50],w=w[50], k_in_hat=K, title="frame 50")
rplotting.plot_atoms(pos[75],w=w[75], k_in_hat=K, title="frame 75")

rplotting.plot_pattern_3d(nx,ny,nz,I[15],stride =3 , title=f"frame 15")
rplotting.plot_pattern_3d(nx,ny,nz,I[50], stride =3, title=f"frame 50" )
rplotting.plot_pattern_3d(nx,ny,nz,I[75], stride = 3, title=f"frame 75")
plt.show()
