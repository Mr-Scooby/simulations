#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rplotting 
import matplotlib.pyplot as plt 
import numpy as np
import helpers as hs


FILE = "sims_runs/N1771561_mc1_nt1_k011_3596ce19"
K = [0,1,1]
N = 1771561
print(f"showing file = {FILE}")

npz = np.load(FILE+'.npz', allow_pickle=True)

#['intensity', 'atom_pos', 'w', 'pcenter']

pos  = npz['atom_pos']
w = npz['w']
I = npz['intensity']


meta = npz["metadata"].item()

print("\n=== metadata ===")
for key, value in meta.items():
    print(f"{key}: {value}")

nt, np_ = I.shape
theta, phi, nx, ny, nz = hs.make_angle_grid(n_theta=nt, n_phi=np_)

rplotting.plot_atoms(pos, w=w)
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

#k_inhat =np.round( meta.get('k_in_hat', 'missing'), 3)

print(meta.keys())
print("optical_size" in meta)

k_inhat = np.round(meta['phys']['k_in_hat'],3)
osize = meta['regime']['optical_size']
ospacing = meta['regime']['optical_spacing']

rplotting.plot_pattern_3d(nx,ny,nz,I, stride = 3, title=f"atoms:{N}. Cube geometry array. k={K}, L/ lambda: {osize}, a/lambda: {ospacing}")
plt.show()

#'optical_size': 30, 'optical_spacing': 0.25
