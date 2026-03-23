#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from radpattern.plotting import pattern_3d as prt
from radpattern.plotting import rplotting
from radpattern.helpers import helpers as hps
from radpattern.geometry import grids

import matplotlib.pyplot as plt 
import numpy as np

#N1771561_mc1_nt1_k110_0976c3c8
#N1771561_mc1_nt1_k011_3596ce19.npz


PATH = "../data/results_sims/"
#exp_numbers_N200_mc1_nt20_k001_6007431b
#test2DslabN100000_mc1_nt20_k001_a2cc34fd
FILE = "test2DslabN10000000_mc1_nt20_k001_4c053b88"
print(f"showing file = {PATH+FILE}")

npz = np.load(PATH+FILE+'.npz', allow_pickle=True)

# Extreact data from file
pos  = npz['atom_pos']
w = npz['w']
I = npz['intensity']
meta = npz["metadata"].item()


print("\n=== metadata ===")
for keys, values in meta.items():
    print(f"\n=== {keys} ===")
    for key, value in meta[keys].items():
        print(f"{key}: {value}")
print("========")

# crewates the grid 
nt, np_ = I.shape
grid = grids.AngleGrid(n_theta = nt, n_phi = np_) 
#theta, phi, nx, ny, nz = hps.make_angle_grid(n_theta=nt, n_phi=np_)

rplotting.plot_atoms(pos, w=w)

# normalize relative to the global max over all entries
I_max = np.max(I)
print( f" Max intensity {I_max}") 

imax = np.unravel_index(np.argmax(I), I.shape)

print(f"Max intensity = {I_max}")
print(f"Grid index of max = {imax}")

print("Direction of max:")
print(f"nx = {grid.nx[imax]}")
print(f"ny = {grid.ny[imax]}")
print(f"nz = {grid.nz[imax]}")
I_max_vec = np.round([grid.nx[imax], grid.ny[imax], grid.nz[imax]])
print(f"normalize max intensity vector = {I_max_vec}")


#

# normalize each frame by its own max
frame_max = np.max(I, axis=tuple(range(1, I.ndim)), keepdims=True)
frame_max[frame_max == 0] = 1.0
I = I / frame_max

#k_inhat =np.round( meta.get('k_in_hat', 'missing'), 3)


K = np.round(meta['phys']['k_in_hat'])
osize = meta['regime']['optical_size']
ospacing = meta['regime']['optical_spacing']
N = meta['sim']['n_atoms']

info = (
    f"file = {FILE}\n"
    f"I_max = {I_max:.3e}, hat I_max = {I_max_vec}"
)


fig,ax = prt.plot_pattern_3d(grid,I, stride = 1, title=f"atoms:{N}. Cube geometry array. k_in={K}, L/ lambda: {osize}, a/lambda: {ospacing}", info_text=info )



#fig.text(
#    0.02, 0.05, info,   # x, y in figure coordinates
#    ha="left",
#    va="top",
#    fontsize=10,
#    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
#)

plt.show()

#'optical_size': 30, 'optical_spacing': 0.25
