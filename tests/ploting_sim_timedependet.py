#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from radpattern.plotting import pattern_3d as prt
from radpattern.helpers import helpers as hps
from radpattern.geometry import grids 
from radpattern.plotting import rplotting 

import matplotlib.pyplot as plt
import numpy as np

PATH = "../data/results_sims/"
#"exp_numbers_N200_mc1_nt20_k001_6007431b"
#exp_numbers_N1000_mc1_nt20_k001_f65429e5
FILE = "exp_numbers_N1000_mc1_nt20_k001_f500e73c"

print(f"showing file = {PATH + FILE}.npz")

npz = np.load(PATH + FILE + ".npz", allow_pickle=True)

# Extract data
pos = npz["atom_pos"]
w = npz["w"]
I = npz["intensity"]          # shape (T, nt, np_)
meta = npz["metadata"].item()



# pos should be (T, N, 3), w should be (T, N)
T = pos.shape[0]

ani = rplotting.animation_atoms_with_pulse(
    pos,
    T,
    weights=w,
    pulse_center=npz["pcenter"] if "pcenter" in npz.files else None,
)

plt.show()

print("\n=== metadata ===")
for section, values in meta.items():
    print(f"\n=== {section} ===")
    if isinstance(values, dict):
        for key, value in values.items():
            print(f"{key}: {value}")
    else:
        print(values)
print("========")

# Angular grid from intensity shape
T, nt, np_ = I.shape
grid = grids.AngleGrid(n_theta = nt, n_phi = np_) 
#theta, phi, nx, ny, nz = hps.make_angle_grid(n_theta=nt, n_phi=np_)

# Global max over all times
I_max = np.max(I)
imax = np.unravel_index(np.argmax(I), I.shape)   # (t_idx, theta_idx, phi_idx)

t_idx, th_idx, ph_idx = imax

print(f"\nMax intensity = {I_max:.3e}")
print(f"Grid index of max = {imax}")
print("Direction of max:")
print(f"nx = {grid.nx[th_idx, ph_idx]}")
print(f"ny = {grid.ny[th_idx, ph_idx]}")
print(f"nz = {grid.nz[th_idx, ph_idx]}")

I_max_vec = np.round([grid.nx[th_idx, ph_idx], grid.ny[th_idx, ph_idx], grid.nz[th_idx, ph_idx]], 3)
print(f"normalized max intensity vector = {I_max_vec}")

# Choose one frame to plot: the frame containing the global max
I_plot = I[t_idx].copy()

# Normalize that frame
frame_max = np.max(I_plot)
if frame_max > 0:
    I_plot /= frame_max

K = np.round(meta["phys"]["k_in_hat"], 3)
osize = meta["regime"]["optical_size"]
ospacing = meta["regime"]["optical_spacing"]
N = meta["sim"]["n_atoms"]

title =f"atoms:{N}, frame:{t_idx} | ,k_in={K}, L/lambda={osize}, a/lambda={ospacing}"


fig, ax = prt.plot_pattern_3d(
    grid, I_plot,
    stride=1,
    title=title ,
)

info = (
    f"file = {FILE}\n"
    f"I_max = {I_max:.3e}\n"
    f"dir_max = {I_max_vec}\n"
    f"frame = {t_idx}"
)

fig.text(
    0.02, 0.02, info,
    ha="left", va="bottom", fontsize=10,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

plt.show()


print("===" ,np.max(np.abs(w[0])))
print (np.max(I[0]))

# optional: normalize each frame for display
I_anim = I.copy()
#frame_max = np.max(I_anim, axis=(1, 2), keepdims=True)
#frame_max[frame_max == 0] = 1.0
#I_anim = I_anim / frame_max

fig_int, ax_int, ani_int = rplotting.animate_pattern_3d(
    grid.nx, grid.ny, grid.nz,
    I_anim,          # shape (T, nt, np_)
    title=f"Intensity animation: {FILE}",
    alpha=1.0,
    stride=2,
    interval=100,
)

plt.show()
