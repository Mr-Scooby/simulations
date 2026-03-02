#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1) USER HOOK: put YOUR far-field E(θ,φ) here
# ---------------------------

def single_dipole(theta, phi): 
    # --- DEMO: dipole along z in far field -> |E| ∝ sin(theta), polarized along e_theta
    # We build E in spherical basis then convert to Cartesian.
    E0 = 1.0 + 0.0j
    E_theta = E0 * np.sin(theta)          # amplitude pattern
    E_phi   = 0.0 + 0.0j                  # no phi component for this simple dipole

    return (E_theta, E_phi)


def AF (theta, phi, spacing = 0.2, atoms = 4): 
    psi = spacing * np.cos(theta) 
    AF  = 0 
    for atom in np.arange(atoms): 
        AF += np.exp( j * atom * psi) 

    return AF

def E_theta_phi(theta, phi):
    """
    Return complex E-field vector at (theta, phi) in Cartesian components [Ex, Ey, Ez].
    theta: polar angle [0..pi]
    phi: azimuth [0..2pi)

    Replace the body of this function with your expression.
    """
    ## --- DEMO: dipole along z in far field -> |E| ∝ sin(theta), polarized along e_theta
    ## We build E in spherical basis then convert to Cartesian.
    #E0 = 1.0 + 0.0j
    #E_theta = E0 * np.sin(theta)          # amplitude pattern
    #E_phi   = 0.0 + 0.0j                  # no phi component for this simple dipole

    E_theta, E_phi = single_dipole(theta, phi)

    E_theta = E_theta * AF(theta, phi, atoms = 2 ) 
    # Convert spherical (E_r=0, E_theta, E_phi) to Cartesian:
    # Unit vectors:
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)

    e_theta = np.array([ct*cp, ct*sp, -st], dtype=complex)
    e_phi   = np.array([-sp,   cp,    0.0], dtype=complex)

    E_cart = E_theta * e_theta + E_phi * e_phi
    return E_cart

# ---------------------------
# 2) Sampling grid on the sphere
# ---------------------------
n_theta = 181 
n_phi = 361 
theta = np.linspace(0, np.pi, n_theta)
phi   = np.linspace(0, 2*np.pi, n_phi)
TH, PH = np.meshgrid(theta, phi, indexing="ij")

# ---------------------------
# 3) Evaluate intensity I(θ,φ) ∝ |E|^2
# ---------------------------
Ex = np.zeros_like(TH, dtype=complex)
Ey = np.zeros_like(TH, dtype=complex)
Ez = np.zeros_like(TH, dtype=complex)

for i in range(n_theta):
    for j in range(n_phi):
        E = E_theta_phi(TH[i, j], PH[i, j])
        Ex[i, j], Ey[i, j], Ez[i, j] = E[0], E[1], E[2]

I = (np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2)

# Normalize for plotting
I /= (I.max() + 1e-15)

# Optional: radius scaling for nicer visuals
alpha = 1          # try 1.0 for true intensity radius
R = I**alpha

# ---------------------------
# 4) Convert to 3D surface coordinates
# ---------------------------
X = R * np.sin(TH) * np.cos(PH)
Y = R * np.sin(TH) * np.sin(PH)
Z = R * np.cos(TH)

# ---------------------------
# 5) Plot
# ---------------------------
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

# Use intensity as colormap
surf = ax.plot_surface(
    X, Y, Z,
    rstride=2, cstride=2,
    facecolors=plt.cm.viridis(I),
    linewidth=0,
    antialiased=True,
    shade=False
)

# Make axes look nice
ax.set_box_aspect([1, 1, 1])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Far-field radiation pattern: r(θ,φ) ∝ |E(θ,φ)|^2")

# Remove panes / grid for cleaner look
ax.grid(False)
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.set_alpha(0.0)

# Colorbar based on intensity
mappable = plt.cm.ScalarMappable(cmap="viridis")
mappable.set_array(I)
cb = plt.colorbar(mappable, shrink=0.75, pad=0.08)
cb.set_label("Normalized intensity |E|^2")

plt.tight_layout()
plt.show()

# ---------------------------
# 6) Orthogonal planar cuts (XY, XZ, YZ) of intensity
# Assumes you already have: theta, phi, I (normalized), TH/PH mesh not required here
# ---------------------------

# Indices for the closest grid values
i_xy = np.argmin(np.abs(theta - np.pi/2))      # theta = pi/2
j_xz = np.argmin(np.abs(phi - 0.0))            # phi = 0
j_yz = np.argmin(np.abs(phi - np.pi/2))        # phi = pi/2

# XY plane cut: I(theta=pi/2, phi) -> as function of phi (0..2pi)
I_xy = I[i_xy, :]          # length n_phi
ang_xy = phi               # use phi directly

# XZ plane cut: I(theta, phi=0) -> as function of theta (0..pi), mirror to 0..2pi
I_xz_half = I[:, j_xz]     # length n_theta
ang_xz_half = theta        # 0..pi

ang_xz = np.concatenate([ang_xz_half, 2*np.pi - ang_xz_half[::-1]])
I_xz  = np.concatenate([I_xz_half,    I_xz_half[::-1]])

# YZ plane cut: I(theta, phi=pi/2) -> mirror to 0..2pi
I_yz_half = I[:, j_yz]
ang_yz_half = theta

ang_yz = np.concatenate([ang_yz_half, 2*np.pi - ang_yz_half[::-1]])
I_yz  = np.concatenate([I_yz_half,    I_yz_half[::-1]])

# Plot all three as polar plots
fig = plt.figure(figsize=(14, 4))

ax1 = fig.add_subplot(131, projection="polar")
ax1.plot(ang_xy, I_xy)
ax1.set_title("XY plane cut (θ = π/2): I(φ)")

ax2 = fig.add_subplot(132, projection="polar")
ax2.plot(ang_xz, I_xz)
ax2.set_title("XZ plane cut (φ = 0): I(plane angle)")

ax3 = fig.add_subplot(133, projection="polar")
ax3.plot(ang_yz, I_yz)
ax3.set_title("YZ plane cut (φ = π/2): I(plane angle)")

plt.tight_layout()
plt.show()
