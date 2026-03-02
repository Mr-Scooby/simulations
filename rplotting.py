#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Plotting functions for the radiation pattern. 

- 3D radiation pattern 
- 2D angular plot of different plane cuts
- 3D plot of the position, velocity, incident wave intensity and direction, direction of the dipole. 

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Plotting
def plot_pattern_3d(nx, ny, nz, I, title="", alpha=1.0, stride=2, cmap="viridis"):
    """
    3D surface plot where radius = I^alpha.

    nx,ny,nz: direction cosines (nt,np)
    I: intensity (nt,np), should be normalized if you want nice scaling
    """
    I = np.asarray(I, dtype=float)
    R = I**alpha
    Xs = R * nx
    Ys = R * ny
    Zs = R * nz

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        Xs, Ys, Zs,
        rstride=stride, cstride=stride,
        facecolors=plt.get_cmap(cmap)(I),
        linewidth=0, antialiased=True, shade=False
    )
    ax.set_box_aspect([1, 1, 1])
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    return fig, ax

def _wrap_to_pi(angle_rad: np.ndarray) -> np.ndarray:
    """Map angles to (-pi, pi]."""
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi


def _nearest_index_periodic(angle_grid: np.ndarray, target: float) -> int:
    """
    Nearest index on a periodic grid (period 2*pi).
    Works even if the grid is not exactly hitting the target value.
    """
    # Compare using wrapped difference so 0 and 2*pi behave correctly
    diff = _wrap_to_pi(angle_grid - target)
    return int(np.argmin(np.abs(diff)))


def _nearest_index(angle_grid: np.ndarray, target: float) -> int:
    """Nearest index on a non-periodic grid."""
    return int(np.argmin(np.abs(angle_grid - target)))


def plot_planar_cuts(theta, phi, I, title_prefix=""):
    """
    Plot planar cuts of an intensity pattern I(theta, phi).

    Inputs
    - theta: (n_theta,) polar angle grid in radians, 0..pi
    - phi:   (n_phi,) azimuth angle grid in radians, 0..2*pi (or similar)
    - I:     (n_theta, n_phi) intensity samples on the grid
    - title_prefix: optional text prefix for subplot titles
    - rmax: optional common radial limit for all polar plots

    Cuts plotted
    1) XY plane: theta = pi/2, angle in the plane is phi
    2) XZ plane: ny = 0, use two half-planes (phi=0 and phi=pi),
       angle in the plane is atan2(nz, nx)
    3) YZ plane: nx = 0, use two half-planes (phi=pi/2 and phi=3pi/2),
       angle in the plane is atan2(nz, ny)

    Returns
    - fig, axes
    """
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    I = np.asarray(I)

    if I.ndim != 2 or I.shape != (theta.size, phi.size):
        raise ValueError(f"I must have shape (len(theta), len(phi)) = "
                         f"({theta.size}, {phi.size}), got {I.shape}")

    # Make sure we do not plot negative radii if I has tiny numerical negatives
    I = np.maximum(I, 0.0)

    # Precompute direction cosines on the grid for consistent angle definitions
    # nx = sin(theta) * cos(phi), ny = sin(theta) * sin(phi), nz = cos(theta)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    # Helper: build a full-plane cut by stitching two half-planes
    def build_plane_cut(phi_targets):
        """
        Stitch two phi slices into a single (angle_in_plane, intensity) curve.
        phi_targets: list/tuple of two target azimuth angles (radians).
        """
        betas = []
        vals = []

        for target in phi_targets:
            j = _nearest_index_periodic(phi, target)

            # Use the actual phi grid value at the chosen index
            phi_j = phi[j]

            # Compute in-plane angle from direction cosines
            # For XZ plane: beta = atan2(nz, nx) where nx depends on phi_j
            # For YZ plane: beta = atan2(nz, ny) where ny depends on phi_j
            # We decide which based on the target values passed in
            # (Caller ensures correct targets for each plane)
            if np.isclose(_wrap_to_pi(target - 0.0), 0.0) or np.isclose(_wrap_to_pi(target - np.pi), 0.0):
                # XZ plane: use (nx, nz)
                nx = sin_t * np.cos(phi_j)
                nz = cos_t
                beta = np.arctan2(nz, nx)
            else:
                # YZ plane: use (ny, nz)
                ny = sin_t * np.sin(phi_j)
                nz = cos_t
                beta = np.arctan2(nz, ny)

            betas.append(beta)
            vals.append(I[:, j])

        beta_all = np.concatenate(betas)
        val_all = np.concatenate(vals)

        # Sort by angle so the polar plot draws a clean curve
        order = np.argsort(beta_all)
        return beta_all[order], val_all[order]

    # --- Cut 1: XY plane (theta = pi/2) ---
    i_xy = _nearest_index(theta, np.pi / 2)
    beta_xy = phi.copy()  # In the XY plane, in-plane angle equals phi
    I_xy = I[i_xy, :].copy()

    # To avoid a duplicate point if phi includes both 0 and 2*pi, we can drop the last point
    # when it is effectively the same as the first.
    if phi.size >= 2 and np.isclose(_wrap_to_pi(phi[-1] - phi[0]), 0.0, atol=1e-12):
        beta_xy = beta_xy[:-1]
        I_xy = I_xy[:-1]

    # --- Cut 2: XZ plane (ny = 0) ---
    beta_xz, I_xz = build_plane_cut((0.0, np.pi))

    # --- Cut 3: YZ plane (nx = 0) ---
    beta_yz, I_yz = build_plane_cut((np.pi / 2, 3 * np.pi / 2))

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, subplot_kw={"projection": "polar"}, figsize=(14, 4.5))

    # XY
    axes[0].plot(beta_xy, I_xy, lw=2)
    axes[0].set_title(f"{title_prefix} XY cut (theta ~= pi/2)".strip())
    #axes[0].set_rlim(0, rmax)

    # XZ
    axes[1].plot(beta_xz, I_xz, lw=2)
    axes[1].set_title(f"{title_prefix} XZ cut (ny = 0)".strip())
    #axes[1].set_rlim(0, rmax)

    # YZ
    axes[2].plot(beta_yz, I_yz, lw=2)
    axes[2].set_title(f"{title_prefix} YZ cut (nx = 0)".strip())
    #axes[2].set_rlim(0, rmax)

    fig.tight_layout()
    return fig, axes
def plot_atoms(
    r_xyz,
    title="Atoms",
    s=3,
    alpha=0.6,
    equal_axes=True,
    # weights
    w=None,
    show_colorbar=True,
    # arrows
    p_hat=None,
    k_in_hat=None,
    arrow_scale=0.25,        # fraction of cloud span used for dipole & k_in arrows
    v_xyz=None,
    v_subsample=20,
    v_arrow_scale=0.08,      # fraction of cloud span used for per-atom velocity arrows
    seed=0,
    # limits by count
    r_subsample=10000,        # max number of atoms to scatter (None means plot all)
):
    """
    3D scatter of atoms with optional weight coloring, dipole arrow, incident k arrow,
    and per-atom velocity arrows.

    Inputs
    - r_xyz: (N,3) positions
    - w: (N,) weights (colored by |w| normalized)
    - p_hat: (3,) dipole direction
    - k_in_hat: (3,) incident propagation direction
    - v_xyz: (N,3) velocities (red arrows, subsampled)
    - r_subsample: maximum number of atoms to scatter (None means all)
    - v_subsample: maximum number of velocity arrows to draw
    """
    r_xyz = np.asarray(r_xyz, dtype=float)
    if r_xyz.ndim != 2 or r_xyz.shape[1] != 3:
        raise ValueError(f"r_xyz must have shape (N,3), got {r_xyz.shape}")

    N = r_xyz.shape[0]

    if w is not None:
        w = np.asarray(w)
        if w.shape != (N,):
            raise ValueError(f"w must be shape (N,), got {w.shape}")

    if v_xyz is not None:
        v_xyz = np.asarray(v_xyz, dtype=float)
        if v_xyz.shape != r_xyz.shape:
            raise ValueError(f"v_xyz must have shape {r_xyz.shape}, got {v_xyz.shape}")

    rng = np.random.default_rng(seed)

    # ---- Choose which atoms to scatter ----
    if (r_subsample is None) or (r_subsample >= N):
        idx_scatter = np.arange(N)
    else:
        idx_scatter = rng.choice(N, size=int(r_subsample), replace=False)

    r_plot = r_xyz[idx_scatter]
    x, y, z = r_plot[:, 0], r_plot[:, 1], r_plot[:, 2]

    # Center of full cloud (so arrows originate from the real cloud center)
    c = np.mean(r_xyz, axis=0)
    cx, cy, cz = c

    # Cloud span computed from full cloud (stable scaling even when scattering a subset)
    xmin, xmax = r_xyz[:, 0].min(), r_xyz[:, 0].max()
    ymin, ymax = r_xyz[:, 1].min(), r_xyz[:, 1].max()
    zmin, zmax = r_xyz[:, 2].min(), r_xyz[:, 2].max()
    span = max(xmax - xmin, ymax - ymin, zmax - zmin)
    span = span if span > 0 else 1.0

    def _norm(v):
        v = np.asarray(v, dtype=float)
        return v / (np.linalg.norm(v) + 1e-15)

    # ----- Figure -----
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    # ----- Scatter (optionally colored by weights amplitude) -----
    sc = None
    if w is None:
        ax.scatter(x, y, z, s=s, alpha=alpha)
    else:
        w_plot = w[idx_scatter]
        amp = np.abs(w_plot).astype(float)
        amp_norm = amp / (amp.max() + 1e-15)
        sc = ax.scatter(x, y, z, s=s, alpha=alpha, c=amp_norm)
        if show_colorbar:
            cb = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.08)
            cb.set_label("|w| (normalized)")

    # ----- Dipole arrow (blue) -----
    L = arrow_scale * span
    if p_hat is not None:
        pdir = _norm(p_hat)
        ax.quiver(cx, cy, cz, pdir[0], pdir[1], pdir[2],
                  length=L, normalize=True, color="blue")

    # ----- Incident wavevector arrow (purple) -----
    if k_in_hat is not None:
        kdir = _norm(k_in_hat)
        ax.quiver(cx, cy, cz, kdir[0], kdir[1], kdir[2],
                  length=L, normalize=True, color="purple")

    # ----- Velocity arrows (red), subsampled -----
    if v_xyz is not None:
        # Pick velocity arrows from the same scatter subset (so arrows match dots)
        M = idx_scatter.size
        if M > v_subsample:
            idx_local = rng.choice(M, size=int(v_subsample), replace=False)
            idx_vel = idx_scatter[idx_local]
        else:
            idx_vel = idx_scatter

        rr = r_xyz[idx_vel]
        vv = v_xyz[idx_vel]

        Lv = v_arrow_scale * span
        ax.quiver(rr[:, 0], rr[:, 1], rr[:, 2],
                  vv[:, 0], vv[:, 1], vv[:, 2],
                  length=Lv, normalize=True, color="red", alpha=0.8)

    # ----- Axes styling -----
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    if equal_axes:
        cx0, cy0, cz0 = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
        half = 0.55 * span
        ax.set_xlim(cx0 - half, cx0 + half)
        ax.set_ylim(cy0 - half, cy0 + half)
        ax.set_zlim(cz0 - half, cz0 + half)
        ax.set_box_aspect((1, 1, 1))

    # ----- Legend (proxy artists) -----
    legend_items = []
    if p_hat is not None:
        legend_items.append(Line2D([0], [0], color="blue", lw=3, label="Dipole direction"))
    if k_in_hat is not None:
        legend_items.append(Line2D([0], [0], color="purple", lw=3, label="Incident k-vector"))
    if v_xyz is not None:
        legend_items.append(Line2D([0], [0], color="red", lw=3, label=f"Velocities (sampled <= {v_subsample})"))
    if r_subsample is not None:
        legend_items.append(Line2D([0], [0], color="black", lw=0, marker="o",
                                   label=f"Atoms plotted: {idx_scatter.size}/{N}"))
    if legend_items:
        ax.legend(handles=legend_items, loc="upper right")

    plt.tight_layout()
    return fig, ax
