#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import logging 

log = logging.getLogger(__name__)

def plot_heatmap_theta_phi(grid, I, title="", use_db=True, db_floor=-40, cmap="viridis"):
    """
    2D far-field heat map using an AngleGrid.
    """
    theta = np.asarray(grid.theta)
    phi = np.asarray(grid.phi)
    I = np.asarray(I, dtype=float)

    if I.shape != grid.shape:
        raise ValueError(f"I must have shape {grid.shape}, got {I.shape}")

    I = np.maximum(I, 0.0)
    I_norm = I / (np.max(I) + 1e-15)

    if use_db:
        Z = 10.0 * np.log10(np.maximum(I_norm, 1e-12))
        Z = np.clip(Z, db_floor, 0.0)
        cbar_label = "Intensity [dB]"
    else:
        Z = I_norm
        cbar_label = "Normalized intensity"

    PHI, THETA = np.meshgrid(np.rad2deg(phi), np.rad2deg(theta), indexing="xy")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    pcm = ax.pcolormesh(PHI, THETA, Z, shading="auto", cmap=cmap)

    ax.set_xlabel(r"$\phi$ [deg]")
    ax.set_ylabel(r"$\theta$ [deg]")
    ax.set_title(title if title else "Far-field heat map")
    fig.colorbar(pcm, ax=ax, label=cbar_label)

    plt.tight_layout()
    log.info("Plotting theta-phi heatmap. Title = %s", title)
    return fig, ax
