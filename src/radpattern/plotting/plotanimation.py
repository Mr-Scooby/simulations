#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors 
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation

#from helpers import (make_angle_grid, build_q_vec) 
import logging
log = logging.getLogger(__name__)


def animate_pattern_3d(
    grid, I_series, title="", stride=2, cmap="viridis",
    interval=50, log_plot=False, clip_db=None
):
    """Animate pattern on a sphere. Color shows intensity, optionally in dB."""
    
    # Correct for data type 
    I_series = np.asarray(I_series, dtype=float)
    if I_series.ndim != 3:
        raise ValueError("I_series must have shape (T, nt, np_)")


    # Create figure plot. 
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    cmap_obj = plt.get_cmap(cmap)

    # fixed sphere
    X = grid.nx
    Y = grid.ny
    Z = grid.nz

    if log_plot:
        C_series = 10 * np.log10(I_series / np.nanmax(I_series) + 1e-12)
        if clip_db is not None:
            C_series = np.clip(C_series, -clip_db, 0)
            norm = plt.Normalize(vmin=-clip_db, vmax=0)
            clabel = f"Intensity [dB] (clipped at -{clip_db} dB)"
        else:
            norm = plt.Normalize(vmin=np.nanmin(C_series), vmax=0)
        clabel = "Intensity [dB]"
    else:
        C_series = I_series / (np.nanmax(I_series) + 1e-15)
        norm = plt.Normalize(vmin=0, vmax=1)
        clabel = "Normalized intensity"

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, shrink=0.7, pad=0.1, label=clabel)

    C0 = C_series[0]
    surf = [ax.plot_surface(
        X, Y, Z,
        rstride=stride, cstride=stride,
        facecolors=cmap_obj(norm(C0)),
        linewidth=0, antialiased=True, shade=False
    )]
    ax.set_title(f"{title} | frame 0")

    def update(frame):
        C = C_series[frame]
        surf[0].remove()
        surf[0] = ax.plot_surface(
            X, Y, Z,
            rstride=stride, cstride=stride,
            facecolors=cmap_obj(norm(C)),
            linewidth=0, antialiased=True, shade=False
        )
        ax.set_title(f"{title} | frame {frame}")
        return surf[0],

    ani = FuncAnimation(
        fig, update, frames=I_series.shape[0], interval=interval, blit=False
    )
    return fig, ax, ani
