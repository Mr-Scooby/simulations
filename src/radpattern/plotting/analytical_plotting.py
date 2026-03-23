#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import logging 
import radpattern.helpers.helpers as hps
log = logging.getLogger(__name__)




def plot_analytic_pattern_3d(
    af_fn,
    k_in_hat=(0.0, 0.0, 1.0),
    af_kwargs=None,
    n_theta=181,
    n_phi=361,
    alpha=1.0,
    title="Analytical radiation pattern",
    cmap="viridis",
    stride=3,
    normalize=True,
):
    """
    Generic 3D plotter for analytical array-factor functions.

    Parameters
    ----------
    af_fn : callable
        Function of the form
            AF = af_fn(q_vec, **af_kwargs)
        returning an array with shape (n_theta, n_phi).
    k_in_hat : tuple or array (3,)
        Incident direction.
    af_kwargs : dict or None
        Extra arguments passed to af_fn.
    n_theta, n_phi : int
        Angular resolution.
    alpha : float
        Radius scaling: R = I**alpha
    title : str
        Plot title.
    cmap : str
        Matplotlib colormap name.
    stride : int
        Surface stride.
    normalize : bool
        Whether to normalize intensity by its max.

    Returns
    -------
    fig, ax, theta, phi, I, AF
    """
    if af_kwargs is None:
        af_kwargs = {}

    theta, phi, nx, ny, nz = hps.make_angle_grid(n_theta=n_theta, n_phi=n_phi)
    n_hat = np.stack([nx, ny, nz], axis=-1)
    q_vec = hps.build_q_vec(n_hat, k_in_hat)

    AF = af_fn(q_vec, **af_kwargs)
    dipole = hps.single_dipole_E(nx,ny,nz,[0,0,1])
    I= hps.intensity_from_field(AF, dipole)
    #gI = np.abs(AF) ** 2

    if normalize and np.max(I) > 0:
        I = I / np.max(I)

    R = I ** alpha

    X = R * n_hat[..., 0]
    Y = R * n_hat[..., 1]
    Z = R * n_hat[..., 2]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X, Y, Z,
        rstride=stride,
        cstride=stride,
        facecolors=plt.get_cmap(cmap)(I),
        linewidth=0,
        antialiased=True,
        shade=False,
    )

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.grid(False)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.0)

    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array(I)
    cb = plt.colorbar(mappable, shrink=0.75, pad=0.08)
    cb.set_label("Normalized intensity")

    plt.tight_layout()
    return fig, ax, theta, phi, I, AF


