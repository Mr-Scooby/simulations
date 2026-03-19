#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def sphere_af_gamma(gamma, k=1.0, R=1.0):
    """
    Normalized array factor of a uniform solid sphere:
        AF(gamma) = 3 j1(x) / x
    with
        x = 2 k R sin(gamma/2)

    Uses the explicit spherical-Bessel form
        j1(x) = (sin x - x cos x)/x^2
    and the limit AF(0)=1.
    """
    x = 2.0 * k * R * np.sin(gamma / 2.0)

    AF = np.empty_like(x, dtype=float)
    small = np.isclose(x, 0.0)

    xs = x[~small]
    AF[~small] = 3.0 * (np.sin(xs) - xs * np.cos(xs)) / (xs**3)
    AF[small] = 1.0

    return AF


def plot_radiation_pattern_3d(
    af_fn,
    k_in_hat,
    af_mode="intensity",          # "amplitude" or "intensity"
    af_kwargs=None,
    ntheta=181,
    nphi=361,
    radial_scale=1.0,             # multiplies plotted radius only
    radial_power=1.0,             # use <1 to compress peaks visually
    radial_offset=0.0,            # adds constant floor to plotted radius only
    show_center=True,
    show_kin_arrow=True,
    arrow_scale=1.2,
    title="3D radiation pattern",
):
    """
    Generic 3D radiation-pattern plotter.

    The user supplies:
        af_fn(q_vec, n_hat, k_in_hat, **af_kwargs) -> AF

    Inputs
    ------
    q_vec     : array (..., 3)
        q = k_in - k_out for each observation direction
    n_hat     : array (..., 3)
        observation direction unit vectors
    k_in_hat  : array (3,)
        incident direction unit vector

    Output of af_fn
    ---------------
    AF : array (...,)
        Can be real or complex.
        If af_mode="intensity", plotted radius = |AF|^2
        If af_mode="amplitude", plotted radius = |AF|

    Returns
    -------
    fig, ax, out
    where out is a dict with X,Y,Z,R,AF,q_vec,n_hat,TH,PH,k_in_hat
    """
    if af_kwargs is None:
        af_kwargs = {}

    k_in_hat = np.asarray(k_in_hat, dtype=float)
    k_in_hat = k_in_hat / np.linalg.norm(k_in_hat)

    print_log(
    af_fn=af_fn,
    k_in_hat=k_in_hat,
    af_mode=af_mode,
    ntheta=ntheta,
    nphi=nphi,
    radial_scale=radial_scale,
    radial_power=radial_power,
    radial_offset=radial_offset,
    show_center=show_center,
    show_kin_arrow=show_kin_arrow,
    arrow_scale=arrow_scale,
    **af_kwargs,
    )

    theta = np.linspace(0.0, np.pi, ntheta)
    phi = np.linspace(0.0, 2.0 * np.pi, nphi)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")

    nx = np.sin(TH) * np.cos(PH)
    ny = np.sin(TH) * np.sin(PH)
    nz = np.cos(TH)

    n_hat = np.stack([nx, ny, nz], axis=-1)          # (ntheta, nphi, 3)
    q_vec = k_in_hat[None, None, :] - n_hat          # assumes |k_in|=|k_out|=1

    AF = af_fn(q_vec=q_vec, n_hat=n_hat, k_in_hat=k_in_hat, **af_kwargs)

    if af_mode == "intensity":
        R = np.abs(AF) ** 2
    elif af_mode == "amplitude":
        R = np.abs(AF)
    else:
        raise ValueError("af_mode must be 'amplitude' or 'intensity'")

    # display-only radial transform
    R_plot = radial_offset + radial_scale * (R ** radial_power)

    X = R_plot * nx
    Y = R_plot * ny
    Z = R_plot * nz

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, rstride=2, cstride=2, linewidth=0, antialiased=True)

    if show_center:
        ax.scatter([0], [0], [0], color="red", s=60)

    if show_kin_arrow:
        arrow_len = arrow_scale * np.max(np.sqrt(X**2 + Y**2 + Z**2))
        ax.quiver(
            0, 0, 0,
            arrow_len * k_in_hat[0],
            arrow_len * k_in_hat[1],
            arrow_len * k_in_hat[2],
            color="red",
            linewidth=2
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])

    out = {
        "X": X, "Y": Y, "Z": Z,
        "R": R, "R_plot": R_plot,
        "AF": AF,
        "q_vec": q_vec,
        "n_hat": n_hat,
        "TH": TH, "PH": PH,
        "k_in_hat": k_in_hat,
    }

    return fig, ax, out


def print_log(
    af_fn,
    k_in_hat,
    af_mode,
    ntheta,
    nphi,
    radial_scale,
    radial_power,
    radial_offset,
    show_center,
    show_kin_arrow,
    arrow_scale,
    **af_kwargs,
):
    print("\n===== Analytical radiation-pattern setup =====")
    print(f"Geometry / AF function : {af_fn.__name__}")
    print(f"k_in_hat (unit)        : {k_in_hat}")
    print(f"|k_in_hat|             : {np.linalg.norm(k_in_hat):.6f}")

    if "kL" in af_kwargs:
        print(f"Regime parameter kL    : {af_kwargs['kL']}")
    if "kR" in af_kwargs:
        print(f"Regime parameter kR    : {af_kwargs['kR']}")

    print(f"AF mode                : {af_mode}")
    print(f"ntheta                 : {ntheta}")
    print(f"nphi                   : {nphi}")
    print(f"radial_scale           : {radial_scale}")
    print(f"radial_power           : {radial_power}")
    print(f"radial_offset          : {radial_offset}")
    print(f"show_center            : {show_center}")
    print(f"show_kin_arrow         : {show_kin_arrow}")
    print(f"arrow_scale            : {arrow_scale}")
    print("==============================================\n")


def sphere_af(q_vec, n_hat, k_in_hat, kR=5.0):
    q = np.linalg.norm(q_vec, axis=-1)
    x = kR * q

    AF = np.empty_like(x, dtype=float)
    small = np.isclose(x, 0.0)
    xs = x[~small]

    AF[~small] = 3.0 * (np.sin(xs) - xs * np.cos(xs)) / (xs**3)
    AF[small] = 1.0
    return AF


def cube_af(q_vec, n_hat, k_in_hat, kL=6.0):
    def sinc(x):
        out = np.ones_like(x, dtype=float)
        m = ~np.isclose(x, 0.0)
        out[m] = np.sin(x[m]) / x[m]
        return out

    qx = q_vec[..., 0]
    qy = q_vec[..., 1]
    qz = q_vec[..., 2]

    return (
        sinc(0.5 * kL * qx) *
        sinc(0.5 * kL * qy) *
        sinc(0.5 * kL * qz)
    )


k_in_hat = [0.0, 1.0, 1.0]

fig, ax, out = plot_radiation_pattern_3d(
        af_fn=sphere_af,              # or cube_af
    k_in_hat=k_in_hat,
    af_mode="intensity",
    af_kwargs={"kR": 20.0},        # for sphere
   #  af_kwargs={"kL": 20.0},      # for cube
    radial_scale=1.0,
    radial_power=0.3,
    title=f"Sphere geometry. 3D radiation pattern {k_in_hat}, kL = 20.0",
)


plt.show()
