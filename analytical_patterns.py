#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Analitical solutions """ 


import numpy as np


def _sinc(x):
    """
    sinc(x) = sin(x)/x with sinc(0)=1
    """
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x, dtype=float)
    m = ~np.isclose(x, 0.0)
    out[m] = np.sin(x[m]) / x[m]
    return out


def sphere_af(q_vec, kR):
    """
    Analytical amplitude for a uniform solid sphere.

    Parameters
    ----------
    q_vec : array (..., 3)
        Scattering vector q = k_in_hat - n_hat
        or any q-vector array with last axis = 3.
    kR : float
        Dimensionless size parameter k * R.

    Returns
    -------
    AF : array (...)
        Real amplitude array.
    """
    q = np.linalg.norm(q_vec, axis=-1)
    x = kR * q

    AF = np.empty_like(x, dtype=float)
    small = np.isclose(x, 0.0)

    xs = x[~small]
    AF[~small] = 3.0 * (np.sin(xs) - xs * np.cos(xs)) / (xs**3)
    AF[small] = 1.0

    return AF


def box_af(q_vec, kLx, kLy=None, kLz=None):
    """
    Analytical amplitude for a uniform box.

    Parameters
    ----------
    q_vec : array (..., 3)
        Scattering vector array.
    kLx, kLy, kLz : float
        Dimensionless lengths k*Lx, k*Ly, k*Lz.
        If kLy or kLz are None, uses cube values.

    Returns
    -------
    AF : array (...)
        Real amplitude array.
    """
    if kLy is None:
        kLy = kLx
    if kLz is None:
        kLz = kLx

    qx = q_vec[..., 0]
    qy = q_vec[..., 1]
    qz = q_vec[..., 2]

    return (
        _sinc(0.5 * kLx * qx) *
        _sinc(0.5 * kLy * qy) *
        _sinc(0.5 * kLz * qz)
    )


def slab_2d_af(q_vec, kb, axis="z"):
    """
    Analytical amplitude for an infinite 2D slab with finite thickness b.

    Infinite in the two transverse directions, finite along `axis`.

    Since an ideal infinite plane gives delta-functions in reciprocal space,
    we cannot represent exact deltas numerically on a regular array.
    So this function returns only the finite-thickness envelope along the
    slab-normal direction:

        AF ~ sinc( q_normal * k*b / 2 )

    This is the useful lobe envelope.

    Parameters
    ----------
    q_vec : array (..., 3)
        Scattering vector array.
    kb : float
        Dimensionless thickness parameter k*b.
    axis : {"x", "y", "z"}
        Slab normal direction.

    Returns
    -------
    AF : array (...)
        Real amplitude array.
    """
    axis_map = {"x": 0, "y": 1, "z": 2}
    i = axis_map[axis]
    qn = q_vec[..., i]
    return _sinc(0.5 * kb * qn)


def slab_2d_gaussian_af(q_vec, ksigma, axis="z"):
    """
    Analytical amplitude for an infinite 2D slab with Gaussian density
    along the normal direction.

        rho(z) ~ exp( -z^2 / (2 sigma^2) )

    Fourier transform envelope:

        AF ~ exp( - (ksigma)^2 q_normal^2 / 2 )

    Parameters
    ----------
    q_vec : array (..., 3)
        Scattering vector array.
    ksigma : float
        Dimensionless width parameter k*sigma.
    axis : {"x", "y", "z"}
        Slab normal direction.

    Returns
    -------
    AF : array (...)
        Real amplitude array.
    """
    axis_map = {"x": 0, "y": 1, "z": 2}
    i = axis_map[axis]
    qn = q_vec[..., i]
    return np.exp(-0.5 * (ksigma**2) * (qn**2))
