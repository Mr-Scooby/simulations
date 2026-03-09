#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Calculates the weights for the atoms for a traveling pulse wave incident in the cloud """

import numpy as np 
import logging 


log = logging.getLogger(__name__)


def upstream_front_position(center, box_size, k_in_hat, margin=1.0):
    """
    Place the pulse front just outside the upstream edge of a box-shaped cloud.

    Parameters
    ----------
    center : array-like, shape (3,)
        Cloud center.
    box_size : array-like, shape (3,)
        Box side lengths (Lx, Ly, Lz).
    k_in_hat : array-like, shape (3,)
        Beam propagation direction.
    margin : float, optional
        Extra distance added upstream.

    Returns
    -------
    np.ndarray, shape (3,)
        Initial pulse-front center.
    """
    center = np.asarray(center, dtype=float)
    box_size = np.asarray(box_size, dtype=float)
    k_in_hat = np.asarray(k_in_hat, dtype=float)
    k_in_hat /= (np.linalg.norm(k_in_hat) + 1e-15)

    half_extent_along_k = 0.5 * np.sum(np.abs(k_in_hat) * box_size)
    return center - (half_extent_along_k + margin) * k_in_hat


def make_weight_fn_gaussian_pulse(
    w0,
    sigma_long,
    k_in_hat,
    k_in=1.0,
    v_front=1.0,
    box_size=(1.0, 1.0, 1.0),
    center=(0.0, 0.0, 0.0),
    margin=0.0,
    pulse_center_t0=0.0,
):
    """
    Build w_fn(r_xyz, t) for a pulsed Gaussian beam propagating through the cloud.

    The weight is
        w(r, t) = env_perp(r, t) * env_long(r, t) * exp(i k_in k_in_hat·r)

    where:
    - env_perp is the transverse Gaussian beam profile
    - env_long is the longitudinal Gaussian pulse envelope
    - the pulse front moves along k_in_hat with speed v_front

    Parameters
    ----------
    w0 : float
        Transverse beam waist.
    sigma_long : float
        Longitudinal pulse width.
    k_in_hat : array-like, shape (3,)
        Beam propagation direction.
    k_in : float, optional
        Incident wave number.
    v_front : float, optional
        Pulse propagation speed.
    box_size : tuple, optional
        Cloud box size used to place the initial front upstream.
    center : tuple, optional
        Cloud center.
    margin : float, optional
        Extra upstream offset.
    pulse_center_t0 : float, optional
        Shift of the pulse center along the propagation direction at t=0.

    Returns
    -------
    callable
        Function w_fn(r_xyz, t) -> complex weights of shape (N,)
    """
    k_in_hat = np.asarray(k_in_hat, dtype=float)
    k_in_hat /= (np.linalg.norm(k_in_hat) + 1e-15)

    # Simulation window. 
    center = np.asarray(center, dtype=float)
    box_size = np.asarray(box_size, dtype=float)

    # position of the center of the pulse. 
    r_front0 = upstream_front_position(
        center=center,
        box_size=box_size,
        k_in_hat=k_in_hat,
        margin=margin,
    )

    # Optional shift of the pulse center at t=0
    r_front0 = r_front0 + float(pulse_center_t0) * k_in_hat

    def w_fn(r_xyz, t, return_pulse_center = False):
        """
        Parameters
        ----------
        r_xyz : np.ndarray, shape (N, 3)
            Atom positions at time t.
        t : float
            Time.

        Returns
        -------
        np.ndarray, shape (N,)
            Complex beam weights.
        """
        r_xyz = np.asarray(r_xyz, dtype=float)

        # Pulse-front center moving along k_in_hat
        r_front_t = r_front0 + v_front * float(t) * k_in_hat

        # Coordinates relative to the moving pulse center
        dr = r_xyz - r_front_t[None, :]

        # Longitudinal coordinate along propagation
        u_par = dr @ k_in_hat

        # Transverse squared distance to the beam axis
        dr2 = np.sum(dr * dr, axis=1)
        u_perp2 = dr2 - u_par**2

        # Gaussian beam envelope
        env_perp = np.exp(-u_perp2 / (w0**2))

        # Gaussian pulse envelope along propagation direction
        env_long = np.exp(-(u_par**2) / (sigma_long**2))

        # Incident optical phase
        phase =  np.exp(1j * k_in * (r_xyz @ k_in_hat))
        if return_pulse_center == True :
            return (env_perp * env_long * phase).astype(np.complex128), r_front_t
        else: 
            return (env_perp * env_long * phase).astype(np.complex128)

    return w_fn


