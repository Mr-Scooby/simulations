#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Build the atoms positions array """

import numpy as np
import logging 

log = logging.getLogger(__name__)

def sample_axis(rng, half_length: float, sigma: float, n: int) -> np.ndarray:
    """ Samples the atom pos for one dimensional axis. random or gaussian distribution"""

    if sigma is None:
        return rng.uniform(-half_length, half_length, size=n)
    return rng.normal(0.0, sigma, size=n)


def generate_candidates_box(cloud, n: int, rng) -> np.ndarray:
    """ generates box with atoms position. size volumen  cloud.Lx * cloud.Ly * cloud.Lz  """

    if cloud.Lx is None or cloud.Ly is None or cloud.Lz is None:
        raise ValueError("Box requires Lx, Ly, Lz")

    n = int(n **( 1/3)) # scale teh number of atoms per axis. 
    x = sample_axis(rng, cloud.Lx / 2.0, cloud.sigma_x, n)
    y = sample_axis(rng, cloud.Ly / 2.0, cloud.sigma_y, n)
    z = sample_axis(rng, cloud.Lz / 2.0, cloud.sigma_z, n)

    return np.column_stack([x, y, z])


def mask_box(xyz: np.ndarray, cloud) -> np.ndarray:
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    return (
        (np.abs(x) <= cloud.Lx / 2.0) &
        (np.abs(y) <= cloud.Ly / 2.0) &
        (np.abs(z) <= cloud.Lz / 2.0)
    )


def mask_sphere(xyz: np.ndarray, cloud) -> np.ndarray:
    if cloud.R is None:
        raise ValueError("Sphere requires R")
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    return x*x + y*y + z*z <= cloud.R**2


def mask_cylinder(xyz: np.ndarray, cloud) -> np.ndarray:
    if cloud.R is None or cloud.Lz is None:
        raise ValueError("Cylinder requires R and Lz")
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    return (x*x + y*y <= cloud.R**2) & (np.abs(z) <= cloud.Lz / 2.0)


def sample_with_mask(cloud, mask_fn, rng) -> np.ndarray:

    """ for complex geometryes i.e not box, we create a mask and keep only atoms lying inside the mask. repeat untill we have the number of atoms that match the desnity """

    parts = []
    n_kept = 0

    while n_kept < cloud.n_atoms:
        xyz_try = generate_candidates_box(cloud,cloud.n_atoms , rng)
        keep = mask_fn(xyz_try, cloud)
        xyz_keep = xyz_try[keep]

        if xyz_keep.size == 0:
            continue

        parts.append(xyz_keep)
        n_kept += xyz_keep.shape[0]

    xyz = np.vstack(parts)
    return xyz[:cloud.n_atoms]


def make_positions(cloud, rng=None) -> np.ndarray:

    rng = np.random.default_rng(rng)

    if cloud.distribution not in {"random", "gaussian"}:
        raise ValueError("distribution must be 'random' or 'gaussian'")

    if cloud.distribution == "gaussian" and not cloud.has_any_sigma:
        raise ValueError(
            "distribution='gaussian' requires at least one of "
            "sigma_x, sigma_y, sigma_z"
        )

    # If distribution is "random", sigmas can still be None on all axes,
    # which means fully uniform.
    if cloud.geometry == "box":
        return sample_with_mask(cloud,mask_box, rng)

    if cloud.geometry == "sphere":
        if cloud.R is None:
            raise ValueError("Sphere requires R")
        # bounding box for sphere
        cloud_local = type(cloud)(**cloud.__dict__)
        cloud_local.Lx = 2.0 * cloud.R
        cloud_local.Ly = 2.0 * cloud.R
        cloud_local.Lz = 2.0 * cloud.R
        return sample_with_mask(cloud_local, mask_sphere, rng)

    if cloud.geometry == "cylinder":
        if cloud.R is None or cloud.Lz is None:
            raise ValueError("Cylinder requires R and Lz")
        cloud_local = type(cloud)(**cloud.__dict__)
        cloud_local.Lx = 2.0 * cloud.R
        cloud_local.Ly = 2.0 * cloud.R
        return sample_with_mask(cloud_local, mask_cylinder, rng)

    raise ValueError(f"Unsupported geometry: {cloud.geometry!r}")
