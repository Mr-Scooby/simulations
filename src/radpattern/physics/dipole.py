#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import logging

log = logging.getLogger(__name__)

# Dipole pattern
PZ_HAT = np.array([0.0, 0.0, 1.0], dtype=float) # +z vector 
def single_dipole_E(nx, ny, nz, p_hat = PZ_HAT):
    """
    Far-field dipole angular dependence:
      E propto n X (n X p) = p - n (n·p)

    p_hat: (3,) unit vector (or will work if not perfectly unit) as array
    Returns Ex,Ey,Ez complex arrays with same shape as nx.
    """

    ndotp = nx * p_hat[0] + ny * p_hat[1] + nz * p_hat[2]
    Ex = p_hat[0] - nx * ndotp
    Ey = p_hat[1] - ny * ndotp
    Ez = p_hat[2] - nz * ndotp
    
    log.info("Construction of single dipole pattern. Dipole vector %s", p_hat)
    return (np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2)


def intensity_from_field(AF, dipole):
    """
    I = |E|^2 where E = AF * E_single_dipole
    """
    log.debug("Computing intensity from field: E shape=%s.", AF.shape)
    return np.abs(AF)**2 * dipole 


