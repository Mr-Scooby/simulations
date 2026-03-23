#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
import numpy as np
from radpattern.helpers import helpers
from radpattern.helpers import io
import logging

log = logging.getLogger(__name__)


@dataclass
class AngleGrid:
    n_theta: int = 91
    n_phi: int = 181

    theta: np.ndarray = field(init=False)
    phi: np.ndarray = field(init=False)
    nx: np.ndarray = field(init=False)
    ny: np.ndarray = field(init=False)
    nz: np.ndarray = field(init=False)
    n_hat: np.ndarray = field(init=False)
    n_hat_flat: np.ndarray = field(init=False)
    shape: tuple = field(init=False)

    def __post_init__(self):
        self.theta, self.phi, self.nx, self.ny, self.nz = helpers.make_angle_grid(
            n_theta=self.n_theta,
            n_phi=self.n_phi,
        )
        self.n_hat = np.stack([self.nx, self.ny, self.nz], axis=-1)
        self.n_hat_flat = self.n_hat.reshape(-1, 3)
        self.shape = (self.n_theta, self.n_phi)

        io.log_attrs(log, self, ["n_theta", "n_phi", "nx", "n_hat", "n_hat_flat"],
        prefix="AngleGrid built: "
    )
