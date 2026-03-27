#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Creates and stores the geometry distribution """


from dataclasses import dataclass, field, asdict
from radpattern.helpers import io
from .sampling import make_positions

import numpy as np
import logging 

log = logging.getLogger(__name__) 


@dataclass
class CloudModel:
    geometry: str                  # "box", "sphere", ...
    distribution: str              # "lattice", "uniform", "gaussian"

    # geometry parameters
    Lx:float   = None
    Ly:float   = None
    Lz:float   = None
    R :float   = None

    # distribution parameters
    density: float = 1
    n_atoms: int  = None #field(init=False) 

    # anisotropic Gaussian widths
    sigma_x:float  = None
    sigma_y:float  = None
    sigma_z:float  = None


    @property
    def volumen(self): 
        if self.geometry == "box": 
            log.debug("Calculating volume for geom. = box") 
            return  self.Lx * self.Ly * self.Lz
        elif self.geometry ==  "sphere":
            log.debug("Calculating volume for geom. = sphere") 
            return  (4/3) * np.pi * self.R**3 
        elif self.geometry == "cylinder": 
            log.debug("Calculating volume for geom. = cylinder") 
            return  np.pi * self.R**2 * self.Lz
        else: 
            raise ValueError(f"No volumen formula define yet for geometry= {self.geometry}")

    @property 
    def spacing(self):
        return self.density ** (-1/3) 

    @property
    def has_any_sigma(self) -> bool:
        return any(s is not None for s in (self.sigma_x, self.sigma_y, self.sigma_z))
    
    @property
    def aspect_ratio(self):
        return self.Lz/ self.Lx


    def make_positions(self, rng=None) -> np.ndarray:
        return make_positions(self, rng=rng)

     def log_info(self):
        """ summary of the cloud being simulated. All lengths in units of wavelength."""
        log.info("====================================================")
        log.info("Cloud model summary")
        log.info("All length units below are relative to wavelength lambda")
        log.info("geometry         = %s", self.geometry)
        log.info("distribution     = %s", self.distribution)

        if self.geometry == "box":
            log.info("Lx               = %.6g lambda", self.Lx)
            log.info("Ly               = %.6g lambda", self.Ly)
            log.info("Lz               = %.6g lambda", self.Lz)
            if self.aspect_ratio is not None:
                log.info("aspect ratio Lz/Lx = %.6g", self.aspect_ratio)

        elif self.geometry == "sphere":
            log.info("R                = %.6g lambda", self.R)
            log.info("diameter         = %.6g lambda", 2 * self.R)

        elif self.geometry == "cylinder":
            log.info("R                = %.6g lambda", self.R)
            log.info("diameter         = %.6g lambda", 2 * self.R)
            log.info("Lz               = %.6g lambda", self.Lz)

        log.info("volume           = %.6g lambda^3", self.volumen)
        log.info("density          = %.6g lambda^-3", self.density)
        log.info("mean spacing      = %.6g lambda", self.spacing)
        log.info("n_atoms          = %d", self.n_atoms)

        if self.distribution == "gaussian":
            log.info("sigma_x          = %s", f"{self.sigma_x:.6g} lambda" if self.sigma_x is not None else "None")
            log.info("sigma_y          = %s", f"{self.sigma_y:.6g} lambda" if self.sigma_y is not None else "None")
            log.info("sigma_z          = %s", f"{self.sigma_z:.6g} lambda" if self.sigma_z is not None else "None")

        log.info("====================================================")

    def __post_init__(self):
        #io.log_attrs(log, self, ["geometry", "distribution"], "Cloud Model: ")
        self.n_atoms = round( self.volumen * self.density )
        self.log_info()


