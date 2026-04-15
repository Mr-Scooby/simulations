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
    distribution: str              # "lattice", "random", "gaussian"

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

    def __post_init__(self):
        #io.log_attrs(log, self, ["geometry", "distribution"], "Cloud Model: ")
        self.n_atoms = round( self.volumen * self.density )

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
            raise ValueError(f"No volumen formula define yet for geometry= {self.geometry}\n Valid current geometries: box, sphere, cylinder")

    #@property 
    #def spacing(self):
    #    return self.density ** (-1/3) 

    @property
    def has_any_sigma(self) -> bool:
        return any(s is not None for s in (self.sigma_x, self.sigma_y, self.sigma_z))
    
    @property
    def aspect_ratio(self):
        return self.Lz/ self.Lx
    
    @property 
    def box_size(self): 
        if self.geometry =="box":
            return np.asarray([self.Lx, self.Ly, self.Lz])
        elif self.geometry == "sphere":
            #take the box to which a sphere is inside. 
            D = 2* self.R
            return np.asarray([D,D,D])
        elif self.geometry =="cylinder":
            return np.asarray([2*self.R, 2*self.R, self.Lz])

    @property
    def spacing(self): 
        return 1 / (self.density ** (1/3))

    def make_positions(self, rng=None) -> np.ndarray:
        log.info("Constructing atom positions...") 
        return make_positions(self, rng=rng)


    ## For later. 
    # def make_velocity_distribution
        # return velocty array 
    
    # def update_position( time): 
        # Ballistic motion update


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
            j.info("diameter         = %.6g lambda", 2 * self.R)

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


    def report_density_profile(
        self,
        r_xyz,
        cloud_fracs=(0.0, 0.25, 0.5, 0.75, 1.0),
        probe_radius= 1 ,
    ):
        """
        Written diagnostic of the sampled atom density profile.
    
        Reports local density at probe points along:
          - x axis (transverse)
          - z axis (longitudinal)
    
        Parameters
        ----------
        r_xyz : np.ndarray, shape (N, 3)
            Sampled atom positions.
        cloud_fracs : tuple
            Fractions of the cloud extent where density is probed.
        probe_radius : float or None
            Radius of spherical probe volume. If None, use 0.5 * spacing.
        """
        r_xyz = np.asarray(r_xyz, dtype=float)
        if r_xyz.ndim != 2 or r_xyz.shape[1] != 3:
            raise ValueError(f"r_xyz must have shape (N, 3), got {r_xyz.shape}")
    
        if probe_radius is None:
            probe_radius = 0.5 * self.spacing
    
        def local_density(point, radius):
            d2 = np.sum((r_xyz - point[None, :]) ** 2, axis=1)
            n_local = int(np.count_nonzero(d2 <= radius**2))
            vol = (4.0 / 3.0) * np.pi * radius**3
            rho_local = n_local / vol
            return n_local, rho_local
    
        # characteristic cloud extents
        if self.geometry in {"cylinder", "sphere"} and self.R is not None:
            R_cloud = float(self.R)
        else:
            R_cloud = 0.5 * float(self.Lx)
    
        Lz = float(self.Lz)
    
        print("\n=== Cloud density profile report ===")
        print(f"geometry = {self.geometry}")
        print(f"n_atoms = {len(r_xyz)}")
        print(f"target density = {self.density:.6g} atoms / lambda^3")
        print(f"probe_radius = {probe_radius:.6g} lambda")
    
        print("\nTransverse density profile (along +x, y=z=0):")
        for f in cloud_fracs:
            x = f * R_cloud
            point = np.array([x, 0.0, 0.0], dtype=float)
            n_local, rho_local = local_density(point, probe_radius)
            print(
                f"  at {100*f:>5.1f}% of cloud radius:"
                f" x={x:>8.4f}, n_local={n_local:>4d},"
                f" rho_local={rho_local:.6g}"
            )
    
        print("\nLongitudinal density profile (along z, x=y=0):")
        for f in cloud_fracs:
            z = -0.5 * Lz + f * Lz
            point = np.array([0.0, 0.0, z], dtype=float)
            n_local, rho_local = local_density(point, probe_radius)
            print(
                f"  at {100*f:>5.1f}% of cloud length:"
                f" z={z:>8.4f}, n_local={n_local:>4d},"
                f" rho_local={rho_local:.6g}"
            )
