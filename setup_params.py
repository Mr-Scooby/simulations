#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field, asdict
import numpy as np
import logging
import hashlib
import hashlib
import json


log = logging.getLogger(__name__) 

@dataclass
class PhysicalRegime: 
    """ Physical regime of the simualtion.  """ 
    # Box size. how large the cloud is compared to the wavelength, so how much spatial phase can build up across it.
    # L/lambda
    optical_size : float = 100.0 

    # Interparticle spacing in units of wavelength. a/ lambda. We want a >sim lambda
    optical_spacing: float = 1.5 

    # illumination ratio: How much of the cloud the beam covers transversely. w0/ L_perp
    illumination_ratio : float = 0.8 

    # Longitudinal filling factor: How much of the cloud is covered along propagation.
    # sigma_long / L_parallel
    filling_factor: float = 0.1

    # Pulse transit. how far the pulse front has traveled through the cloud. for time parameters. 
    # v_front t / L_parrallel. 
    pulse_transit : float = 1.5 


# ------------------------------------------------------------------
# Actual physical parameters
@dataclass
class PhysicalParams:
    """
    Physical scales used by the simulation.
    By default we work in units where lambda = 1.
    """

    regime: PhysicalRegime = field(default_factory=PhysicalRegime)

    # Optical units
    wavelength: float = 1.0
    k0: float = field(init=False)

    # Input polarization
    p_hat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))

    # Cloud geometry
    # Simulation box size
    L: float = field( init = False) 
    box_size: np.ndarray = field(init=False)

    # Microscopic scale
    # Interparticle distance a
    spacing: float = field(init=False)
    density: float = field(init=False)
    atoms: int  = field(init=False) 

    # Beam / pulse scales
    beam_waist: float = field(init=False)
    sigma_long: float = field(init=False)
    # Incident beam direction
    k_in_hat: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    # Position of the beam at t=0 in the direction of k_in. 
    beam_r0: float = 0
    pcenter_atOrigin: bool = False


    # Motion / time 
    v_front: float = 1.0
    v_thermal: float = 0.001

    # Dinamics
    #characteristic time. time to cross the one atom cloud.
    t_char: float = field(init=False)
    # dephase due to thermal motion. 
    mot_dephase: float = field(init=False)



    ## Once class is called. __init__ is done. __post_init__ does the math. 
    def __post_init__(self):
        r = self.regime

        self.k0 = 2 * np.pi / self.wavelength
        
        # Normalize propagation / polarization directions
        self.k_in_hat = np.asarray(self.k_in_hat, dtype=float)
        self.k_in_hat /= np.linalg.norm(self.k_in_hat)

        self.p_hat = np.asarray(self.p_hat, dtype=float)
        self.p_hat /= np.linalg.norm(self.p_hat)
        
        # Characteristic cloud size
        self.L = r.optical_size * self.wavelength 
        # Simulation box size
        self.box_size = np.array([self.L, self.L, self.L], dtype=float)

        # Microscopic spacing and density
        self.spacing = r.optical_spacing * self.wavelength
        self.density = 1.0 / self.spacing**3

        # Illumination / pulse scales
        self.beam_waist = r.illumination_ratio * self.L
        self.sigma_long = r.filling_factor * self.L

        # Atom number
        self.atoms = int(round(self.L / self.spacing )**3)

        # Choose thermal speed so that k v_th t_char = mot_dephase,
        # with t_char taken from pulse transit through the cloud => the time to cross the clod length. 
        try: 
            self.t_char = r.pulse_transit * self.L /self.v_front
        except: 
            # in case we want v =0. we set a default t_char of 10. 
            self.t_char = 10

        # Motional dephasing accumulated over t_char
        self.mot_dephase = self.k0 * self.v_thermal * self.t_char

# 3) Simulation / numerical parameters
# ------------------------------------------------------------------
@dataclass
class SimParams:
    """Numerical controls for the Monte Carlo simulation."""

    n_atoms: int = 5000
    n_mc: int = 100

    # Time sampling
    t_max_factor: float = 1.5
    t_char : float = 1
    n_times: int = 100
    t_max: float = field(init = False) 

    # Angular grid
    n_theta: int = 91
    n_phi: int = 181
    grid_shape: int  = field(init = False) 

    # Performance / implementation
    chunk_atoms: int = 2000
    normalize_each_time: bool = False
    plane_restricted: bool = False
    seed: int = None

    # File naming 
    # Computed run name: human-readable + hash from all params

    def __post_init__(self):
        self.t_max = self.t_max_factor * self.t_char
        self.grid_shape = (self.n_theta, self.n_phi) 

    @property
    def times(self) -> np.ndarray:
        return np.linspace(0.0, self.t_max, self.n_times)

    


def _k_tag(k_hat) -> str:
    return "k" + "".join(str(round(x)) for x in k_hat)

@dataclass
class SetupParams:
    """ Stores metadata and creates run naming """
    regime: PhysicalRegime
    phys: PhysicalParams
    sim: SimParams

    # Computed run name: human-readable + hash from all params
    @property
    def run_name(self) -> str:
        # hash full setup
        d = {
            "regime": asdict(self.regime),
            "phys": asdict(self.phys),
            "sim": asdict(self.sim),
        }
        h = hashlib.sha1(
            json.dumps(d, sort_keys=True, default=str).encode()
        ).hexdigest()[:8]

        return (
            f"N{self.sim.n_atoms}"
            f"_mc{self.sim.n_mc}"
            f"_nt{self.sim.n_times}"
            f"_{_k_tag(self.phys.k_in_hat)}"
            f"_{h}"
        )


def log_main_params(log, main) -> None:
    log.info("==== Main control parameters ====")
    log.info("k_in_hat           = %s", main.k_in_hat)
    log.info("density            = %.6g", main.density)
    log.info("pulse_duration     = %.6g", main.pulse_duration)
    log.info("pulse_speed        = %.6g", main.pulse_speed)
    log.info("pulse_waist        = %.6g", main.pulse_waist)
    log.info("thermal_velocity   = %.6g", main.thermal_velocity)
    log.info("beam_cloud_overlap = %.6g", main.beam_cloud_overlap)
    log.info("=================================")
