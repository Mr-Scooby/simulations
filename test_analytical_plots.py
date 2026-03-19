#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import analytical_patterns as ap
import rplotting as rpt 
import matplotlib.pyplot as plt 
import helpers as hps


# normal incidence
k_in_hat = np.array([0.0, 0.0, 1.0])

n_theta = 2*181
n_phi = 2*361

# Sphere
fig, ax, theta, phi, I, AF = rpt.plot_analytic_pattern_3d(
    ap.box_af,
    k_in_hat=k_in_hat,
    af_kwargs={"kLx": 10000, "kLy":1000, "kLz":1, "kLz":100},
    title="Sphere analytical pattern",
    alpha=0.5,
)

plt.show()
