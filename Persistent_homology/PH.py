#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Achraf Atila
Date: 12.11.2025
email: achraf.atila@gmail.com

Topological Analysis of Silica Glass via Persistent Homology
------------------------------------------------------------

This script computes persistence diagrams using the HomCloud package
through an alpha-complex filtration of atomic configurations.

Methodology Details
-------------------
- Software: HomCloud (v4.4.0; https://homcloud.dev/)
- Complex type: Alpha complex (squared radii)
- Filtration range: Automatically determined from data
- Weights: Atomic radii set to zero for unweighted analysis
- Periodic boundaries: Enabled, using ASE cell vectors for periodic embedding
- Histogram binning: 256x256 in birth-death space
- Input: ASE-readable atomic configuration
- Output: .pdgm persistence diagram
- Visualization: Seaborn + Matplotlib

"""

import numpy as np
import ase.io
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import ListedColormap
import homcloud.interface as hc

# -------------------------------------------------------------------------
# Plotting configuration
# -------------------------------------------------------------------------
plt.rc('font', family='serif', serif='palatino', size=12)
plt.rc('text', usetex=True)
sns.set_theme(style="ticks", rc={
    'axes.linewidth': 1,
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    "figure.figsize": (5, 5),
})
border = mpe.withStroke(linewidth=.5, foreground='black')

# -------------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------------
pressure = 0                # Pressure index
data_path = f"SiO2_0.xyz"

# -------------------------------------------------------------------------
# Input structure and atomic weights
# -------------------------------------------------------------------------
atoms = ase.io.read(data_path)
radii = {"O": 0.0, "Si": 0.0}  # unweighted alpha complex, can be replaced with radii obtained from rdf
weights = np.array([radii[a] for a in atoms.get_chemical_symbols()])

# Extract cell for periodicity
cell = atoms.cell

periodicity = [(0, cell[0][0]), (0, cell[1][1]), (0, cell[2][2])]

# -------------------------------------------------------------------------
# Persistence diagram computation (with periodic boundaries)
# -------------------------------------------------------------------------
hc.PDList.from_alpha_filtration(
    atoms.get_positions(),
    weight=weights,
    save_to=f"SiO2_{pressure}.pdgm",
    vertex_symbols=atoms.get_chemical_symbols(),
    save_boundary_map=True,
    squared=True,
    periodicity=periodicity
)

# Load diagram
pdlist = hc.PDList(f"SiO2_{pressure}.pdgm")
pd1 = pdlist.dth_diagram(1)
pd2 = pdlist.dth_diagram(2)

# -------------------------------------------------------------------------
# Plot persistence diagram
# -------------------------------------------------------------------------
min_val, max_val = 0.25, 10.25
g = sns.JointGrid(
    x=pd1.births, y=pd1.deaths,
    height=5, ratio=3, space=0,
    xlim=(min_val, max_val), ylim=(min_val, max_val)
)
g.plot_joint(sns.histplot, bins=256, cmap='Spectral_r', pmax=0.9, rasterized=True)
g.ax_joint.plot([min_val, max_val], [min_val, max_val], ':k', linewidth=2.5)
g.set_axis_labels(r"Birth ${\rm (\AA^2)}$", r"Death ${\rm (\AA^2)}$")
g.ax_joint.xaxis.set_minor_locator(AutoMinorLocator(5))
g.ax_joint.yaxis.set_minor_locator(AutoMinorLocator(5))
g.plot_marginals(sns.histplot, kde=False, edgecolor='k', bins=256, color='#000075')
g.ax_joint.text(0.85, 0.06, rf"H$_{{{1}}}$", fontsize=14, transform=g.ax_joint.transAxes)
g.ax_joint.text(0.08, 1.0, rf"{str(pressure * 10)} GPa", fontsize=14, transform=g.ax_joint.transAxes)
plt.tight_layout()
plt.show()


g = sns.JointGrid(
    x=pd2.births, y=pd2.deaths,
    height=5, ratio=3, space=0,
    xlim=(min_val, max_val), ylim=(min_val, max_val)
)
g.plot_joint(sns.histplot, bins=256, cmap='Spectral_r', pmax=0.9, rasterized=True)
g.ax_joint.plot([min_val, max_val], [min_val, max_val], ':k', linewidth=2.5)
g.set_axis_labels(r"Birth ${\rm (\AA^2)}$", r"Death ${\rm (\AA^2)}$")
g.ax_joint.xaxis.set_minor_locator(AutoMinorLocator(5))
g.ax_joint.yaxis.set_minor_locator(AutoMinorLocator(5))
g.plot_marginals(sns.histplot, kde=False, edgecolor='k', bins=256, color='#000075')
g.ax_joint.text(0.85, 0.06, rf"H$_{{{2}}}$", fontsize=14, transform=g.ax_joint.transAxes)
g.ax_joint.text(0.08, 1.0, rf"{str(pressure * 10)} GPa", fontsize=14, transform=g.ax_joint.transAxes)
plt.tight_layout()
plt.show()
