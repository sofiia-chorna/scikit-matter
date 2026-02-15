#!/usr/bin/env python
# coding: utf-8
"""
Sketch-Map
==========

This example demonstrates the :class:`~skmatter.decomposition.SketchMap` estimator for
nonlinear dimensionality reduction.
"""

# %%
# Example 1: Swiss roll dataset
# -----------------------------
#
# First, let's demonstrate SketchMap on the classic swiss roll dataset.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll

from skmatter.decomposition import SketchMap

X, color = make_swiss_roll(n_samples=500, noise=0.5, random_state=42)
print(f"Swiss roll data shape: {X.shape}")

# %%
# Fit SketchMap with automatic parameter estimation

sm = SketchMap(n_components=2, random_state=42, verbose=True)
embedding = sm.fit_transform(X)

print(f"\nEstimated parameters: {sm.params_}")

# %%
# Visualize the embedding

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Original data
axes[0].scatter(X[:, 0], X[:, 2], c=color, cmap="viridis", s=20)
axes[0].set_title("Original Swiss Roll (X vs Z)")
axes[0].set_xlabel("X")
axes[0].set_ylabel("Z")

# SketchMap embedding
sc = axes[1].scatter(embedding[:, 0], embedding[:, 1], c=color, cmap="viridis", s=20)
axes[1].set_title(f"SketchMap Embedding (stress={sm.stress_:.4f})")
axes[1].set_xlabel("Dimension 1")
axes[1].set_ylabel("Dimension 2")

plt.colorbar(sc, ax=axes[1], label="Position along roll")
plt.tight_layout()
plt.show()


# %%
# Example 2: Comparison with C++ sketchmap reference
# --------------------------------------------------
#
# This example loads high-dimensional landmark data and compares the Python
# implementation with the original C++ sketchmap output.

# Load high-dimensional landmarks (1000 samples, 1024 features + weight)
data = np.loadtxt("highd-landmarks")
X_hd = data[:, :-1]
weights = data[:, -1]

print(f"Loaded {X_hd.shape[0]} landmarks with {X_hd.shape[1]} features")

# Load C++ reference embedding for comparison
lowd_cpp = np.loadtxt("low_landmarks.dat", comments="#")
lowd_cpp = lowd_cpp[:, :2]
print(f"C++ reference embedding shape: {lowd_cpp.shape}")

# %%
# Fit SketchMap with same parameters as C++ reference

sm_py = SketchMap(
    n_components=2,
    params={"sigma": 7.0, "a_hd": 4.0, "b_hd": 2.0, "a_ld": 2.0, "b_ld": 2.0},
    verbose=True,
)
lowd_py = sm_py.fit_transform(X_hd, sample_weights=weights)

print(f"\nPython stress: {sm_py.stress_:.6f}")

# Plot comparison

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Python embedding
axes[0].scatter(
    lowd_py[:, 0], lowd_py[:, 1], c=weights, cmap="viridis", s=20, edgecolor="k", lw=0.3
)
axes[0].set_title(f"Python SketchMap\n(stress={sm_py.stress_:.4f})")
axes[0].set_xlabel("Dimension 1")
axes[0].set_ylabel("Dimension 2")

# C++ embedding
axes[1].scatter(
    lowd_cpp[:, 0],
    lowd_cpp[:, 1],
    c=weights,
    cmap="viridis",
    s=20,
    edgecolor="k",
    lw=0.3,
)
axes[1].set_title("C++ Sketchmap")
axes[1].set_xlabel("Dimension 1")
axes[1].set_ylabel("Dimension 2")

plt.tight_layout()
plt.show()
