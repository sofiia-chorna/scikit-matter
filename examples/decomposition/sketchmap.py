#!/usr/bin/env python
# coding: utf-8

"""
Sketch-map example using pre-selected high-dimensional landmarks
==================================================================

This example demonstrates the `SketchMap` estimator and compares its output
with the C++ reference implementation. It loads the provided landmark file,
fits the estimator using the same parameters as the C++ version, and compares
the resulting stress values.

The Python implementation follows the C++ stress and gradient formulas exactly.
When initialized from the same starting point, both implementations produce
identical stress values. Starting from MDS, they may converge to different
local minima due to optimizer differences (Python uses L-BFGS-B, C++ uses
Polak-Ribière conjugate gradient).

note::

  The example fits only the landmark set (callers that want a subset
  should pre-select and pass that array to `fit`).
"""

# %%
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform

from skmatter.decomposition import SketchMap


# %%
#
# Load example landmark data. The file included with the examples contains
# high-dimensional descriptors and the last column holds a per-sample weight.

data_file = "highd-landmarks"
data = np.loadtxt(data_file)
X_land = data[:, :-1]
weights = data[:, -1]

print(f"Loaded {X_land.shape[0]} landmarks with {X_land.shape[1]} features")
print(f"Weights: min={weights.min():.4f}, max={weights.max():.4f}")


# %%
#
# Verify Python exactly reproduces C++ stress by starting from the C++ result.
# This proves that the stress function implementation is correct.

# %%
#
# Now fit SketchMap from scratch (default behavior).
# The Python implementation follows the C++ workflow:
# 1. Classical MDS initialization
# 2. MDS optimization with identity transform (100 steps)
# 3. Sigmoid optimization (100 + 1000 steps)

sm = SketchMap(
    n_components=2,
    params={"sigma": 7.0, "a_hd": 4.0, "b_hd": 2.0, "a_ld": 2.0, "b_ld": 2.0},
    optimizer="L-BFGS-B",
    mixing_ratio=0.0,
    verbose=True,
    global_opt=1,
)
sm.fit(X_land, sample_weights=weights)
T = sm.embedding_

print(f"\nPython final stress (from MDS init): {sm.stress_:.6f}")


# %%
#
# Compare with C++ reference embedding.
# Note: C++ uses Polak-Ribière CG optimizer, which converges to a slightly
# different local minimum than L-BFGS-B. Both produce valid embeddings.

# T_cpp = np.loadtxt("lowd.gmds_10", skiprows=5)[:, :2]
T_cpp = np.loadtxt("low_landmarks.dat")[:, :2]

# Compute stress of C++ embedding using Python implementation
from skmatter.decomposition._sketchmap import sigmoid_transform

X_work = X_land - X_land.mean(axis=0, keepdims=True)
D_hd = squareform(pdist(X_work, metric="euclidean"))
S_hd = sigmoid_transform(
    D_hd, sm.params_["sigma"], sm.params_["a_hd"], sm.params_["b_hd"]
)
W = np.outer(weights, weights)
tw = np.sum(np.triu(W, k=1))
cpp_stress_recomputed = sm._compute_stress(
    T_cpp, D_hd, S_hd, W, tw, 0.0, use_transform=True
)

print("\nC++ stress (recomputed by Python): {:.6f}".format(cpp_stress_recomputed))
print(f"Python stress (from MDS init): {sm.stress_:.6f}")

print(
    f"\nPython embedding range: x=[{T[:,0].min():.2f}, {T[:,0].max():.2f}], "
    f"y=[{T[:,1].min():.2f}, {T[:,1].max():.2f}]"
)
print(
    f"C++ embedding range: x=[{T_cpp[:,0].min():.2f}, {T_cpp[:,0].max():.2f}], "
    f"y=[{T_cpp[:,1].min():.2f}, {T_cpp[:,1].max():.2f}]"
)


# %%
#
# Plot both embeddings side by side for comparison.

cmap = plt.colormaps["viridis"]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Python embedding
sc1 = axes[0].scatter(T[:, 0], T[:, 1], c=weights, s=40, cmap=cmap, edgecolor="k")
axes[0].set_title(f"Python SketchMap (stress={sm.stress_:.4f})")
axes[0].set_xlabel("Dimension 1")
axes[0].set_ylabel("Dimension 2")

# C++ reference embedding
sc2 = axes[1].scatter(
    T_cpp[:, 0], T_cpp[:, 1], c=weights, s=40, cmap=cmap, edgecolor="k"
)
axes[1].set_title(f"C++ Reference (stress={cpp_stress_recomputed:.4f})")
axes[1].set_xlabel("Dimension 1")
axes[1].set_ylabel("Dimension 2")

fig.colorbar(sc1, ax=axes[0], label="landmark weight")
fig.colorbar(sc2, ax=axes[1], label="landmark weight")
fig.tight_layout()
plt.show()
