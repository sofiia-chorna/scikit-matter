"""
Sketch-Map dimensionality reduction.

This module implements the Sketch-Map algorithm for nonlinear dimensionality
reduction, which uses sigmoid transformations to focus on intermediate-range
distances while ignoring very short (Gaussian fluctuations) and very long
(high-dimensional topology) distances.

References
----------
.. [1] Ceriotti, M., Tribello, G. A., & Parrinello, M. (2011).
       Simplifying the representation of complex free-energy landscapes
       using sketch-map. PNAS, 108(32), 13023-13028.
"""

import warnings

import numpy as np
from scipy import sparse
from scipy.optimize import basinhopping, minimize, curve_fit
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


# ================
# Helper functions
# ================


def sigmoid_transform(distances, sigma, a, b):
    """Apply the sketch-map sigmoid transformation to distances.

    The sigmoid function is: s(r) = 1 - (1 + (2^(a/b) - 1) * (r/sigma)^a)^(-b/a)

    This function maps:
    - r => 0  when r << sigma  (close points)
    - r => 1  when r >> sigma  (far points)
    - r = sigma  maps to 0.5  (switching distance)

    Parameters
    ----------
    distances : ndarray
        Pairwise distances to transform.
    sigma : float
        Switching distance where s(sigma) = 0.5.
    a : float
        Controls steepness at short range (how fast s→0 for r << sigma).
    b : float
        Controls steepness at long range (how fast s→1 for r >> sigma).

    Returns
    -------
    transformed : ndarray
        Sigmoid-transformed distances in [0, 1].
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        # Normalized distance
        r_normalized = distances / sigma

        # Sigmoid coefficient: A = 2^(a/b) - 1
        A = 2 ** (a / b) - 1.0

        # Compute sigmoid: s(r) = 1 - (1 + A * (r/sigma)^a)^(-b/a)
        term = A * (r_normalized ** a)
        transformed = 1.0 - (1.0 + term) ** (-b / a)

        # Handle edge case: r = 0 should map to s = 0
        transformed[distances <= 0.0] = 0.0

    return transformed


def sigmoid_inverse(y, sigma, a, b):
    """Inverse of the sigmoid transformation.

    Maps transformed distance y back to raw distance r.

    Parameters
    ----------
    y : ndarray
        Transformed distances in [0, 1].
    sigma : float
        Switching distance.
    a : float
        Short-range steepness parameter.
    b : float
        Long-range steepness parameter.

    Returns
    -------
    distances : ndarray
        Raw distances corresponding to transformed values.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        A = 2 ** (a / b) - 1.0

        # Inverse formula: r = sigma * (((1-y)^(-a/b) - 1) / A)^(1/a)
        one_minus_y = np.maximum(1.0 - y, 1e-12)
        inner = np.power(one_minus_y, -a / b) - 1.0
        distances = sigma * np.power(np.maximum(inner / A, 0.0), 1.0 / a)

        # Handle edge cases
        distances[y <= 0.0] = 0.0
        distances[y >= 1.0] = np.inf

    return distances


def sigmoid_derivative(distances, sigma, a, b):
    """Compute derivative of sigmoid with respect to distance.

    Used for gradient computation in optimization.

    Parameters
    ----------
    distances : ndarray
        Pairwise distances.
    sigma : float
        Switching distance.
    a : float
        Short-range steepness parameter.
    b : float
        Long-range steepness parameter.

    Returns
    -------
    derivative : ndarray
        ds/dr at each distance.
    """
    A = 2 ** (a / b) - 1.0
    derivative = np.zeros_like(distances, dtype=float)

    positive_mask = distances > 0.0
    if np.any(positive_mask):
        r = distances[positive_mask]
        u = A * (r / sigma) ** a
        prefactor = b * A * (r ** (a - 1.0)) / (sigma ** a)
        derivative[positive_mask] = prefactor * (1.0 + u) ** (-b / a - 1.0)

    return derivative


def classical_mds(distances, n_components):
    """Classical MDS initialization.

    Computes an initial embedding using eigendecomposition of the
    double-centered distance matrix.

    Parameters
    ----------
    distances : ndarray, shape (n_samples, n_samples)
        Pairwise distance matrix.
    n_components : int
        Number of dimensions for the embedding.

    Returns
    -------
    coordinates : ndarray, shape (n_samples, n_components)
        Initial embedding coordinates.
    """
    # Compute Gram matrix with double centering: B = -0.5 * H @ D^2 @ H
    # where H = I - 1/n * ones is the centering matrix
    gram_matrix = -0.5 * (distances ** 2)
    gram_matrix -= gram_matrix.mean(axis=0, keepdims=True)  # Center columns
    gram_matrix -= gram_matrix.mean(axis=1, keepdims=True)  # Center rows

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(gram_matrix)

    # Take top n_components (largest eigenvalues)
    top_indices = np.argsort(eigenvalues)[::-1][:n_components]
    top_eigenvalues = eigenvalues[top_indices]
    top_eigenvectors = eigenvectors[:, top_indices]

    # Compute coordinates
    coordinates = top_eigenvectors * np.sqrt(np.maximum(top_eigenvalues, 0.0))

    # Consistent sign convention
    for i in range(n_components):
        col = coordinates[:, i]
        if col[np.argmax(np.abs(col))] < 0:
            coordinates[:, i] *= -1

    return coordinates


def _gaussian(x, amplitude, center, std_dev):
    """Gaussian function for curve fitting."""
    return amplitude * np.exp(-((x - center) ** 2) / (2 * std_dev ** 2))


def analyze_distance_distribution(distances, n_bins=200):
    """Analyze the distribution of pairwise distances.

    This function computes a histogram of distances and identifies:
    1. Peak distance: where the bulk of pairwise distances lie
    2. Gaussian range: short-range fluctuations (to be ignored)
    3. Uniform cutoff: where high-D topology dominates (to be ignored)

    Parameters
    ----------
    distances : ndarray
        Pairwise distance matrix (square) or flattened upper triangle.
    n_bins : int, default=200
        Number of histogram bins.

    Returns
    -------
    analysis : dict
        Dictionary containing:
        - 'peak_distance': distance at histogram maximum
        - 'gaussian_std': estimated std of Gaussian fluctuations
        - 'gaussian_range': upper bound of Gaussian regime
        - 'uniform_cutoff': where density drops significantly
        - 'bin_centers': histogram bin centers
        - 'prob_density': probability density values
    """
    # Extract upper triangle if square matrix
    if distances.ndim == 2:
        d = distances[np.triu_indices_from(distances, k=1)]
    else:
        d = distances.copy()

    d = d[np.isfinite(d) & (d >= 0)]
    if d.size == 0:
        raise ValueError("Empty distances array")

    # Compute histogram
    max_distance = np.percentile(d, 99.9)
    bin_edges = np.linspace(0, max_distance, n_bins + 1)
    prob_density, _ = np.histogram(d, bins=bin_edges, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Find peak of distribution
    peak_idx = np.argmax(prob_density)
    peak_distance = bin_centers[peak_idx]

    # Initialize results
    analysis = {
        "peak_distance": peak_distance,
        "gaussian_std": None,
        "gaussian_range": None,
        "uniform_cutoff": None,
        "bin_centers": bin_centers,
        "prob_density": prob_density,
        "max_distance": max_distance,
    }

    # === Estimate Gaussian fluctuation range (left side of peak) ===
    left_mask = bin_centers <= peak_distance
    if np.sum(left_mask) > 3:
        try:
            initial_guess = [np.max(prob_density), peak_distance, 1.0]
            optimal_params, _ = curve_fit(
                _gaussian,
                bin_centers[left_mask],
                prob_density[left_mask],
                p0=initial_guess,
                maxfev=5000,
            )
            analysis["gaussian_std"] = abs(optimal_params[2])
            # Gaussian range: ~3 sigma from peak
            analysis["gaussian_range"] = peak_distance + 3 * analysis["gaussian_std"]
        except (RuntimeError, ValueError):
            # Fallback estimate
            analysis["gaussian_std"] = peak_distance / 3.0
            analysis["gaussian_range"] = peak_distance * 2.0

    # === Estimate uniform/high-D cutoff (right side of peak) ===
    right_mask = bin_centers > peak_distance
    right_density = prob_density[right_mask]

    if len(right_density) > 3:
        # Find where density drops to 10% of peak
        peak_density = prob_density[peak_idx]
        threshold = 0.1 * peak_density
        below_threshold = right_density < threshold

        if np.any(below_threshold):
            first_below = np.argmax(below_threshold)
            analysis["uniform_cutoff"] = bin_centers[right_mask][first_below]
        else:
            analysis["uniform_cutoff"] = np.percentile(d, 90)

    # Ensure gaussian_range < uniform_cutoff
    if analysis["gaussian_range"] is not None and analysis["uniform_cutoff"] is not None:
        if analysis["gaussian_range"] >= analysis["uniform_cutoff"]:
            analysis["gaussian_range"] = peak_distance + 0.2 * max_distance
            analysis["uniform_cutoff"] = peak_distance + 0.6 * max_distance

    return analysis


def suggest_sigmoid_params(distances, n_components, n_features, n_bins=200):
    """Suggest sigmoid parameters based on distance distribution analysis.

    Following sketchmap.org guidelines:
    - sigma: placed just before the peak (90% of peak distance)
    - a_hd, b_hd: control high-D sigmoid shape
    - a_ld, b_ld: typically 1-2 for low-D (dimension equalization)

    Parameters
    ----------
    distances : ndarray
        Pairwise distance matrix.
    n_components : int
        Target embedding dimensionality.
    n_features : int
        Original feature dimensionality.
    n_bins : int, default=200
        Number of histogram bins for analysis.

    Returns
    -------
    params : dict
        Suggested parameters: sigma, a_hd, b_hd, a_ld, b_ld
    analysis : dict
        Distance distribution analysis results.
    """
    # Analyze distance distribution
    analysis = analyze_distance_distribution(distances, n_bins=n_bins)

    # === Estimate sigma ===
    # Place sigma just before the peak (90% of peak distance)
    # This ensures bulk of distances are in the sigmoid's sensitive region
    sigma = 0.9 * analysis["peak_distance"]

    # === Estimate a_hd, b_hd (high-dimensional sigmoid) ===
    if analysis["gaussian_range"] is not None and analysis["uniform_cutoff"] is not None:
        # Sharper transition for wider informative range
        range_ratio = analysis["uniform_cutoff"] / max(analysis["gaussian_range"], 1e-10)
        a_hd = np.clip(2.0 + np.log(range_ratio), 2.0, 6.0)
        b_hd = np.clip(a_hd * 2, 4.0, 12.0)
    else:
        # defaults
        a_hd = 2.0
        b_hd = 6.0

    # === Estimate a_ld, b_ld (low-dimensional sigmoid) ===
    # Rule of thumb: a_ld/d ≈ a_hd/D for volume equalization
    if n_features > 0:
        a_ld = np.clip(a_hd * n_components / n_features, 1.0, 2.0)
    else:
        a_ld = 2.0
    b_ld = np.clip(a_ld, 1.0, 2.0)

    params = {
        "sigma": sigma,
        "a_hd": a_hd,
        "b_hd": b_hd,
        "a_ld": a_ld,
        "b_ld": b_ld,
    }

    return params, analysis


# =============================================================================
# Main SketchMap class
# =============================================================================


class SketchMap(TransformerMixin, BaseEstimator):
    """Sketch-Map dimensionality reduction.

    Sketch-Map transforms distances using sigmoid functions to focus on
    intermediate-range structure, ignoring both very short distances
    (Gaussian fluctuations) and very long distances (high-D topology effects).

    Parameters
    ----------
    n_components : int, default=2
        Dimensionality of the target embedding.

    params : dict or None, default=None
        Sigmoid parameters dictionary with keys:
        - 'sigma': switching distance where s(sigma) = 0.5
        - 'a_hd': high-D sigmoid short-range steepness
        - 'b_hd': high-D sigmoid long-range steepness
        - 'a_ld': low-D sigmoid short-range steepness
        - 'b_ld': low-D sigmoid long-range steepness

        If None, parameters are estimated automatically from data.
        Partial dicts are allowed; missing keys will be estimated.

        Example: params={"sigma": 7.0, "a_hd": 4.0, "b_hd": 2.0,
                         "a_ld": 2.0, "b_ld": 2.0}

    mds_opt_steps : int, default=100
        Number of MDS optimization steps (identity transform) before
        applying sigmoid. Set to 0 to skip.

    optimizer : str, default="L-BFGS-B"
        Optimization method: "L-BFGS-B" (fast) or "CG" (matches C++).

    preopt_steps : int, default=100
        Number of pre-optimization steps with sigmoid transform.

    max_iter : int, default=1000
        Number of main optimization steps with sigmoid transform.

    global_opt : int or None, default=None
        Number of basin-hopping iterations for global optimization.
        Set to None to disable.

    mixing_ratio : float, default=0.0
        Balance between direct distance stress (1.0) and transformed
        distance stress (0.0).

    center : bool, default=True
        If True, center input data before computing distances.

    init : array-like or None, default=None
        Initial embedding coordinates. If None, use classical MDS.

    random_state : int or None, default=None
        Random seed for reproducibility.

    verbose : bool, default=False
        If True, print progress information.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
        The low-dimensional embedding.

    stress_ : float
        Final stress value.

    params_ : dict
        The sigmoid parameters used (estimated or provided).

    suggested_params_ : dict
        Auto-suggested parameters (available after fit).

    distance_analysis_ : dict
        Distance distribution analysis results.

    Examples
    --------
    >>> from skmatter.decomposition import SketchMap
    >>> import numpy as np
    >>> X = np.random.randn(100, 50)
    >>> # Auto-estimate all parameters
    >>> sm = SketchMap(n_components=2)
    >>> embedding = sm.fit_transform(X)
    >>> # Or provide specific parameters
    >>> sm = SketchMap(
    ...     n_components=2,
    ...     params={"sigma": 7.0, "a_hd": 4.0, "b_hd": 2.0,
    ...             "a_ld": 2.0, "b_ld": 2.0}
    ... )
    >>> embedding = sm.fit_transform(X)
    """

    def __init__(
        self,
        n_components=2,
        params=None,
        mds_opt_steps=100,
        optimizer="L-BFGS-B",
        preopt_steps=100,
        max_iter=1000,
        global_opt=None,
        mixing_ratio=0.0,
        center=True,
        init=None,
        random_state=None,
        verbose=False,
    ):
        self.n_components = n_components
        self.params = params
        self.mds_opt_steps = mds_opt_steps
        self.optimizer = optimizer
        self.preopt_steps = preopt_steps
        self.max_iter = max_iter
        self.global_opt = global_opt
        self.mixing_ratio = mixing_ratio
        self.center = center
        self.init = init
        self.random_state = random_state
        self.verbose = verbose

    def _compute_stress(
        self, embedding, hd_distances, hd_transformed, weights, total_weight,
        mixing_ratio, use_transform=True
    ):
        """Compute the sketch-map stress function.

        Stress measures how well the low-D embedding preserves the
        (transformed) high-D distances.

        Parameters
        ----------
        embedding : ndarray, shape (n*d,) or (n, d)
            Low-dimensional coordinates.
        hd_distances : ndarray, shape (n, n)
            High-dimensional pairwise distances.
        hd_transformed : ndarray, shape (n, n)
            Sigmoid-transformed high-dimensional distances.
        weights : ndarray, shape (n, n)
            Pairwise weight matrix.
        total_weight : float
            Sum of upper-triangle weights.
        mixing_ratio : float
            Balance between direct (1.0) and transformed (0.0) stress.
        use_transform : bool
            Whether to apply sigmoid to low-D distances.

        Returns
        -------
        stress : float
            Normalized stress value.
        """
        # Reshape if flattened
        if embedding.ndim == 1:
            n_samples = hd_distances.shape[0]
            embedding = embedding.reshape((n_samples, self.n_components))

        # Compute low-dimensional distances
        ld_distances = squareform(pdist(embedding, metric="euclidean"))

        # Transform low-D distances if requested
        if use_transform:
            ld_transformed = sigmoid_transform(
                ld_distances,
                self.params_["sigma"],
                self.params_["a_ld"],
                self.params_["b_ld"],
            )
        else:
            ld_transformed = ld_distances

        # Compute stress on upper triangle only
        triu_idx = np.triu_indices_from(hd_distances, k=1)

        diff_transformed = (hd_transformed[triu_idx] - ld_transformed[triu_idx]) ** 2
        diff_direct = (hd_distances[triu_idx] - ld_distances[triu_idx]) ** 2
        weights_triu = weights[triu_idx]

        stress = np.sum(
            weights_triu * (
                (1.0 - mixing_ratio) * diff_transformed +
                mixing_ratio * diff_direct
            )
        )

        return stress / total_weight

    def _compute_gradient(
        self, embedding, hd_distances, hd_transformed, weights, total_weight,
        mixing_ratio, use_transform=True
    ):
        """Compute gradient of the stress function.

        Parameters
        ----------
        embedding : ndarray
            Low-dimensional coordinates (flat or matrix).
        hd_distances : ndarray
            High-dimensional distances.
        hd_transformed : ndarray
            Transformed high-dimensional distances.
        weights : ndarray
            Weight matrix.
        total_weight : float
            Total weight.
        mixing_ratio : float
            Mixing ratio.
        use_transform : bool
            Whether to use sigmoid transform.

        Returns
        -------
        gradient : ndarray
            Flattened gradient vector.
        """
        # Reshape if flattened
        if embedding.ndim == 1:
            n_samples = hd_distances.shape[0]
            embedding = embedding.reshape((n_samples, self.n_components))

        # Compute pairwise differences and distances
        diff_vectors = embedding[:, None, :] - embedding[None, :, :]
        ld_distances = np.sqrt(np.sum(diff_vectors ** 2, axis=2))

        # Transform and derivatives
        if use_transform:
            ld_transformed = sigmoid_transform(
                ld_distances,
                self.params_["sigma"],
                self.params_["a_ld"],
                self.params_["b_ld"],
            )
            ld_derivative = sigmoid_derivative(
                ld_distances,
                self.params_["sigma"],
                self.params_["a_ld"],
                self.params_["b_ld"],
            )
        else:
            ld_transformed = ld_distances
            ld_derivative = np.ones_like(ld_distances)

        # Compute gradient coefficients
        if mixing_ratio == 0.0:
            # Pure transformed stress
            coefficients = weights * (hd_transformed - ld_transformed) * ld_derivative
        else:
            # Mixed stress
            eps = 1e-100
            inv_distance = 1.0 / np.maximum(ld_distances, eps)
            coefficients = (
                weights * (
                    (1.0 - mixing_ratio) * (hd_transformed - ld_transformed) * ld_derivative +
                    mixing_ratio * (hd_distances - ld_distances)
                ) * inv_distance
            )

        np.fill_diagonal(coefficients, 0.0)

        # Compute gradient
        row_sums = coefficients.sum(axis=1)
        gradient = (row_sums[:, None] * embedding) - (coefficients @ embedding)
        gradient = -2.0 * gradient / total_weight

        return gradient.ravel()

    def _optimize(
        self, initial_embedding, hd_distances, hd_transformed, weights,
        total_weight, mixing_ratio, n_steps, use_transform=True
    ):
        """Run optimization to minimize stress.

        Parameters
        ----------
        initial_embedding : ndarray
            Starting coordinates.
        hd_distances : ndarray
            High-dimensional distances.
        hd_transformed : ndarray
            Transformed high-dimensional distances.
        weights : ndarray
            Weight matrix.
        total_weight : float
            Total weight.
        mixing_ratio : float
            Mixing ratio.
        n_steps : int
            Maximum iterations.
        use_transform : bool
            Whether to use sigmoid transform.

        Returns
        -------
        optimized_embedding : ndarray
            Optimized coordinates.
        final_stress : float
            Final stress value.
        """
        n_samples = initial_embedding.shape[0]
        x0 = initial_embedding.ravel()

        def objective(x):
            return self._compute_stress(
                x, hd_distances, hd_transformed, weights, total_weight,
                mixing_ratio, use_transform
            )

        def gradient(x):
            return self._compute_gradient(
                x, hd_distances, hd_transformed, weights, total_weight,
                mixing_ratio, use_transform
            )

        # Run optimization
        if self.optimizer == "CG":
            result = minimize(
                objective, x0, method="CG", jac=gradient,
                options={"maxiter": n_steps, "disp": False, "gtol": 1e-8}
            )
        else:
            result = minimize(
                objective, x0, method="L-BFGS-B", jac=gradient,
                options={"maxiter": n_steps, "disp": False, "gtol": 1e-8}
            )

        if self.verbose:
            print(f"  Optimization finished: stress = {result.fun:.6f}")

        optimized_embedding = result.x.reshape((n_samples, self.n_components))
        return optimized_embedding, result.fun

    def _global_optimize(
        self, embedding, hd_distances, hd_transformed, weights, total_weight,
        n_iterations
    ):
        """Global optimization using basin hopping.

        Parameters
        ----------
        embedding : ndarray
            Current embedding.
        hd_distances : ndarray
            High-dimensional distances.
        hd_transformed : ndarray
            Transformed distances.
        weights : ndarray
            Weight matrix.
        total_weight : float
            Total weight.
        n_iterations : int
            Number of basin hopping iterations.

        Returns
        -------
        optimized_embedding : ndarray
            Optimized coordinates.
        final_stress : float
            Final stress value.
        """
        n_samples = embedding.shape[0]
        x0 = embedding.ravel()

        def objective(x):
            return self._compute_stress(
                x, hd_distances, hd_transformed, weights, total_weight,
                self.mixing_ratio, use_transform=True
            )

        def gradient(x):
            return self._compute_gradient(
                x, hd_distances, hd_transformed, weights, total_weight,
                self.mixing_ratio, use_transform=True
            )

        minimizer_kwargs = {
            "method": self.optimizer,
            "jac": gradient,
            "options": {"maxiter": 100, "disp": False},
        }

        if self.verbose:
            print(f"\n=== Global optimization (basin hopping, {n_iterations} iter) ===")

        # Random displacement step
        rng = np.random.default_rng(self.random_state)

        class RandomDisplacement:
            def __init__(self, stepsize, rng):
                self.stepsize = stepsize
                self.rng = rng

            def __call__(self, x):
                return x + self.rng.uniform(-self.stepsize, self.stepsize, x.shape)

        scale = np.std(embedding)
        take_step = RandomDisplacement(stepsize=scale * 0.5, rng=rng)

        result = basinhopping(
            objective, x0, niter=n_iterations,
            minimizer_kwargs=minimizer_kwargs,
            take_step=take_step,
            seed=int(rng.integers(0, 2**31)) if self.random_state else None,
        )

        if self.verbose:
            print(f"  Basin hopping finished: stress = {result.fun:.6f}")

        return result.x.reshape((n_samples, self.n_components)), result.fun

    def fit(self, X, y=None, sample_weights=None):
        """Fit the Sketch-Map model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.
        sample_weights : array-like, shape (n_samples,), optional
            Per-sample weights.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Validate input
        X = validate_data(self, X, reset=True, dtype=np.float64)
        self.X_ = X.copy()

        if sparse.issparse(X):
            raise ValueError("Sparse input is not supported")
        if np.any(~np.isfinite(X)):
            raise ValueError("Input contains NaN or infinity")

        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError(
                f"Found array with {n_samples} sample(s) while minimum of 2 required."
            )

        self.n_samples_ = n_samples
        self.n_features_ = n_features

        if self.verbose:
            print(f"Fitting Sketch-Map: {n_samples} samples, {n_features} features")

        # -----------------------------------------------------------------
        # Step 1: Preprocess data and compute distances
        # -----------------------------------------------------------------
        X_centered = X.copy()
        if self.center:
            X_centered = X_centered - X_centered.mean(axis=0, keepdims=True)
            if self.verbose:
                print("Data centered")

        if self.verbose:
            print("Computing pairwise distances...")
        hd_distances = squareform(pdist(X_centered, metric="euclidean"))

        # -----------------------------------------------------------------
        # Step 2: Determine sigmoid parameters
        # -----------------------------------------------------------------
        # Always compute suggested parameters for reference
        suggested, analysis = suggest_sigmoid_params(
            hd_distances, self.n_components, n_features
        )
        self.suggested_params_ = suggested
        self.distance_analysis_ = analysis

        # Merge user-provided params with suggested params
        if self.params is None:
            # Use all suggested parameters
            self.params_ = suggested.copy()
            if self.verbose:
                print("Using auto-estimated sigmoid parameters:")
        else:
            # Start with suggested, override with user-provided
            self.params_ = suggested.copy()
            for key in ["sigma", "a_hd", "b_hd", "a_ld", "b_ld"]:
                if key in self.params and self.params[key] is not None:
                    self.params_[key] = self.params[key]
            if self.verbose:
                print("Using sigmoid parameters (user + auto-estimated):")

        if self.verbose:
            print(f"  sigma = {self.params_['sigma']:.4f}")
            print(f"  a_hd = {self.params_['a_hd']:.2f}, b_hd = {self.params_['b_hd']:.2f}")
            print(f"  a_ld = {self.params_['a_ld']:.2f}, b_ld = {self.params_['b_ld']:.2f}")
            print(f"  (peak distance = {analysis['peak_distance']:.4f})")

        # -----------------------------------------------------------------
        # Step 3: Transform high-dimensional distances
        # -----------------------------------------------------------------
        hd_transformed = sigmoid_transform(
            hd_distances,
            self.params_["sigma"],
            self.params_["a_hd"],
            self.params_["b_hd"],
        )

        # -----------------------------------------------------------------
        # Step 4: Setup weight matrix
        # -----------------------------------------------------------------
        if sample_weights is not None:
            w = np.asarray(sample_weights)
            if w.shape[0] != n_samples:
                raise ValueError(
                    f"sample_weights length {w.shape[0]} != n_samples {n_samples}"
                )
            weights = np.outer(w, w)
            total_weight = np.sum(np.triu(weights, k=1))
            if self.verbose:
                print(f"Using sample weights (total_weight = {total_weight:.4f})")
        else:
            weights = np.ones_like(hd_distances)
            total_weight = n_samples * (n_samples - 1) / 2.0
            if self.verbose:
                print(f"Using uniform weights (total_weight = {total_weight:.0f})")

        # -----------------------------------------------------------------
        # Step 5: Initialize embedding
        # -----------------------------------------------------------------
        if self.init is not None:
            embedding = np.asarray(self.init).copy()
            if self.verbose:
                print(f"Using provided initialization, shape = {embedding.shape}")
        else:
            if self.verbose:
                print("Initializing with classical MDS...")
            embedding = classical_mds(hd_distances, self.n_components)

        # -----------------------------------------------------------------
        # Step 6: MDS optimization (identity transform)
        # -----------------------------------------------------------------
        if self.mds_opt_steps > 0 and self.init is None:
            if self.verbose:
                print(f"\n=== Stage 0: MDS optimization ({self.mds_opt_steps} steps) ===")

            embedding, mds_stress = self._optimize(
                embedding, hd_distances, hd_distances,  # No transform
                weights, total_weight,
                mixing_ratio=1.0,  # Pure distance stress
                n_steps=self.mds_opt_steps,
                use_transform=False,
            )

            if self.verbose:
                print(f"MDS stress: {mds_stress:.6f}")

        # -----------------------------------------------------------------
        # Step 7: Pre-optimization (sigmoid transform)
        # -----------------------------------------------------------------
        stress = 0.0
        if self.preopt_steps > 0:
            if self.verbose:
                print(f"\n=== Stage 1: Pre-optimization ({self.preopt_steps} steps) ===")

            embedding, stress = self._optimize(
                embedding, hd_distances, hd_transformed,
                weights, total_weight,
                self.mixing_ratio, self.preopt_steps,
                use_transform=True,
            )

            if self.verbose:
                print(f"Pre-optimization stress: {stress:.6f}")

        # -----------------------------------------------------------------
        # Step 8: Global optimization (optional)
        # -----------------------------------------------------------------
        if self.global_opt is not None:
            if not isinstance(self.global_opt, int) or self.global_opt < 1:
                raise ValueError(f"global_opt must be positive int, got {self.global_opt}")

            embedding, stress = self._global_optimize(
                embedding, hd_distances, hd_transformed,
                weights, total_weight, self.global_opt
            )

        # -----------------------------------------------------------------
        # Step 9: Main optimization
        # -----------------------------------------------------------------
        if self.max_iter > 0:
            if self.verbose:
                print(f"\n=== Stage 2: Main optimization ({self.max_iter} steps) ===")

            embedding, stress = self._optimize(
                embedding, hd_distances, hd_transformed,
                weights, total_weight,
                self.mixing_ratio, self.max_iter,
                use_transform=True,
            )

            if self.verbose:
                print(f"Final stress: {stress:.6f}")

        self.embedding_ = embedding
        self.stress_ = stress

        if self.verbose:
            print("\nSketch-Map fitting complete!")

        return self

    def fit_transform(self, X, y=None, sample_weights=None):
        """Fit the model and return the embedding.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.
        sample_weights : array-like, shape (n_samples,), optional
            Per-sample weights.

        Returns
        -------
        embedding : ndarray, shape (n_samples, n_components)
            Low-dimensional embedding.
        """
        self.fit(X, y, sample_weights=sample_weights)
        return self.embedding_

    def transform(self, X):
        """Project data to the embedding space.

        Note: Only supports in-sample transformation (same data used for fit).
        Out-of-sample projection is not implemented.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        embedding : ndarray, shape (n_samples, n_components)
            Embedded coordinates.
        """
        check_is_fitted(self, ["embedding_", "X_"])
        X = validate_data(self, X, reset=False)

        # Match rows to training data
        indices = []
        for row in X:
            matches = np.all(np.isclose(self.X_, row, rtol=1e-8, atol=1e-12), axis=1)
            if not np.any(matches):
                warnings.warn(
                    "SketchMap.transform only supports in-sample rows. "
                    "Out-of-sample projection is not implemented.",
                    UserWarning,
                )
            indices.append(int(np.argmax(matches)))

        return self.embedding_[indices]

    def predict(self, X):
        """Alias for transform."""
        return self.transform(X)

    def score(self, X, y=None):
        """Return negative stress as score (higher is better).

        Parameters
        ----------
        X : array-like
            Ignored, present for API consistency.
        y : array-like
            Ignored, present for API consistency.

        Returns
        -------
        score : float
            Negative of final stress value.
        """
        check_is_fitted(self, ["stress_"])
        X = validate_data(self, X, reset=False)
        return -self.stress_
