import warnings

import numpy as np
from scipy import sparse
from scipy.optimize import basinhopping, minimize, curve_fit
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class SketchMap(TransformerMixin, BaseEstimator):
    """Sketch-Map dimensionality reducer.

    Sketch-Map transforms distances using sigmoid functions to focus on
    intermediate-range structure, ignoring both very short distances
    (Gaussian fluctuations) and very long distances (high-D topology effects).

    Parameters
    ----------
    n_components : int, default=2
        Dimensionality of the target embedding.

    sigma : float, default=None
        Sigmoid switching distance. Points closer than sigma are considered
        "close", points farther are "far". If None and auto_params=True,
        estimated from pairwise distance histogram.

    a_hd : float, default=None
        High-dimensional sigmoid exponent a (controls short-range steepness).
        If None and auto_params=True, estimated from data.

    b_hd : float, default=None
        High-dimensional sigmoid exponent b (controls long-range steepness).
        If None and auto_params=True, estimated from data.

    a_ld : float, default=None
        Low-dimensional sigmoid exponent a. Typically 1-2 for better
        accommodation of high-D quirks. If None and auto_params=True,
        estimated based on dimension ratio.

    b_ld : float, default=None
        Low-dimensional sigmoid exponent b. Typically 1-2.
        If None and auto_params=True, estimated from data.

    auto_params : bool, default=True
        If True, automatically estimate sigma, a, and b parameters from
        the pairwise distance histogram following sketchmap.org guidelines.

    mds_opt_steps : int, default=100
        Number of optimization steps with identity transform after classical MDS.
        This matches the C++ iterative MDS step that optimizes raw distance
        stress before applying sigmoid transforms. Set to 0 to skip.

    optimizer : str, default="L-BFGS-B"
        Optimization method. Options:
        - "L-BFGS-B": Limited-memory BFGS with bounds (default, fast)
        - "CG": Conjugate Gradient using Polak-Ribière formula
          (matches C++ implementation more closely)

    preopt_steps : int, default=100
        Number of pre-optimization steps with sigmoid transform.

    opt_steps : int, default=100
        Number of main optimization steps with sigmoid transform.

    global_opt : int or None, default=None
        Number of basin-hopping iterations for global optimization.
        If provided, uses scipy's basin-hopping algorithm to escape local
        minima by combining local optimization with random perturbations.
        Typical values: 5-20 iterations. Set to None to disable.

    mixing_ratio : float, default=0.0
        Balance between direct (1.0) and transformed (0.0) stress.

    center : bool, default=True
        If True, center the input data around origin.

    init : array-like or None, default=None
        Initial embedding coordinates. If provided, skip MDS initialization
        and start optimization from this embedding. Useful for reproducing
        C++ results by using the same initialization.

    random_state : int or None
        Random seed.

    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
        The low-dimensional embedding.

    stress_ : float
        Final stress value.

    sigma_ : float
        The sigma parameter used (estimated or provided).

    a_hd_, b_hd_, a_ld_, b_ld_ : float
        The sigmoid parameters used (estimated or provided).

    gaussian_range_ : float
        Estimated range of Gaussian fluctuations (short distances to ignore).

    uniform_cutoff_ : float
        Estimated cutoff where high-D topology dominates (long distances to ignore).
    """

    def __init__(
        self,
        n_components=2,
        sigma=None,
        a_hd=None,
        b_hd=None,
        a_ld=None,
        b_ld=None,
        auto_params=True,
        mds_opt_steps=100,
        optimizer="L-BFGS-B",
        preopt_steps=100,
        opt_steps=100,
        global_opt=None,
        mixing_ratio=0.0,
        center=True,
        init=None,
        random_state=None,
        verbose=False,
    ):
        self.n_components = n_components
        self.sigma = sigma
        self.a_hd = a_hd
        self.b_hd = b_hd
        self.a_ld = a_ld
        self.b_ld = b_ld
        self.auto_params = auto_params
        self.mds_opt_steps = mds_opt_steps
        self.optimizer = optimizer
        self.preopt_steps = preopt_steps
        self.opt_steps = opt_steps
        self.global_opt = global_opt
        self.mixing_ratio = mixing_ratio
        self.center = center
        self.init = init
        self.random_state = random_state
        self.verbose = verbose
        self._first_stress_call = True

    def _sigmoid_transform(self, distances, sigma, a, b):
        """Apply xsigmoid transformation: 1-(1+(2^(a/b)-1)(x/s)^a)^(-b/a)"""
        with np.errstate(divide="ignore", invalid="ignore"):
            x = distances / sigma
            A = 2 ** (a / b) - 1.0
            term = A * (x**a)
            val = 1.0 - (1.0 + term) ** (-b / a)
            val[distances <= 0.0] = 0.0
        return val

    def _sigmoid_inverse(self, y, sigma, a, b):
        """Inverse of sigmoid: maps transformed distance back to raw distance.

        Given y = 1-(1+(2^(a/b)-1)(x/s)^a)^(-b/a)
        Return x = s * (((1-y)^(-a/b) - 1) / (2^(a/b)-1))^(1/a)
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            A = 2 ** (a / b) - 1.0
            # (1-y)^(-a/b) = (1+term)
            # term = (1-y)^(-a/b) - 1
            # A * (x/s)^a = term
            # x = s * (term/A)^(1/a)
            one_minus_y = np.maximum(1.0 - y, 1e-12)  # Avoid division by zero
            inner = np.power(one_minus_y, -a / b) - 1.0
            val = sigma * np.power(np.maximum(inner / A, 0.0), 1.0 / a)
            val[y <= 0.0] = 0.0
            val[y >= 1.0] = np.inf
        return val

    def _warp_distances(self, D_hd):
        """Warp HD distances to target LD distances: f_ld^{-1}(f_hd(D_hd)).

        This is used for MDS initialization (like C++ -warp flag).
        """
        # Apply HD sigmoid
        S_hd = self._sigmoid_transform(D_hd, self.sigma_, self.a_hd_, self.b_hd_)
        # Apply inverse LD sigmoid to get target LD distances
        D_ld_target = self._sigmoid_inverse(S_hd, self.sigma_, self.a_ld_, self.b_ld_)
        return D_ld_target

    def _sigmoid_derivative(self, distances, sigma, a, b):
        """Derivative of sigmoid with respect to distance."""
        A = 2 ** (a / b) - 1.0
        out = np.zeros_like(distances, dtype=float)
        pos = distances > 0.0

        if np.any(pos):
            r = distances[pos]
            u = A * (r / sigma) ** a
            pref = b * A * (r ** (a - 1.0)) / (sigma**a)
            out[pos] = pref * (1.0 + u) ** (-b / a - 1.0)

        return out

    def _classical_mds(self, distances):
        """Classical MDS initialization matching C++ implementation."""
        D = distances

        # Compute Gram matrix with centering as in C++
        M = -0.5 * (D**2)
        M -= M.mean(axis=0, keepdims=True)  # subtract column means
        M -= M.mean(axis=1, keepdims=True)  # subtract row means

        # Eigendecomposition
        w, V = np.linalg.eigh(M)

        # Sort descending and take top n_components
        idx = np.argsort(w)[::-1]
        top_idx = idx[: self.n_components]

        top_eigenvalues = w[top_idx]
        top_eigenvectors = V[:, top_idx]

        # Compute coordinates
        coordinates = top_eigenvectors * np.sqrt(np.maximum(top_eigenvalues, 0.0))

        # Align signs to match C++ convention: make max abs value positive
        for i in range(self.n_components):
            col = coordinates[:, i]
            max_id = np.argmax(np.abs(col))
            if col[max_id] < 0:
                coordinates[:, i] *= -1

        return coordinates

    def _gaussian_func(self, x, amplitude, center, std_dev):
        """Gaussian function for curve fitting."""
        return amplitude * np.exp(-((x - center) ** 2) / (2 * std_dev**2))

    def _estimate_sigma(self, distances, n_bins=200):
        """Estimate sigma from pairwise distance histogram.

        Following sketchmap.org guidelines:
        - sigma should be placed just before the peak of the distance histogram
        - This ensures that the bulk of pairwise distances (around the peak)
          are mapped to ~0.5 or higher, preserving their relative ordering
        - Distances much smaller than sigma → 0 ("close")
        - Distances much larger than sigma → 1 ("far")

        Parameters
        ----------
        distances : ndarray, shape (n_samples, n_samples)
            Pairwise distance matrix.
        n_bins : int, default=200
            Number of bins for histogram.

        Returns
        -------
        sigma : float
            Estimated sigma parameter.
        """
        # Analyze distance distribution
        self._analyze_distance(distances, n_bins=n_bins)

        # Sigma should be placed just before the peak of the distribution
        # This way, the bulk of distances (around and after the peak) are
        # in the "transition" region where sigmoid is most sensitive
        if self.peak_distance_ is not None:
            # Place sigma slightly before the peak (e.g., 90% of peak distance)
            # This ensures most distances are mapped above 0.5
            sigma = 0.9 * self.peak_distance_
        else:
            # Fall back to median distance
            d = distances[np.triu_indices_from(distances, k=1)]
            sigma = 0.9 * np.median(d[np.isfinite(d) & (d > 0)])

        return sigma

    def _estimate_a_b_params(self, n_features_ld=None):
        """Estimate a and b parameters for high and low dimensional sigmoids.

        Following sketchmap.org guidelines:
        - a_hd, b_hd: control how fast sigmoid goes to 0 and 1 in high-D
        - b_hd should ensure s(x)→1 by the time n(R) becomes very small
        - a_ld, b_ld: typically smaller (1-2) to accommodate high-D quirks
        - For dimension equalization: a_ld/d ≈ a_hd/D

        Parameters
        ----------
        n_features_ld : int or None
            Number of low-dimensional components. If None, uses n_components.

        Returns
        -------
        params : dict
            Dictionary with 'a_hd', 'b_hd', 'a_ld', 'b_ld'.
        """
        if n_features_ld is None:
            n_features_ld = self.n_components

        n_features_hd = getattr(self, "n_features_in_", None)
        if n_features_hd is None:
            # Default values if dimensionality unknown
            return {"a_hd": 2.0, "b_hd": 6.0, "a_ld": 2.0, "b_ld": 2.0}

        # High-dimensional parameters
        # a_hd controls steepness at short range (going to 0)
        # b_hd controls steepness at long range (going to 1)
        # b_hd should be large enough that s(x)→1 when n(R) is small

        # Estimate based on the ratio of ranges
        if (
            hasattr(self, "gaussian_range_")
            and hasattr(self, "uniform_cutoff_")
            and self.gaussian_range_ is not None
            and self.uniform_cutoff_ is not None
        ):
            # The wider the gap between Gaussian range and cutoff,
            # the sharper we can make the transition
            range_ratio = self.uniform_cutoff_ / max(self.gaussian_range_, 1e-10)
            # Typical values: a_hd=2-4, b_hd=4-8
            a_hd = np.clip(2.0 + np.log(range_ratio), 2.0, 6.0)
            b_hd = np.clip(a_hd * 2, 4.0, 12.0)
        else:
            # Default conservative values
            a_hd = 2.0
            b_hd = 6.0

        # Low-dimensional parameters
        # Rule of thumb from sketchmap.org: a_ld/d ≈ a_hd/D for volume equalization
        # Typical good starting values: a_ld=1-2, b_ld=1-2
        if n_features_hd > 0:
            a_ld = np.clip(a_hd * n_features_ld / n_features_hd, 1.0, 2.0)
        else:
            a_ld = 2.0
        # b_ld is typically smaller than b_hd
        b_ld = np.clip(a_ld, 1.0, 2.0)

        return {"a_hd": a_hd, "b_hd": b_hd, "a_ld": a_ld, "b_ld": b_ld}

    def _analyze_distance(self, distances, n_bins=200, max_distance=None):
        """Analyze pairwise distance distribution.

        Following sketchmap.org analysis:
        1. Histogram shows low probability at very short distances (high-D effect)
        2. Peak/shoulder at Gaussian fluctuation scale (~std*sqrt(D/d))
        3. Long-range part dominated by high-D topology (similar to uniform)

        Parameters
        ----------
        distances : ndarray, shape (n_samples, n_samples)
            Pairwise distance matrix.
        n_bins : int, default=200
            Number of bins for histogram.
        max_distance : float or None, default=None
            Maximum distance for histogram. If None, uses 99.9th percentile.
        """
        # Extract upper triangle (pairwise distances without diagonal)
        d = distances[np.triu_indices_from(distances, k=1)]
        d = d[np.isfinite(d) & (d >= 0)]

        if d.size == 0:
            raise ValueError("Empty distances array")

        self.distances_ = d

        if max_distance is None:
            max_distance = np.percentile(d, 99.9)
        self.max_distance_ = max_distance

        # Create histogram (probability density n(R))
        self.bin_edges_ = np.linspace(0, self.max_distance_, n_bins + 1)
        bin_counts, _ = np.histogram(d, bins=self.bin_edges_, density=True)

        self.prob_density_ = bin_counts

        right_edges = self.bin_edges_[1:]
        left_edges = self.bin_edges_[:-1]
        self.bin_centers_ = 0.5 * (left_edges + right_edges)

        # Find the peak of the distribution
        peak_id = np.argmax(bin_counts)
        self.peak_distance_ = self.bin_centers_[peak_id]

        # Initialize attributes
        self.gaussian_std_ = None
        self.gaussian_range_ = None
        self.uniform_cutoff_ = None

        # === Estimate Gaussian fluctuation range ===
        # The left side of the peak (short distances) reflects Gaussian correlations
        # Fit a Gaussian to estimate the standard deviation
        left_mask = self.bin_centers_ <= self.peak_distance_
        if np.sum(left_mask) > 3:
            try:
                initial_guess = [np.max(bin_counts), self.peak_distance_, 1.0]
                optimal_params, _ = curve_fit(
                    self._gaussian_func,
                    self.bin_centers_[left_mask],
                    bin_counts[left_mask],
                    p0=initial_guess,
                    maxfev=5000,
                )
                _amplitude, _center, std_dev = optimal_params
                self.gaussian_std_ = abs(std_dev)
                # Gaussian range: ~3 std deviations from peak encompasses most
                # of the Gaussian fluctuations (these distances are "too close")
                self.gaussian_range_ = self.peak_distance_ + 3 * self.gaussian_std_
            except (RuntimeError, ValueError):
                # Curve fitting failed, estimate from peak position
                self.gaussian_std_ = self.peak_distance_ / 3.0
                self.gaussian_range_ = self.peak_distance_ * 2.0

        # === Estimate uniform/high-D dominated cutoff ===
        # The right tail of the distribution becomes dominated by high-D topology
        # Find where n(R) drops significantly (these distances are "too far")
        right_mask = self.bin_centers_ > self.peak_distance_
        right_bins = self.bin_centers_[right_mask]
        right_counts = bin_counts[right_mask]

        if len(right_counts) > 3:
            # Find where the density drops to a small fraction of the peak
            peak_density = bin_counts[peak_id]
            threshold = 0.1 * peak_density  # 10% of peak density

            # Find the distance where density drops below threshold
            below_threshold = right_counts < threshold
            if np.any(below_threshold):
                first_below = np.argmax(below_threshold)
                self.uniform_cutoff_ = right_bins[first_below]
            else:
                # If never drops below, use the 90th percentile of distances
                self.uniform_cutoff_ = np.percentile(d, 90)

        # Ensure gaussian_range < uniform_cutoff
        if self.gaussian_range_ is not None and self.uniform_cutoff_ is not None:
            if self.gaussian_range_ >= self.uniform_cutoff_:
                # Adjust: place gaussian_range at 1/3 and cutoff at 2/3 of range
                range_span = self.max_distance_
                self.gaussian_range_ = self.peak_distance_ + 0.2 * range_span
                self.uniform_cutoff_ = self.peak_distance_ + 0.6 * range_span


    def _compute_stress(
        self, X_ld, D_hd, S_hd, W, tw, mixing_ratio, use_transform=True
    ):
        """Compute sketch-map stress function (C++ compatible).

        Parameters
        ----------
        X_ld : array, shape (n*d,) or (n, d)
            Low-dimensional coordinates (flattened or matrix)
        D_hd : array, shape (n, n)
            High-dimensional distances
        S_hd : array, shape (n, n)
            Transformed high-dimensional distances
        W : array, shape (n, n)
            Weight matrix (pair weights w_i * w_j)
        tw : float
            Total weight (sum of upper-triangle weights)
        mixing_ratio : float
            Balance between direct (1.0) and transformed (0.0) stress
        use_transform : bool
            Whether to apply sigmoid transform to low-dim distances
        """
        # Handle both flat and matrix inputs
        if X_ld.ndim == 1:
            n = D_hd.shape[0]
            X_ld_mat = X_ld.reshape((n, self.n_components))
        else:
            X_ld_mat = X_ld

        # Low-dimensional distances
        D_ld = squareform(pdist(X_ld_mat, metric="euclidean"))

        # Transform low-dimensional distances if requested
        if use_transform:
            S_ld = self._sigmoid_transform(D_ld, self.sigma_, self.a_ld_, self.b_ld_)
        else:
            S_ld = D_ld

        # Compute stress using upper triangle only (like C++)
        # C++ formula: sum_{i>j} [(S_hd-S_ld)^2*(1-mix) + mix*(D_hd-D_ld)^2] * W
        triu_idx = np.triu_indices_from(D_hd, k=1)
        diff_transformed = (S_hd[triu_idx] - S_ld[triu_idx]) ** 2
        diff_direct = (D_hd[triu_idx] - D_ld[triu_idx]) ** 2
        weights_triu = W[triu_idx]

        combined_stress = np.sum(
            weights_triu
            * ((1.0 - mixing_ratio) * diff_transformed + mixing_ratio * diff_direct)
        )

        # Normalize by total weight
        return combined_stress / tw

    def _compute_gradient(
        self, X_ld, D_hd, S_hd, W, tw, mixing_ratio, use_transform=True
    ):
        """Compute gradient of stress function (C++ compatible).

        For imix=0 (pure transformed stress):
            gij = (S_hd - S_ld) * dS_dD
            grad[i] += gij * (x_i - x_j)  [no division by D_ld]

        For imix>0 (mixed stress):
            gij = ((S_hd - S_ld)*dS_dD*(1-mix) + mix*(D_hd - D_ld)) / D_ld
            grad[i] += gij * (x_i - x_j)

        Final: grad *= -2.0 / tw
        """
        # Handle both flat and matrix inputs
        if X_ld.ndim == 1:
            n = D_hd.shape[0]
            X_ld_mat = X_ld.reshape((n, self.n_components))
        else:
            X_ld_mat = X_ld

        # Pairwise differences
        dif = X_ld_mat[:, None, :] - X_ld_mat[None, :, :]
        D_ld = np.sqrt(np.sum(dif**2, axis=2))

        # Transform and derivatives
        if use_transform:
            S_ld = self._sigmoid_transform(D_ld, self.sigma_, self.a_ld_, self.b_ld_)
            dS_dD = self._sigmoid_derivative(D_ld, self.sigma_, self.a_ld_, self.b_ld_)
        else:
            S_ld = D_ld
            dS_dD = np.ones_like(D_ld)

        # C++ gradient formula depends on mixing_ratio:
        # For imix=0: gij = (S_hd - S_ld) * dS_dD  (no /D_ld)
        # For imix>0: gij = ((S_hd - S_ld)*dS_dD*(1-imix) + imix*(D_hd - D_ld))/D_ld
        if mixing_ratio == 0.0:
            M = W * (S_hd - S_ld) * dS_dD
        else:
            eps = 1e-100
            inv_D = 1.0 / np.maximum(D_ld, eps)
            M = (
                W
                * (
                    (1.0 - mixing_ratio) * (S_hd - S_ld) * dS_dD
                    + mixing_ratio * (D_hd - D_ld)
                )
                * inv_D
            )
        np.fill_diagonal(M, 0.0)

        row_sums = M.sum(axis=1)
        G = (row_sums[:, None] * X_ld_mat) - (M @ X_ld_mat)

        # C++ normalization: *= -2.0/tw
        G = -2.0 * G / tw

        return G.ravel()

    def _optimize(
        self, X_init, D_hd, S_hd, W, tw, mixing_ratio, n_steps, use_transform=True
    ):
        """Run optimization using the configured optimizer.

        Parameters
        ----------
        X_init : ndarray
            Initial embedding coordinates.
        D_hd : ndarray
            High-dimensional pairwise distances.
        S_hd : ndarray
            Transformed high-dimensional distances.
        W : ndarray
            Weight matrix.
        tw : float
            Total weight (sum of upper triangle of W).
        mixing_ratio : float
            Balance between direct and transformed stress.
        n_steps : int
            Maximum number of optimization iterations.
        use_transform : bool
            Whether to apply sigmoid transform to low-dimensional distances.

        Returns
        -------
        X_opt : ndarray
            Optimized embedding coordinates.
        stress : float
            Final stress value.
        """
        n = X_init.shape[0]
        x0 = X_init.ravel()

        def objective(x):
            return self._compute_stress(
                x, D_hd, S_hd, W, tw, mixing_ratio, use_transform
            )

        def gradient(x):
            return self._compute_gradient(
                x, D_hd, S_hd, W, tw, mixing_ratio, use_transform
            )

        # Choose optimizer
        if self.optimizer == "CG":
            # Conjugate Gradient with Polak-Ribière (matches C++ implementation)
            result = minimize(
                objective,
                x0,
                method="CG",
                jac=gradient,
                options={"maxiter": n_steps, "disp": False, "gtol": 1e-8},
            )
        else:
            # L-BFGS-B (default, faster but may converge to different minimum)
            result = minimize(
                objective,
                x0,
                method="L-BFGS-B",
                jac=gradient,
                options={"maxiter": n_steps, "disp": False, "gtol": 1e-8},
            )

        if self.verbose:
            print(f"  Optimization finished: stress = {result.fun:.6f}")

        return result.x.reshape((n, self.n_components)), result.fun

    def _global_optimize_basinhopping(
        self, X_ld, D_hd, S_hd, W, tw, n_iterations
    ):
        """Perform global optimization using basin hopping.

        Basin hopping combines local optimization with random perturbations
        to escape local minima.

        Parameters
        ----------
        X_ld : ndarray, shape (n_samples, n_components)
            Current embedding.
        D_hd : ndarray, shape (n_samples, n_samples)
            High-dimensional distances.
        S_hd : ndarray, shape (n_samples, n_samples)
            Transformed high-dimensional distances.
        W : ndarray, shape (n_samples, n_samples)
            Weight matrix.
        tw : float
            Total weight.
        n_iterations : int
            Number of basin hopping iterations.

        Returns
        -------
        X_ld : ndarray
            Optimized embedding.
        stress : float
            Final stress value.
        """
        n = X_ld.shape[0]
        x0 = X_ld.ravel()

        def objective(x):
            return self._compute_stress(
                x, D_hd, S_hd, W, tw, self.mixing_ratio, use_transform=True
            )

        def gradient(x):
            return self._compute_gradient(
                x, D_hd, S_hd, W, tw, self.mixing_ratio, use_transform=True
            )

        # Local minimizer options
        minimizer_kwargs = {
            "method": self.optimizer,
            "jac": gradient,
            "options": {"maxiter": 100, "disp": False},
        }

        if self.verbose:
            print(f"\n=== Global optimization (basin hopping, {n_iterations} iter) ===")

        # Set random state for reproducibility
        rng = np.random.default_rng(self.random_state)

        # Custom step function with controlled step size
        class RandomDisplacement:
            def __init__(self, stepsize=1.0, rng=None):
                self.stepsize = stepsize
                self.rng = rng if rng is not None else np.random.default_rng()

            def __call__(self, x):
                return x + self.rng.uniform(-self.stepsize, self.stepsize, x.shape)

        # Estimate step size from current embedding scale
        scale = np.std(X_ld)
        take_step = RandomDisplacement(stepsize=scale * 0.5, rng=rng)

        result = basinhopping(
            objective,
            x0,
            niter=n_iterations,
            minimizer_kwargs=minimizer_kwargs,
            take_step=take_step,
            seed=int(rng.integers(0, 2**31)) if self.random_state else None,
        )

        if self.verbose:
            print(f"  Basin hopping finished: stress = {result.fun:.6f}")

        return result.x.reshape((n, self.n_components)), result.fun

    def fit(self, X, y=None, sample_weights=None):
        """Fit the Sketch-Map model following PyTorch two-stage approach.

        Stage 1: Pre-optimization with identity transform
        Stage 2: Refinement with sigmoid transform

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
        X = validate_data(self, X, reset=True, dtype=np.float64)
        self.X_ = X.copy()

        if sparse.issparse(X):
            raise ValueError("Sparse input is not supported")
        if np.any(~np.isfinite(X)):
            raise ValueError("Input contains NaN or infinity")

        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError(
                f"Found array with {n_samples} sample(s) (shape={X.shape}) while a "
                "minimum of 2 is required."
            )
        self.n_samples_ = n_samples
        self.n_features_ = n_features

        if self.verbose:
            print(f"Fitting Sketch-Map with {n_samples} samples, {n_features} features")
            if sample_weights is not None:
                print(
                    f"Using sample weights: min={np.min(sample_weights):.4f}, "
                    f"max={np.max(sample_weights):.4f}, "
                    f"mean={np.mean(sample_weights):.4f}"
                )

        X_work = X.copy()
        if self.center:
            if self.verbose:
                print("Centering the data")
            X_work = X_work - X_work.mean(axis=0, keepdims=True)

        # Compute pairwise distances
        if self.verbose:
            print("Computing pairwise distances...")
        D_hd = squareform(pdist(X_work, metric="euclidean"))

        # Estimate or set sigmoid parameters (sigma, a, b)
        # Following sketchmap.org guidelines for parameter selection
        if self.auto_params and self.sigma is None:
            self.sigma_ = self._estimate_sigma(D_hd)
            if self.verbose:
                print(f"Estimated sigma: {self.sigma_:.4f}")
                if hasattr(self, "gaussian_range_") and self.gaussian_range_ is not None:
                    print(f"  Gaussian range (short-range cutoff): {self.gaussian_range_:.4f}")
                if hasattr(self, "uniform_cutoff_") and self.uniform_cutoff_ is not None:
                    print(f"  Uniform cutoff (long-range cutoff): {self.uniform_cutoff_:.4f}")
        else:
            self.sigma_ = self.sigma if self.sigma is not None else 1.0

        # Estimate or set a/b parameters
        if self.auto_params:
            auto_ab = self._estimate_a_b_params()
            self.a_hd_ = self.a_hd if self.a_hd is not None else auto_ab["a_hd"]
            self.b_hd_ = self.b_hd if self.b_hd is not None else auto_ab["b_hd"]
            self.a_ld_ = self.a_ld if self.a_ld is not None else auto_ab["a_ld"]
            self.b_ld_ = self.b_ld if self.b_ld is not None else auto_ab["b_ld"]
            if self.verbose:
                print(f"Sigmoid parameters: a_hd={self.a_hd_:.2f}, b_hd={self.b_hd_:.2f}, "
                      f"a_ld={self.a_ld_:.2f}, b_ld={self.b_ld_:.2f}")
        else:
            # Use provided values or defaults
            self.a_hd_ = self.a_hd if self.a_hd is not None else 2.0
            self.b_hd_ = self.b_hd if self.b_hd is not None else 6.0
            self.a_ld_ = self.a_ld if self.a_ld is not None else 2.0
            self.b_ld_ = self.b_ld if self.b_ld is not None else 6.0

        # Transform high-dimensional distances (for stage 2)
        S_hd = self._sigmoid_transform(D_hd, self.sigma_, self.a_hd_, self.b_hd_)

        # Setup weight matrix: W_ij = w_i * w_j (like C++)
        if sample_weights is not None:
            w = np.asarray(sample_weights)
            if w.shape[0] != n_samples:
                raise ValueError(
                    f"sample_weights must have length {n_samples}, got {w.shape[0]}"
                )
            W = np.outer(w, w)
            # Total weight is sum of upper triangle (like C++)
            tw = np.sum(np.triu(W, k=1))
            if self.verbose:
                print(f"Weight matrix: tw={tw:.4f}, shape={W.shape}")
        else:
            W = np.ones_like(D_hd)
            # For unweighted case, tw = n*(n-1)/2 (like C++)
            tw = n_samples * (n_samples - 1) / 2.0
            if self.verbose:
                print(f"Using uniform weights, tw={tw:.4f}")

        # Reset debug flag
        self._first_stress_call = True

        # Initialize embedding
        if self.init is not None:
            # Use provided initialization (e.g., from C++ lowd.imds)
            X_ld = np.asarray(self.init).copy()
            if self.verbose:
                print(f"Using provided initialization, shape={X_ld.shape}")
        else:
            # Initialize with classical MDS on raw distances
            if self.verbose:
                print("Initializing with classical MDS on raw distances...")
            X_ld = self._classical_mds(D_hd)

        # Stage 0: MDS optimization with identity transform (like C++ lowd.imds)
        # This matches C++ iterative MDS that optimizes raw distance stress
        if self.mds_opt_steps > 0 and self.init is None:
            if self.verbose:
                print("\n=== Stage 0: MDS optimization (identity transform) ===")
                print(f"Running {self.mds_opt_steps} optimization steps...")

            X_ld, mds_stress = self._optimize(
                X_ld,
                D_hd,
                D_hd,  # Use raw distances (identity transform)
                W,
                tw,
                mixing_ratio=1.0,  # 100% direct distance stress
                n_steps=self.mds_opt_steps,
                use_transform=False,  # No sigmoid transform
            )

            if self.verbose:
                print(f"MDS optimization stress: {mds_stress:.6f}")

        # Stage 1: Pre-optimization with sigmoid transform
        if self.preopt_steps > 0:
            if self.verbose:
                print("\n=== Stage 1: Pre-optimization (sigmoid transform) ===")
                print(f"Running {self.preopt_steps} optimization steps...")

            X_ld, stress = self._optimize(
                X_ld,
                D_hd,
                S_hd,  # Use sigmoid-transformed HD distances
                W,
                tw,
                self.mixing_ratio,
                self.preopt_steps,
                use_transform=True,  # Use sigmoid transform
            )

            if self.verbose:
                print(f"Pre-optimization stress: {stress:.6f}")

        # Global optimization (basin hopping)
        if self.global_opt is not None:
            if not isinstance(self.global_opt, int) or self.global_opt < 1:
                raise ValueError(
                    f"global_opt must be a positive integer (number of iterations), "
                    f"got {self.global_opt}"
                )

            X_ld, stress = self._global_optimize_basinhopping(
                X_ld, D_hd, S_hd, W, tw, self.global_opt
            )

        # Stage 2: Main optimization (continuation)
        if self.opt_steps > 0:
            if self.verbose:
                print("\n=== Stage 2: Main optimization (sigmoid transform) ===")
                print(f"Running {self.opt_steps} optimization steps...")

            # Reset debug flag for second stage
            self._first_stress_call = True

            X_ld, stress = self._optimize(
                X_ld,
                D_hd,
                S_hd,
                W,
                tw,
                self.mixing_ratio,
                self.opt_steps,
                use_transform=True,
            )

            if self.verbose:
                print(f"Final stress: {stress:.6f}")

        self.embedding_ = X_ld
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
        X_new : array, shape (n_samples, n_components)
            Embedded coordinates.
        """
        self.fit(X, y, sample_weights=sample_weights)
        return self.embedding_

    def transform(self, X):
        """Project data to the embedding space.

        Only supports in-sample transformation (same data used for fit).
        """
        check_is_fitted(self, ["embedding_", "X_"])
        X = validate_data(self, X, reset=False)

        # Allow in-sample transformation (full or subset) by matching rows.
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
        """Project data to the embedding space (alias for transform).

        Currently only supports in-sample transformation (same data used for fit).
        Out-of-sample projection is not implemented.
        """
        return self.transform(X)

    def score(self, X, y=None):
        """Return the negative stress as a score.

        Parameters
        ----------
        X : array-like
            Ignored, present for API consistency.
        y : array-like
            Ignored, present for API consistency.

        Returns
        -------
        score : float
            Negative of the final stress value.
        """
        check_is_fitted(self, ["stress_"])
        X = validate_data(self, X, reset=False)
        return -self.stress_
