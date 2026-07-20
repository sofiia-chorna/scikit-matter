import time

import numpy as np
from scipy.spatial.distance import cdist

from skmatter.decomposition import SketchMap
from skmatter.decomposition._sketchmap_torch import TiledStress, closed_form_total_weight
from skmatter.decomposition._sketchmap_utils import sigmoid_transform

analysis = "/capstor/scratch/cscs/schorna/merge-datasets/phace-atomic/voronoi/out/analysis"
SIGMOID_PARAMS = dict(sigma=3.0, a_high=4.0, b_high=2.0, a_low=2.0, b_low=2.0)


def _inputs(n_samples, n_components=8):
    features = np.load(f"{analysis}/features_first65536.npy")[:n_samples]
    features = features.astype(np.float64)

    populations = np.load(f"{analysis}/populations_level14.npy")
    weights = populations.astype(np.float64)[:n_samples] ** 0.25

    return features, weights / weights.sum(), n_components


def _dense_reference(features, weights, n_components, flat_embedding):
    sketchmap = SketchMap(n_components=n_components, center=False, **SIGMOID_PARAMS)

    high_dim_distances = cdist(features, features)
    sketchmap._resolve_params(high_dim_distances)
    high_dim_transformed = sigmoid_transform(
        high_dim_distances,
        SIGMOID_PARAMS["sigma"],
        SIGMOID_PARAMS["a_high"],
        SIGMOID_PARAMS["b_high"],
    )

    pair_weights = np.outer(weights, weights)
    total_weight = float(np.sum(np.triu(pair_weights, k=1)))
    problem = (high_dim_distances, high_dim_transformed, pair_weights, total_weight)

    return (
        sketchmap._stress_and_grad(flat_embedding, problem, 0.0, True),
        total_weight,
    )


def test_closed_form_total_weight():
    generator = np.random.default_rng(0)
    weights = generator.random(500)
    weights /= weights.sum()

    reference = float(np.sum(np.triu(np.outer(weights, weights), k=1)))

    assert abs(closed_form_total_weight(weights) - reference) / reference < 1e-10


def test_tiled_matches_dense():
    n_samples, n_components = 3000, 8
    features, weights, n_components = _inputs(n_samples, n_components)

    generator = np.random.default_rng(0)
    flat_embedding = generator.normal(size=n_samples * n_components)

    (reference_stress, reference_gradient), reference_total_weight = _dense_reference(
        features, weights, n_components, flat_embedding
    )

    engine = TiledStress(
        features.astype(np.float32),
        weights.astype(np.float32),
        n_components,
        SIGMOID_PARAMS,
        row_tile=512,
        col_tile=1024,
        compile_kernel=False,
    )
    assert (
        abs(engine.total_weight - reference_total_weight) / reference_total_weight < 1e-7
    )

    stress, gradient = engine.stress_and_grad(flat_embedding, 0.0, True)

    assert abs(stress - reference_stress) / abs(reference_stress) < 1e-5, (
        stress,
        reference_stress,
    )
    assert (
        np.abs(gradient - reference_gradient).max() / np.abs(reference_gradient).max()
        < 1e-4
    )

    cosine = (
        gradient
        @ reference_gradient
        / (np.linalg.norm(gradient) * np.linalg.norm(reference_gradient))
    )
    assert cosine > 1 - 1e-9, cosine


def test_tiling_is_invariant():
    n_samples, n_components = 2048, 8
    features, weights, n_components = _inputs(n_samples, n_components)

    generator = np.random.default_rng(1)
    flat_embedding = generator.normal(size=n_samples * n_components)

    results = []
    for row_tile, col_tile in ((256, 512), (1024, 2048), (2048, 2048)):
        engine = TiledStress(
            features.astype(np.float32),
            weights.astype(np.float32),
            n_components,
            SIGMOID_PARAMS,
            row_tile=row_tile,
            col_tile=col_tile,
            compile_kernel=False,
        )
        results.append(engine.stress_and_grad(flat_embedding, 0.0, True))

    first_stress, first_gradient = results[0]
    for stress, gradient in results[1:]:
        assert abs(stress - first_stress) / abs(first_stress) < 1e-6
        assert np.abs(gradient - first_gradient).max() / np.abs(first_gradient).max() < 1e-5


def test_compile_speedup():
    n_samples, n_components = 8192, 8
    features, weights, n_components = _inputs(n_samples, n_components)

    generator = np.random.default_rng(0)
    flat_embedding = generator.normal(size=n_samples * n_components)

    seconds_per_eval = {}
    for compile_kernel in (False, True):
        engine = TiledStress(
            features.astype(np.float32),
            weights.astype(np.float32),
            n_components,
            SIGMOID_PARAMS,
            row_tile=2048,
            col_tile=4096,
            compile_kernel=compile_kernel,
        )
        engine.stress_and_grad(flat_embedding, 0.0, True)

        started = time.time()
        for _ in range(3):
            engine.stress_and_grad(flat_embedding, 0.0, True)
        label = "compiled" if compile_kernel else "eager"
        seconds_per_eval[label] = (time.time() - started) / 3

    print(
        f"\neager={seconds_per_eval['eager']*1000:.1f} ms  "
        f"compiled={seconds_per_eval['compiled']*1000:.1f} ms  "
        f"speedup={seconds_per_eval['eager']/seconds_per_eval['compiled']:.2f}x"
    )

    assert seconds_per_eval["compiled"] < seconds_per_eval["eager"]


def test_row_range_is_additive():
    n_samples, n_components = 2048, 8
    features, weights, n_components = _inputs(n_samples, n_components)

    generator = np.random.default_rng(2)
    flat_embedding = generator.normal(size=n_samples * n_components)

    features32 = features.astype(np.float32)
    weights32 = weights.astype(np.float32)

    whole = TiledStress(
        features32,
        weights32,
        n_components,
        SIGMOID_PARAMS,
        row_tile=512,
        col_tile=1024,
        compile_kernel=False,
    )
    whole_stress, whole_gradient = whole.stress_and_grad(flat_embedding, 0.0, True)

    for n_parts in (2, 4):
        boundaries = np.linspace(0, n_samples, n_parts + 1).astype(int)
        summed_stress = 0.0
        summed_gradient = np.zeros_like(whole_gradient)

        for part in range(n_parts):
            engine = TiledStress(
                features32,
                weights32,
                n_components,
                SIGMOID_PARAMS,
                row_tile=512,
                col_tile=1024,
                compile_kernel=False,
                row_range=(int(boundaries[part]), int(boundaries[part + 1])),
            )
            stress, gradient = engine.stress_and_grad(flat_embedding, 0.0, True)

            assert gradient.shape == whole_gradient.shape
            summed_stress += stress
            summed_gradient += gradient

        assert abs(summed_stress - whole_stress) / abs(whole_stress) < 1e-6, (
            n_parts,
            summed_stress,
            whole_stress,
        )
        assert (
            np.abs(summed_gradient - whole_gradient).max()
            / np.abs(whole_gradient).max()
            < 1e-6
        ), n_parts
