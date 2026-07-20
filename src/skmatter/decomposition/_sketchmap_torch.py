import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _require_torch():
    if torch is None:
        raise ImportError("backend='torch' requires PyTorch")


def closed_form_total_weight(weights):
    weights = np.asarray(weights, dtype=np.float64)
    return float(0.5 * (weights.sum() ** 2 - (weights**2).sum()))


def _sigmoid_value(radius, sigma, exponent_a, exponent_b):
    offset = 2.0 ** (exponent_a / exponent_b) - 1.0
    scaled_radius = radius / sigma
    return 1.0 - (1.0 + offset * scaled_radius**exponent_a) ** (
        -exponent_b / exponent_a
    )


def _sigmoid_value_and_derivative(radius, sigma, exponent_a, exponent_b):
    offset = 2.0 ** (exponent_a / exponent_b) - 1.0
    scaled_radius = radius / sigma
    scaled_power = offset * scaled_radius**exponent_a
    decay = (1.0 + scaled_power) ** (-exponent_b / exponent_a)

    value = 1.0 - decay
    derivative = (
        (exponent_b * offset / sigma)
        * scaled_radius ** (exponent_a - 1.0)
        * decay
        / (1.0 + scaled_power)
    )

    nonpositive = radius <= 0.0
    zero_value = torch.zeros((), dtype=value.dtype, device=value.device)

    return (
        torch.where(nonpositive, zero_value, value),
        torch.where(nonpositive, zero_value, derivative),
    )


def _pair_kernel(
    features_rows,
    features_cols,
    feature_sqnorm_rows,
    feature_sqnorm_cols,
    embedding_rows,
    embedding_cols,
    embedding_sqnorm_rows,
    embedding_sqnorm_cols,
    weights_rows,
    weights_cols,
    sigma,
    a_high,
    b_high,
    a_low,
    b_low,
    mixing_ratio,
    use_transform,
):
    squared_high_dim = torch.clamp(
        feature_sqnorm_rows[:, None]
        + feature_sqnorm_cols[None, :]
        - 2.0 * (features_rows @ features_cols.T),
        min=0.0,
    )
    high_dim_distances = torch.sqrt(squared_high_dim)

    squared_low_dim = torch.clamp(
        embedding_sqnorm_rows[:, None]
        + embedding_sqnorm_cols[None, :]
        - 2.0 * (embedding_rows @ embedding_cols.T),
        min=0.0,
    )
    low_dim_distances = torch.sqrt(squared_low_dim)

    if use_transform:
        high_dim_transformed = _sigmoid_value(high_dim_distances, sigma, a_high, b_high)
        low_dim_transformed, low_dim_derivative = _sigmoid_value_and_derivative(
            low_dim_distances, sigma, a_low, b_low
        )
    else:
        high_dim_transformed = high_dim_distances
        low_dim_transformed = low_dim_distances
        low_dim_derivative = 1.0

    transformed_residual = high_dim_transformed - low_dim_transformed
    raw_residual = high_dim_distances - low_dim_distances
    pair_weights = weights_rows[:, None] * weights_cols[None, :]

    stress = (
        pair_weights
        * (
            (1.0 - mixing_ratio) * transformed_residual * transformed_residual
            + mixing_ratio * raw_residual * raw_residual
        )
    ).sum(dtype=torch.float64)

    coefficients = pair_weights * (
        (1.0 - mixing_ratio) * transformed_residual * low_dim_derivative
        + mixing_ratio * raw_residual
    )
    coefficients = coefficients / torch.clamp(
        low_dim_distances, min=torch.finfo(low_dim_distances.dtype).tiny
    )

    return stress, coefficients


class TiledStress:
    def __init__(
        self,
        features,
        weights,
        n_components,
        sigmoid_params,
        devices=None,
        row_tile=8192,
        col_tile=16384,
        compile_kernel=True,
        dtype="float32",
        row_range=None,
    ):
        _require_torch()

        self.n_samples, self.n_components = features.shape[0], n_components
        self.sigmoid_params = sigmoid_params
        self.row_tile, self.col_tile = row_tile, col_tile
        self.total_weight = (
            closed_form_total_weight(weights)
            if weights is not None
            else self.n_samples * (self.n_samples - 1) / 2.0
        )

        if devices is None:
            device_count = torch.cuda.device_count()
            devices = (
                [f"cuda:{index}" for index in range(device_count)]
                if device_count
                else ["cpu"]
            )
        self.devices = list(devices)
        torch_dtype = getattr(torch, dtype)

        self.row_start, self.row_end = (
            row_range if row_range is not None else (0, self.n_samples)
        )
        boundaries = np.linspace(
            self.row_start, self.row_end, len(self.devices) + 1
        ).astype(int)
        self.device_row_ranges = [
            (int(boundaries[index]), int(boundaries[index + 1]))
            for index in range(len(self.devices))
        ]

        self.features, self.feature_sqnorm, self.weights = {}, {}, {}
        contiguous_features = np.ascontiguousarray(features)

        for device in self.devices:
            self.features[device] = torch.as_tensor(
                contiguous_features, device=device, dtype=torch_dtype
            )
            self.feature_sqnorm[device] = (self.features[device] ** 2).sum(1)
            self.weights[device] = (
                torch.as_tensor(
                    np.ascontiguousarray(weights), device=device, dtype=torch_dtype
                )
                if weights is not None
                else torch.full(
                    (self.n_samples,), 1.0, device=device, dtype=torch_dtype
                )
            )

        self.kernel = (
            torch.compile(_pair_kernel, dynamic=False) if compile_kernel else _pair_kernel
        )
        self.n_evaluations = 0

    def _device_block(
        self, device, block_start, block_end, embedding, mixing_ratio, use_transform
    ):
        params = self.sigmoid_params
        device_features = self.features[device]
        device_feature_sqnorm = self.feature_sqnorm[device]
        device_weights = self.weights[device]

        device_embedding = torch.as_tensor(
            embedding, device=device, dtype=device_features.dtype
        )
        device_embedding_sqnorm = (device_embedding**2).sum(1)

        gradient = torch.zeros(
            (block_end - block_start, self.n_components),
            device=device,
            dtype=device_embedding.dtype,
        )
        stress = torch.zeros((), device=device, dtype=torch.float64)

        for row_start in range(block_start, block_end, self.row_tile):
            row_end = min(row_start + self.row_tile, block_end)
            row_gradient = torch.zeros(
                (row_end - row_start, self.n_components),
                device=device,
                dtype=device_embedding.dtype,
            )

            for col_start in range(0, self.n_samples, self.col_tile):
                col_end = min(col_start + self.col_tile, self.n_samples)

                tile_stress, coefficients = self.kernel(
                    device_features[row_start:row_end],
                    device_features[col_start:col_end],
                    device_feature_sqnorm[row_start:row_end],
                    device_feature_sqnorm[col_start:col_end],
                    device_embedding[row_start:row_end],
                    device_embedding[col_start:col_end],
                    device_embedding_sqnorm[row_start:row_end],
                    device_embedding_sqnorm[col_start:col_end],
                    device_weights[row_start:row_end],
                    device_weights[col_start:col_end],
                    params["sigma"],
                    params["a_high"],
                    params["b_high"],
                    params["a_low"],
                    params["b_low"],
                    mixing_ratio,
                    use_transform,
                )

                overlap_start = max(row_start, col_start)
                overlap_end = min(row_end, col_end)
                if overlap_end > overlap_start:
                    diagonal = torch.arange(overlap_start, overlap_end, device=device)
                    coefficients[diagonal - row_start, diagonal - col_start] = 0.0

                stress = stress + tile_stress
                row_gradient += (
                    coefficients.sum(1, keepdim=True) * device_embedding[row_start:row_end]
                    - coefficients @ device_embedding[col_start:col_end]
                )

            gradient[row_start - block_start : row_end - block_start] = row_gradient

        return stress, gradient * (-2.0 / self.total_weight)

    def stress_and_grad(self, flat_embedding, mixing_ratio=0.0, use_transform=True):
        embedding = np.ascontiguousarray(
            flat_embedding.reshape(self.n_samples, self.n_components), dtype=np.float32
        )

        device_results = [
            self._device_block(
                device, block_start, block_end, embedding, mixing_ratio, use_transform
            )
            for device, (block_start, block_end) in zip(
                self.devices, self.device_row_ranges
            )
        ]

        stress = (
            0.5
            * float(sum(float(stress) for stress, _ in device_results))
            / self.total_weight
        )
        gradient_rows = np.concatenate(
            [block.to("cpu").numpy() for _, block in device_results], axis=0
        )

        if (self.row_start, self.row_end) == (0, self.n_samples):
            gradient = gradient_rows
        else:
            gradient = np.zeros(
                (self.n_samples, self.n_components), dtype=gradient_rows.dtype
            )
            gradient[self.row_start : self.row_end] = gradient_rows

        self.n_evaluations += 1
        return stress, gradient.ravel().astype(np.float64)
