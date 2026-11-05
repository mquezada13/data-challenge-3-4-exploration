"""Descriptive statistics for prepared observed and expected counts."""

import numpy as np


def _poisson_deviance(observed, expected, residuals):
    positive_observed = observed > 0
    terms = expected.copy()
    terms[positive_observed] += observed[positive_observed] * (
        np.log(observed[positive_observed] / expected[positive_observed]) - 1.0
    )
    return float(2 * np.sum(terms))


_COMMON_STATISTICS = {
    "Valid bins (N)": lambda observed, expected, residuals: residuals.size,
    "Mean residual": lambda observed, expected, residuals: float(np.mean(residuals)),
    "Median residual": lambda observed, expected, residuals: float(np.median(residuals)),
    "Std. dev.": lambda observed, expected, residuals: float(np.std(residuals)),
    "Residual RMS": lambda observed, expected, residuals: float(np.sqrt(np.mean(residuals**2))),
    "Minimum residual": lambda observed, expected, residuals: float(np.min(residuals)),
    "Maximum residual": lambda observed, expected, residuals: float(np.max(residuals)),
}

_PEARSON_STATISTICS = {
    "|R| > 2 [%]": lambda observed, expected, residuals: float(100 * np.mean(np.abs(residuals) > 2)),
    "|R| > 3 [%]": lambda observed, expected, residuals: float(100 * np.mean(np.abs(residuals) > 3)),
    "Pearson chi-square": lambda observed, expected, residuals: float(np.sum(residuals**2)),
    "Poisson deviance": _poisson_deviance,
}

_STATISTICS_BY_TYPE = {
    "raw": _COMMON_STATISTICS,
    "pearson": {**_COMMON_STATISTICS, **_PEARSON_STATISTICS},
    "fractional": _COMMON_STATISTICS,
}


def residual_statistics(observed, expected, residual_type):
    """Return the existing descriptive statistics for prepared valid bins."""
    difference = observed - expected
    if residual_type == "raw":
        residuals = difference
    elif residual_type == "pearson":
        residuals = difference / np.sqrt(expected)
    elif residual_type == "fractional":
        residuals = difference / expected
    else:
        raise ValueError("Unknown residual type. Choose 'raw', 'pearson', or 'fractional'.")

    return {
        name: calculate(observed, expected, residuals)
        for name, calculate in _STATISTICS_BY_TYPE[residual_type].items()
    }


def build_pearson_references(
    requests, references, *, masked_expected, expected_shape,
    temporal_probabilities_input, temporal_total_counts, project_simulated_counts,
    reference_cache, reference_type, n_simulations, effective_seed,
):
    """Build and cache the requested fixed-fit Pearson reference summaries."""
    expected_full = np.asarray(
        masked_expected.contents,
        dtype=float,
    )

    multinomial_probabilities = None
    if reference_type == "multinomial":
        temporal_probabilities = np.asarray(
            temporal_probabilities_input,
            dtype=float,
        )
        if temporal_probabilities.shape != expected_shape:
            raise ValueError(
                "Temporal reference probabilities do not match the "
                "analysis shape."
            )
        if (
            not np.all(np.isfinite(temporal_probabilities))
            or np.any(temporal_probabilities < 0)
        ):
            raise ValueError(
                "Temporal reference probabilities must be finite and "
                "non-negative."
            )

        flattened_probabilities = temporal_probabilities.reshape(-1).copy()
        retained_probability = float(flattened_probabilities.sum())
        if retained_probability > 1.0:
            excess = retained_probability - 1.0
            if excess > 1e-12:
                raise ValueError(
                    "Temporal reference probabilities exceed one."
                )
            largest_index = int(np.argmax(flattened_probabilities))
            flattened_probabilities[largest_index] -= excess
            retained_probability = 1.0

        outside_probability = 1.0 - retained_probability
        multinomial_probabilities = np.append(
            flattened_probabilities,
            outside_probability,
        )

    def simulated_residuals(
        simulated_full: np.ndarray,
        request: dict[str, object],
    ) -> np.ndarray:
        # Apply the same mask and projection used for the observed data.
        simulated_projection = project_simulated_counts(
            simulated_full,
            request["projection_id"],
        )
        projected_counts = np.asarray(
            simulated_projection.contents,
            dtype=float,
        )
        if request["phi_bin_index"] is not None:
            projected_counts = projected_counts[request["phi_bin_index"]]
        elif request["energy_bin_index"] is not None:
            projected_counts = projected_counts[
                request["energy_bin_index"]
            ]
        expected_values = request["expected"]
        valid_mask = request["valid_mask"]
        return (
            projected_counts[valid_mask] - expected_values[valid_mask]
        ) / np.sqrt(expected_values[valid_mask])

    # Generate each full-CDS realization once for every requested view.
    generator = np.random.default_rng(effective_seed)
    for _ in range(n_simulations):
        if reference_type == "multinomial":
            simulated_with_outside = generator.multinomial(
                temporal_total_counts,
                multinomial_probabilities,
            )
            simulated_full = simulated_with_outside[:-1].reshape(
                expected_shape
            )
        else:
            simulated_full = generator.poisson(expected_full)
        for request in requests.values():
            values = simulated_residuals(simulated_full, request)
            request["samples"].append(values)
            request["minimum"] = min(
                request["minimum"], float(np.min(values))
            )
            request["maximum"] = max(
                request["maximum"], float(np.max(values))
            )

    # Histogram the saved projected residuals; full-CDS datasets are gone.
    density_samples = {}
    bin_edges_by_request = {}
    for reference_key, request in requests.items():
        minimum = request["minimum"]
        maximum = request["maximum"]
        if minimum == maximum:
            minimum -= 0.5
            maximum += 0.5
        bin_edges = np.linspace(minimum, maximum, request["bins"] + 1)
        bin_edges_by_request[reference_key] = bin_edges
        samples = request["samples"]
        bin_widths = np.diff(bin_edges)
        n_valid = np.count_nonzero(request["valid_mask"])
        density_samples[reference_key] = np.empty(
            (n_simulations, request["bins"]),
            dtype=float,
        )

        for simulation_index, values in enumerate(samples):
            counts = np.histogram(values, bins=bin_edges)[0]
            density_samples[reference_key][simulation_index] = (
                counts / (n_valid * bin_widths)
            )

    for reference_key, request in requests.items():
        samples = density_samples[reference_key]
        lower, upper = np.quantile(samples, [0.025, 0.975], axis=0)
        reference = {
            "bin_edges": bin_edges_by_request[reference_key],
            "density": np.mean(samples, axis=0),
            "lower": lower,
            "upper": upper,
            "seed": effective_seed,
        }
        references[reference_key] = reference
        reference_cache[request["cache_key"]] = reference

    return references
