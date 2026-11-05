"""Checks for grouping physical temporal predictions."""

import astropy.units as u
import numpy as np
import pytest
from histpy import Axis, Histogram

from src.residuals import ResidualAnalysis
from src.time_resolved_residuals_v1 import TimeResolvedResidualAnalysis


@pytest.mark.parametrize("sparse_observed", [False, True])
def test_grouping_uses_original_edges_and_cached_counts(sparse_observed):
    """Group cached counts without changing the native-resolution result."""
    time = Axis([0, 1, 3, 6, 10], label="Time", unit=u.s)
    phi = Axis([0, 1, 2], label="Phi", unit=u.deg)
    values = np.array([[200, 1], [100, 2], [4, 3], [5, 4]], dtype=np.uint8)
    observed = Histogram([time, phi], contents=values)
    if sparse_observed:
        observed = observed.to_sparse()
    integrated = Histogram([phi], contents=np.ones(2))
    builder = TimeResolvedResidualAnalysis(observed, integrated)
    assert builder.observed_hist.is_sparse == sparse_observed
    builder.configure_physical_model(None, "unused.h5", None, observed, 1, earth_occ=True)
    builder._physical_cache = {i: np.array([i + 1.0, 2.0]) for i in range(4)}

    native_d, native_m = builder.build_physical_time_range()
    grouped_d, grouped_m = builder.build_physical_time_range(bins_per_group=2)
    np.testing.assert_array_equal(native_d.contents, values)
    np.testing.assert_array_equal(native_m.contents[:, 0], [1, 2, 3, 4])
    np.testing.assert_array_equal(grouped_d.axes["Time"].edges.to_value(u.s), [0, 3, 10])
    np.testing.assert_array_equal(grouped_d.contents, [[300, 3], [9, 7]])
    np.testing.assert_array_equal(grouped_m.contents, [[3, 4], [7, 4]])
    analysis = ResidualAnalysis(grouped_d, grouped_m)
    np.testing.assert_array_equal(analysis.select_time_bin(0).expected_hist.contents, [3, 4])

    for invalid in (0, -1, 3):
        with pytest.raises(ValueError):
            builder.build_physical_time_range(bins_per_group=invalid)
    for invalid in (True, 1.5):
        with pytest.raises(TypeError):
            builder.build_physical_time_range(bins_per_group=invalid)


def test_parametric_dispatch_reuses_bins_and_checks_background(monkeypatch):
    """Route a fitted model through the shared range and reject incompatible backgrounds."""
    from astromodels import ExtendedSource, Gaussian_on_sphere, Model, PointSource, Powerlaw
    from cosipy.response import FullDetectorResponse

    time = Axis([0, 1, 2, 3], label="Time", unit=u.s)
    phi = Axis([0, 1, 2], label="Phi", unit=u.deg)
    temporal = Histogram([time, phi], contents=np.ones((3, 2)))
    integrated = Histogram([phi], contents=np.array([3.0, 3.0]))
    fitted_background = Histogram([phi], contents=np.array([6.0, 6.0]))
    model = Model(PointSource("source", ra=0, dec=0, spectral_shape=Powerlaw()))
    builder = TimeResolvedResidualAnalysis(temporal, integrated)
    builder.configure_physical_model(
        fitted_image=model, response_path="unused", orientation=None,
        temporal_background=temporal, fitted_background=fitted_background,
        fitted_energy_axis=object(), earth_occ=True,
    )
    calls = []
    def fake_bin(time_bin_index, **kwargs):
        calls.append(time_bin_index)
        return {"total": Histogram([phi], contents=np.full(2, time_bin_index + 1.0))}

    from unittest.mock import MagicMock
    monkeypatch.setattr(FullDetectorResponse, "open", MagicMock())
    monkeypatch.setattr(builder, "_build_parametric_time_bin", fake_bin)
    builder.build_physical_time_range(0, 2)
    _, result = builder.build_physical_time_range(1, 3)
    np.testing.assert_array_equal(result.contents[:, 0], [2, 3])
    assert calls == [0, 1, 2]
    assert builder._physical_configuration["background_normalization"] == 2.0

    extended_model = Model(ExtendedSource(
        "extended", spatial_shape=Gaussian_on_sphere(), spectral_shape=Powerlaw()
    ))
    with pytest.raises(ValueError, match="Extended sources require"):
        builder.configure_physical_model(
            fitted_model=extended_model, response_path="unused", orientation=None,
            temporal_background=temporal, fitted_background=fitted_background,
            fitted_energy_axis=object(), earth_occ=True,
        )
    builder.configure_physical_model(
        fitted_model=extended_model, response_path="unused", orientation=None,
        temporal_background=temporal, fitted_background=fitted_background,
        extended_sky_axis=object(), fitted_energy_axis=object(), earth_occ=True,
    )

    incompatible = Histogram([phi], contents=np.array([6.0, 7.0]))
    with pytest.raises(ValueError, match="single scaled"):
        builder.configure_physical_model(
            fitted_model=model, response_path="unused", orientation=None,
            temporal_background=temporal, fitted_background=incompatible,
            extended_sky_axis=object(), fitted_energy_axis=object(), earth_occ=True,
        )
