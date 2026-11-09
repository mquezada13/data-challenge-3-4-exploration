"""Checks for grouping physical temporal predictions."""

import astropy.units as u
from astropy.time import Time
import numpy as np
import pytest
from histpy import Axis, Histogram

from src.residuals import ResidualAnalysis
from src.time_resolved_residuals_v1 import TemporalPredictionBuilder


@pytest.mark.parametrize(
    "edges,start,stop,expected_edges",
    [
        ([0, 3600, 7200, 10800, 14400], 1800, 9000, [0, 3600, 7200, 10800]),
        ([0, 30, 60, 90, 120], 15, 60, [0, 30, 60, 90]),
        ([0, 30, 75, 120], 35, 40, [30, 75]),
        ([0, 30, 75, 120], 0, 119, [0, 30, 75, 120]),
    ],
)
def test_timestamp_selection_uses_native_bins_and_cache(edges, start, stop, expected_edges):
    """Select complete bins for hourly, sub-minute, and uneven time axes."""
    time = Axis(edges, label="Time", unit=u.s)
    phi = Axis([0, 1], label="Phi", unit=u.deg)
    observed = Histogram([time, phi], contents=np.arange(1, len(edges))[:, None])
    integrated = Histogram([phi], contents=np.ones(1))
    builder = TemporalPredictionBuilder(observed, integrated)
    builder.configure_physical_model(integrated, "unused.h5", None, observed, 1, earth_occ=True)
    builder._physical_cache = {i: np.array([10.0 + i]) for i in range(len(edges) - 1)}

    selected, expected = builder.timestamps_interval(
        Time(start, format="unix", scale="utc").isot,
        Time(stop, format="unix", scale="utc"),
    )

    np.testing.assert_array_equal(selected.axes["Time"].edges.to_value(u.s), expected_edges)
    np.testing.assert_array_equal(expected.axes["Time"].edges.to_value(u.s), expected_edges)
    first_index = edges.index(expected_edges[0])
    last_index = edges.index(expected_edges[-1])
    np.testing.assert_array_equal(selected.contents[:, 0], np.arange(first_index + 1, last_index + 1))
    np.testing.assert_array_equal(expected.contents[:, 0], np.arange(10 + first_index, 10 + last_index))


def test_timestamp_selection_rejects_invalid_ranges():
    """Reject invalid timestamps and the excluded final observation edge."""
    time = Axis([0, 30, 60], label="Time", unit=u.s)
    phi = Axis([0, 1], label="Phi", unit=u.deg)
    observed = Histogram([time, phi], contents=np.ones((2, 1)))
    builder = TemporalPredictionBuilder(observed, Histogram([phi], contents=np.ones(1)))

    def utc(seconds):
        return Time(seconds, format="unix", scale="utc")

    for start, stop in ((utc(-1), utc(1)), (utc(1), utc(60)), (utc(31), utc(30))):
        with pytest.raises(ValueError):
            builder.timestamps_interval(start, stop)
    with pytest.raises(ValueError, match="valid UTC timestamp"):
        builder.timestamps_interval("not a timestamp", utc(1))
    with pytest.raises(TypeError, match="timestamp string or astropy Time"):
        builder.timestamps_interval(0, utc(1))
    with pytest.raises(ValueError, match="scalar"):
        builder.timestamps_interval(utc([0, 1]), utc(2))


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
    builder = TemporalPredictionBuilder(observed, integrated)
    assert builder.observed_hist.is_sparse == sparse_observed
    # The cached-count fixture does not perform physical image folding.
    builder.configure_physical_model(integrated, "unused.h5", None, observed, 1, earth_occ=True)
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
    builder = TemporalPredictionBuilder(temporal, integrated)
    builder.configure_physical_model(
        fitted_sky=model, response_path="unused", orientation=None,
        temporal_background=temporal, fitted_background=fitted_background,
        fitted_energy_axis=object(), earth_occ=True,
    )
    calls = []
    def fake_bin(time_bin_index, **kwargs):
        calls.append(time_bin_index)
        return {"total": Histogram([phi], contents=np.full(2, time_bin_index + 1.0))}

    from unittest.mock import MagicMock
    monkeypatch.setattr(FullDetectorResponse, "open", MagicMock())
    monkeypatch.setattr(builder, "build_parametric_time_bin", fake_bin)
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
            fitted_sky=extended_model, response_path="unused", orientation=None,
            temporal_background=temporal, fitted_background=fitted_background,
            fitted_energy_axis=object(), earth_occ=True,
        )
    builder.configure_physical_model(
        fitted_sky=extended_model, response_path="unused", orientation=None,
        temporal_background=temporal, fitted_background=fitted_background,
        extended_sky_axis=object(), fitted_energy_axis=object(), earth_occ=True,
    )

    incompatible = Histogram([phi], contents=np.array([6.0, 7.0]))
    with pytest.raises(ValueError, match="single scaled"):
        builder.configure_physical_model(
            fitted_sky=model, response_path="unused", orientation=None,
            temporal_background=temporal, fitted_background=incompatible,
            extended_sky_axis=object(), fitted_energy_axis=object(), earth_occ=True,
        )


def test_fitted_sky_selects_imaging_and_rejects_unsupported_inputs(monkeypatch):
    """Check configuration and dispatch with a mock, not a physical prediction."""
    from unittest.mock import MagicMock
    from cosipy.response import FullDetectorResponse

    time = Axis([0, 1], label="Time", unit=u.s)
    phi = Axis([0, 1, 2], label="Phi", unit=u.deg)
    temporal = Histogram([time, phi], contents=np.ones((1, 2)))
    integrated = Histogram([phi], contents=np.ones(2))
    builder = TemporalPredictionBuilder(temporal, integrated)

    # This histogram is a dispatch fixture, not a fitted sky image.
    builder.configure_physical_model(
        fitted_sky=integrated, response_path="unused", orientation=None,
        temporal_background=temporal, background_normalization=1.0, earth_occ=True,
    )
    assert builder._physical_configuration["fitted_image"] is integrated
    assert "fitted_model" not in builder._physical_configuration

    calls = []
    def fake_imaging_bin(time_bin_index, **kwargs):
        calls.append(time_bin_index)
        assert kwargs["fitted_image"] is integrated
        return {"total": Histogram([phi], contents=np.array([2.0, 3.0]))}

    monkeypatch.setattr(FullDetectorResponse, "open", MagicMock())
    monkeypatch.setattr(builder, "build_imaging_time_bin", fake_imaging_bin)
    _, expected = builder.build_physical_time_range()
    assert calls == [0]
    np.testing.assert_array_equal(expected.contents, [[2.0, 3.0]])

    with pytest.raises(TypeError, match="Histogram or an astromodels Model"):
        builder.configure_physical_model(fitted_sky=object(), earth_occ=True)
