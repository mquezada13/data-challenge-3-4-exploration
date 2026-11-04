"""Checks for grouping physical temporal predictions."""

import astropy.units as u
import numpy as np
import pytest
from histpy import Axis, Histogram

from src.residuals import ResidualAnalysis
from src.time_resolved_residuals_v1 import TimeResolvedResidualAnalysis


def test_grouping_uses_original_edges_and_cached_counts():
    """Group cached counts without changing the native-resolution result."""
    time = Axis([0, 1, 3, 6, 10], label="Time", unit=u.s)
    phi = Axis([0, 1, 2], label="Phi", unit=u.deg)
    values = np.array([[200, 1], [100, 2], [4, 3], [5, 4]], dtype=np.uint8)
    observed = Histogram([time, phi], contents=values)
    integrated = Histogram([phi], contents=np.ones(2))
    builder = TimeResolvedResidualAnalysis(observed, integrated)
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
