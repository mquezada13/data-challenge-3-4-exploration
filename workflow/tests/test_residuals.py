"""Numerical tests for COSI residual analysis."""

import numpy as np
import pytest
import astropy.units as u
from histpy import Axis, HealpixAxis, Histogram

from src.residuals import ResidualAnalysis


def make_histogram(contents, axis_label="Phi", edges=None):
    """Create a small histogram with a labeled angular axis."""
    if edges is None:
        edges = np.arange(len(contents) + 1, dtype=float)
    axis = Axis(edges, label=axis_label, unit=u.deg)
    return Histogram([axis], contents=np.asarray(contents, dtype=float))


def test_residual_definitions_use_projected_counts():
    """Check raw, Pearson, and fractional residuals after aggregation."""
    phi_axis = Axis([0.0, 1.0, 2.0], label="Phi", unit=u.deg)
    other_axis = Axis([0.0, 1.0, 2.0, 3.0], label="Other", unit=u.deg)
    observed = Histogram(
        [phi_axis, other_axis],
        contents=np.array([[4.0, 0.0, 1.0], [0.0, 6.0, 2.0]]),
    )
    expected = Histogram(
        [phi_axis, other_axis],
        contents=np.array([[3.0, 0.0, 2.0], [1.0, 3.0, 1.0]]),
    )
    analysis = ResidualAnalysis(observed, expected)
    phi_products = analysis.projection_function("Phi")
    analysis._projected_data_cache = {"Phi_projection": phi_products}

    np.testing.assert_allclose(
        analysis.residual_function("Phi_projection", "raw").contents,
        [0.0, 3.0],
    )
    np.testing.assert_allclose(
        analysis.residual_function("Phi_projection", "pearson").contents,
        [0.0, 3.0 / np.sqrt(5.0)],
    )
    np.testing.assert_allclose(
        analysis.residual_function("Phi_projection", "fractional").contents,
        [0.0, 0.6],
    )


def test_normalized_residuals_mark_zero_expectations_undefined():
    """Zero expected counts remain undefined for normalized residuals."""
    observed = make_histogram([2.0, 0.0])
    expected = make_histogram([1.0, 0.0])
    analysis = ResidualAnalysis(observed, expected)
    analysis._projected_data_cache = {
        "Phi_projection": {"observed": observed, "expected": expected}
    }

    pearson = analysis.residual_function("Phi_projection", "pearson").contents
    fractional = analysis.residual_function(
        "Phi_projection",
        "fractional",
    ).contents

    assert pearson[0] == pytest.approx(1.0)
    assert fractional[0] == pytest.approx(1.0)
    assert np.isnan(pearson[1])
    assert np.isnan(fractional[1])


def test_mask_excludes_only_numerical_floor_bins_without_events():
    """The numerical-floor mask excludes empty bins and reports the choice."""
    numerical_floor = 10 * np.finfo(float).tiny
    observed = make_histogram([3.0, 0.0])
    expected = make_histogram([2.0, numerical_floor / 2])
    analysis = ResidualAnalysis(observed, expected)

    np.testing.assert_array_equal(analysis.mask, [True, False])
    assert analysis.mask_report["excluded_bins"] == 1
    assert analysis.mask_report["observed_counts_excluded"] == 0.0


def test_mask_rejects_observed_events_with_zero_expectation():
    """Observed events cannot be silently masked when expectation is zero."""
    observed = make_histogram([0.0, 1.0])
    expected = make_histogram([2.0, 0.0])

    with pytest.raises(ValueError, match="contain observed events"):
        ResidualAnalysis(observed, expected)


def test_mask_rejects_histograms_with_no_valid_bins():
    """At least one positive, non-floor expectation must remain."""
    observed = make_histogram([0.0, 0.0])
    expected = make_histogram([0.0, np.finfo(float).tiny])

    with pytest.raises(ValueError, match="No valid bins remain"):
        ResidualAnalysis(observed, expected)


def test_histograms_must_have_matching_axes():
    """Equal shapes with different axis labels are not compatible."""
    observed = make_histogram([1.0, 2.0], axis_label="Phi")
    expected = make_histogram([1.0, 2.0], axis_label="Chi")

    with pytest.raises(ValueError, match="matching axes"):
        ResidualAnalysis(observed, expected)


def test_statistics_table_calculates_pearson_chi_square_and_deviance():
    """Check full-CDS chi-square and Poisson deviance against known values."""
    observed = make_histogram([5.0, 0.0])
    expected = make_histogram([4.0, 0.0])
    analysis = ResidualAnalysis(observed, expected)
    table = analysis.statistics_table().set_index("Statistic")

    assert table.loc["Valid CDS bins", "Value"] == 1
    assert table.loc["Pearson chi-square (χ²)", "Value"] == pytest.approx(0.25)
    assert table.loc["Mean Pearson contribution per valid bin", "Value"] == pytest.approx(0.25)
    expected_deviance = 2 * (5 * np.log(5 / 4) - (5 - 4))
    assert table.loc["Poisson deviance", "Value"] == pytest.approx(expected_deviance)


def test_projection_statistics_report_projected_and_full_cds_bins():
    """The projection table distinguishes the full CDS from a projection."""
    observed = make_histogram([5.0, 0.0])
    expected = make_histogram([4.0, 0.0])
    analysis = ResidualAnalysis(observed, expected)
    analysis._projected_data_cache = {
        "Phi_projection": {"observed": observed, "expected": expected}
    }

    table = analysis.projection_statistics_table().set_index("Projection")
    assert table.loc["Full CDS", "valid_bins"] == 1
    assert table.loc["Phi_projection", "valid_bins"] == 1
    assert table.loc["Phi_projection", "pearson_chi_square"] == pytest.approx(0.25)
    assert table.loc["Phi_projection", "fraction_abs_gt_2_percent"] == 0.0


def test_psichi_coordinate_split_preserves_counts():
    """Center-based Chi and Psi binning preserves the input event total."""
    psichi_axis = HealpixAxis(nside=1, coordsys="C", label="PsiChi")
    observed = Histogram([psichi_axis], contents=np.arange(12, dtype=float))
    expected = observed.copy()
    analysis = ResidualAnalysis(observed, expected)

    split_products = analysis.split_psichi_function(
        {"observed": observed, "expected": expected}
    )

    for coordinate in ("Chi", "Psi"):
        assert split_products[coordinate]["observed"].contents.sum() == pytest.approx(
            observed.contents.sum()
        )
        assert split_products[coordinate]["expected"].contents.sum() == pytest.approx(
            expected.contents.sum()
        )


def test_psichi_coordinate_split_requires_positive_bin_width():
    """Coordinate rebinning rejects a zero or negative angular width."""
    psichi_axis = HealpixAxis(nside=1, coordsys="C", label="PsiChi")
    histogram = Histogram([psichi_axis], contents=np.ones(12))
    analysis = ResidualAnalysis(histogram, histogram.copy())

    with pytest.raises(ValueError, match="bin_width must be positive"):
        analysis.split_psichi_function(
            {"observed": histogram, "expected": histogram.copy()},
            bin_width=0.0,
        )
