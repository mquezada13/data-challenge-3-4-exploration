"""Numerical tests for COSI residual analysis."""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import astropy.units as u
from histpy import Axis, HealpixAxis, Histogram

from src.plot_residuals import PlottingResiduals
from src.residuals import ResidualAnalysis


def make_histogram(contents, axis_label="Phi", edges=None):
    """Create a small histogram with a labeled angular axis."""
    if edges is None:
        edges = np.arange(len(contents) + 1, dtype=float)
    axis = Axis(edges, label=axis_label, unit=u.deg)
    return Histogram([axis], contents=np.asarray(contents, dtype=float))


def make_poisson_reference_analysis(expected_values, observed_values=None):
    """Create a one-dimensional analysis for Poisson-reference tests."""
    expected_values = np.asarray(expected_values, dtype=float)
    if observed_values is None:
        observed_values = expected_values.copy()
    observed = make_histogram(observed_values)
    expected = make_histogram(expected_values)
    analysis = ResidualAnalysis(observed, expected)
    analysis._projected_data_cache = {
        "Phi_projection": analysis.projection_function("Phi")
    }
    return analysis


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


def test_configured_expected_floor_excludes_empty_bins_and_rejects_events():
    """A configured artificial floor is masked only for empty bins."""
    expected_floor = 1e-12
    expected = make_histogram([2.0, expected_floor])

    # Empty bins at the configured floor are excluded from the analysis.
    observed_empty_floor = make_histogram([3.0, 0.0])
    analysis = ResidualAnalysis(
        observed_empty_floor,
        expected,
        expected_floor=expected_floor,
    )
    np.testing.assert_array_equal(analysis.mask, [True, False])
    assert analysis.mask_report["excluded_bins"] == 1
    assert analysis.mask_report["expected_floor"] == expected_floor

    # A floor bin containing observed events must raise instead of hiding data.
    observed_with_floor_events = make_histogram([3.0, 1.0])
    with pytest.raises(ValueError, match="contain observed events"):
        ResidualAnalysis(
            observed_with_floor_events,
            expected,
            expected_floor=expected_floor,
        )


def test_pearson_poisson_reference_is_reproducible():
    """A fixed random seed reproduces the complete reference estimate."""
    analysis = make_poisson_reference_analysis(np.full(12, 4.0))

    first = analysis.pearson_poisson_reference(
        "Phi_projection",
        n_simulations=20,
        bins=10,
        seed=17,
    )
    second = analysis.pearson_poisson_reference(
        "Phi_projection",
        n_simulations=20,
        bins=10,
        seed=17,
    )

    for key in ("bin_edges", "density", "lower", "upper"):
        np.testing.assert_array_equal(first[key], second[key])
    assert first["seed"] == 17


def test_shared_pearson_references_reuse_each_full_cds_draw(monkeypatch):
    """One simulated CDS realization serves every requested projection."""
    phi_axis = Axis([0.0, 1.0, 2.0], label="Phi", unit=u.deg)
    psichi_axis = HealpixAxis(nside=1, coordsys="C", label="PsiChi")
    expected = Histogram(
        [phi_axis, psichi_axis],
        contents=np.full((2, 12), 4.0),
    )
    observed = Histogram(
        [phi_axis, psichi_axis],
        contents=np.full((2, 12), 4.0),
    )
    analysis = ResidualAnalysis(observed, expected)
    analysis._projected_data_cache = {
        "Phi_projection": analysis.projection_function("Phi"),
        "PsiChi_projection": analysis.projection_function("PsiChi"),
    }

    original_default_rng = np.random.default_rng
    poisson_call_count = 0

    class CountingGenerator:
        """Count full-CDS Poisson draws while retaining NumPy behavior."""

        def __init__(self, seed):
            self.generator = original_default_rng(seed)

        def poisson(self, expected_counts):
            nonlocal poisson_call_count
            poisson_call_count += 1
            return self.generator.poisson(expected_counts)

    monkeypatch.setattr(
        np.random,
        "default_rng",
        lambda seed: CountingGenerator(seed),
    )
    n_simulations = 8
    references = analysis.prepare_pearson_poisson_references(
        {"Phi_projection": 15, "PsiChi_projection": 40},
        n_simulations=n_simulations,
        seed=23,
    )

    # Each simulated full-CDS dataset is generated once for both projections.
    assert poisson_call_count == n_simulations
    assert set(references) == {
        ("Phi_projection", None),
        ("PsiChi_projection", None),
    }

    # Plot requests reuse the cached reference without generating new draws.
    analysis.pearson_poisson_reference(
        "Phi_projection",
        n_simulations=n_simulations,
        bins=15,
        seed=23,
    )
    assert poisson_call_count == n_simulations


def test_pearson_poisson_reference_respects_mask_and_zero_expectations():
    """Masked floor and zero-expectation bins do not enter the distribution."""
    expected_floor = 1e-12
    observed = make_histogram([4.0, 0.0, 0.0])
    expected = make_histogram([4.0, expected_floor, 0.0])
    analysis = ResidualAnalysis(
        observed,
        expected,
        expected_floor=expected_floor,
    )
    analysis._projected_data_cache = {
        "Phi_projection": analysis.projection_function("Phi")
    }

    reference = analysis.pearson_poisson_reference(
        "Phi_projection",
        n_simulations=30,
        bins=12,
        seed=5,
    )
    bin_widths = np.diff(reference["bin_edges"])

    # Each simulated density integrates to one over the sole retained bin.
    assert np.sum(reference["density"] * bin_widths) == pytest.approx(1.0)
    assert analysis.mask_report["excluded_bins"] == 2


def test_simulated_counts_conserve_after_projection():
    """Projection and coordinate splitting preserve masked simulated counts."""
    phi_axis = Axis([0.0, 1.0, 2.0], label="Phi", unit=u.deg)
    psichi_axis = HealpixAxis(nside=1, coordsys="C", label="PsiChi")
    expected_values = np.full((2, 12), 2.0)
    observed_values = np.full((2, 12), 2.0)
    observed = Histogram([phi_axis, psichi_axis], contents=observed_values)
    expected = Histogram([phi_axis, psichi_axis], contents=expected_values)
    analysis = ResidualAnalysis(observed, expected)
    analysis.projected_data()
    simulated_counts = np.arange(24, dtype=float).reshape(2, 12)
    retained_total = simulated_counts[analysis.mask].sum()

    for projection_id in (
        "Phi_projection",
        "Chi_projection",
        "Psi_projection",
        "PsiChi_projection",
        "Phi_Chi_projection",
        "Phi_Psi_projection",
        "Phi_PsiChi_projection",
    ):
        projected = analysis._project_simulated_counts(
            simulated_counts,
            projection_id,
        )
        assert np.sum(projected.contents) == pytest.approx(retained_total)

        reference = analysis.pearson_poisson_reference(
            projection_id,
            n_simulations=4,
            bins=6,
            seed=9,
            phi_bin_index=(0 if projection_id == "Phi_PsiChi_projection" else None),
        )
        assert np.sum(
            reference["density"] * np.diff(reference["bin_edges"])
        ) == pytest.approx(1.0)

    # The selected fixed-Phi HEALPix view uses only that slice's expected bins.
    phi_slice_reference = analysis.pearson_poisson_reference(
        "Phi_PsiChi_projection",
        n_simulations=12,
        bins=10,
        seed=13,
        phi_bin_index=1,
    )
    assert np.sum(
        phi_slice_reference["density"]
        * np.diff(phi_slice_reference["bin_edges"])
    ) == pytest.approx(1.0)


def test_pearson_poisson_reference_matches_simple_poisson_moments():
    """The simulated standardized residuals approach zero mean and unit spread."""
    analysis = make_poisson_reference_analysis(np.full(80, 4.0))
    reference = analysis.pearson_poisson_reference(
        "Phi_projection",
        n_simulations=500,
        bins=80,
        seed=42,
    )
    edges = reference["bin_edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    density = reference["density"]
    mean = np.sum(centers * density * widths)
    variance = np.sum((centers - mean) ** 2 * density * widths)

    assert mean == pytest.approx(0.0, abs=0.15)
    assert variance == pytest.approx(1.0, abs=0.25)


def test_pearson_plot_uses_the_same_histogram_edges_for_observed_and_reference(
    monkeypatch,
):
    """Observed histogram bars and simulated reference use identical edges."""
    monkeypatch.setattr(plt, "show", lambda: None)
    analysis = make_poisson_reference_analysis(np.full(12, 4.0))
    plotting = PlottingResiduals(analysis)
    figure = plotting.plot(
        "Phi_projection",
        residual_type="pearson",
        n_simulations=20,
        random_seed=7,
    )
    distribution_axis = figure.axes[-1]
    observed_bars = distribution_axis.containers[0].patches
    observed_edges = np.array(
        [observed_bars[0].get_x()]
        + [bar.get_x() + bar.get_width() for bar in observed_bars]
    )
    simulated_step = next(
        patch
        for patch in distribution_axis.patches
        if patch.get_label() == "Poisson MC mean density"
    )
    simulated_edges = simulated_step.get_data()[1]

    np.testing.assert_allclose(observed_edges, simulated_edges)
    plt.close(figure)


def test_plotter_prepares_pearson_reference_automatically_and_reuses_it(
    monkeypatch,
):
    """Plotter settings prepare the reference once without an explicit call."""
    monkeypatch.setattr(plt, "show", lambda: None)
    analysis = make_poisson_reference_analysis(np.full(12, 4.0))
    projection_calls = 0
    original_project = analysis._project_simulated_counts

    def count_projection_calls(simulated_counts, projection_id):
        nonlocal projection_calls
        projection_calls += 1
        return original_project(simulated_counts, projection_id)

    monkeypatch.setattr(analysis, "_project_simulated_counts", count_projection_calls)
    plotting = PlottingResiduals(
        analysis,
        n_simulations=5,
        random_seed=13,
    )

    first_figure = plotting.plot("Phi_projection", residual_type="pearson")
    second_figure = plotting.plot("Phi_projection", residual_type="pearson")

    assert projection_calls == 5
    assert len(analysis._poisson_reference_cache) == 1
    plt.close(first_figure)
    plt.close(second_figure)


def test_constructor_phi_slice_does_not_affect_integrated_psichi_plot(monkeypatch):
    """A configured slice index is only passed to the Phi-sliced map plot."""
    analysis = make_poisson_reference_analysis(np.full(12, 4.0))
    products = analysis._projected_data_cache["Phi_projection"]
    analysis._projected_data_cache["PsiChi_projection"] = products
    plotting = PlottingResiduals(analysis, phi_bin_index=30)
    monkeypatch.setattr(
        plotting,
        "_plot_psichi_projection",
        lambda *args: args[2],
    )

    result = plotting.plot("PsiChi_projection", residual_type="pearson")

    assert result is None


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


def test_pearson_table_reports_compact_full_cds_statistics():
    """The Pearson table reports only the key full-CDS fit summaries."""
    observed = make_histogram([5.0, 0.0])
    expected = make_histogram([4.0, 0.0])
    analysis = ResidualAnalysis(observed, expected)
    analysis._projected_data_cache = {
        "Phi_projection": {"observed": observed, "expected": expected}
    }
    table = analysis.statistics_table("pearson")

    assert len(table) == 1
    assert table.loc[0, "Valid bins (N)"] == 1
    assert table.loc[0, "Mean residual"] == pytest.approx(0.5)
    assert table.loc[0, "Median residual"] == pytest.approx(0.5)
    assert table.loc[0, "Std. dev."] == pytest.approx(0.0)
    assert table.loc[0, "Residual RMS"] == pytest.approx(0.5)
    assert table.loc[0, "Pearson chi-square"] == pytest.approx(0.25)
    expected_deviance = 2 * (5 * np.log(5 / 4) - (5 - 4))
    assert table.loc[0, "Poisson deviance"] == pytest.approx(expected_deviance)
    assert table.loc[0, "|R| > 2 [%]"] == 0.0
    assert table.loc[0, "|R| > 3 [%]"] == 0.0


def test_statistics_tables_report_one_compact_summary_per_residual_type():
    """Each residual table has one row with only useful summary metrics."""
    observed = make_histogram([5.0, 0.0])
    expected = make_histogram([4.0, 0.0])
    analysis = ResidualAnalysis(observed, expected)
    analysis._projected_data_cache = {
        "Phi_projection": {"observed": observed, "expected": expected}
    }

    raw_table = analysis.statistics_table("raw")
    pearson_table = analysis.statistics_table("pearson")
    fractional_table = analysis.statistics_table("fractional")

    assert len(raw_table) == len(pearson_table) == len(fractional_table) == 1
    assert raw_table.loc[0, "Valid bins (N)"] == 1
    assert raw_table.loc[0, "Mean residual"] == pytest.approx(1.0)
    assert raw_table.loc[0, "Median residual"] == pytest.approx(1.0)
    assert raw_table.loc[0, "Std. dev."] == pytest.approx(0.0)
    assert raw_table.loc[0, "Residual RMS"] == pytest.approx(1.0)
    assert raw_table.loc[0, "Minimum residual"] == pytest.approx(1.0)
    assert raw_table.loc[0, "Maximum residual"] == pytest.approx(1.0)
    assert "Pearson chi-square" not in raw_table
    assert pearson_table.loc[0, "Pearson chi-square"] == pytest.approx(0.25)
    assert fractional_table.loc[0, "Mean residual"] == pytest.approx(0.25)
    assert fractional_table.loc[0, "Residual RMS"] == pytest.approx(0.25)
    assert "Pearson chi-square" not in fractional_table

    with pytest.raises(ValueError, match="Unknown residual type"):
        analysis.statistics_table("unsupported")


def test_public_projection_and_pearson_workflow_preserves_aggregated_counts():
    """Public projection and residual methods use aggregated CDS counts."""
    phi_axis = Axis([0.0, 1.0, 2.0], label="Phi", unit=u.deg)
    psichi_axis = HealpixAxis(nside=1, coordsys="C", label="PsiChi")
    observed_values = np.arange(1.0, 25.0).reshape(2, 12)
    expected_values = np.full((2, 12), 2.0)
    observed = Histogram(
        [phi_axis, psichi_axis],
        contents=observed_values,
    )
    expected = Histogram(
        [phi_axis, psichi_axis],
        contents=expected_values,
    )
    analysis = ResidualAnalysis(observed, expected)

    projected_data = analysis.projected_data()
    for products in projected_data.values():
        assert products["observed"].contents.sum() == pytest.approx(
            observed_values.sum()
        )
        assert products["expected"].contents.sum() == pytest.approx(
            expected_values.sum()
        )

    products = projected_data["Phi_Chi_projection"]
    pearson = analysis.residual_function("Phi_Chi_projection", "pearson")
    expected_values = products["expected"].contents
    expected_pearson = np.full(expected_values.shape, np.nan)
    valid_bins = expected_values > 0
    expected_pearson[valid_bins] = (
        products["observed"].contents[valid_bins] - expected_values[valid_bins]
    ) / np.sqrt(expected_values[valid_bins])
    np.testing.assert_allclose(
        pearson.contents,
        expected_pearson,
        equal_nan=True,
    )


def test_multi_energy_projection_preserves_energy_in_cds_views():
    """Energy-resolved CDS projections retain every original Em bin."""
    em_axis = Axis([300.0, 400.0, 600.0], label="Em", unit=u.keV)
    phi_axis = Axis([0.0, 1.0, 2.0], label="Phi", unit=u.deg)
    psichi_axis = HealpixAxis(nside=1, coordsys="C", label="PsiChi")

    observed_values = np.arange(1.0, 49.0).reshape(2, 2, 12)
    expected_values = np.full((2, 2, 12), 2.0)
    observed = Histogram(
        [em_axis, phi_axis, psichi_axis],
        contents=observed_values,
    )
    expected = Histogram(
        [em_axis, phi_axis, psichi_axis],
        contents=expected_values,
    )
    analysis = ResidualAnalysis(observed, expected)

    projected_data = analysis.projected_data()
    em_products = projected_data["Em_projection"]
    raw_residual = analysis.residual_function("Em_projection", "raw")
    expected_observed_counts = observed_values.sum(axis=(1, 2))
    expected_model_counts = expected_values.sum(axis=(1, 2))
    expected_projection_ids = {
        "Em_projection",
        "Em_Phi_projection",
        "Em_Chi_projection",
        "Em_Psi_projection",
        "Em_PsiChi_projection",
    }

    assert list(em_products["observed"].axes.labels) == ["Em"]
    np.testing.assert_allclose(
        em_products["observed"].contents,
        expected_observed_counts,
    )
    np.testing.assert_allclose(
        em_products["expected"].contents,
        expected_model_counts,
    )
    np.testing.assert_allclose(
        raw_residual.contents,
        expected_observed_counts - expected_model_counts,
    )
    assert set(projected_data) == expected_projection_ids

    np.testing.assert_allclose(
        projected_data["Em_Phi_projection"]["observed"].contents,
        observed_values.sum(axis=2),
    )
    np.testing.assert_allclose(
        projected_data["Em_Phi_projection"]["expected"].contents,
        expected_values.sum(axis=2),
    )
    np.testing.assert_allclose(
        projected_data["Em_PsiChi_projection"]["observed"].contents,
        observed_values.sum(axis=1),
    )

    original_totals_by_energy = observed_values.sum(axis=(1, 2))
    for projection_id in ("Em_Chi_projection", "Em_Psi_projection"):
        np.testing.assert_allclose(
            projected_data[projection_id]["observed"].contents.sum(axis=1),
            original_totals_by_energy,
        )


def test_multi_energy_residuals_and_plots_use_new_projection_ids(monkeypatch):
    """Energy-resolved products use existing residual and plotting interfaces."""
    monkeypatch.setattr(plt, "show", lambda: None)
    em_axis = Axis([300.0, 400.0, 600.0], label="Em", unit=u.keV)
    phi_axis = Axis([0.0, 1.0, 2.0], label="Phi", unit=u.deg)
    psichi_axis = HealpixAxis(nside=1, coordsys="C", label="PsiChi")
    expected_values = np.full((2, 2, 12), 4.0)
    observed_values = expected_values + 2.0
    analysis = ResidualAnalysis(
        Histogram(
            [em_axis, phi_axis, psichi_axis],
            contents=observed_values,
        ),
        Histogram(
            [em_axis, phi_axis, psichi_axis],
            contents=expected_values,
        ),
    )
    plotting = PlottingResiduals(analysis)

    for projection_id in (
        "Em_Phi_projection",
        "Em_Chi_projection",
        "Em_Psi_projection",
    ):
        residual = analysis.residual_function(projection_id, "pearson")
        assert np.any(np.isfinite(residual.contents))
        assert not np.any(np.isinf(residual.contents))
        figure = plotting.plot(projection_id, residual_type="pearson")
        assert figure is not None
        plt.close(figure)

    with pytest.raises(ValueError, match="Provide energy_bin_index"):
        plotting.plot("Em_PsiChi_projection", residual_type="raw")

    figure = plotting.plot(
        "Em_PsiChi_projection",
        residual_type="raw",
        energy_bin_index=1,
    )
    assert figure is not None
    plt.close(figure)


def test_multi_energy_psichi_plot_supports_poisson_reference(monkeypatch):
    """An Em-PsiChi slice reuses the fixed-fit Pearson simulation pipeline."""
    monkeypatch.setattr(plt, "show", lambda: None)
    em_axis = Axis([300.0, 400.0, 600.0], label="Em", unit=u.keV)
    phi_axis = Axis([0.0, 1.0, 2.0], label="Phi", unit=u.deg)
    psichi_axis = HealpixAxis(nside=1, coordsys="C", label="PsiChi")
    expected_values = np.full((2, 2, 12), 4.0)
    analysis = ResidualAnalysis(
        Histogram(
            [em_axis, phi_axis, psichi_axis],
            contents=expected_values.copy(),
        ),
        Histogram(
            [em_axis, phi_axis, psichi_axis],
            contents=expected_values,
        ),
    )
    plotting = PlottingResiduals(
        analysis,
        n_simulations=2,
        random_seed=42,
        energy_bin_index=1,
    )

    figure = plotting.plot(
        "Em_PsiChi_projection",
        residual_type="pearson",
    )

    assert figure is not None
    assert len(analysis._poisson_reference_cache) == 1
    assert next(iter(analysis._poisson_reference_cache)) == (
        "Em_PsiChi_projection",
        2,
        40,
        42,
        None,
        1,
        "poisson",
    )
    plt.close(figure)


def test_select_energy_range_aggregates_selected_cds_counts():
    """An energy interval is aggregated before making CDS projections."""
    em_axis = Axis([300.0, 400.0, 600.0], label="Em", unit=u.keV)
    phi_axis = Axis([0.0, 1.0, 2.0], label="Phi", unit=u.deg)
    psichi_axis = HealpixAxis(nside=1, coordsys="C", label="PsiChi")
    observed_values = np.arange(1.0, 49.0).reshape(2, 2, 12)
    expected_values = np.full((2, 2, 12), 2.0)
    analysis = ResidualAnalysis(
        Histogram(
            [em_axis, phi_axis, psichi_axis],
            contents=observed_values,
        ),
        Histogram(
            [em_axis, phi_axis, psichi_axis],
            contents=expected_values,
        ),
    )

    selected_analysis = analysis.select_energy_range(300.0, 600.0)
    selected_products = selected_analysis.projected_data()

    assert list(selected_analysis.observed_hist.axes.labels) == [
        "Em",
        "Phi",
        "PsiChi",
    ]
    assert selected_analysis.observed_hist.shape == (1, 2, 12)
    np.testing.assert_allclose(
        selected_analysis.observed_hist.contents,
        observed_values.sum(axis=0, keepdims=True),
    )
    np.testing.assert_allclose(
        selected_analysis.expected_hist.contents,
        expected_values.sum(axis=0, keepdims=True),
    )
    np.testing.assert_allclose(
        selected_products["Phi_projection"]["observed"].contents,
        observed_values.sum(axis=(0, 2)),
    )
    np.testing.assert_allclose(
        selected_products["Phi_projection"]["expected"].contents,
        expected_values.sum(axis=(0, 2)),
    )
    assert selected_products["Em_projection"]["observed"].shape == (1,)


def test_select_energy_range_validates_limits_and_preserves_energy_edges():
    """Energy selection validates and retains exact Em edge intervals."""
    em_axis = Axis([300.0, 400.0, 600.0], label="Em", unit=u.keV)
    phi_axis = Axis([0.0, 1.0, 2.0], label="Phi", unit=u.deg)
    psichi_axis = HealpixAxis(nside=1, coordsys="C", label="PsiChi")
    values = np.ones((2, 2, 12))
    histogram = Histogram(
        [em_axis, phi_axis, psichi_axis],
        contents=values,
    )
    analysis = ResidualAnalysis(histogram, histogram.copy())

    for energy_min, energy_max in ((300.0, 400.0), (400.0, 600.0)):
        selected = analysis.select_energy_range(energy_min, energy_max)
        np.testing.assert_allclose(
            selected.observed_hist.axes["Em"].edges.to_value(u.keV),
            [energy_min, energy_max],
        )

    with pytest.raises(TypeError, match="must be numeric values"):
        analysis.select_energy_range(True, 400.0)
    with pytest.raises(ValueError, match="must be finite"):
        analysis.select_energy_range(300.0, np.inf)
    with pytest.raises(ValueError, match="must be smaller"):
        analysis.select_energy_range(400.0, 300.0)
    with pytest.raises(ValueError, match="Available edges in keV"):
        analysis.select_energy_range(350.0, 600.0)

    no_energy_analysis = ResidualAnalysis(
        make_histogram([1.0, 1.0]),
        make_histogram([1.0, 1.0]),
    )
    with pytest.raises(ValueError, match="do not contain an Em axis"):
        no_energy_analysis.select_energy_range(300.0, 400.0)


def test_data_summary_describes_geometry_mask_and_counts():
    """Distinguish native dimensions, excluded bins, and retained counts."""
    axes = [
        Axis([300.0, 400.0, 600.0], label="Em", unit=u.keV),
        Axis([0.0, 1.0, 2.0], label="Phi", unit=u.deg),
        HealpixAxis(nside=1, coordsys="C", label="PsiChi"),
    ]
    observed = np.full((2, 2, 12), 3.0)
    expected = np.full((2, 2, 12), 4.0)
    observed[0, 0, 0] = 0.0
    expected[0, 0, 0] = 0.5
    analysis = ResidualAnalysis(
        Histogram(axes, contents=observed),
        Histogram(axes, contents=expected),
        expected_floor=0.5,
    )
    summary = analysis.data_summary_table().set_index("Statistic")["Value"]
    assert summary["Em bins"] == 2
    assert summary["Em edges [keV]"] == (300.0, 400.0, 600.0)
    assert summary["Phi bins"] == 2
    assert summary["Phi range [deg]"] == (0.0, 2.0)
    assert summary["PsiChi bins"] == 12
    assert summary["PsiChi HEALPix nside"] == 1
    assert summary["Total bins"] == 48
    assert summary["Valid bins"] == 47
    assert summary["Excluded bins"] == 1
    assert summary["Observed counts (all bins)"] == 141.0
    assert summary["Expected counts (all bins)"] == 188.5
    assert summary["Observed counts (valid bins)"] == 141.0
    assert summary["Expected counts (valid bins)"] == 188.0

    # A selected analysis reports its already aggregated data.
    selected = analysis.select_energy_range(300.0, 600.0)
    selected_summary = selected.data_summary_table().set_index("Statistic")["Value"]
    assert selected_summary["Em bins"] == 1
    assert selected_summary["Total bins"] == 24
    assert selected_summary["Observed counts (valid bins)"] == 141.0
    assert selected_summary["Expected counts (valid bins)"] == 188.0
    assert analysis.observed_hist.axes["Em"].nbins == 2


def test_selected_energy_statistics_use_only_the_selected_range():
    """Full-CDS summaries exclude counts outside the selected energy range."""
    em_axis = Axis([300.0, 400.0, 600.0], label="Em", unit=u.keV)
    phi_axis = Axis([0.0, 1.0, 2.0], label="Phi", unit=u.deg)
    psichi_axis = HealpixAxis(nside=1, coordsys="C", label="PsiChi")
    expected_values = np.full((2, 2, 12), 4.0)
    observed_values = expected_values.copy()
    observed_values[1] = 6.0
    analysis = ResidualAnalysis(
        Histogram(
            [em_axis, phi_axis, psichi_axis],
            contents=observed_values,
        ),
        Histogram(
            [em_axis, phi_axis, psichi_axis],
            contents=expected_values,
        ),
    )

    with pytest.raises(ValueError, match="Select an energy interval"):
        analysis.statistics_table("pearson")

    selected_table = analysis.select_energy_range(
        400.0,
        600.0,
    ).statistics_table("pearson")

    assert selected_table.loc[0, "Valid bins (N)"] == 24
    assert selected_table.loc[0, "Mean residual"] == pytest.approx(1.0)
    assert selected_table.loc[0, "Pearson chi-square"] == pytest.approx(24.0)


def test_selected_energy_poisson_draw_uses_only_selected_expectation(monkeypatch):
    """Poisson simulations use only the selected energy-range expectation."""
    em_axis = Axis([300.0, 400.0, 600.0], label="Em", unit=u.keV)
    phi_axis = Axis([0.0, 1.0, 2.0], label="Phi", unit=u.deg)
    psichi_axis = HealpixAxis(nside=1, coordsys="C", label="PsiChi")
    expected_values = np.stack(
        [np.full((2, 12), 2.0), np.full((2, 12), 9.0)]
    )
    expected = Histogram(
        [em_axis, phi_axis, psichi_axis],
        contents=expected_values,
    )
    analysis = ResidualAnalysis(expected.copy(), expected)
    selected = analysis.select_energy_range(400.0, 600.0)
    poisson_inputs = []

    class RecordingGenerator:
        """Record each Poisson expectation and return its mean counts."""

        def poisson(self, expected_counts):
            poisson_inputs.append(expected_counts.copy())
            return expected_counts.copy()

    monkeypatch.setattr(
        np.random,
        "default_rng",
        lambda seed: RecordingGenerator(),
    )
    selected.pearson_poisson_reference(
        "Phi_projection",
        n_simulations=2,
        bins=4,
        seed=7,
    )

    assert len(poisson_inputs) == 2
    for poisson_input in poisson_inputs:
        assert poisson_input.shape == (1, 2, 12)
        np.testing.assert_allclose(poisson_input, 9.0)


def test_single_energy_bin_preserves_existing_cds_projections():
    """A singleton Em axis keeps the existing 511-keV CDS workflow."""
    em_axis = Axis([509.0, 513.0], label="Em", unit=u.keV)
    phi_axis = Axis([0.0, 1.0, 2.0], label="Phi", unit=u.deg)
    psichi_axis = HealpixAxis(nside=1, coordsys="C", label="PsiChi")
    values = np.full((1, 2, 12), 2.0)
    histogram = Histogram(
        [em_axis, phi_axis, psichi_axis],
        contents=values,
    )

    analysis = ResidualAnalysis(
        histogram,
        histogram.copy(),
    )
    projected_data = analysis.projected_data()
    selected_analysis = analysis.select_energy_range(509.0, 513.0)

    assert "Em_projection" in projected_data
    assert "Phi_projection" in projected_data
    assert "PsiChi_projection" in projected_data
    np.testing.assert_allclose(
        selected_analysis.observed_hist.contents,
        histogram.contents,
    )


def test_energy_plot_uses_native_units_and_true_bin_edges(monkeypatch):
    """The Em count plot uses keV and the histogram's actual boundaries."""
    monkeypatch.setattr(plt, "show", lambda: None)
    energy_edges = np.array([300.0, 400.0, 600.0, 800.0])
    em_axis = Axis(energy_edges, label="Em", unit=u.keV)
    observed = Histogram(
        [em_axis],
        contents=np.array([12.0, 18.0, 13.0]),
    )
    expected = Histogram(
        [em_axis],
        contents=np.array([11.0, 19.0, 14.0]),
    )
    analysis = ResidualAnalysis(observed, expected)
    analysis._projected_data_cache = {
        "Em_projection": {"observed": observed, "expected": expected}
    }

    figure = PlottingResiduals(analysis).plot(
        "Em_projection",
        residual_type="raw",
    )
    observed_stairs = figure.axes[0].patches[0]

    np.testing.assert_allclose(observed_stairs.get_data().edges, energy_edges)
    assert figure.axes[1].get_xlabel() == "Em [keV]"
    plt.close(figure)


def test_energy_pearson_plot_supports_poisson_reference(monkeypatch):
    """The Em Pearson plot uses the matching projected Poisson simulations."""
    monkeypatch.setattr(plt, "show", lambda: None)
    em_axis = Axis([300.0, 400.0, 600.0, 800.0], label="Em", unit=u.keV)
    observed = Histogram([em_axis], contents=np.array([12.0, 18.0, 13.0]))
    expected = Histogram([em_axis], contents=np.array([11.0, 19.0, 14.0]))
    analysis = ResidualAnalysis(observed, expected)
    analysis._projected_data_cache = {
        "Em_projection": {"observed": observed, "expected": expected}
    }

    figure = PlottingResiduals(
        analysis,
        n_simulations=4,
        random_seed=42,
    ).plot("Em_projection", residual_type="pearson")

    assert len(analysis._poisson_reference_cache) == 1
    legend_labels = figure.axes[-1].get_legend_handles_labels()[1]
    assert "Poisson MC mean density" in legend_labels
    assert r"Standard normal $N(0,1)$" not in legend_labels
    plt.close(figure)

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
def test_time_residual_selects_and_normalizes_requested_interval():
    """Select one Time bin and normalize the integrated model to its counts."""
    time_axis = Axis([0.0, 10.0, 20.0], label="Time", unit=u.s)
    phi_axis = Axis([0.0, 1.0, 2.0], label="Phi", unit=u.deg)

    # The Time projection reproduces the integrated observed counts [5, 5].
    time_binned_observed = Histogram(
        [time_axis, phi_axis],
        contents=np.array(
            [
                [2.0, 4.0],
                [3.0, 1.0],
            ]
        ),
    )

    integrated_observed = Histogram(
        [phi_axis],
        contents=np.array([5.0, 5.0]),
    )
    integrated_expected = Histogram(
        [phi_axis],
        contents=np.array([3.0, 7.0]),
    )

    analysis = ResidualAnalysis(
        observed=integrated_observed,
        expected=integrated_expected,
        time_residual=True,
        time_binned_observed=time_binned_observed,
        time_bin_index=1,
    )

    np.testing.assert_array_equal(
        analysis.observed_hist.contents,
        [3.0, 1.0],
    )
    np.testing.assert_allclose(
        analysis.expected_hist.contents,
        [1.2, 2.8],
    )

    assert tuple(analysis.observed_hist.axes.labels) == ("Phi",)
    assert analysis.time_bin_index == 1
    assert analysis.time_start == 10.0 * u.s
    assert analysis.time_stop == 20.0 * u.s

def test_time_residual_requires_temporal_inputs():
    """Require time-binned data and a valid integer Time-bin index."""
    histogram = make_histogram([2.0, 3.0])

    with pytest.raises(
        ValueError,
        match="time_binned_observed is required",
    ):
        ResidualAnalysis(
            observed=histogram,
            expected=histogram.copy(),
            time_residual=True,
            time_bin_index=0,
        )

    with pytest.raises(
        TypeError,
        match="time_bin_index must be an integer",
    ):
        ResidualAnalysis(
            observed=histogram,
            expected=histogram.copy(),
            time_residual=True,
            time_binned_observed=histogram,
            time_bin_index=True,
        )


def test_temporal_1d_plot_labels_model_and_utc_interval():
    """Temporal count plots identify D(t,i), M_tilde(t,i), and UTC bounds."""
    analysis = make_poisson_reference_analysis(
        expected_values=[3.0, 4.0, 5.0],
        observed_values=[2.0, 5.0, 5.0],
    )
    analysis.time_bin_index = 5
    analysis.time_start = 3600.0 * u.s
    analysis.time_stop = 7200.0 * u.s
    analysis._temporal_total_counts = 12
    analysis._temporal_reference_probabilities = np.array(
        [0.25, 1.0 / 3.0, 5.0 / 12.0]
    )

    figure = PlottingResiduals(analysis).plot(
        "Phi_projection",
        residual_type="raw",
    )

    counts_axis = figure.axes[0]
    title = counts_axis.get_title()
    legend_labels = [
        text.get_text()
        for text in counts_axis.get_legend().get_texts()
    ]

    assert "Time bin 5" in title
    assert "1970-01-01 01:00:00–1970-01-01 02:00:00 UTC" in title
    assert r"Observed $D_{t,i}$" in legend_labels
    assert r"Normalized model $\widetilde{M}_{t,i}$" in legend_labels
    assert r"Multinomial uncertainty ($\pm 1\sigma$)" in legend_labels
    plt.close(figure)


def test_pearson_plot_uses_limegreen_sigma_guides():
    """Pearson plots use a three-sigma band and distinct inner guide lines."""
    analysis = make_poisson_reference_analysis(
        expected_values=[3.0, 4.0, 5.0],
        observed_values=[2.0, 5.0, 5.0],
    )

    figure = PlottingResiduals(analysis).plot(
        "Phi_projection",
        residual_type="pearson",
        n_simulations=4,
        random_seed=17,
    )
    residual_axis = figure.axes[1]

    sigma_band = next(
        patch
        for patch in residual_axis.patches
        if patch.get_label() == r"$\pm3\sigma$ range"
    )
    np.testing.assert_allclose(
        sigma_band.get_facecolor(),
        matplotlib.colors.to_rgba("limegreen", alpha=0.15),
    )
    assert residual_axis.get_ylim() == (-5.0, 5.0)

    labeled_lines = {
        line.get_label(): line
        for line in residual_axis.lines
    }
    assert labeled_lines[r"$\pm1\sigma$"].get_color() == "limegreen"
    assert labeled_lines[r"$\pm1\sigma$"].get_linestyle() == ":"
    assert labeled_lines[r"$\pm2\sigma$"].get_color() == "limegreen"
    assert labeled_lines[r"$\pm2\sigma$"].get_linestyle() == "--"
    counts_legend_labels = [
        text.get_text()
        for text in figure.axes[0].get_legend().get_texts()
    ]
    assert r"Poisson uncertainty ($\pm 1\sigma$)" in counts_legend_labels
    plt.close(figure)


def test_temporal_reference_uses_fixed_total_multinomial_after_energy_selection(
    monkeypatch,
):
    """Temporal MC keeps full N(t) and the probability outside an Em range."""
    time_axis = Axis([0.0, 10.0, 20.0], label="Time", unit=u.s)
    em_axis = Axis([100.0, 200.0, 300.0], label="Em", unit=u.keV)
    phi_axis = Axis([0.0, 1.0, 2.0], label="Phi", unit=u.deg)

    time_binned_observed = Histogram(
        [time_axis, em_axis, phi_axis],
        contents=np.array(
            [
                [[1.0, 1.0], [2.0, 2.0]],
                [[1.0, 3.0], [2.0, 4.0]],
            ]
        ),
    )
    integrated_observed = Histogram(
        [em_axis, phi_axis],
        contents=np.array([[2.0, 4.0], [4.0, 6.0]]),
    )
    integrated_expected = Histogram(
        [em_axis, phi_axis],
        contents=np.array([[1.0, 2.0], [3.0, 4.0]]),
    )

    temporal_analysis = ResidualAnalysis(
        observed=integrated_observed,
        expected=integrated_expected,
        time_residual=True,
        time_binned_observed=time_binned_observed,
        time_bin_index=1,
    )
    selected_analysis = temporal_analysis.select_energy_range(100.0, 200.0)
    selected_analysis._projected_data_cache = {
        "Phi_projection": selected_analysis.projection_function("Phi")
    }

    original_default_rng = np.random.default_rng
    multinomial_calls = []

    class CountingGenerator:
        """Record multinomial inputs while retaining NumPy draws."""

        def __init__(self, seed):
            self.generator = original_default_rng(seed)

        def multinomial(self, total, probabilities):
            multinomial_calls.append(
                (total, np.asarray(probabilities).copy())
            )
            return self.generator.multinomial(total, probabilities)

        def poisson(self, expected_counts):
            raise AssertionError(
                "Temporal references must not use Poisson draws."
            )

    monkeypatch.setattr(
        np.random,
        "default_rng",
        lambda seed=None: CountingGenerator(seed),
    )

    selected_analysis.pearson_poisson_reference(
        "Phi_projection",
        n_simulations=4,
        bins=5,
        seed=17,
    )

    assert len(multinomial_calls) == 4
    for total, probabilities in multinomial_calls:
        assert total == 10
        np.testing.assert_allclose(probabilities, [0.1, 0.2, 0.7])

    plotter = PlottingResiduals(
        selected_analysis,
        n_simulations=4,
        random_seed=17,
    )
    figure = plotter.plot(
        "Phi_projection",
        residual_type="pearson",
    )
    distribution_labels = [
        text.get_text()
        for text in figure.axes[2].get_legend().get_texts()
    ]
    assert "Multinomial MC mean density" in distribution_labels
    plt.close(figure)
