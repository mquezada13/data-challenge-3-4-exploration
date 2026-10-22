"""Plot residual diagnostics for histogram projections."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
import astropy.units as u
from histpy import Histogram
import mhealpy as mhp


class PlottingResiduals:
    """Create residual diagnostic figures from a residual analysis.

    The class supports one-dimensional, rectangular two-dimensional, and
    HEALPix residual views prepared by a ``ResidualAnalysis`` instance.
    """

    _PEARSON_REFERENCE_BINS = {
        "Em_projection": 15,
        "Em_Phi_projection": 40,
        "Em_Chi_projection": 40,
        "Em_Psi_projection": 40,
        "Em_PsiChi_projection": 40,
        "Phi_projection": 15,
        "Chi_projection": 15,
        "Psi_projection": 15,
        "Phi_Chi_projection": 40,
        "Phi_Psi_projection": 40,
        "PsiChi_projection": 40,
        "Phi_PsiChi_projection": 40,
    }

    def __init__(
        self,
        analysis,
        n_simulations: int | None = None,
        random_seed: int | None = None,
        phi_bin_index: int | None = None,
        energy_bin_index: int | None = None,
    ) -> None:
        """Store the analysis and optional Pearson simulation settings.

        Args:
            analysis: A ``ResidualAnalysis`` instance containing the observed
                and expected histograms.
            n_simulations: Optional number of Poisson realizations used for
                Pearson reference histograms. ``None`` keeps simulations off.
            random_seed: Optional seed shared by the simulated references.
            phi_bin_index: Optional Phi slice prepared with the other
                references and used by ``Phi_PsiChi_projection`` by default.
            energy_bin_index: Optional Em slice used by
                ``Em_PsiChi_projection`` by default.
        """
        self.analysis = analysis
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        self.phi_bin_index = phi_bin_index
        self.energy_bin_index = energy_bin_index
        self._generated_seed = None
        self._prepared_reference_configs = set()

    def _format_energy_interval_suffix(self) -> str:
        """Format the analysis energy interval for angular plot titles."""
        observed_hist = self.analysis.observed_hist
        if "Em" not in observed_hist.axes.labels:
            return ""

        energy_edges = observed_hist.axes["Em"].edges.to_value(u.keV)
        return f" — {energy_edges[0]:g}–{energy_edges[-1]:g} keV"

    def _prepare_pearson_references(
        self,
        n_simulations: int,
        random_seed: int,
        phi_bin_index: int | None,
        energy_bin_index: int | None,
    ) -> None:
        """Prepare and cache the supported Pearson references once per setup."""
        configuration = (
            n_simulations,
            random_seed,
            phi_bin_index,
            energy_bin_index,
        )
        if configuration in self._prepared_reference_configs:
            return

        available_projections = self.analysis.projected_data()
        projection_bins = {
            projection_id: bins
            for projection_id, bins in self._PEARSON_REFERENCE_BINS.items()
            if projection_id in available_projections
            and (
                projection_id != "Phi_PsiChi_projection"
                or phi_bin_index is not None
            )
            and (
                projection_id != "Em_PsiChi_projection"
                or energy_bin_index is not None
            )
        }
        phi_bin_indices = (
            {"Phi_PsiChi_projection": phi_bin_index}
            if "Phi_PsiChi_projection" in projection_bins
            else None
        )
        energy_bin_indices = (
            {"Em_PsiChi_projection": energy_bin_index}
            if "Em_PsiChi_projection" in projection_bins
            else None
        )
        self.analysis.prepare_pearson_poisson_references(
            projection_bins,
            n_simulations=n_simulations,
            seed=random_seed,
            phi_bin_indices=phi_bin_indices,
            energy_bin_indices=energy_bin_indices,
        )
        self._prepared_reference_configs.add(configuration)


    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    # Plot function
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------

    def plot(
        self,
        projection_id: str,
        residual_type: str = "raw",
        phi_bin_index: int | None = None,
        running_mean_window: int | None = None,
        n_simulations: int | None = None,
        random_seed: int | None = None,
        show_gaussian_reference: bool | None = None,
        energy_bin_index: int | None = None,
    ) -> Figure:
        """Create a residual figure for a selected projection.

        Args:
            projection_id: Key identifying a projection returned by
                ``ResidualAnalysis.projected_data``.
            residual_type: Residual definition: ``"raw"``, ``"pearson"``, or
                ``"fractional"``.
            phi_bin_index: Phi-bin index for a slice of
                ``"Phi_PsiChi_projection"``. Leave as ``None`` for
                ``"PsiChi_projection"`` and other projections.
            running_mean_window: Optional centered window size in bins for a
                Pearson residual running mean. Windows are truncated at the
                first and last bins; they do not wrap around the coordinate.
            n_simulations: Number of fixed-fit Poisson realizations for the
                Pearson residual distribution. If omitted, the constructor
                setting is used; ``None`` keeps simulations off.
            random_seed: Optional seed for reproducible Poisson realizations.
                If omitted, the constructor setting is used. When simulations
                are enabled without a seed, one seed is generated and reused.
            show_gaussian_reference: Whether to overlay the theoretical
                standard normal curve. By default it is shown without
                simulations and hidden when simulations are requested.
            energy_bin_index: Em-bin index for a slice of
                ``"Em_PsiChi_projection"``.

        Returns:
            The Matplotlib figure containing the requested diagnostic.

        Raises:
            ValueError: If the projection is unknown or a required slice index
                is missing.
            IndexError: If a Phi or Em slice index is outside the available
                bins.
            NotImplementedError: If the projection does not have a plotter.
        """
        projected_data = self.analysis.projected_data()

        if running_mean_window is not None:
            if (
                isinstance(running_mean_window, bool)
                or not isinstance(running_mean_window, int)
                or running_mean_window < 1
            ):
                raise ValueError("running_mean_window must be a positive integer.")
            if residual_type != "pearson":
                raise ValueError(
                    "running_mean_window is only available for Pearson residuals."
                )

        if n_simulations is not None and residual_type != "pearson":
            raise ValueError(
                "Poisson simulations are only available for Pearson residuals."
            )
        if show_gaussian_reference is not None and residual_type != "pearson":
            raise ValueError(
                "show_gaussian_reference is only available for Pearson residuals."
            )
        plot_n_simulations = None
        plot_random_seed = None
        reference_phi_bin_index = self.phi_bin_index
        reference_energy_bin_index = self.energy_bin_index
        if projection_id == "Phi_PsiChi_projection":
            if phi_bin_index is None:
                phi_bin_index = self.phi_bin_index
            reference_phi_bin_index = phi_bin_index
        if projection_id == "Em_PsiChi_projection":
            if energy_bin_index is None:
                energy_bin_index = self.energy_bin_index
            reference_energy_bin_index = energy_bin_index
        if residual_type == "pearson":
            plot_n_simulations = (
                n_simulations if n_simulations is not None else self.n_simulations
            )
            plot_random_seed = (
                random_seed if random_seed is not None else self.random_seed
            )
            if plot_n_simulations is not None:
                if plot_random_seed is None:
                    if self._generated_seed is None:
                        self._generated_seed = int(
                            np.random.SeedSequence().generate_state(1)[0]
                        )
                    plot_random_seed = self._generated_seed
                self._prepare_pearson_references(
                    plot_n_simulations,
                    plot_random_seed,
                    reference_phi_bin_index,
                    reference_energy_bin_index,
                )

        if show_gaussian_reference is None:
            show_gaussian = residual_type == "pearson" and plot_n_simulations is None
        else:
            show_gaussian = show_gaussian_reference

        if projection_id not in projected_data:
            raise ValueError(f"Unknown projection: {projection_id}")

        # Route either joint rectangular projection to the paired 2D plot.
        two_dimensional_ids = {
            "Phi_Chi_projection",
            "Phi_Psi_projection",
        }
        if projection_id in two_dimensional_ids:
            if running_mean_window is not None:
                raise ValueError("The running mean is only available for 1D plots.")
            return self._plot_2d_projections(
                residual_type,
                plot_n_simulations,
                plot_random_seed,
                show_gaussian,
            )

        energy_coordinate_ids = {
            "Em_Phi_projection",
            "Em_Chi_projection",
            "Em_Psi_projection",
        }
        if projection_id in energy_coordinate_ids:
            if running_mean_window is not None:
                raise ValueError("The running mean is only available for 1D plots.")
            return self._plot_energy_coordinate_projection(
                projection_id,
                residual_type,
                plot_n_simulations,
                plot_random_seed,
                show_gaussian,
            )

        # Route HEALPix projections to their shared plotter.
        healpix_ids = {
            "PsiChi_projection",
            "Phi_PsiChi_projection",
            "Em_PsiChi_projection",
        }
        if projection_id in healpix_ids:
            if running_mean_window is not None:
                raise ValueError("The running mean is only available for 1D plots.")
            return self._plot_psichi_projection(
                projection_id,
                residual_type,
                phi_bin_index,
                energy_bin_index,
                plot_n_simulations,
                plot_random_seed,
                show_gaussian,
            )

        # Route supported one-dimensional projections to their plotter.
        one_dimensional_ids = {
            "Em_projection",
            "Phi_projection",
            "Chi_projection",
            "Psi_projection",
        }
        if projection_id not in one_dimensional_ids:
            raise NotImplementedError(
                "This projection is not plotted yet."
            )

        projection_products = projected_data[projection_id]
        observed_hist = projection_products["observed"]
        expected_hist = projection_products["expected"]
        residual_hist = self.analysis.residual_function(
            projection_id,
            residual_type,
        )

        return self._plot_1d_projection(
            observed_hist,
            expected_hist,
            residual_hist,
            projection_id,
            residual_type,
            running_mean_window,
            plot_n_simulations,
            plot_random_seed,
            show_gaussian,
        )

    #---------------------------------------------------------------------------
    # Helper function for 1D plotting
    #---------------------------------------------------------------------------
    def _plot_1d_projection(
        self,
        observed_hist: Histogram,
        expected_hist: Histogram,
        residual_hist: Histogram,
        projection_id: str,
        residual_type: str,
        running_mean_window: int | None = None,
        n_simulations: int | None = None,
        random_seed: int | None = None,
        show_gaussian_reference: bool = True,
    ) -> Figure:
        """Plot observed and expected counts with one-dimensional residuals.

        Args:
            observed_hist: Observed counts projected onto one axis.
            expected_hist: Expected counts projected onto the same axis.
            residual_hist: Selected residual values for that projection.
            residual_type: Residual definition used for labels and layout.

        Returns:
            The Matplotlib figure containing the one-dimensional diagnostic.

        Raises:
            ValueError: If ``residual_type`` is not supported.
        """
        
        # Get the single coordinate axis from this one-dimensional histogram.
        axis_name = observed_hist.axes.labels[0]
        axis = observed_hist.axes[axis_name]
        # Use the native unit of the selected histogram axis.
        axis_unit = axis.centers.unit
        x_values = axis.centers.to_value(axis_unit)
        # Use the true bin boundaries for step plots.
        bin_edges = axis.edges.to_value(axis_unit)

        # Get the counts and residual values from the input histograms.
        observed_values = observed_hist.contents
        expected_values = expected_hist.contents
        residual_values = residual_hist.contents

        if residual_type == "raw":
            figure, axes = plt.subplots(
                nrows=2,
                ncols=1,
                figsize=(10, 7),
                sharex=True,
                gridspec_kw={"height_ratios": [2, 1]},
            )
            counts_axis, residual_axis = axes

        elif residual_type in {"pearson", "fractional"}:
            figure = plt.figure(figsize=(12, 8))
            grid = figure.add_gridspec(
                nrows=2,
                ncols=2,
                height_ratios=[2, 1],
            )
            counts_axis = figure.add_subplot(grid[0, :])
            residual_axis = figure.add_subplot(grid[1, 0])
            distribution_axis = figure.add_subplot(grid[1, 1])

        else:
            raise ValueError(f"Unknown residual type: {residual_type}")

        # Compare the projected observed and expected counts.
        counts_axis.stairs(
            observed_values,
            bin_edges,
            color="black",
            linewidth=1.5,
            label="Observed",
        )
        # Show the nominal one-sigma Poisson uncertainty around the prediction.
        poisson_sigma = np.sqrt(expected_values)
        # Draw the Poisson band using the true histogram boundaries.
        counts_axis.stairs(
            expected_values + poisson_sigma,
            bin_edges,
            baseline=expected_values - poisson_sigma,
            fill=True,
            color="tab:orange",
            alpha=0.3,
            label=r"Poisson uncertainty ($\pm 1\sigma$)",
        )
        counts_axis.stairs(
            expected_values,
            bin_edges,
            color="red",
            linewidth=1.5,
            linestyle=":",
            label="Fitted expectation",
        )
        counts_axis.set_ylabel("Counts")
        energy_suffix = (
            self._format_energy_interval_suffix()
            if projection_id
            in {"Phi_projection", "Chi_projection", "Psi_projection"}
            else ""
        )
        counts_axis.set_title(
            f"{residual_type.capitalize()} residual as a function of {axis_name}"
            f"{energy_suffix}"
        )
        if residual_type in {"pearson", "fractional"}:
            # Put the summary in the large count panel, separate from its legend.
            statistics_text = self._format_residual_statistics(
                residual_values,
                residual_type,
            )
            counts_axis.text(
                0.02,
                0.97,
                statistics_text,
                transform=counts_axis.transAxes,
                verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
            )
        counts_axis.legend(loc="upper right")
        counts_axis.grid(alpha=0.2)

        # Show the corresponding residual values.
        residual_axis.scatter(
            x_values,
            residual_values,
            color="tab:purple",
            s=25,
        )
        residual_axis.axhline(
            0.0,
            color="black",
            linewidth=1,
            linestyle="--",
        )

        residual_axis.set_xlabel(f"{axis_name} [{axis_unit}]")
        residual_axis.set_ylabel(f"{residual_type.capitalize()} residual")
        residual_axis.grid(alpha=0.2)

        if running_mean_window is not None:
            if residual_type != "pearson":
                raise ValueError(
                    "running_mean_window is only available for Pearson residuals."
                )

            # Use a centered window, truncating it rather than wrapping at edges.
            running_mean = np.full(residual_values.shape, np.nan, dtype=float)
            left_width = (running_mean_window - 1) // 2
            right_width = running_mean_window // 2
            for index in range(residual_values.size):
                start = max(0, index - left_width)
                stop = min(residual_values.size, index + right_width + 1)
                window_values = residual_values[start:stop]
                finite_window_values = window_values[np.isfinite(window_values)]
                if finite_window_values.size:
                    running_mean[index] = np.mean(finite_window_values)

            residual_axis.plot(
                x_values,
                running_mean,
                color="tab:orange",
                linewidth=2,
                label=f"Running mean ({running_mean_window} bins)",
            )
            residual_axis.legend(loc="upper right")

        if residual_type in {"pearson", "fractional"}:
            # Keep undefined residual bins out of the distribution.
            finite_residuals = residual_values[np.isfinite(residual_values)]

            if finite_residuals.size == 0:
                raise ValueError("No finite residual values are available to plot.")

            if residual_type == "pearson" and n_simulations is not None:
                reference = self.analysis.pearson_poisson_reference(
                    projection_id,
                    n_simulations=n_simulations,
                    bins=15,
                    seed=random_seed,
                )
                bin_edges = reference["bin_edges"]
                distribution_axis.hist(
                    finite_residuals,
                    bins=bin_edges,
                    density=True,
                    color="tab:blue",
                    alpha=0.55,
                    label="Observed Pearson residuals",
                )
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                distribution_axis.stairs(
                    reference["density"],
                    bin_edges,
                    color="black",
                    linewidth=1.5,
                    label="Poisson MC mean density",
                )
                distribution_axis.fill_between(
                    bin_centers,
                    reference["lower"],
                    reference["upper"],
                    step="mid",
                    color="tab:orange",
                    alpha=0.25,
                    label="95% pointwise MC interval",
                )
            else:
                distribution_axis.hist(
                    finite_residuals,
                    bins=15,
                    density=True,
                    color="tab:blue",
                    alpha=0.65,
                    label=f"{residual_type.capitalize()} residuals",
                )
            distribution_axis.axvline(
                0.0,
                color="black",
                linestyle=":",
                linewidth=1,
            )

            if residual_type == "pearson" and show_gaussian_reference:
                # Mark the commonly used range of two standard deviations.
                # residual_axis.axhspan(-2.0, 2.0, color="gray", alpha=0.15)
                # residual_axis.set_ylim(-5.0, 5.0)

                # Show Pearson residual reference levels in nominal sigma units.
                for level, color, linestyle in (
                    (1.0, "gray", ":"),
                    (2.0, "black", "--"),
                    (3.0, "gray", "-."),
                ):
                    residual_axis.axhline(
                        level,
                        color=color,
                        linestyle=linestyle,
                        linewidth=0.9,
                        label=rf"$\pm{level:.0f}\sigma$",
                    )
                    residual_axis.axhline(
                        -level,
                        color=color,
                        linestyle=linestyle,
                        linewidth=0.9,
                    )
                residual_axis.legend(
                    loc="upper right",
                    ncol=3,
                    fontsize="small",
                )

                # Compare Pearson residuals with the standard normal density.
                distribution_limit = max(
                    4.0,
                    1.1 * np.max(np.abs(finite_residuals)),
                )
                reference_x = np.linspace(
                    -distribution_limit,
                    distribution_limit,
                    500,
                )
                distribution_axis.plot(
                    reference_x,
                    stats.norm.pdf(reference_x),
                    color="black",
                    linestyle="--",
                    linewidth=1.5,
                    label=r"Standard normal $N(0,1)$",
                )

            distribution_axis.set_xlabel(f"{residual_type.capitalize()} residual")
            distribution_axis.set_ylabel("Probability density")
            distribution_axis.legend(loc="upper right")
            distribution_axis.grid(alpha=0.2)

        figure.tight_layout()
        plt.show()
        return figure

    @staticmethod
    def _format_residual_statistics(
        values: np.ndarray,
        residual_type: str,
    ) -> str:
        """Format descriptive residual statistics for plot annotations."""
        values = np.asarray(values, dtype=float)
        finite_values = values[np.isfinite(values)]
        statistics = [
            f"N = {finite_values.size}",
            f"Mean = {np.mean(finite_values):.3f}",
            f"Median = {np.median(finite_values):.3f}",
            f"Std. dev. = {np.std(finite_values):.3f}",
            f"RMS = {np.sqrt(np.mean(finite_values**2)):.3f}",
            f"Min = {np.min(finite_values):.3f}",
            f"Max = {np.max(finite_values):.3f}",
        ]

        if residual_type == "pearson":
            absolute_values = np.abs(finite_values)
            statistics.extend(
                [
                    f"|R| > 2 = {100 * np.mean(absolute_values > 2):.1f}%",
                    f"|R| > 3 = {100 * np.mean(absolute_values > 3):.1f}%",
                    rf"Pearson $\chi^2$ = {np.sum(finite_values**2):.2f}",
                ]
            )

        return "\n".join(statistics)

    def _plot_energy_coordinate_projection(
        self,
        projection_id: str,
        residual_type: str,
        n_simulations: int | None = None,
        random_seed: int | None = None,
        show_gaussian_reference: bool = True,
    ) -> Figure:
        """Plot one rectangular residual view with energy on the vertical axis."""
        coordinate_names = {
            "Em_Phi_projection": "Phi",
            "Em_Chi_projection": "Chi",
            "Em_Psi_projection": "Psi",
        }
        if projection_id not in coordinate_names:
            raise ValueError(f"Unknown energy-coordinate projection: {projection_id}")
        if residual_type not in {"raw", "pearson", "fractional"}:
            raise ValueError(f"Unknown residual type: {residual_type}")

        coordinate_name = coordinate_names[projection_id]
        projected_data = self.analysis.projected_data()
        products = projected_data[projection_id]
        residual_hist = self.analysis.residual_function(
            projection_id,
            residual_type,
        )

        energy_axis = products["observed"].axes["Em"]
        coordinate_axis = products["observed"].axes[coordinate_name]
        energy_unit = energy_axis.edges.unit
        coordinate_unit = coordinate_axis.edges.unit
        energy_edges = energy_axis.edges.to_value(energy_unit)
        coordinate_edges = coordinate_axis.edges.to_value(coordinate_unit)
        residual_values = np.asarray(residual_hist.contents, dtype=float)
        finite_values = residual_values[np.isfinite(residual_values)]

        if finite_values.size == 0:
            raise ValueError(
                f"No finite residual values are available for {projection_id}."
            )

        if residual_type == "fractional":
            color_limit = np.quantile(np.abs(finite_values), 0.995)
        else:
            color_limit = np.max(np.abs(finite_values))
        if color_limit == 0:
            color_limit = 1.0

        color_norm = TwoSlopeNorm(
            vmin=-color_limit,
            vcenter=0.0,
            vmax=color_limit,
        )

        if residual_type == "raw":
            figure, map_axis = plt.subplots(figsize=(9, 6))
            distribution_axis = None
        else:
            figure, (map_axis, distribution_axis) = plt.subplots(
                nrows=1,
                ncols=2,
                figsize=(13, 6),
            )

        image = map_axis.pcolormesh(
            coordinate_edges,
            energy_edges,
            residual_values,
            cmap="RdBu_r",
            norm=color_norm,
            shading="auto",
        )
        map_axis.set_xlabel(f"{coordinate_name} [{coordinate_unit}]")
        map_axis.set_ylabel(f"Measured energy [{energy_unit}]")
        map_axis.set_title(
            f"{residual_type.capitalize()} residual: Em–{coordinate_name}"
        )
        if energy_axis.axis_scale == "log":
            map_axis.set_yscale("log")

        colorbar_label = f"{residual_type.capitalize()} residual"
        if residual_type == "raw":
            colorbar_label += " [counts]"
        figure.colorbar(
            image,
            ax=map_axis,
            label=colorbar_label,
            extend="both" if residual_type == "fractional" else "neither",
        )

        if distribution_axis is not None:
            reference = None
            if residual_type == "pearson" and n_simulations is not None:
                reference = self.analysis.pearson_poisson_reference(
                    projection_id,
                    n_simulations=n_simulations,
                    bins=40,
                    seed=random_seed,
                )

            if reference is None:
                distribution_axis.hist(
                    finite_values,
                    bins=40,
                    density=True,
                    color="tab:blue",
                    alpha=0.65,
                    label=f"{residual_type.capitalize()} residuals",
                )
            else:
                bin_edges = reference["bin_edges"]
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                distribution_axis.hist(
                    finite_values,
                    bins=bin_edges,
                    density=True,
                    color="tab:blue",
                    alpha=0.55,
                    label="Observed Pearson residuals",
                )
                distribution_axis.stairs(
                    reference["density"],
                    bin_edges,
                    color="black",
                    linewidth=1.5,
                    label="Poisson MC mean density",
                )
                distribution_axis.fill_between(
                    bin_centers,
                    reference["lower"],
                    reference["upper"],
                    step="mid",
                    color="tab:orange",
                    alpha=0.25,
                    label="95% pointwise MC interval",
                )

            if residual_type == "pearson" and show_gaussian_reference:
                distribution_limit = max(
                    4.0,
                    1.1 * np.max(np.abs(finite_values)),
                )
                reference_x = np.linspace(
                    -distribution_limit,
                    distribution_limit,
                    500,
                )
                distribution_axis.plot(
                    reference_x,
                    stats.norm.pdf(reference_x),
                    color="black",
                    linestyle="--",
                    linewidth=1.5,
                    label=r"Standard normal $N(0,1)$",
                )

            distribution_axis.axvline(
                0.0,
                color="black",
                linestyle=":",
                linewidth=1,
            )
            distribution_axis.text(
                0.03,
                0.95,
                self._format_residual_statistics(
                    finite_values,
                    residual_type,
                ),
                transform=distribution_axis.transAxes,
                verticalalignment="top",
                bbox={
                    "boxstyle": "round",
                    "facecolor": "white",
                    "alpha": 0.85,
                },
            )
            distribution_axis.set_xlabel(
                f"{residual_type.capitalize()} residual"
            )
            distribution_axis.set_ylabel("Probability density")
            distribution_axis.legend(loc="upper right")
            distribution_axis.grid(alpha=0.2)

        figure.tight_layout()
        plt.show()
        return figure

    #---------------------------------------------------------------------------
    # Helper for plotting 2D
    #---------------------------------------------------------------------------
    
    def _plot_2d_projections(
        self,
        residual_type: str,
        n_simulations: int | None = None,
        random_seed: int | None = None,
        show_gaussian_reference: bool = True,
    ) -> Figure:
        """Plot Phi-Chi and Phi-Psi residual views side by side.

        Args:
            residual_type: Residual definition: ``"raw"``, ``"pearson"``, or
                ``"fractional"``.
            n_simulations: Optional number of fixed-fit Poisson realizations.
            random_seed: Optional seed for reproducible simulations.
            show_gaussian_reference: Whether to overlay a Gaussian reference.

        Returns:
            The Matplotlib figure containing both rectangular residual maps.

        Raises:
            ValueError: If ``residual_type`` is not supported or a projection
                has no finite residual values.
        """
        if residual_type not in {"raw", "pearson", "fractional"}:
            raise ValueError(f"Unknown residual type: {residual_type}")

        # Get the two rectangular projections prepared by the analysis.
        projected_data = self.analysis.projected_data()
        projection_specs = [
            ("Phi_Chi_projection", r"$\Phi$–$\chi$", "Chi"),
            ("Phi_Psi_projection", r"$\Phi$–$\psi$", "Psi"),
        ]

        residual_views = []

        for projection_id, view_name, x_axis_name in projection_specs:
            products = projected_data[projection_id]
            residual_hist = self.analysis.residual_function(
                projection_id,
                residual_type,
            )

            # Get the bin edges and residual values for this view.
            phi_edges = products["observed"].axes["Phi"].edges.to_value(u.deg)
            x_edges = products["observed"].axes[x_axis_name].edges.to_value(u.deg)
            residual_values = np.asarray(residual_hist.contents, dtype=float)
            finite_values = residual_values[np.isfinite(residual_values)]
            simulation_reference = None
            if residual_type == "pearson" and n_simulations is not None:
                simulation_reference = self.analysis.pearson_poisson_reference(
                    projection_id,
                    n_simulations=n_simulations,
                    bins=40,
                    seed=random_seed,
                )

            if finite_values.size == 0:
                raise ValueError(
                    f"No finite residual values are available for {projection_id}."
                )

            residual_views.append(
                {
                    "view_name": view_name,
                    "x_axis_name": x_axis_name,
                    "x_edges": x_edges,
                    "phi_edges": phi_edges,
                    "values": residual_values,
                    "finite_values": finite_values,
                    "simulation_reference": simulation_reference,
                }
            )

        # Use a robust shared scale for fractional maps so one outlier does not
        # wash out the remaining spatial structure.
        if residual_type == "fractional":
            all_absolute_values = np.concatenate(
                [np.abs(view["finite_values"]) for view in residual_views]
            )
            color_limit = np.quantile(all_absolute_values, 0.995)
        else:
            # Keep the full symmetric range for raw and Pearson residual maps.
            color_limit = max(
                np.max(np.abs(view["finite_values"]))
                for view in residual_views
            )
        if color_limit == 0:
            color_limit = 1.0

        color_norm = TwoSlopeNorm(
            vmin=-color_limit,
            vcenter=0.0,
            vmax=color_limit,
        )

        if residual_type == "raw":
            # Reserve a separate column for the shared colorbar.
            figure = plt.figure(figsize=(15, 5))
            grid = figure.add_gridspec(
                nrows=1,
                ncols=3,
                width_ratios=[1, 1, 0.05],
                wspace=0.25,
            )
            map_axes = [
                figure.add_subplot(grid[0, 0]),
                figure.add_subplot(grid[0, 1]),
            ]
            map_axes[1].sharey(map_axes[0])
            colorbar_axis = figure.add_subplot(grid[0, 2])
            distribution_axes = None
        else:
            # Reserve space for two distributions and one shared colorbar.
            figure = plt.figure(figsize=(14, 10))
            grid = figure.add_gridspec(
                nrows=2,
                ncols=3,
                width_ratios=[1, 1, 0.05],
                height_ratios=[1, 1],
                wspace=0.25,
                hspace=0.25,
            )
            map_axes = [
                figure.add_subplot(grid[0, 0]),
                figure.add_subplot(grid[0, 1]),
            ]
            colorbar_axis = figure.add_subplot(grid[0, 2])
            distribution_axes = [
                figure.add_subplot(grid[1, 0]),
                figure.add_subplot(grid[1, 1]),
            ]

        # Draw both residual maps with the same axes and color scale.
        map_images = []

        for map_axis, view in zip(map_axes, residual_views):
            image = map_axis.pcolormesh(
                view["x_edges"],
                view["phi_edges"],
                view["values"],
                cmap="RdBu_r",
                norm=color_norm,
                shading="auto",
            )
            map_images.append(image)
            map_axis.set_xlabel(f"{view['x_axis_name']} [deg]")
            map_axis.set_ylabel(r"$\Phi$ [deg]")
            map_axis.set_title(
                f"{residual_type.capitalize()} residual: {view['view_name']}"
                f"{self._format_energy_interval_suffix()}"
            )

        # Add a distribution and summary statistics for normalized residuals.
        if distribution_axes is not None:
            for distribution_axis, view in zip(
                distribution_axes,
                residual_views,
            ):
                finite_values = view["finite_values"]

                reference = view["simulation_reference"]
                if reference is not None:
                    bin_edges = reference["bin_edges"]
                    distribution_axis.hist(
                        finite_values,
                        bins=bin_edges,
                        density=True,
                        color="tab:blue",
                        alpha=0.55,
                        label="Observed Pearson residuals",
                    )
                    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                    distribution_axis.stairs(
                        reference["density"],
                        bin_edges,
                        color="black",
                        linewidth=1.5,
                        label="Poisson MC mean density",
                    )
                    distribution_axis.fill_between(
                        bin_centers,
                        reference["lower"],
                        reference["upper"],
                        step="mid",
                        color="tab:orange",
                        alpha=0.25,
                        label="95% pointwise MC interval",
                    )
                else:
                    distribution_axis.hist(
                        finite_values,
                        bins=40,
                        density=True,
                        color="tab:blue",
                        alpha=0.65,
                        label=f"{residual_type.capitalize()} residuals",
                    )
                distribution_axis.axvline(
                    0.0,
                    color="black",
                    linestyle=":",
                    linewidth=1,
                )

                if residual_type == "pearson" and show_gaussian_reference:
                    distribution_limit = max(
                        4.0,
                        1.1 * np.max(np.abs(finite_values)),
                    )
                    reference_x = np.linspace(
                        -distribution_limit,
                        distribution_limit,
                        500,
                    )
                    distribution_axis.plot(
                        reference_x,
                        stats.norm.pdf(reference_x),
                        color="black",
                        linestyle="--",
                        linewidth=1.5,
                        label=r"Standard normal $N(0,1)$",
                    )

                statistics_text = self._format_residual_statistics(
                    finite_values,
                    residual_type,
                )
                distribution_axis.text(
                    0.03,
                    0.95,
                    statistics_text,
                    transform=distribution_axis.transAxes,
                    verticalalignment="top",
                    bbox={
                        "boxstyle": "round",
                        "facecolor": "white",
                        "alpha": 0.85,
                    },
                )
                distribution_axis.set_title(
                    f"Residual distribution: {view['view_name']}"
                )
                distribution_axis.set_xlabel(
                    f"{residual_type.capitalize()} residual"
                )
                distribution_axis.set_ylabel("Probability density")
                distribution_axis.legend(loc="upper right")
                distribution_axis.grid(alpha=0.2)

        # Add one colorbar shared by both maps in its reserved axis.
        colorbar_label = f"{residual_type.capitalize()} residual"
        if residual_type == "raw":
            colorbar_label += " [counts]"
        figure.colorbar(
            map_images[-1],
            cax=colorbar_axis,
            label=colorbar_label,
            extend="both" if residual_type == "fractional" else "neither",
        )

        figure.tight_layout()
        plt.show()
        return figure


    #---------------------------------------------------------------------------
    # Helper for plotting HEALPix
    #---------------------------------------------------------------------------

    def _plot_psichi_projection(
        self,
        projection_id: str,
        residual_type: str,
        phi_bin_index: int | None = None,
        energy_bin_index: int | None = None,
        n_simulations: int | None = None,
        random_seed: int | None = None,
        show_gaussian_reference: bool = True,
    ) -> Figure:
        """Plot a PsiChi residual map, optionally for one Phi or Em bin.

        Args:
            projection_id: Either ``"PsiChi_projection"`` for the view
                integrated over Phi, or ``"Phi_PsiChi_projection"`` for a
                selected Phi slice, or ``"Em_PsiChi_projection"`` for a
                selected Em slice.
            residual_type: Residual definition: ``"raw"``, ``"pearson"``, or
                ``"fractional"``.
            phi_bin_index: Phi-bin index required for
                ``"Phi_PsiChi_projection"``; leave as ``None`` for the
                Phi-integrated projection.
            energy_bin_index: Em-bin index required for
                ``"Em_PsiChi_projection"``.
            n_simulations: Optional number of fixed-fit Poisson realizations.
            random_seed: Optional seed for reproducible simulations.
            show_gaussian_reference: Whether to overlay a Gaussian reference.

        Returns:
            The Matplotlib figure containing the HEALPix map and distribution.

        Raises:
            ValueError: If the residual type or projection is unsupported, or
                if a required slice index is missing or not applicable.
            IndexError: If a Phi or Em slice index is outside the available
                bins.
        """
        if residual_type not in {"raw", "pearson", "fractional"}:
            raise ValueError(f"Unknown residual type: {residual_type}")

        if projection_id not in {
            "PsiChi_projection",
            "Phi_PsiChi_projection",
            "Em_PsiChi_projection",
        }:
            raise ValueError(f"Unknown HEALPix projection: {projection_id}")

        # Calculate the selected residual for the requested HEALPix projection.
        residual_hist = self.analysis.residual_function(
            projection_id,
            residual_type,
        )

        # Get the HEALPix axis and select a Phi slice only when requested.
        psichi_axis = residual_hist.axes["PsiChi"]
        if projection_id == "Phi_PsiChi_projection":
            if energy_bin_index is not None:
                raise ValueError(
                    "energy_bin_index applies only to Em_PsiChi_projection."
                )
            if phi_bin_index is None:
                raise ValueError(
                    "Provide phi_bin_index to select a Phi slice."
                )

            phi_axis = residual_hist.axes["Phi"]
            phi_centers = phi_axis.centers.to_value(u.deg)
            if not 0 <= phi_bin_index < len(phi_centers):
                raise IndexError(
                    "phi_bin_index is outside the available Phi bins."
                )

            map_values = np.asarray(
                residual_hist.contents[phi_bin_index],
                dtype=float,
            )
            view_label = f"at Phi = {phi_centers[phi_bin_index]:.1f} deg"
        elif projection_id == "Em_PsiChi_projection":
            if phi_bin_index is not None:
                raise ValueError(
                    "phi_bin_index applies only to Phi_PsiChi_projection."
                )
            if energy_bin_index is None:
                raise ValueError(
                    "Provide energy_bin_index to select an Em slice."
                )

            energy_axis = residual_hist.axes["Em"]
            energy_edges = energy_axis.edges.to_value(u.keV)
            if not 0 <= energy_bin_index < energy_axis.nbins:
                raise IndexError(
                    "energy_bin_index is outside the available Em bins."
                )

            map_values = np.asarray(
                residual_hist.contents[energy_bin_index],
                dtype=float,
            )
            view_label = (
                f"for {energy_edges[energy_bin_index]:g}–"
                f"{energy_edges[energy_bin_index + 1]:g} keV"
            )
        else:
            if phi_bin_index is not None:
                raise ValueError(
                    "phi_bin_index applies only to Phi_PsiChi_projection."
                )
            if energy_bin_index is not None:
                raise ValueError(
                    "energy_bin_index applies only to Em_PsiChi_projection."
                )

            map_values = np.asarray(residual_hist.contents, dtype=float)
            view_label = "integrated over Phi"

        # Keep only defined residuals for the distribution.
        finite_values = map_values[np.isfinite(map_values)]

        if finite_values.size == 0:
            raise ValueError("This HEALPix view has no finite residual values.")

        simulation_reference = None
        if residual_type == "pearson" and n_simulations is not None:
            simulation_reference = self.analysis.pearson_poisson_reference(
                projection_id,
                n_simulations=n_simulations,
                bins=40,
                seed=random_seed,
                phi_bin_index=phi_bin_index,
                energy_bin_index=energy_bin_index,
            )

        # Center the map color scale on zero.
        color_limit = np.max(np.abs(finite_values))
        if color_limit == 0:
            color_limit = 1.0

        color_norm = TwoSlopeNorm(
            vmin=-color_limit,
            vcenter=0.0,
            vmax=color_limit,
        )

        # Build the map using the original HEALPix geometry.
        sky_map = mhp.HealpixMap(
            data=map_values,
            nside=psichi_axis.nside,
            scheme=psichi_axis.scheme,
            coordsys=psichi_axis.coordsys,
        )

        figure = plt.figure(figsize=(13, 6))
        map_axis = figure.add_subplot(
            1,
            2,
            1,
            projection="mollview",
        )
        distribution_axis = figure.add_subplot(1, 2, 2)

        # Plot the residual map.
        map_image, map_axis = sky_map.plot(
            ax=map_axis,
            cmap="RdBu_r",
            norm=color_norm,
            cbar=False,
        )
        energy_suffix = (
            self._format_energy_interval_suffix()
            if projection_id
            in {"PsiChi_projection", "Phi_PsiChi_projection"}
            else ""
        )
        map_axis.set_title(
            f"{residual_type.capitalize()} residual on PsiChi "
            f"{view_label}{energy_suffix}"
        )

        colorbar_label = (
            "Raw residual [counts]"
            if residual_type == "raw"
            else f"{residual_type.capitalize()} residual"
        )
        figure.colorbar(
            map_image,
            ax=map_axis,
            orientation="horizontal",
            pad=0.08,
            label=colorbar_label,
        )

        # Plot the empirical residual distribution.
        if simulation_reference is not None:
            bin_edges = simulation_reference["bin_edges"]
            distribution_axis.hist(
                finite_values,
                bins=bin_edges,
                density=True,
                color="tab:blue",
                alpha=0.55,
                label="Observed Pearson residuals",
            )
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            distribution_axis.stairs(
                simulation_reference["density"],
                bin_edges,
                color="black",
                linewidth=1.5,
                label="Poisson MC mean density",
            )
            distribution_axis.fill_between(
                bin_centers,
                simulation_reference["lower"],
                simulation_reference["upper"],
                step="mid",
                color="tab:orange",
                alpha=0.25,
                label="95% pointwise MC interval",
            )
        else:
            distribution_axis.hist(
                finite_values,
                bins=40,
                density=True,
                color="tab:blue",
                alpha=0.65,
                label=f"{residual_type.capitalize()} residuals",
            )
        distribution_axis.axvline(
            0.0,
            color="black",
            linestyle=":",
            linewidth=1,
        )

        if residual_type == "pearson" and show_gaussian_reference:
            # Compare Pearson residuals with the standard normal density.
            distribution_limit = max(
                4.0,
                1.1 * np.max(np.abs(finite_values)),
            )
            reference_x = np.linspace(
                -distribution_limit,
                distribution_limit,
                500,
            )
            distribution_axis.plot(
                reference_x,
                stats.norm.pdf(reference_x),
                color="black",
                linestyle="--",
                linewidth=1.5,
                label=r"Standard normal $N(0,1)$",
            )

        # Show descriptive statistics for the finite residuals.
        statistics_text = self._format_residual_statistics(
            finite_values,
            residual_type,
        )
        distribution_axis.text(
            0.03,
            0.95,
            statistics_text,
            transform=distribution_axis.transAxes,
            verticalalignment="top",
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "alpha": 0.85,
            },
        )
        distribution_axis.set_title(
            f"Residual distribution {view_label}"
        )
        distribution_axis.set_xlabel(f"{residual_type.capitalize()} residual")
        distribution_axis.set_ylabel("Probability density")
        distribution_axis.legend(loc="upper right")
        distribution_axis.grid(alpha=0.2)

        figure.tight_layout()
        plt.show()
        return figure
