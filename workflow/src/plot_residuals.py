import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
import astropy.units as u
from histpy import Histogram


class PlottingResiduals:
    """Create residual diagnostic plots from a ResidualAnalysis object."""

    def __init__(self, analysis) -> None:
        """Store the analysis object used by the plotting methods."""
        # Keep the analysis object available to this instance.
        self.analysis = analysis



    def plot(
        self,
        projection_id: str,
        residual_type: str = "raw",
    ) -> Figure:
        """Create a residual plot for the selected projection and residual type."""
        # Get the available projections from the analysis object.
        projected_data = self.analysis.projected_data()

        if projection_id not in projected_data:
            raise ValueError(f"Unknown projection: {projection_id}")

        # Select the observed and expected histograms for this projection.
        projection_products = projected_data[projection_id]
        observed_hist = projection_products["observed"]
        expected_hist = projection_products["expected"]

        # Ask the analysis object for the selected residual.
        residual_hist = self.analysis.residual_function(
            projection_id,
            residual_type,
        )

        # Route supported raw one-dimensional projections to their plotter.
        one_dimensional_ids = {
            "Phi_projection",
            "Chi_projection",
            "Psi_projection",
        }

        if projection_id in one_dimensional_ids:
            return self._plot_1d_projection(
                observed_hist,
                expected_hist,
                residual_hist,
                residual_type,
            )

        raise NotImplementedError(
            "This projection and residual combination is not plotted yet."
        )


# Heler function for 1D plotting
    def _plot_1d_projection(
        self,
        observed_hist: Histogram,
        expected_hist: Histogram,
        residual_hist: Histogram,
        residual_type: str,
    ) -> Figure:
        """Plot observed and expected counts with one-dimensional residuals."""
        
        # Get the single coordinate axis from this one-dimensional histogram.
        axis_name = observed_hist.axes.labels[0]
        axis = observed_hist.axes[axis_name]
        x_values = axis.centers.to_value(u.deg)

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
        counts_axis.step(
            x_values,
            observed_values,
            where="mid",
            color="black",
            linewidth=1.5,
            label="Observed",
        )
        counts_axis.step(
            x_values,
            expected_values,
            where="mid",
            color="red",
            linewidth=1.5,
            linestyle=":",
            label="Fitted expectation",
        )
        counts_axis.set_ylabel("Counts")
        counts_axis.set_title(
            f"{residual_type.capitalize()} residual as a function of {axis_name}"
        )
        counts_axis.legend()
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

        residual_axis.set_xlabel(f"{axis_name} [deg]")
        residual_axis.set_ylabel(f"{residual_type.capitalize()} residual")
        residual_axis.grid(alpha=0.2)

        if residual_type in {"pearson", "fractional"}:
            # Keep undefined residual bins out of the distribution.
            finite_residuals = residual_values[np.isfinite(residual_values)]

            if finite_residuals.size == 0:
                raise ValueError("No finite residual values are available to plot.")

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
                label="Zero",
            )

            if residual_type == "pearson":
                # Mark the commonly used range of two standard deviations.
                residual_axis.axhspan(-2.0, 2.0, color="gray", alpha=0.15)
                residual_axis.set_ylim(-5.0, 5.0)

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

            residual_mean = np.mean(finite_residuals)
            residual_median = np.median(finite_residuals)
            residual_std = np.std(finite_residuals)
            residual_rms = np.sqrt(np.mean(finite_residuals**2))
            statistics_text = (
                f"N = {finite_residuals.size}\n"
                f"Mean = {residual_mean:.3f}\n"
                f"Median = {residual_median:.3f}\n"
                f"Std. dev. = {residual_std:.3f}\n"
                f"RMS = {residual_rms:.3f}"
            )
            distribution_axis.text(
                0.03,
                0.95,
                statistics_text,
                transform=distribution_axis.transAxes,
                verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
            )
            distribution_axis.set_xlabel(f"{residual_type.capitalize()} residual")
            distribution_axis.set_ylabel("Probability density")
            distribution_axis.legend()
            distribution_axis.grid(alpha=0.2)

        figure.tight_layout()
        plt.show()
        return figure
