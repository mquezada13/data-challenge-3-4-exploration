import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
import astropy.units as u
from histpy import Histogram
import mhealpy as mhp


class PlottingResiduals:
    """Create residual diagnostic plots from a ResidualAnalysis object."""

    def __init__(self, analysis) -> None:
        """Store the analysis object used by the plotting methods."""
        # Keep the analysis object available to this instance.
        self.analysis = analysis


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
    ) -> Figure:
        """Create a residual plot for the selected projection and residual type."""
        projected_data = self.analysis.projected_data()

        if projection_id not in projected_data:
            raise ValueError(f"Unknown projection: {projection_id}")

        # Route either joint rectangular projection to the paired 2D plot.
        two_dimensional_ids = {
            "Phi_Chi_projection",
            "Phi_Psi_projection",
        }
        if projection_id in two_dimensional_ids:
            return self._plot_2d_projections(residual_type)

        # Route one Phi slice of the joint Phi-PsiChi projection to its HEALPix plot.
        if projection_id == "Phi_PsiChi_projection":
            if phi_bin_index is None:
                raise ValueError(
                    "Provide phi_bin_index to select a Phi slice."
                )
            return self._plot_psichi_slice(
                phi_bin_index,
                residual_type,
            )

        # Route supported one-dimensional projections to their plotter.
        one_dimensional_ids = {
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
            residual_type,
        )

    #---------------------------------------------------------------------------
    # Heler function for 1D plotting
    #---------------------------------------------------------------------------
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

    #---------------------------------------------------------------------------
    # Helper for plotting 2D
    #---------------------------------------------------------------------------
    
    def _plot_2d_projections(
        self,
        residual_type: str,
    ) -> Figure:
        """Plot the Phi-Chi and Phi-Psi residual views side by side."""
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
                }
            )

        # Use one symmetric color scale to compare the two maps.
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
            figure, map_axes = plt.subplots(
                nrows=1,
                ncols=2,
                figsize=(15, 5),
                sharey=True,
            )
            distribution_axes = None
            colorbar_axis = None
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
            )

        # Add a distribution and summary statistics for normalized residuals.
        if distribution_axes is not None:
            for distribution_axis, view in zip(
                distribution_axes,
                residual_views,
            ):
                finite_values = view["finite_values"]

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

                if residual_type == "pearson":
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

                statistics_text = (
                    f"N = {finite_values.size}\n"
                    f"Mean = {np.mean(finite_values):.3f}\n"
                    f"Median = {np.median(finite_values):.3f}\n"
                    f"Std. dev. = {np.std(finite_values):.3f}\n"
                    f"RMS = {np.sqrt(np.mean(finite_values**2)):.3f}"
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
                distribution_axis.legend()
                distribution_axis.grid(alpha=0.2)

        # Add one colorbar shared by both maps.
        if colorbar_axis is None:
            figure.colorbar(
                map_images[-1],
                ax=map_axes,
                label=f"{residual_type.capitalize()} residual [counts]",
            )
        else:
            figure.colorbar(
                map_images[-1],
                cax=colorbar_axis,
                label=f"{residual_type.capitalize()} residual",
            )

        figure.tight_layout()
        plt.show()
        return figure


    #---------------------------------------------------------------------------
    # Helper for plotting HEALPix
    #---------------------------------------------------------------------------

    def _plot_psichi_slice(
        self,
        phi_bin_index: int,
        residual_type: str,
    ) -> Figure:
        """Plot a HEALPix residual map and distribution for one Phi bin."""
        if residual_type not in {"raw", "pearson", "fractional"}:
            raise ValueError(f"Unknown residual type: {residual_type}")

        # Calculate the selected residual for the joint Phi-PsiChi projection.
        residual_hist = self.analysis.residual_function(
            "Phi_PsiChi_projection",
            residual_type,
        )

        # Get the Phi-bin center and the HEALPix axis.
        phi_axis = residual_hist.axes["Phi"]
        phi_centers = phi_axis.centers.to_value(u.deg)

        if not 0 <= phi_bin_index < len(phi_centers):
            raise IndexError("phi_bin_index is outside the available Phi bins.")

        psichi_axis = residual_hist.axes["PsiChi"]
        phi_value = phi_centers[phi_bin_index]

        # Select this Phi bin and keep only defined residuals for the distribution.
        slice_values = np.asarray(
            residual_hist.contents[phi_bin_index],
            dtype=float,
        )
        finite_values = slice_values[np.isfinite(slice_values)]

        if finite_values.size == 0:
            raise ValueError("This Phi bin has no finite residual values.")

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
            data=slice_values,
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

        # Plot the residual map for this Phi bin.
        map_image, map_axis = sky_map.plot(
            ax=map_axis,
            cmap="RdBu_r",
            norm=color_norm,
            cbar=False,
        )
        map_axis.set_title(
            f"{residual_type.capitalize()} residual on PsiChi "
            f"at Phi = {phi_value:.1f} deg"
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

        if residual_type == "pearson":
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
        statistics_text = (
            f"N = {finite_values.size}\n"
            f"Mean = {np.mean(finite_values):.3f}\n"
            f"Median = {np.median(finite_values):.3f}\n"
            f"Std. dev. = {np.std(finite_values):.3f}\n"
            f"RMS = {np.sqrt(np.mean(finite_values**2)):.3f}"
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
            f"Residual distribution at Phi = {phi_value:.1f} deg"
        )
        distribution_axis.set_xlabel(f"{residual_type.capitalize()} residual")
        distribution_axis.set_ylabel("Probability density")
        distribution_axis.legend()
        distribution_axis.grid(alpha=0.2)

        figure.tight_layout()
        plt.show()
        return figure