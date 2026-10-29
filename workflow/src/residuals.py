import numpy as np
import pandas as pd
from histpy import Axis, Histogram
import astropy.units as u
from .time_resolved_residuals_v1 import TimeResolvedResidualAnalysis
from collections.abc import Sequence
from pathlib import Path

class ResidualAnalysis:
    def __init__(
        self,
        observed: Histogram,
        expected: Histogram,
        expected_floor: float | None = None,
        time_residual: bool = False,
        time_binned_observed: (
            Histogram | str | Path | Sequence[str | Path] | None
        ) = None,
        time_bin_index: int | None = None,
    ) -> None:
        """Initialize the analysis with observed and expected histograms.

        Args:
            observed: Histogram containing observed counts.
            expected: Histogram containing fitted expected counts.
            expected_floor: Optional artificial floor used when preparing the
                expected histogram. Bins at or below this value are excluded
                only when they contain no observed counts.
            time_residual: Whether to prepare a time-resolved residual analysis.
            time_binned_observed: Time-binned observed histogram or paths to
                the existing weekly time-binned histograms. Required when
                time_residual is True.
            time_bin_index: Zero-based Time-bin index to select when
                time_residual is True.
        """
        # Validate the optional time-resolved configuration.
        if not isinstance(time_residual, bool):
            raise TypeError("time_residual must be a boolean.")

        self._temporal_total_counts = None
        self._temporal_reference_probabilities = None

        if time_residual:
            observed, expected = self._prepare_time_resolved_inputs(
                observed,
                expected,
                time_binned_observed,
                time_bin_index,
            )

        else:
            self.time_bin_index = None
            self.time_start = None
            self.time_stop = None

        self.observed_hist = observed
        self.expected_hist = expected
        self.expected_floor = expected_floor

        if expected_floor is not None and (
            not np.isfinite(expected_floor) or expected_floor < 0
        ):
            raise ValueError("expected_floor must be finite and non-negative.")

        products = {
            "observed": self.observed_hist,
            "expected": self.expected_hist,
        }

        if self.expected_hist.axes != self.observed_hist.axes:
            raise ValueError(
                "Observed and expected histograms must have matching axes."
            )
        if self.observed_hist.shape != self.expected_hist.shape:
            raise ValueError(
                "Observed and expected histograms must have the same shape."
            )

        for product_name, histogram in products.items():
            values = np.asanyarray(
                histogram.contents,
                dtype=float,
                copy=True,
            )

            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"{product_name} contains non-finite values."
                )
            if np.any(values < 0):
                raise ValueError(
                    f"{product_name} contains negative values."
                )
        # Build and store the mask and its report.
        self.mask, self.mask_report = self.mask_function()

        # Create masked copies without changing the original histograms.
        self.masked_observed_hist = Histogram(
            list(self.observed_hist.axes),
            contents=np.where(
                self.mask,
                self.observed_hist.contents,
                0.0,
            ),
        )

        self.masked_expected_hist = Histogram(
            list(self.expected_hist.axes),
            contents=np.where(
                self.mask,
                self.expected_hist.contents,
                0.0,
            ),
        )

        # Initialize the projection cache.
        self._projected_data_cache = None
        self._poisson_reference_cache = {}

    def _prepare_time_resolved_inputs(
        self,
        observed: Histogram,
        expected: Histogram,
        time_binned_observed: (
            Histogram | str | Path | Sequence[str | Path] | None
        ),
        time_bin_index: int | None,
    ) -> tuple[Histogram, Histogram]:
        """Prepare one selected temporal interval for ordinary analysis."""
        if time_binned_observed is None:
            raise ValueError(
                "time_binned_observed is required when "
                "time_residual=True."
            )

        if (
            isinstance(time_bin_index, bool)
            or not isinstance(time_bin_index, (int, np.integer))
        ):
            raise TypeError(
                "time_bin_index must be an integer when "
                "time_residual=True."
            )

        if time_bin_index < 0:
            raise ValueError("time_bin_index must be non-negative.")

        time_analysis = TimeResolvedResidualAnalysis(
            integrated_observed=observed,
            integrated_expected=expected,
            time_binned_observed=time_binned_observed,
        )

        # Construct the complete D(t,i) histogram.
        time_observed = time_analysis.build_observed()
        number_of_time_bins = time_observed.axes["Time"].nbins

        # Validate the upper index after the Time axis is available.
        if time_bin_index >= number_of_time_bins:
            raise IndexError(
                "time_bin_index is outside the available Time bins."
            )

        # Construct the complete normalized M_tilde(t,i) histogram.
        time_expected = time_analysis.build_expected()

        # Preserve the fixed total and original normalized CDS model for
        # temporal multinomial reference simulations.
        temporal_model = np.asarray(
            time_analysis.integrated_expected_hist.to_dense(
                copy=True
            ).contents,
            dtype=float,
        )
        self._temporal_reference_probabilities = (
            temporal_model / temporal_model.sum()
        )
        self._temporal_total_counts = int(
            time_observed.contents[time_bin_index].sum()
        )

        # Preserve the physical time boundaries for later plot labels.
        time_axis = time_observed.axes["Time"]
        selected_time_start = time_axis.edges[time_bin_index]
        selected_time_stop = time_axis.edges[time_bin_index + 1]

        # Remove the Time axis and retain the CDS axes.
        selected_cds_axes = list(time_observed.axes)[1:]

        # Copy the selected interval so the full temporal arrays can be released.
        observed = Histogram(
            selected_cds_axes,
            contents=time_observed.contents[time_bin_index].copy(),
            copy_contents=False,
        )

        expected = Histogram(
            selected_cds_axes,
            contents=time_expected.contents[time_bin_index].copy(),
            copy_contents=False,
        )

        self.time_bin_index = int(time_bin_index)
        self.time_start = selected_time_start
        self.time_stop = selected_time_stop
        return observed, expected

    def mask_function(self) -> tuple[np.ndarray, dict[str, object]]:
        """Build the valid-bin mask and summarize its effect."""
        # Define a numerical-zero threshold near the float64 limit.
        numerical_floor = 10 * np.finfo(float).tiny
        exclusion_threshold = numerical_floor

        # Include a known artificial floor when one was supplied.
        if self.expected_floor is not None:
            exclusion_threshold = max(numerical_floor, self.expected_floor)

        # Identify bins at or below the selected floor.
        floor_mask = self.expected_hist.contents <= exclusion_threshold

        # Do not silently exclude bins that contain observed events.
        observed_at_floor = self.observed_hist.contents[
            floor_mask
        ].sum()

        # Ensure that the mask retains data for the residual calculation.
        if observed_at_floor > 0:
            raise ValueError(
                "Expected-floor bins contain observed events and cannot "
                "be excluded automatically."
            )
        # Exclude only floor bins that contain no observed events.
        valid_mask = ~floor_mask

        # Ensure that the mask retains data for the residual calculation.
        if not np.any(valid_mask):
            raise ValueError(
                "No valid bins remain after applying the expected-floor mask."
            )
        # Summarize the automatic masking decision.
        mask_report = {
            "total_bins": valid_mask.size,
            "valid_bins": np.count_nonzero(valid_mask),
            "excluded_bins": np.count_nonzero(~valid_mask),
            "observed_counts_excluded": observed_at_floor,
            "expected_floor": self.expected_floor,
            "minimum_expected": self.expected_hist.contents.min(),
            "minimum_retained_expected": self.expected_hist.contents[
                valid_mask
            ].min(),
        }
        return valid_mask, mask_report

    def projection_function(self, *axis_names: str) -> dict[str, Histogram]:
        """Project masked histograms onto the requested axes.

        Args:
            *axis_names: Histogram axis labels to retain in the projection.

        Returns:
            A dictionary containing the projected observed and expected histograms.
        """

        if not axis_names:
            raise ValueError("Provide at least one axis for the projection.")

        available_axes = set(self.observed_hist.axes.labels)
        unknown_axes = set(axis_names) - available_axes

        if unknown_axes:
            raise ValueError(
                f"Unknown projection axes: {sorted(unknown_axes)}"
            )

        projected_obs = self.masked_observed_hist.project(*axis_names)
        projected_exp = self.masked_expected_hist.project(*axis_names)

        # Return both projections under their product names.
        return {
            "observed": projected_obs,
            "expected": projected_exp,
        }

    def select_energy_range(
        self,
        energy_min: float,
        energy_max: float,
    ) -> "ResidualAnalysis":
        """Create a residual analysis for an exact measured-energy interval.

        The requested limits must coincide with existing Em bin edges.

        Args:
            energy_min: Lower energy boundary in keV.
            energy_max: Upper energy boundary in keV.

        Returns:
            A residual analysis with all selected Em bins aggregated into
            one energy bin.
        """
        if "Em" not in self.observed_hist.axes.labels:
            raise ValueError("The input histograms do not contain an Em axis.")

        numeric_types = (int, float, np.integer, np.floating)
        if (
            isinstance(energy_min, bool)
            or isinstance(energy_max, bool)
            or not isinstance(energy_min, numeric_types)
            or not isinstance(energy_max, numeric_types)
        ):
            raise TypeError(
                "energy_min and energy_max must be numeric values in keV."
            )

        energy_min = float(energy_min)
        energy_max = float(energy_max)

        if not np.isfinite(energy_min) or not np.isfinite(energy_max):
            raise ValueError("energy_min and energy_max must be finite.")
        if energy_min >= energy_max:
            raise ValueError("energy_min must be smaller than energy_max.")

        energy_axis = self.observed_hist.axes["Em"]
        energy_edges = energy_axis.edges.to_value(u.keV)
        available_edges = ", ".join(f"{edge:g}" for edge in energy_edges)

        # Locate limits that coincide with existing energy-bin edges.
        lower_matches = np.flatnonzero(
            np.isclose(energy_edges, energy_min, rtol=0.0, atol=1e-9)
        )
        upper_matches = np.flatnonzero(
            np.isclose(energy_edges, energy_max, rtol=0.0, atol=1e-9)
        )

        if lower_matches.size == 0 or upper_matches.size == 0:
            raise ValueError(
                "energy_min and energy_max must match existing Em bin edges. "
                f"Available edges in keV: [{available_edges}]"
            )

        lower_edge_index = int(lower_matches[0])
        upper_edge_index = int(upper_matches[0])

        if lower_edge_index >= upper_edge_index:
            raise ValueError(
                "energy_min must correspond to an edge below energy_max."
            )

        # Select the same energy bins from observed and expected counts.
        energy_axis_index = list(self.observed_hist.axes.labels).index("Em")
        selection = [slice(None)] * self.observed_hist.ndim
        selection[energy_axis_index] = slice(
            lower_edge_index,
            upper_edge_index,
        )
        selection = tuple(selection)

        # Aggregate the selected energy bins while retaining a length-one Em axis.
        selected_observed_contents = np.sum(
            self.masked_observed_hist.contents[selection],
            axis=energy_axis_index,
            keepdims=True,
        )
        selected_expected_contents = np.sum(
            self.masked_expected_hist.contents[selection],
            axis=energy_axis_index,
            keepdims=True,
        )

        selected_axes = list(self.observed_hist.axes)
        selected_axes[energy_axis_index] = Axis(
            [energy_min, energy_max],
            label="Em",
            scale=energy_axis.axis_scale,
            unit=u.keV,
        )

        selected_observed = Histogram(
            selected_axes,
            contents=selected_observed_contents,
        )
        selected_expected = Histogram(
            selected_axes,
            contents=selected_expected_contents,
        )

        selected_analysis = ResidualAnalysis(
            selected_observed,
            selected_expected,
            expected_floor=self.expected_floor,
        )

        # Preserve the selected time interval in the energy-selected analysis.
        selected_analysis.time_bin_index = self.time_bin_index
        selected_analysis.time_start = self.time_start
        selected_analysis.time_stop = self.time_stop

        # Preserve the temporal multinomial reference without renormalizing
        # the probability retained by the selected energy interval.
        if self._temporal_reference_probabilities is not None:
            selected_analysis._temporal_reference_probabilities = np.sum(
                self._temporal_reference_probabilities[selection],
                axis=energy_axis_index,
                keepdims=True,
            )
            selected_analysis._temporal_total_counts = (
                self._temporal_total_counts
            )

        return selected_analysis

    def split_psichi_function(
        self,
        psichi_products: dict[str, Histogram],
        bin_width: float = 5.0,
    ) -> dict[str, dict[str, Histogram]]:
        """Split a PsiChi projection into Chi and Psi histograms.

        Args:
            psichi_products: Histograms sharing the same PsiChi axis and binning.
            bin_width: Angular bin width in degrees for Chi and Psi.

        Returns:
            Chi and Psi products, retaining every non-PsiChi axis.
        """
        if not psichi_products:
            raise ValueError("Provide at least one PsiChi histogram to split.")

        # Use the first product to get the shared HEALPix and retained axes.
        reference_histogram = next(iter(psichi_products.values()))
        psichi_axis = reference_histogram.axes["PsiChi"]
        retained_axis_names = [
            axis_name
            for axis_name in reference_histogram.axes.labels
            if axis_name != "PsiChi"
        ]
        retained_axes = [
            reference_histogram.axes[axis_name]
            for axis_name in retained_axis_names
        ]

        # Get the index of every HEALPix pixel.
        psichi_pixels = np.arange(psichi_axis.npix)

        # Get the sky coordinates of each pixel center.
        psichi_coordinates = psichi_axis.pix2skycoord(psichi_pixels)

        # Extract Chi and Psi coordinates in degrees.
        chi_centers = psichi_coordinates.spherical.lon.to_value(u.deg)
        psi_centers = psichi_coordinates.spherical.lat.to_value(u.deg)

        # Define the angular bin width for coordinate splitting.
        coordinate_bin_width_deg = bin_width

        if coordinate_bin_width_deg <= 0:
            raise ValueError("bin_width must be positive.")

        # Create Chi bin edges and include the full coordinate range.
        chi_edges = np.arange(
            0.0,
            360.0,
            coordinate_bin_width_deg,
        )
        if np.isclose(chi_edges[-1], 360.0):
            chi_edges[-1] = 360.0
        else:
            chi_edges = np.append(chi_edges, 360.0)

        # Create Psi bin edges and include the full coordinate range.
        psi_edges = np.arange(
            -90.0,
            90.0,
            coordinate_bin_width_deg,
        )
        if np.isclose(psi_edges[-1], 90.0):
            psi_edges[-1] = 90.0
        else:
            psi_edges = np.append(psi_edges, 90.0)

        # Calculate the centers of the new one-dimensional coordinate bins.
        chi_bin_centers = 0.5 * (
            chi_edges[:-1] + chi_edges[1:]
        )
        psi_bin_centers = 0.5 * (
            psi_edges[:-1] + psi_edges[1:]
        )

        # Define labeled HistPy axes for the separated coordinates.
        chi_axis = Axis(chi_edges, label="Chi", unit=u.deg)
        psi_axis = Axis(psi_edges, label="Psi", unit=u.deg)

        # Prepare containers for the separated data products.
        chi_products = {}
        psi_products = {}

        # Apply the same Chi and Psi binning to every input product.
        for product_name, histogram in psichi_products.items():
            values = np.asarray(
                histogram.contents,
                dtype=float,
            )

            psichi_axis_index = list(histogram.axes.labels).index("PsiChi")
            values_by_pixel = np.moveaxis(values, psichi_axis_index, -1)
            retained_shape = values_by_pixel.shape[:-1]
            flattened_values = values_by_pixel.reshape(-1, psichi_axis.npix)

            # Rebin every retained-axis row using the same sky coordinates.
            chi_values = np.stack(
                [
                    np.histogram(
                        chi_centers,
                        bins=chi_edges,
                        weights=row,
                    )[0]
                    for row in flattened_values
                ]
            ).reshape(*retained_shape, len(chi_edges) - 1)
            psi_values = np.stack(
                [
                    np.histogram(
                        psi_centers,
                        bins=psi_edges,
                        weights=row,
                    )[0]
                    for row in flattened_values
                ]
            ).reshape(*retained_shape, len(psi_edges) - 1)

            chi_axes = [*retained_axes, chi_axis]
            psi_axes = [*retained_axes, psi_axis]

            chi_products[product_name] = Histogram(
                chi_axes,
                contents=chi_values,
                track_overflow=False,
            )
            psi_products[product_name] = Histogram(
                psi_axes,
                contents=psi_values,
                track_overflow=False,
            )

            # Confirm that the separation preserves the counts.
            if not np.isclose(
                chi_values.sum(),
                values.sum(),
            ):
                raise ValueError(
                    f"The Chi separation does not preserve {product_name}."
                )
            if not np.isclose(
                psi_values.sum(),
                values.sum(),
            ):
                raise ValueError(
                    f"The Psi separation does not preserve {product_name}."
                )

        axis_prefix = "_".join(retained_axis_names)
        chi_name = f"{axis_prefix}_Chi" if axis_prefix else "Chi"
        psi_name = f"{axis_prefix}_Psi" if axis_prefix else "Psi"
        return {
            chi_name: chi_products,
            psi_name: psi_products,
        }


    def projected_data(self) -> dict[str, dict[str, Histogram]]:
        """Prepare the supported projected and split data products."""

        # Reuse the projections if they have already been built.
        if self._projected_data_cache is not None:
            return self._projected_data_cache

        # Require an explicit energy interval before making CDS projections
        # from input containing more than one measured-energy bin.
        has_energy_axis = "Em" in self.observed_hist.axes.labels
        if has_energy_axis and self.observed_hist.axes["Em"].nbins > 1:
            em_psichi_products = self.projection_function("Em", "PsiChi")
            em_split_products = self.split_psichi_function(em_psichi_products)
            self._projected_data_cache = {
                "Em_projection": self.projection_function("Em"),
                "Em_Phi_projection": self.projection_function("Em", "Phi"),
                "Em_Chi_projection": em_split_products["Em_Chi"],
                "Em_Psi_projection": em_split_products["Em_Psi"],
                "Em_PsiChi_projection": em_psichi_products,
            }
            return self._projected_data_cache

        phi_products = self.projection_function("Phi")
        psichi_products = self.projection_function("PsiChi")
        phi_psichi_products = self.projection_function("Phi", "PsiChi")

        split_products = self.split_psichi_function(psichi_products)
        phi_split_products = self.split_psichi_function(phi_psichi_products)

        projection_dictionary = {
            "Phi_projection": phi_products,
            "Chi_projection": split_products["Chi"],
            "Psi_projection": split_products["Psi"],
            "PsiChi_projection": psichi_products,
            "Phi_Chi_projection": phi_split_products["Phi_Chi"],
            "Phi_Psi_projection": phi_split_products["Phi_Psi"],
            "Phi_PsiChi_projection": phi_psichi_products,
        }
        # Add the energy projection only when the input contains an Em axis.
        if "Em" in self.observed_hist.axes.labels:
            projection_dictionary["Em_projection"] = self.projection_function("Em")

        # Save the projections so later calls can reuse them.
        self._projected_data_cache = projection_dictionary
        return self._projected_data_cache


    def residual_function(
        self,
        projection_id: str,
        residual_type: str = "raw",
    ) -> Histogram:
        """Calculate a selected residual for one named projection.

        Args:
            projection_id: Key identifying a supported projected data product.
            residual_type: Residual definition: ``raw``, ``pearson``, or
                ``fractional``.

        Returns:
            A histogram containing the selected residual values.
        """
        projected_data = self.projected_data()

        if projection_id not in projected_data:
            raise ValueError(f"Unknown projection: {projection_id}")

        projection_products = projected_data[projection_id]
        observed_hist = projection_products["observed"]
        expected_hist = projection_products["expected"]

        # Calculate the observed-minus-expected difference.
        difference = observed_hist - expected_hist
        # Keep only bins with positive projected expectation for normalized residuals.
        valid_mask = expected_hist.contents > 0

        if residual_type == "raw":
            residual = difference

        elif residual_type == "pearson":
            # Use the observed-minus-expected histogram as the numerator.
            pearson_numerator = difference

            # Leave undefined bins as NaN while preserving the histogram shape.
            pearson_values = np.full(
                expected_hist.shape,
                np.nan,
                dtype=float,
            )

            pearson_values[valid_mask] = (
                pearson_numerator.contents[valid_mask]
                / np.sqrt(expected_hist.contents[valid_mask])
            )

            # Preserve the axes of the selected projection.
            residual = Histogram(
                list(expected_hist.axes),
                contents=pearson_values,
                track_overflow=False,
            )

        elif residual_type == "fractional":
            # Calculate fractional residuals in bins with positive expectation.
            fractional_values = np.full(
                expected_hist.shape,
                np.nan,
                dtype=float,
            )

            fractional_values[valid_mask] = (
                difference.contents[valid_mask]
                / expected_hist.contents[valid_mask]
            )

            residual = Histogram(
                list(expected_hist.axes),
                contents=fractional_values,
                track_overflow=False,
            )
        else:
            raise ValueError(
                f"Unknown residual type {residual_type!r}. "
                "Choose 'raw', 'pearson', or 'fractional'."
            )
            
        return residual

    def _project_simulated_counts(
        self,
        simulated_counts: np.ndarray,
        projection_id: str,
    ) -> Histogram:
        """Apply the analysis mask and existing binning to simulated CDS counts."""
        masked_histogram = Histogram(
            list(self.expected_hist.axes),
            contents=np.where(self.mask, simulated_counts, 0.0),
            track_overflow=False,
        )

        direct_axes = {
            "Em_projection": ("Em",),
            "Em_Phi_projection": ("Em", "Phi"),
            "Em_PsiChi_projection": ("Em", "PsiChi"),
            "Phi_projection": ("Phi",),
            "PsiChi_projection": ("PsiChi",),
            "Phi_PsiChi_projection": ("Phi", "PsiChi"),
        }
        split_axes = {
            "Em_Chi_projection": (("Em", "PsiChi"), "Em_Chi"),
            "Em_Psi_projection": (("Em", "PsiChi"), "Em_Psi"),
            "Chi_projection": (("PsiChi",), "Chi"),
            "Psi_projection": (("PsiChi",), "Psi"),
            "Phi_Chi_projection": (("Phi", "PsiChi"), "Phi_Chi"),
            "Phi_Psi_projection": (("Phi", "PsiChi"), "Phi_Psi"),
        }

        if projection_id in direct_axes:
            return masked_histogram.project(*direct_axes[projection_id])

        if projection_id in split_axes:
            axes, split_name = split_axes[projection_id]
            psichi_histogram = masked_histogram.project(*axes)
            split_products = self.split_psichi_function(
                {"simulated": psichi_histogram}
            )
            return split_products[split_name]["simulated"]

        raise ValueError(f"Unknown projection: {projection_id}")

    def pearson_poisson_reference(
        self,
        projection_id: str,
        n_simulations: int = 1000,
        bins: int = 15,
        seed: int | None = None,
        phi_bin_index: int | None = None,
        energy_bin_index: int | None = None,
    ) -> dict[str, np.ndarray | int]:
        """Get a simulated Pearson reference for one projection."""
        references = self.prepare_pearson_poisson_references(
            {projection_id: bins},
            n_simulations=n_simulations,
            seed=seed,
            phi_bin_indices=(
                {projection_id: phi_bin_index}
                if phi_bin_index is not None
                else None
            ),
            energy_bin_indices=(
                {projection_id: energy_bin_index}
                if energy_bin_index is not None
                else None
            ),
        )
        slice_index = (
            phi_bin_index
            if phi_bin_index is not None
            else energy_bin_index
        )
        return references[(projection_id, slice_index)]

    def prepare_pearson_poisson_references(
        self,
        projection_bins: dict[str, int],
        n_simulations: int = 1000,
        seed: int | None = None,
        phi_bin_indices: dict[str, int] | None = None,
        energy_bin_indices: dict[str, int] | None = None,
    ) -> dict[tuple[str, int | None], dict[str, np.ndarray | int]]:
        """Prepare references for several plots from shared CDS simulations.

        Each full-CDS reference realization is generated once and then
        projected to every requested view. Ordinary analyses use independent
        Poisson counts. Temporal shape analyses use multinomial counts with
        the observed interval total fixed. Projected residual values are
        temporarily retained to establish common histogram edges; full-CDS
        datasets are not stored. The resulting summaries are cached for the
        plotting methods.

        Args:
            projection_bins: Mapping from projection IDs to distribution
                histogram bin counts. For example, use 15 for 1D views and 40
                for 2D or HEALPix views.
            n_simulations: Number of independent reference realizations.
            seed: Optional seed for reproducibility.
            phi_bin_indices: Optional Phi-bin index for each requested
                ``"Phi_PsiChi_projection"`` slice.
            energy_bin_indices: Optional Em-bin index for each requested
                ``"Em_PsiChi_projection"`` slice.

        Returns:
            References keyed by ``(projection_id, slice_index)``. The slice
            index is a Phi index for ``Phi_PsiChi_projection``, an Em index for
            ``Em_PsiChi_projection``, and ``None`` otherwise. Each reference
            contains common ``bin_edges``, mean ``density``, pointwise 95%
            ``lower`` and ``upper`` limits, and the seed used.

        Notes:
            This fixed-fit reference is not a goodness-of-fit test. The
            original mask is applied before projection, including its handling
            of artificial expected-count floors.
        """
        if (
            isinstance(n_simulations, bool)
            or not isinstance(n_simulations, int)
            or n_simulations < 2
        ):
            raise ValueError("n_simulations must be an integer of at least 2.")
        if not projection_bins:
            raise ValueError("Provide at least one projection to simulate.")

        phi_bin_indices = phi_bin_indices or {}
        energy_bin_indices = energy_bin_indices or {}
        projected_data = self.projected_data()
        reference_type = (
            "multinomial"
            if self._temporal_reference_probabilities is not None
            else "poisson"
        )
        effective_seed = (
            int(seed)
            if seed is not None
            else int(np.random.SeedSequence().generate_state(1)[0])
        )

        requests = {}
        references = {}
        for projection_id, bins in projection_bins.items():
            if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
                raise ValueError("Each histogram bin count must be a positive integer.")
            if projection_id not in projected_data:
                raise ValueError(f"Unknown projection: {projection_id}")

            phi_bin_index = phi_bin_indices.get(projection_id)
            energy_bin_index = energy_bin_indices.get(projection_id)
            if phi_bin_index is not None and energy_bin_index is not None:
                raise ValueError(
                    "A projection cannot use both Phi and Em slice indices."
                )
            if phi_bin_index is not None and projection_id != "Phi_PsiChi_projection":
                raise ValueError(
                    "phi_bin_indices applies only to Phi_PsiChi_projection."
                )
            if (
                energy_bin_index is not None
                and projection_id != "Em_PsiChi_projection"
            ):
                raise ValueError(
                    "energy_bin_indices applies only to Em_PsiChi_projection."
                )
            if projection_id == "Phi_PsiChi_projection" and phi_bin_index is None:
                raise ValueError(
                    "Provide a Phi-bin index for Phi_PsiChi_projection."
                )
            if (
                projection_id == "Em_PsiChi_projection"
                and energy_bin_index is None
            ):
                raise ValueError(
                    "Provide an Em-bin index for Em_PsiChi_projection."
                )

            slice_index = (
                phi_bin_index
                if phi_bin_index is not None
                else energy_bin_index
            )

            cache_key = (
                projection_id,
                n_simulations,
                bins,
                effective_seed,
                phi_bin_index,
                energy_bin_index,
                reference_type,
            )
            reference_key = (projection_id, slice_index)
            if cache_key in self._poisson_reference_cache:
                references[reference_key] = self._poisson_reference_cache[cache_key]
                continue

            expected_values = np.asarray(
                projected_data[projection_id]["expected"].contents,
                dtype=float,
            )
            observed_values = np.asarray(
                self.residual_function(projection_id, "pearson").contents,
                dtype=float,
            )
            if phi_bin_index is not None:
                if not 0 <= phi_bin_index < expected_values.shape[0]:
                    raise IndexError(
                        "phi_bin_index is outside the available Phi bins."
                    )
                expected_values = expected_values[phi_bin_index]
                observed_values = observed_values[phi_bin_index]
            elif energy_bin_index is not None:
                if not 0 <= energy_bin_index < expected_values.shape[0]:
                    raise IndexError(
                        "energy_bin_index is outside the available Em bins."
                    )
                expected_values = expected_values[energy_bin_index]
                observed_values = observed_values[energy_bin_index]

            valid_mask = expected_values > 0
            observed_values = observed_values[np.isfinite(observed_values)]
            if observed_values.size == 0 or not np.any(valid_mask):
                raise ValueError("No valid Pearson residual bins are available.")

            requests[reference_key] = {
                "projection_id": projection_id,
                "phi_bin_index": phi_bin_index,
                "energy_bin_index": energy_bin_index,
                "bins": bins,
                "expected": expected_values,
                "valid_mask": valid_mask,
                "observed": observed_values,
                "minimum": float(np.min(observed_values)),
                "maximum": float(np.max(observed_values)),
                "cache_key": cache_key,
                "samples": [],
            }

        if not requests:
            return references

        expected_full = np.asarray(
            self.masked_expected_hist.contents,
            dtype=float,
        )

        multinomial_probabilities = None
        if reference_type == "multinomial":
            temporal_probabilities = np.asarray(
                self._temporal_reference_probabilities,
                dtype=float,
            )
            if temporal_probabilities.shape != self.expected_hist.shape:
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
            simulated_projection = self._project_simulated_counts(
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
                    self._temporal_total_counts,
                    multinomial_probabilities,
                )
                simulated_full = simulated_with_outside[:-1].reshape(
                    self.expected_hist.shape
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
            self._poisson_reference_cache[request["cache_key"]] = reference

        return references

    def data_summary_table(self) -> pd.DataFrame:
        """Describe the current observed/model histograms without projections.

        Counts labeled 'all bins' include excluded bins; counts labeled
        'valid bins' use the same analysis mask for observed and expected.
        A selected-range analysis describes its already aggregated input.
        """
        rows = []
        for axis in self.observed_hist.axes:
            rows.append((f"{axis.label} bins", int(axis.nbins)))
            if axis.label == "Em":
                edges = axis.edges.to_value(u.keV)
                rows.append(("Em edges [keV]", tuple(float(x) for x in edges)))
            elif axis.label == "Phi":
                edges = axis.edges.to_value(u.deg)
                rows.append(("Phi range [deg]", (float(edges[0]), float(edges[-1]))))
            elif axis.label == "PsiChi":
                rows.append(("PsiChi HEALPix nside", int(axis.nside)))

        # Report geometry and masking separately from the count totals.
        valid_bins = int(np.count_nonzero(self.mask))
        rows.extend([
            ("Total bins", int(self.mask.size)),
            ("Valid bins", valid_bins),
            ("Excluded bins", int(self.mask.size) - valid_bins),
            ("Observed counts (all bins)", float(self.observed_hist.contents.sum())),
            ("Expected counts (all bins)", float(self.expected_hist.contents.sum())),
            ("Observed counts (valid bins)", float(self.observed_hist.contents[self.mask].sum())),
            ("Expected counts (valid bins)", float(self.expected_hist.contents[self.mask].sum())),
        ])
        return pd.DataFrame(rows, columns=["Statistic", "Value"])

    def statistics_table(self, residual_type: str) -> pd.DataFrame:
        """Summarize one residual type across valid full-CDS bins.

        The compact table contains one row with descriptive residual
        statistics. Pearson residuals also report the fractions beyond two
        and three standard deviations, Pearson chi-square, and Poisson
        deviance. These are descriptive summaries; this method does not
        calculate degrees of freedom or a goodness-of-fit probability.

        Args:
            residual_type: ``"raw"``, ``"pearson"``, or ``"fractional"``.

        Returns:
            A one-row DataFrame with the global full-CDS summary.
        """
        residual_types = {"raw", "pearson", "fractional"}
        if residual_type not in residual_types:
            raise ValueError(
                "Unknown residual type. Choose 'raw', 'pearson', "
                "or 'fractional'."
            )
        if (
            "Em" in self.observed_hist.axes.labels
            and self.observed_hist.axes["Em"].nbins > 1
        ):
            raise ValueError(
                "Select an energy interval before calculating full-CDS "
                "statistics."
            )

        observed = np.asarray(
            self.masked_observed_hist.contents[self.mask],
            dtype=float,
        )
        expected = np.asarray(
            self.masked_expected_hist.contents[self.mask],
            dtype=float,
        )
        difference = observed - expected

        if residual_type == "raw":
            residual_values = difference
        elif residual_type == "pearson":
            residual_values = difference / np.sqrt(expected)
        else:
            residual_values = difference / expected

        summary = {
            "Valid bins (N)": residual_values.size,
            "Mean residual": float(np.mean(residual_values)),
            "Median residual": float(np.median(residual_values)),
            "Std. dev.": float(np.std(residual_values)),
            "Residual RMS": float(np.sqrt(np.mean(residual_values**2))),
            "Minimum residual": float(np.min(residual_values)),
            "Maximum residual": float(np.max(residual_values)),
        }

        if residual_type == "pearson":
            positive_observed = observed > 0
            deviance_terms = expected.copy()
            deviance_terms[positive_observed] += observed[positive_observed] * (
                np.log(observed[positive_observed] / expected[positive_observed])
                - 1.0
            )
            summary.update(
                {
                    "|R| > 2 [%]": float(
                        100 * np.mean(np.abs(residual_values) > 2)
                    ),
                    "|R| > 3 [%]": float(
                        100 * np.mean(np.abs(residual_values) > 3)
                    ),
                    "Pearson chi-square": float(np.sum(residual_values**2)),
                    "Poisson deviance": float(2 * np.sum(deviance_terms)),
                }
            )

        return pd.DataFrame([summary])
