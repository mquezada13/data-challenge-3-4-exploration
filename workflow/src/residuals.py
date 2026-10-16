import numpy as np
import pandas as pd
from histpy import Axis, Histogram
import astropy.units as u


class ResidualAnalysis:
    def __init__(
        self,
        observed: Histogram,
        expected: Histogram,
        expected_floor: float | None = None,
    ) -> None:
        """Initialize the analysis with observed and expected histograms.

        Args:
            observed: Histogram containing observed counts.
            expected: Histogram containing fitted expected counts.
            expected_floor: Optional artificial floor used when preparing the
                expected histogram. Bins at or below this value are excluded
                only when they contain no observed counts.
        """
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

    def select_energy_bin(self, energy_bin_index: int) -> "ResidualAnalysis":
        """Create a residual analysis for one measured-energy bin.

        Args:
            energy_bin_index: Zero-based index of the Em bin to analyze.

        Returns:
            A residual analysis retaining the selected Em bin as a
            one-bin energy axis.
        """
        if "Em" not in self.observed_hist.axes.labels:
            raise ValueError("The input histograms do not contain an Em axis.")
        if (
            isinstance(energy_bin_index, bool)
            or not isinstance(energy_bin_index, (int, np.integer))
        ):
            raise TypeError("energy_bin_index must be an integer.")

        number_of_energy_bins = self.observed_hist.axes["Em"].nbins
        if not 0 <= energy_bin_index < number_of_energy_bins:
            raise IndexError("energy_bin_index is outside the available Em bins.")

        selected_observed = self.observed_hist.slice[
            {"Em": energy_bin_index}
        ].copy()
        selected_expected = self.expected_hist.slice[
            {"Em": energy_bin_index}
        ].copy()

        return ResidualAnalysis(
            selected_observed,
            selected_expected,
            expected_floor=self.expected_floor,
        )

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
            Chi and Psi products, retaining Phi when it is present.
        """
        if not psichi_products:
            raise ValueError("Provide at least one PsiChi histogram to split.")

        # Use the first product to get the shared HEALPix and optional Phi axes.
        reference_histogram = next(iter(psichi_products.values()))
        psichi_axis = reference_histogram.axes["PsiChi"]
        has_phi = "Phi" in reference_histogram.axes.labels

        # Keep the Phi axis when it is present in the projection.
        if has_phi:
            phi_axis = reference_histogram.axes["Phi"]

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

            if has_phi:
                # Rebin each Phi row separately and preserve the Phi axis.
                chi_values = np.stack(
                    [
                        np.histogram(
                            chi_centers,
                            bins=chi_edges,
                            weights=phi_row,
                        )[0]
                        for phi_row in values
                    ]
                )
                psi_values = np.stack(
                    [
                        np.histogram(
                            psi_centers,
                            bins=psi_edges,
                            weights=phi_row,
                        )[0]
                        for phi_row in values
                    ]
                )
                chi_axes = [phi_axis, chi_axis]
                psi_axes = [phi_axis, psi_axis]
            else:
                # Rebin the one-dimensional PsiChi histogram directly.
                chi_values, _ = np.histogram(
                    chi_centers,
                    bins=chi_edges,
                    weights=values,
                )
                psi_values, _ = np.histogram(
                    psi_centers,
                    bins=psi_edges,
                    weights=values,
                )
                chi_axes = [chi_axis]
                psi_axes = [psi_axis]

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

        if has_phi:
            return {
                "Phi_Chi": chi_products,
                "Phi_Psi": psi_products,
            }

        return {
            "Chi": chi_products,
            "Psi": psi_products,
        }


    def projected_data(self) -> dict[str, dict[str, Histogram]]:
        """Prepare the supported projected and split data products."""

        # Reuse the projections if they have already been built.
        if self._projected_data_cache is not None:
            return self._projected_data_cache

        # Require an explicit selection before making CDS projections from
        # input containing more than one measured-energy bin.
        has_energy_axis = "Em" in self.observed_hist.axes.labels
        if has_energy_axis and self.observed_hist.axes["Em"].nbins > 1:
            self._projected_data_cache = {
                "Em_projection": self.projection_function("Em")
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
            "Phi_projection": ("Phi",),
            "PsiChi_projection": ("PsiChi",),
            "Phi_PsiChi_projection": ("Phi", "PsiChi"),
        }
        split_axes = {
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
        )
        return references[(projection_id, phi_bin_index)]

    def prepare_pearson_poisson_references(
        self,
        projection_bins: dict[str, int],
        n_simulations: int = 1000,
        seed: int | None = None,
        phi_bin_indices: dict[str, int] | None = None,
    ) -> dict[tuple[str, int | None], dict[str, np.ndarray | int]]:
        """Prepare references for several plots from shared CDS simulations.

        Each full-CDS Poisson realization is generated once and then projected
        to every requested view. Projected residual values are temporarily
        retained to establish common histogram edges; full-CDS datasets are
        not stored. The resulting summaries are cached for the plotting methods.

        Args:
            projection_bins: Mapping from projection IDs to distribution
                histogram bin counts. For example, use 15 for 1D views and 40
                for 2D or HEALPix views.
            n_simulations: Number of independent Poisson realizations.
            seed: Optional seed for reproducibility.
            phi_bin_indices: Optional Phi-bin index for each requested
                ``"Phi_PsiChi_projection"`` slice.

        Returns:
            References keyed by ``(projection_id, phi_bin_index)``. Each
            reference contains common ``bin_edges``, mean ``density``,
            pointwise 95% ``lower`` and ``upper`` limits, and the seed used.

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
        projected_data = self.projected_data()
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
            if phi_bin_index is not None and projection_id != "Phi_PsiChi_projection":
                raise ValueError(
                    "phi_bin_indices applies only to Phi_PsiChi_projection."
                )
            if projection_id == "Phi_PsiChi_projection" and phi_bin_index is None:
                raise ValueError(
                    "Provide a Phi-bin index for Phi_PsiChi_projection."
                )

            cache_key = (
                projection_id,
                n_simulations,
                bins,
                effective_seed,
                phi_bin_index,
            )
            reference_key = (projection_id, phi_bin_index)
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

            valid_mask = expected_values > 0
            observed_values = observed_values[np.isfinite(observed_values)]
            if observed_values.size == 0 or not np.any(valid_mask):
                raise ValueError("No valid Pearson residual bins are available.")

            requests[reference_key] = {
                "projection_id": projection_id,
                "phi_bin_index": phi_bin_index,
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
            self.expected_hist.contents,
            dtype=float,
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
            expected_values = request["expected"]
            valid_mask = request["valid_mask"]
            return (
                projected_counts[valid_mask] - expected_values[valid_mask]
            ) / np.sqrt(expected_values[valid_mask])

        # Generate each full-CDS realization once for every requested view.
        generator = np.random.default_rng(effective_seed)
        for _ in range(n_simulations):
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
                "Select one Em bin before calculating full-CDS statistics."
            )

        observed = np.asarray(self.observed_hist.contents[self.mask], dtype=float)
        expected = np.asarray(self.expected_hist.contents[self.mask], dtype=float)
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
