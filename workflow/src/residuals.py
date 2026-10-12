from pathlib import Path
import sys

import numpy as np
import pandas as pd
from histpy import Axis, Histogram 
import astropy.units as u
import mhealpy as mhp
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from astropy.coordinates import SkyCoord
from scipy import stats
# Inspect the internal structure of HDF5 files.
import h5py


class ResidualAnalysis:
    def __init__(
        self,
        observed: Histogram,
        expected: Histogram,
    ) -> None:
        """Initialize the analysis with observed and expected histograms."""
        self.observed_hist = observed
        self.expected_hist = expected

        products = {
                    'observed' : self.observed_hist,
                    "expected" : self.expected_hist,
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
                copy=True
            )

            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"{product_name} contains non-finite values."
                )
            if np.any(values < 0):
                raise ValueError(
                    f"{product_name} contains negative values."
                )
        #Build and store the mask and its report.
        self.mask, self.mask_report = self.mask_function()

        # Create masked copies without changing the original histograms
        self.masked_observed_hist = Histogram(
            list(self.observed_hist.axes),
            contents = np.where(
                self.mask,
                self.observed_hist.contents,
                0.0,
            )
        )

        self.masked_expected_hist = Histogram(
            list(self.expected_hist.axes),
            contents=np.where(
                self.mask,
                self.expected_hist.contents,
                0.0
            )
        )

    #-------------------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------
    ## Masking funtion       
    #-------------------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------
    def mask_function(self) -> tuple[np.ndarray, dict[str, object]]:
        """Build the valid-bin mask and summarize its effect."""
         # Define a numerical-zero threshold near the float64 limit 
        numerical_floor = 10 * np.finfo(float).tiny

        # identify bins whose expectation is effectively numerical zero
        numerical_floor_mask = self.expected_hist.contents <= numerical_floor

        # Do not silently exclude bins that contain observed events.
        observed_at_numerical_floor = self.observed_hist.contents[numerical_floor_mask].sum()

        # Ensure that the mask retains data for the residual calculation
        if observed_at_numerical_floor > 0:
            raise ValueError(
                        "Numerical-floor bins contain observed events and cannot "
                        "be excluded automatically."
                    )
        # Exclude only safe numerical floor bins
        valid_mask = ~numerical_floor_mask

        # Ensure that the mask retains data for the residual calculation
        if not np.any(valid_mask):
             raise ValueError(
                  "No valid bins remain after applying the numerical-floor mask"
             )
        # Summarise the mask authomatic masking decision
        mask_report = {
                "total_bins": valid_mask.size,
                "valid_bins": np.count_nonzero(valid_mask),
                "excluded_bins": np.count_nonzero(~valid_mask),
                "observed_counts_excluded": observed_at_numerical_floor,
                "minimum_expected": self.expected_hist.contents.min(),
                "minimum_retained_expected": self.expected_hist.contents[valid_mask].min(),
            }
        return valid_mask, mask_report


    #-------------------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------
    ## Projection function      
    #-------------------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------

    def projection_function(self, *axis_names: str) -> dict[str, Histogram]:
         """Project the masked observed and expected histograms onto selected axes."""

         if not axis_names : 
              raise ValueError(
                   "Provide at least one axis for the projection"
              )

         reference_histograms = self.observed_hist
         available_axes = set(reference_histograms.axes.labels)
         unknown_axes = set(axis_names) - available_axes

         if unknown_axes:
              raise ValueError(
                          f"Unknown projection axes: {sorted(unknown_axes)}"
                      )

         
         projected_obs = self.masked_observed_hist.project(*axis_names)
         projected_exp = self.masked_expected_hist.project(*axis_names)
         
        # Return both projections under their product names
         return {
            "observed": projected_obs,
            "expected": projected_exp,
        }

    #-------------------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------
    ## Split PsiChi function      
    #-------------------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------
    def split_psichi_function(
        self,
        psichi_products: dict[str, Histogram],
        bin_width: float = 5.0,
    ) -> dict[str, dict[str, Histogram]]:
        """Split a PsiChi projection into Chi and Psi histograms."""
        # Get the HEALPix axis from the projected observed histogram.
        observed_histogram = psichi_products["observed"]
        psichi_axis = observed_histogram.axes["PsiChi"]
        has_phi = "Phi" in observed_histogram.axes.labels

        # Keep the Phi axis when it is present in the projection.
        if has_phi:
            phi_axis = observed_histogram.axes["Phi"]

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

    #-------------------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------
    ## Obtaining projected data  
    #-------------------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------
        
    def projected_data(self) -> dict[str, dict[str, Histogram]]:
        """Prepare the supported projected and split data products."""
        phi_products = self.projection_function("Phi")
        psichi_products = self.projection_function("PsiChi")
        phi_psichi_products = self.projection_function("Phi", "PsiChi")

        split_products = self.split_psichi_function(psichi_products)
        phi_split_products = self.split_psichi_function(phi_psichi_products)

        projection_dictionary = {
            "Phi_projection": phi_products,
            "PsiChi_projection": psichi_products,
            "Chi_projection": split_products["Chi"],
            "Psi_projection": split_products["Psi"],
            "Phi_PsiChi_projection": phi_psichi_products,
            "Phi_Chi_projection": phi_split_products["Phi_Chi"],
            "Phi_Psi_projection": phi_split_products["Phi_Psi"],
        }

        return projection_dictionary


    #-------------------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------
    ## Residual functions    
    #-------------------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------

    
    def ResidualFunction(
        self,
        projection_id: str,
        residual_type: str = "raw",
    ) -> Histogram:
        """
        Calculate a selected residual for one named projection.
        """


        projected_data = self.projected_data()

        if projection_id not in projected_data:
            raise ValueError(f"Unknown projection: {projection_id}")

        projection_products = projected_data[projection_id]
        observed_hist = projection_products["observed"]
        expected_hist = projection_products["expected"]

        # General data difference used in every residual function 
        difference = observed_hist - expected_hist
        # Creating the mask to keep only valid bins
        valid_mask = expected_hist.contents > 0

        if residual_type == "raw":
            residual = difference

        elif residual_type == "pearson":
            # Calculate the numerator as a projected histogram.
            pearson_numerator = difference


            # Select valid bins
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
            # Calculate fracrtional residuals in bins with positive expectation
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
                     contents = fractional_values,
                     track_overflow = False,
            )
        else:
            raise ValueError(
                f"Unknown residual type {residual_type!r}. "
                "Choose 'raw', 'pearson', or 'fractional'."
            )
            
        return residual

    
    def statistics_table(self) -> pd.DataFrame:
        """
        Build a table of global model-fit statistics
        """

        # Select the original observed and expected counts in valid bins.
        observed_values = np.asarray(
            self.observed_hist.contents[self.mask],
            dtype=float,
        )

        expected_values = np.asarray(
            self.expected_hist.contents[self.mask],
            dtype=float,
        )

        # Count the valid and excluded bins in the original histograms.
        valid_bin_count = int(np.count_nonzero(self.mask))
        excluded_bin_count = int(np.count_nonzero(~self.mask))

        # Sum the observed and expected counts included in the comparison.
        observed_count_total = float(np.sum(observed_values))
        expected_count_total = float(np.sum(expected_values))

        # Sum the squared Pearson residual contribution from every valid bin.
        pearson_chi_square = float(
            np.sum(
                (observed_values - expected_values) ** 2
                / expected_values
            )
        )

        # Calculate the descriptive mean contribution per valid bin.
        mean_pearson_contribution = pearson_chi_square / valid_bin_count

        # Separate bins with zero observed counts to avoid taking log(0).
        positive_observed = observed_values > 0

        # For zero observed counts, the deviance contribution is the expectation.
        deviance_sum = np.sum(expected_values[~positive_observed])

        # Calculate the remaining Poisson deviance contributions.
        deviance_sum += np.sum(
            observed_values[positive_observed]
            * np.log(
                observed_values[positive_observed]
                / expected_values[positive_observed]
            )
            - (
                observed_values[positive_observed]
                - expected_values[positive_observed]
            )
        )

        poisson_deviance = float(2.0 * deviance_sum)
        # Collect the global statistics and their interpretations.
        statistics_rows = [
            {
                "Statistic": "Valid CDS bins",
                "Value": valid_bin_count,
                "Interpretation": "Bins included in the global comparison.",
            },
            {
                "Statistic": "Excluded CDS bins",
                "Value": excluded_bin_count,
                "Interpretation": "Numerical-floor bins removed by the mask.",
            },
            {
                "Statistic": "Observed counts in valid bins",
                "Value": observed_count_total,
                "Interpretation": "Observed events included in the comparison.",
            },
            {
                "Statistic": "Expected counts in valid bins",
                "Value": expected_count_total,
                "Interpretation": "Model-predicted events in the same bins.",
            },
            {
                "Statistic": "Pearson chi-square (χ²)",
                "Value": pearson_chi_square,
                "Interpretation": "Global sum of squared Pearson residuals.",
            },
            {
                "Statistic": "Mean Pearson contribution per valid bin",
                "Value": mean_pearson_contribution,
                "Interpretation": "Descriptive average; not a reduced chi-square.",
            },
            {
                "Statistic": "Poisson deviance",
                "Value": poisson_deviance,
                "Interpretation": "Global likelihood-based discrepancy from the model.",
            },
            {
                "Statistic": "Global goodness-of-fit p-value",
                "Value": "Not calibrated",
                "Interpretation": (
                    "Requires a justified reference distribution accounting "
                    "for model fitting and low expected counts."
                ),
            },
        ]

        return pd.DataFrame(statistics_rows)
            