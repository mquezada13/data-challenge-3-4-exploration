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
