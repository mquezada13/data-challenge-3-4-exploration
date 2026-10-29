"""Construct time-resolved observed counts from existing binned data."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import numpy as np
from histpy import Histogram


class TimeResolvedResidualAnalysis:
    """Construct D(t,i) from the original time-binned observations."""

    # ----------------------------------------------------------------------
    # Input data and configuration
    # ----------------------------------------------------------------------
    def __init__(
        self,
        integrated_observed: Histogram,
        integrated_expected: Histogram,
        time_binned_observed: Histogram | str | Path | Sequence[str | Path],
        observed_time_histogram_name: str = "hist",
    ) -> None:
        if integrated_expected.axes != integrated_observed.axes:
            raise ValueError("Integrated D and M must have matching CDS axes.")

        self.integrated_observed_hist = integrated_observed
        self.integrated_expected_hist = integrated_expected

        if isinstance(time_binned_observed, Histogram):
            self.time_binned_observed = time_binned_observed
        elif isinstance(time_binned_observed, (str, Path)):
            self.time_binned_observed = (Path(time_binned_observed),)
        else:
            self.time_binned_observed = tuple(
                Path(path) for path in time_binned_observed
            )

        self.observed_time_histogram_name = observed_time_histogram_name
        self.observed_hist: Histogram | None = None
        self.expected_hist: Histogram | None = None

    # ----------------------------------------------------------------------
    # Observed counts D(t,i)
    # ----------------------------------------------------------------------
    def build_observed(self) -> Histogram:
        """Build observed counts using the original time-binned data."""
        observed_histograms = self._load_observed_time_histograms()

        self._validate_observed_time_axes(observed_histograms)
        self._validate_observed_time_binning(observed_histograms)

        combined = self._combine_observed_time_histograms(
            observed_histograms
        )

        self._validate_observed_time_consistency(combined)
        self.observed_hist = self._densify_observed_counts(combined)

        return self.observed_hist

    def build_expected(self) -> Histogram:
        """Normalize the integrated model to the observed counts in each time bin."""
        # Build D(t,i) first if it is not already available.
        if self.observed_hist is None:
            self.build_observed()

        # Convert the integrated model to a dense floating-point array.
        reference = self.integrated_expected_hist.to_dense(copy=True)
        model_values = np.asarray(reference.contents, dtype=float)

        # Validate the integrated model before normalizing it.
        if not np.all(np.isfinite(model_values)):
            raise ValueError("Integrated M contains non-finite values.")

        if np.any(model_values < 0):
            raise ValueError("Integrated M contains negative values.")

        model_total = model_values.sum()

        if model_total <= 0:
            raise ValueError("Integrated M must have a positive total.")

        # Sum D(t,i) over every CDS axis while preserving the Time axis.
        cds_axes = tuple(range(1, self.observed_hist.ndim))
        interval_totals = np.asarray(
            self.observed_hist.contents.sum(axis=cds_axes),
            dtype=float,
        )

        # Convert M(i) into the normalized CDS shape p(i).
        normalized_model = model_values / model_total

        # Reshape N(t) so it can multiply every CDS bin in each time interval.
        interval_shape = (len(interval_totals),) + (1,) * reference.ndim

        # Construct M_tilde(t,i) = N(t) M(i) / sum_j M(j).
        expected_contents = (
            interval_totals.reshape(interval_shape) * normalized_model
        )

        # Verify that each normalized model has the observed interval total.
        reconstructed_interval_totals = expected_contents.sum(axis=cds_axes)

        if not np.allclose(
            reconstructed_interval_totals,
            interval_totals,
            rtol=1e-10,
            atol=1e-10,
        ):
            raise ValueError(
                "Normalized M does not reproduce the observed total "
                "in every time bin."
            )

        # Store the complete M_tilde(t,i) histogram with the D(t,i) axes.
        self.expected_hist = Histogram(
            self.observed_hist.axes,
            contents=expected_contents,
            copy_contents=False,
        )

        return self.expected_hist



    def _load_observed_time_histograms(self) -> tuple[Histogram, ...]:
        """Load all time-binned observed histograms."""
        if isinstance(self.time_binned_observed, Histogram):
            return (self.time_binned_observed,)

        if not self.time_binned_observed:
            raise ValueError("No time-binned observations supplied.")

        return tuple(
            Histogram.open(path, name=self.observed_time_histogram_name)
            for path in self.time_binned_observed
        )

    def _validate_observed_time_axes(
        self,
        histograms: tuple[Histogram, ...],
    ) -> None:
        """Check that time and CDS axes are present and ordered correctly."""
        expected_labels = (
            "Time",
            *(str(label) for label in self.integrated_observed_hist.axes.labels),
        )

        for index, histogram in enumerate(histograms):
            labels = tuple(str(label) for label in histogram.axes.labels)
            if labels != expected_labels:
                raise ValueError(
                    f"Observed histogram {index} has incompatible axis ordering."
                )

    def _validate_observed_time_binning(
        self,
        histograms: tuple[Histogram, ...],
    ) -> None:
        """Check compatible time and CDS binning."""
        reference_time_axis = histograms[0].axes["Time"]

        for index, histogram in enumerate(histograms):
            if histogram.axes["Time"] != reference_time_axis:
                raise ValueError(
                    f"Observed histogram {index} has incompatible time bins."
                )

            for label in self.integrated_observed_hist.axes.labels:
                if (
                    histogram.axes[str(label)]
                    != self.integrated_observed_hist.axes[str(label)]
                ):
                    raise ValueError(
                        f"Observed histogram {index} has incompatible {label} bins."
                    )

    @staticmethod
    def _combine_observed_time_histograms(
        histograms: tuple[Histogram, ...],
    ) -> Histogram:
        """Combine observed counts across all input histograms."""
        contents = histograms[0].contents.copy()

        for histogram in histograms[1:]:
            contents = contents + histogram.contents

        return Histogram(
            histograms[0].axes,
            contents=contents,
            copy_contents=False,
        )

    @staticmethod
    def _densify_observed_counts(histogram: Histogram) -> Histogram:
        """Preserve integer counts while reducing dense-memory usage."""
        if not histogram.is_sparse:
            return histogram.to_dense(copy=True)

        contents = histogram.contents
        values = contents.data
        fill_value = contents.fill_value

        integer_counts = (
            np.all(np.isfinite(values))
            and np.isfinite(fill_value)
            and np.all(values >= 0)
            and fill_value >= 0
            and np.all(values == np.floor(values))
            and fill_value == np.floor(fill_value)
        )

        if not integer_counts:
            return histogram.to_dense(copy=True)

        maximum = max(
            float(fill_value),
            float(values.max()) if values.size else 0.0,
        )

        for dtype in (np.uint8, np.uint16, np.uint32):
            if maximum <= np.iinfo(dtype).max:
                return histogram.astype(
                    dtype, copy=False
                ).to_dense(copy=False)

        return histogram.to_dense(copy=True)

    def _validate_observed_time_consistency(
        self,
        observed: Histogram,
    ) -> None:
        """Verify that summing D(t,i) reproduces integrated D."""
        labels = tuple(
            str(label) for label in self.integrated_observed_hist.axes.labels
        )

        summed = observed.project(*labels).to_dense(copy=True)
        reference = self.integrated_observed_hist.to_dense(copy=True)

        if not np.array_equal(summed.contents, reference.contents):
            raise ValueError(
                "Time-resolved D does not reproduce integrated D."
            )