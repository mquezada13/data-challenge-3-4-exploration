"""Analyze time-resolved observed counts with an integrated model."""

from __future__ import annotations

import numpy as np
from histpy import Axis, Histogram


class TimeResolvedResidualAnalysis:
    """Prepare physical temporal predictions from observed D(t,i)."""

    # ----------------------------------------------------------------------
    # Input data and configuration
    # ----------------------------------------------------------------------
    def __init__(
        self,
        observed: Histogram,
        integrated_expected: Histogram,
    ) -> None:
        if "Time" not in observed.axes.labels:
            raise ValueError("Temporal analysis requires observed D(t,i) with a Time axis.")
        expected_labels = ("Time", *(str(label) for label in integrated_expected.axes.labels))
        if tuple(str(label) for label in observed.axes.labels) != expected_labels:
            raise ValueError("Observed D(t,i) must have Time first and match the integrated model axes.")
        for label in integrated_expected.axes.labels:
            if observed.axes[str(label)] != integrated_expected.axes[str(label)]:
                raise ValueError(f"Observed D(t,i) and integrated M have incompatible {label} bins.")

        self.integrated_expected_hist = integrated_expected
        self.observed_hist = self._densify_observed_counts(observed)
        self._physical_configuration = None
        self._physical_cache = {}

    def configure_physical_model(
        self,
        fitted_image: Histogram,
        response_path,
        orientation,
        temporal_background: Histogram,
        background_normalization: float,
        *,
        earth_occ: bool,
        nside_scatt_map: int | None = None,
    ) -> None:
        """Set the inputs for physical predictions and clear cached bins.

        Keep the supplied objects unchanged while using cached predictions.
        Call this method again after changing any input or response setting.
        The response file is opened only when new bins are calculated.
        """
        self._physical_configuration = {
            "fitted_image": fitted_image,
            "response_path": response_path,
            "orientation": orientation,
            "temporal_background": temporal_background,
            "background_normalization": background_normalization,
            "earth_occ": earth_occ,
            "nside_scatt_map": nside_scatt_map,
        }
        self._physical_cache.clear()

    def build_physical_time_range(
        self, start: int = 0, stop: int | None = None, *, bins_per_group: int = 1
    ) -> tuple[Histogram, Histogram]:
        """Return observed and physical expected counts for consecutive bins.

        ``start`` is inclusive and ``stop`` is exclusive. The default range
        covers the complete Time axis. Already calculated bins are reused in
        memory; their indices refer to the original observed histogram.
        Returned histograms have matching Time and CDS axes and can be passed
        directly to ResidualAnalysis. Their time-bin indices start at zero.
        ``bins_per_group`` optionally combines that many consecutive original
        bins, summing observed and expected counts before residuals are formed.
        The selected range must contain a whole number of groups. Original
        bins are calculated and cached individually even when grouped.
        """
        if self._physical_configuration is None:
            raise ValueError("Configure the physical model before building a time range.")
        n_bins = self.observed_hist.axes["Time"].nbins
        if stop is None:
            stop = n_bins
        for name, value in (("start", start), ("stop", stop)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer.")
        if not 0 <= start < stop <= n_bins:
            raise ValueError("The time range must contain available consecutive bins.")
        if isinstance(bins_per_group, bool) or not isinstance(bins_per_group, (int, np.integer)):
            raise TypeError("bins_per_group must be an integer.")
        if bins_per_group < 1 or (stop - start) % bins_per_group:
            raise ValueError("bins_per_group must be positive and divide the selected range exactly.")

        missing = [index for index in range(start, stop) if index not in self._physical_cache]
        if missing:
            from cosipy.response import FullDetectorResponse

            configuration = self._physical_configuration.copy()
            response_path = configuration.pop("response_path")
            with FullDetectorResponse.open(response_path) as detector_response:
                for index in missing:
                    products = self.build_physical_time_bin(
                        time_bin_index=index,
                        detector_response=detector_response,
                        **configuration,
                    )
                    self._physical_cache[index] = products["total"].contents.copy()

        time_axis = self.observed_hist.axes["Time"][start:stop]
        observed_values = self.observed_hist.contents[start:stop].copy()
        expected_values = np.stack([self._physical_cache[index] for index in range(start, stop)])
        if bins_per_group > 1:
            # Keep the original time boundaries and sum complete groups.
            time_axis = Axis(
                time_axis.edges[::bins_per_group], label="Time", unit=time_axis.unit
            )
            grouped_shape = ((stop - start) // bins_per_group, bins_per_group, *observed_values.shape[1:])
            observed_values = observed_values.reshape(grouped_shape).sum(axis=1)
            expected_values = expected_values.reshape(grouped_shape).sum(axis=1)
        axes = [time_axis, *list(self.observed_hist.axes)[1:]]
        observed = Histogram(
            axes, contents=observed_values, copy_contents=False
        )
        expected = Histogram(
            axes,
            contents=expected_values,
            copy_contents=False,
        )
        return observed, expected

    # ----------------------------------------------------------------------
    # Expected counts M(t,i)
    # ----------------------------------------------------------------------
    def build_physical_time_bin(
        self,
        time_bin_index: int,
        fitted_image: Histogram,
        detector_response,
        orientation,
        temporal_background: Histogram,
        background_normalization: float,
        *,
        earth_occ: bool,
        nside_scatt_map: int | None = None,
    ) -> dict[str, Histogram]:
        """Forward-fold a fitted sky and temporal background for one bin.

        The background must be a time-binned template matching the observed
        histogram, and its normalization must come from the integrated fit.
        The interval response is constructed from the supplied spacecraft
        history. Results are provisional until the response configuration is
        checked against the one used in the integrated fit.
        """
        from astropy.time import Time
        import astropy.units as u

        if isinstance(time_bin_index, bool) or not isinstance(
            time_bin_index, (int, np.integer)
        ):
            raise TypeError("time_bin_index must be an integer.")
        if not 0 <= time_bin_index < self.observed_hist.axes["Time"].nbins:
            raise IndexError("time_bin_index is outside the available Time bins.")
        if temporal_background.axes != self.observed_hist.axes:
            raise ValueError("Temporal background and observed data must have matching axes.")
        if not np.isfinite(background_normalization) or background_normalization < 0:
            raise ValueError("Background normalization must be finite and non-negative.")
        if tuple(fitted_image.axes.labels) != ("lb", "Ei"):
            raise ValueError("The fitted sky image must have axes (lb, Ei).")

        # The histogram Time axis stores Unix seconds, matching the notebook.
        edges = self.observed_hist.axes["Time"].edges
        start = Time(edges[time_bin_index].to_value(u.s), format="unix")
        stop = Time(edges[time_bin_index + 1].to_value(u.s), format="unix")
        interval_orientation = orientation.select_interval(start, stop)

        # Fold one sky pixel at a time to avoid storing the full response matrix.
        model_values = np.asarray(fitted_image.contents, dtype=float)
        if not np.all(np.isfinite(model_values)) or np.any(model_values < 0):
            raise ValueError("The fitted sky image contains invalid values.")
        if fitted_image.axes["Ei"] != detector_response.axes["Ei"]:
            raise ValueError("Fitted sky and detector response energy bins do not match.")
        if fitted_image.unit is None:
            raise ValueError("The fitted sky image must have physical units.")

        source_values = np.zeros(self.integrated_expected_hist.shape, dtype=float)
        source_axes = None
        pixel_area = fitted_image.axes["lb"].pixarea()
        for pixel_index, pixel_values in enumerate(model_values):
            if not np.any(pixel_values):
                continue
            pixel_response = detector_response.get_point_source_response_per_image_pixel(
                pixel_index,
                interval_orientation,
                coordsys="galactic",
                nside_image=fitted_image.axes["lb"].nside,
                nside_scatt_map=nside_scatt_map,
                earth_occ=earth_occ,
            )
            pixel_axes = pixel_response.axes
            if pixel_axes["Ei"] != fitted_image.axes["Ei"]:
                raise ValueError("Pixel response and fitted sky energy bins do not match.")
            if pixel_axes[1:] != self.integrated_expected_hist.axes:
                raise ValueError("Interval source response and fitted CDS axes do not match.")
            conversion = (fitted_image.unit * pixel_response.unit * pixel_area).to_value(
                u.dimensionless_unscaled
            )
            source_values += conversion * np.tensordot(
                pixel_values, np.asarray(pixel_response.contents), axes=(0, 0)
            )
            source_axes = pixel_axes[1:]

        if source_axes is None:
            source_axes = self.integrated_expected_hist.axes
        source = Histogram(source_axes, contents=source_values, copy_contents=False)
        expected_axes = list(self.observed_hist.axes)[1:]

        # Scale the actual simulated background in the same interval as the data.
        background_values = temporal_background.contents[time_bin_index]
        if hasattr(background_values, "todense"):
            background_values = background_values.todense()
        background_values = np.asarray(background_values, dtype=float)
        if not np.all(np.isfinite(background_values)) or np.any(background_values < 0):
            raise ValueError("Temporal background contains invalid counts.")
        background = Histogram(
            expected_axes,
            contents=background_normalization * background_values,
            copy_contents=False,
        )
        total = Histogram(
            expected_axes,
            contents=np.asarray(source.contents) + background.contents,
            copy_contents=False,
        )
        return {"source": source, "background": background, "total": total}

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
