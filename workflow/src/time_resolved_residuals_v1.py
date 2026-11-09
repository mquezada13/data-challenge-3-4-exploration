"""Build time-resolved observed and physical expected count histograms."""

from __future__ import annotations

import numpy as np
from histpy import Axis, Histogram

# Make these types available to the editor without importing them at runtime.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astromodels import Model
    from cosipy.response import FullDetectorResponse
    from cosipy.spacecraftfile import SpacecraftHistory
    from histpy import HealpixAxis


class TemporalPredictionBuilder:
    """Prepare matching observed and physical expected counts over time.

    Start with time-binned observations and the integrated expectation from
    a fit. Configure the fitted sky, detector response, spacecraft history,
    and temporal background before requesting a time range. For each original
    time bin, the prediction is the sum of the fitted sky folded through the
    interval response and the scaled background template.

    The returned observed and expected histograms have matching Time and CDS
    axes and can be passed to ResidualAnalysis. The observed count values do
    not normalize the physical prediction. The prediction is provisional until
    its response configuration is validated against the integrated fit.

    Workflow:
        1. Initialize with the temporal observation and integrated expectation.
        2. Configure either a fitted image or a parametric fitted model.
        3. Request consecutive bins with ``build_physical_time_range``.

    An image is folded pixel by pixel. A parametric model uses COSIpy's
    point-source response for point sources and image folding for extended
    sources. Both paths add the scaled temporal background afterward.
    """

    # ----------------------------------------------------------------------
    # Public methods
    # ----------------------------------------------------------------------

    def __init__(
        self,
        observed: Histogram,
        integrated_expected: Histogram,
    ) -> None:
        """Store the observation and check its axes against the integrated fit.

        Args:
            observed: Histogram of observed counts with Time first and CDS
                axes afterward. Its time edges define the prediction intervals.
            integrated_expected: Time-integrated fitted expectation. Its CDS
                axes and binning must match those of the observation; its count
                values do not set the temporal prediction's normalization.

        Sparse observations remain sparse until a time range is selected.
        """
        # ------------------------------------------------------------------
        # Validate the observed Time axis and the matching CDS axes
        # ------------------------------------------------------------------
        self._validate_observed_axes(observed, integrated_expected)

        # ------------------------------------------------------------------
        # Store the inputs and initialize configuration and prediction cache
        # ------------------------------------------------------------------
        # These attributes make the histograms available to the other methods.
        self.integrated_expected_hist = integrated_expected
        # Keep sparse counts sparse; copy dense inputs through the helper.
        self.observed_hist = observed if observed.is_sparse else self._densify_observed_counts(observed)
        # No physical inputs or predictions have been supplied yet.
        self._physical_configuration = None
        self._physical_cache = {}

    def configure_physical_model(
        self,
        fitted_sky,
        response_path=None,
        orientation=None,
        temporal_background: Histogram | None = None,
        background_normalization: float | None = None,
        *,
        earth_occ: bool,
        nside_scatt_map: int | None = None,
        extended_sky_axis=None,
        fitted_energy_axis=None,
        fitted_background: Histogram | None = None,
    ) -> None:
        """Set the inputs for physical predictions and clear cached bins.

        Args:
            fitted_sky: Fitted sky histogram with axes (lb, Ei), for imaging,
                or an optimized astromodels Model for the parametric path.
            response_path: File path to the full detector response.
            orientation: Spacecraft history covering the requested intervals.
            temporal_background: Simulated background with Time and CDS axes
                matching the observation.
            background_normalization: Fitted scale for the image-based path.
            earth_occ: Whether to account for Earth occultation.
            nside_scatt_map: Optional resolution for the response calculation.
            extended_sky_axis: Fitted response sky axis, required for extended
                sources but not for a model containing only point sources.
            fitted_energy_axis: Incident-energy axis used by the parametric fit.
            fitted_background: Integrated fitted background for the parametric
                path, used to determine and check the background scale.

        Keep the supplied objects unchanged while using cached predictions.
        Call this method again after changing any input or response setting.
        The response file is opened only when new bins are calculated.

        For a parametric COSIpy/3ML fit, supply the optimized astromodels Model
        as ``fitted_sky``, along with ``fitted_energy_axis`` and the integrated
        ``fitted_background``. Extended sources also require ``extended_sky_axis``
        from the fitted extended response. Background
        normalization is derived from the temporal background and checked
        against every CDS bin. The parametric path currently requires Earth
        occultation, as in COSIpy's point-source response implementation.
        """
        # ------------------------------------------------------------------
        # Identify the fitted sky: an image or a parametric model
        # Optional dependencies are loaded only when they are needed.
        # ------------------------------------------------------------------
        if isinstance(fitted_sky, Histogram):
            # An image is already in the representation used by the imaging path.
            fitted_image = fitted_sky
            fitted_model = None
        else:
            # Import astromodels only for a non-histogram sky input.
            from astromodels import Model

            if isinstance(fitted_sky, Model):
                fitted_image = None
                fitted_model = fitted_sky
            else:
                raise TypeError("The fitted sky must be a Histogram or an astromodels Model.")
        # ------------------------------------------------------------------
        # Parametric path: validate the model, axes and background inputs
        # ------------------------------------------------------------------
        if fitted_model is not None:
            # First check that the model and background inputs are compatible.
            self._validate_parametric_inputs(
                fitted_image, fitted_model, fitted_energy_axis, extended_sky_axis,
                temporal_background, fitted_background, background_normalization, earth_occ,
            )
            # Then derive the scale that matches the fitted background.
            background_normalization = self._calculate_background_normalization(
                temporal_background, fitted_background,
            )
        # ------------------------------------------------------------------
        # Store the configuration shared by the image and parametric paths
        # ------------------------------------------------------------------
        # Store common inputs by name; other methods read this dictionary later.
        self._physical_configuration = {
            "fitted_image": fitted_image,
            "response_path": response_path,
            "orientation": orientation,
            "temporal_background": temporal_background,
            "background_normalization": background_normalization,
            "earth_occ": earth_occ,
            "nside_scatt_map": nside_scatt_map,
        }
        # Only the parametric calculation needs these additional model and axes.
        if fitted_model is not None:
            self._physical_configuration["fitted_model"] = fitted_model
            self._physical_configuration["extended_sky_axis"] = extended_sky_axis
            self._physical_configuration["fitted_energy_axis"] = fitted_energy_axis
        # ------------------------------------------------------------------
        # Clear predictions calculated with the previous configuration
        # ------------------------------------------------------------------
        self._physical_cache.clear()

    def build_physical_time_range(
        self, 
        start: int = 0, 
        stop: int | None = None, 
        *, 
        bins_per_group: int = 1
    ) -> tuple[Histogram, Histogram]:
        """Return observed and physical expected counts for consecutive bins.

        Args:
            start: First original time-bin index, included in the selection.
            stop: Original time-bin index after the selection; None selects
                through the end of the observed Time axis.
            bins_per_group: Number of consecutive original bins to sum into
                each output bin. The selection must contain complete groups.

        Returns:
            A pair (observed, expected) of matching Time and CDS histograms.
            Expected counts are physical source plus scaled background counts.

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
        # ------------------------------------------------------------------
        # Validate the requested range and optional grouping
        # ------------------------------------------------------------------
        stop = self._validate_time_range(start, stop, bins_per_group)

        # ------------------------------------------------------------------
        # Calculate missing original bins and reuse cached predictions
        # ------------------------------------------------------------------
        # range includes start and excludes stop. Cache keys are original indices.
        missing = [index for index in range(start, stop) if index not in self._physical_cache]
        if missing:
            # Load COSIpy only when at least one new prediction is needed.
            from cosipy.response import FullDetectorResponse

            # Copy the saved inputs so removing call-specific entries does not
            # change the stored configuration used by later requests.
            configuration = self._physical_configuration.copy()

            # The response path opens the file; the per-bin builder receives
            # the resulting open response object instead of the path.
            response_path = configuration.pop("response_path")

            # The presence of the fitted model selects the parametric path.
            is_parametric = "fitted_model" in configuration
            if is_parametric:
                configuration.pop("fitted_image")
            # Open once for all missing bins; the context manager closes it after.
            with FullDetectorResponse.open(response_path) as detector_response:
                for index in missing:
                    # ------------------------------------------------------
                    # Choose which per-bin prediction method to call:
                    # Parametric fitted model -> build_parametric_time_bin.
                    # Fitted image -> build_imaging_time_bin.
                    # Both return source, background and total counts.
                    # ------------------------------------------------------
                    builder = self.build_parametric_time_bin if is_parametric else self.build_imaging_time_bin
                    
                    # **configuration passes the saved dictionary entries as
                    # named arguments to the selected per-bin method.
                    products = builder(
                        time_bin_index=index,
                        detector_response=detector_response,
                        **configuration,
                    )
                    # Cache only the total counts for reuse in later selections.
                    self._physical_cache[index] = products["total"].contents.copy()

        # ------------------------------------------------------------------
        # Select observed counts and collect matching expected counts
        # ------------------------------------------------------------------
        # Slice the original Time axis and counts using the same [start:stop].
        time_axis = self.observed_hist.axes["Time"][start:stop]
        observed_values = self.observed_hist.contents[start:stop].copy()
        # Densify only the selected observation, not the full sparse histogram.
        if hasattr(observed_values, "todense"):
            observed_values = observed_values.todense()
        # Stack cached CDS arrays in chokronological order to restore a Time axis.
        expected_values = np.stack([self._physical_cache[index] for index in range(start, stop)])
        # ------------------------------------------------------------------
        # Optionally combine consecutive observed and expected bins
        # ------------------------------------------------------------------
        if bins_per_group > 1:
            # Keep the original time boundaries and sum complete groups.
            time_axis = Axis(
                time_axis.edges[::bins_per_group], label="Time", unit=time_axis.unit
            )
            # Reshape into (number of groups, bins per group, CDS axes), then
            # sum the second dimension for both observation and expectation.
            grouped_shape = ((stop - start) // bins_per_group, bins_per_group, *observed_values.shape[1:])
            observed_values = observed_values.reshape(grouped_shape).sum(axis=1)
            expected_values = expected_values.reshape(grouped_shape).sum(axis=1)
        
        # ------------------------------------------------------------------
        # Return histograms with the same Time and CDS axes
        # ------------------------------------------------------------------
        # Use the selected Time axis followed by the original CDS axes for both.
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

    def timestamps_interval(
        self, start_time, stop_time, *, bins_per_group: int = 1
    ) -> tuple[Histogram, Histogram]:
        """Build physical predictions for the bins containing two timestamps.

        Args:
            start_time: UTC ISO timestamp string or scalar astropy Time.
            stop_time: UTC ISO timestamp string or scalar astropy Time.
            bins_per_group: Number of consecutive original bins per output bin.

        Returns:
            Matching observed and expected histograms. Their Time axis contains
            the actual selected bin edges.

        Both timestamps are included. A timestamp on an internal bin edge
        belongs to the bin beginning at that edge. The final observation edge
        is excluded because no bin begins there. Both timestamps may be within
        the same bin. Grouping requires a whole number of groups.
        """
        from astropy.time import Time

        # Convert the validated timestamps and native edges to bin indices.
        start_time, stop_time = self._validate_timestamps(start_time, stop_time)
        edges = Time(
            self.observed_hist.axes["Time"].edges.to_value("s"),
            format="unix", scale="utc",
        )
        start_index = int(np.count_nonzero(edges <= start_time)) - 1
        stop_index = int(np.count_nonzero(edges <= stop_time))

        # Use the existing prediction, grouping, and cache implementation.
        return self.build_physical_time_range(
            start=start_index, stop=stop_index, bins_per_group=bins_per_group
        )

    def build_parametric_time_bin(
        self, 
        time_bin_index : int, 
        fitted_model: Model, 
        extended_sky_axis: HealpixAxis | None, 
        fitted_energy_axis: Axis,
        detector_response: FullDetectorResponse, 
        orientation: SpacecraftHistory, 
        temporal_background: Histogram,
        background_normalization: float, 
        *, 
        earth_occ: bool, 
        nside_scatt_map: int | None = None,
    ):
        """Fold an unpolarized COSIpy/3ML model for one time interval.

        The original observation supplies interval boundaries and CDS axes.
        The fitted model supplies the sky sources; the detector response and
        interval spacecraft history turn them into expected detector counts.
        The temporal background is selected in the same bin and scaled by the
        normalization determined during configuration.

        Returns:
            A dictionary with ``source``, ``background``, and ``total`` CDS
            histograms for this interval. Total is source plus background.

        Point sources use COSIpy's point-source folding; extended sources
        use its integrated flux maps and the existing pixel-wise folding.
        The result is provisional until its response is checked against the
        integrated fit. Three-dimensional spatial models are not yet supported.
        """
        # These imports are needed only when folding a parametric model.
        import astropy.units as u
        from astropy.coordinates import Galactic
        from astropy.time import Time
        from astromodels import Function3D
        from astromodels.sources import PointSource, ExtendedSource
        from histpy import HealpixAxis
        from cosipy.data_io import EmCDSBinnedData
        from cosipy.response import BinnedInstrumentResponse, BinnedThreeMLPointSourceResponse
        from cosipy.response.functions import get_integrated_extended_model
        from cosipy.threeml.util import to_linear_polarization

        # ------------------------------------------------------------------
        # Validate the incident-energy axis and extended-source sky grid
        # ------------------------------------------------------------------
        image_axis = self._validate_parametric_response_axes(
            fitted_model, extended_sky_axis, fitted_energy_axis, detector_response,
        )

        # ------------------------------------------------------------------
        # Select the spacecraft history for this observed time interval
        # ------------------------------------------------------------------
        # Read the original bin edges in seconds and interpret them as Unix time.
        edges = self.observed_hist.axes["Time"].edges.to_value(u.s)
        # COSIpy selects the portion of the spacecraft history between these edges.
        interval = orientation.select_interval(
            Time(edges[time_bin_index], format="unix"),
            Time(edges[time_bin_index + 1], format="unix"),
        )

        # Wrap the integrated CDS histogram for COSIpy's response interface.
        # Its axes define the measured bins in which source counts are predicted.
        data = EmCDSBinnedData(self.integrated_expected_hist)

        # Accumulate predicted counts from all sources into a CDS-shaped array.
        source_values = np.zeros(self.integrated_expected_hist.shape, dtype=float)

        # Extended-source maps are collected before their shared image folding.
        extended_values = None
        extended_unit = None

        # ------------------------------------------------------------------
        # Fold point sources and collect extended sources into a sky map
        # ------------------------------------------------------------------
        for source in fitted_model.sources.values():
            self._validate_parametric_source(source)
            if isinstance(source, PointSource):
                # Fold the fitted point source through its interval response.
                expected = self._fold_parametric_point_source(
                    source, data, detector_response, interval,
                    fitted_energy_axis, nside_scatt_map,
                )
            else:
                # Create a sky flux map; all extended maps are folded together.
                flux, flux_values = self._parametric_extended_flux_map(
                    source, extended_sky_axis, fitted_energy_axis,
                )
                if extended_values is None:
                    # The first extended source establishes the common map unit.
                    extended_unit = flux.unit
                    extended_values = flux_values.copy()
                else:
                    # Convert subsequent maps to the first unit before summing.
                    extended_values += flux_values * flux.unit.to(extended_unit)
                # Extended maps are folded together after the source loop.
                continue
            self._validate_parametric_source_axes(expected)
            # Add this point source's CDS prediction to the running source total.
            source_values += np.asarray(expected.contents, dtype=float)

        # ------------------------------------------------------------------
        # Fold the extended sky map with the image-based calculation
        # ------------------------------------------------------------------
        if extended_values is not None:
            # Turn the combined extended-source map into the image histogram
            # expected by the image-based forward-folding method.
            image = Histogram([image_axis, fitted_energy_axis],
                              contents=extended_values, unit=extended_unit)
            
            # Fold the map pixel by pixel. A zero background scale isolates the
            # source component; the actual background is added once below.
            expected = self.build_imaging_time_bin(
                time_bin_index=time_bin_index, 
                fitted_image=image,
                detector_response=detector_response, 
                orientation=orientation,
                temporal_background=temporal_background, 
                background_normalization=0.0,
                earth_occ=earth_occ, 
                nside_scatt_map=nside_scatt_map,
            )["source"]

            self._validate_parametric_source_axes(expected)

            # Include the folded extended-source counts with any point sources.
            source_values += np.asarray(expected.contents, dtype=float)

        # ------------------------------------------------------------------
        # Scale the interval background and assemble the three products
        # ------------------------------------------------------------------
        # Slice the simulated template at the same original time-bin index.
        background_values = temporal_background.contents[time_bin_index]

        # A sparse slice is made dense only after selecting this interval.
        if hasattr(background_values, "todense"):
            background_values = background_values.todense()

        background_values = np.asarray(background_values, dtype=float)
        self._validate_temporal_background_values(background_values)

        # Source, background and total share the integrated fit's CDS axes.
        axes = list(self.integrated_expected_hist.axes)
        source = Histogram(axes, contents=source_values)

        # Apply the single fitted scale: B[t, i] = alpha * template[t, i].
        background = Histogram(axes, contents=background_normalization * background_values)

        # Add source and scaled background bin by bin: M[t, i] = S[t, i] + B[t, i].
        total = Histogram(axes, contents=source_values + background.contents)
        return {"source": source, "background": background, "total": total}

    def build_imaging_time_bin(
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

        Args:
            time_bin_index: Index on the original observed Time axis.
            fitted_image: Fitted sky intensity histogram with axes (lb, Ei).
            detector_response: Open COSIpy full detector response.
            orientation: Spacecraft history covering the selected interval.
            temporal_background: Time-binned background template on the
                observed Time and CDS axes.
            background_normalization: Scale obtained from the integrated fit.
            earth_occ: Whether to account for Earth occultation.
            nside_scatt_map: Optional resolution for the response calculation.

        Returns:
            A dictionary with ``source``, ``background``, and ``total`` CDS
            histograms. Total is source plus scaled background.

        The background must be a time-binned template matching the observed
        histogram, and its normalization must come from the integrated fit.
        The interval response is constructed from the supplied spacecraft
        history. Results are provisional until the response configuration is
        checked against the one used in the integrated fit.
        """
        # Load time utilities only when an image-based prediction is requested.
        from astropy.time import Time
        import astropy.units as u

        # ------------------------------------------------------------------
        # Validate the interval, fitted image and background inputs
        # ------------------------------------------------------------------
        self._validate_imaging_time_bin_inputs(
            time_bin_index, fitted_image, temporal_background, background_normalization,
        )

        # ------------------------------------------------------------------
        # Select the spacecraft history for the observed time interval
        # The Time axis stores Unix seconds.
        # ------------------------------------------------------------------
        # The two consecutive bin edges give this interval's start and stop.
        edges = self.observed_hist.axes["Time"].edges

        # Convert the edges to seconds and construct Astropy Unix times.
        start = Time(edges[time_bin_index].to_value(u.s), format="unix")
        stop = Time(edges[time_bin_index + 1].to_value(u.s), format="unix")

        # Select the part of the spacecraft history used for the interval response.
        interval_orientation = orientation.select_interval(start, stop)

        # ------------------------------------------------------------------
        # Validate the image values, units and incident-energy axis
        # ------------------------------------------------------------------
        # Extract image values as a floating-point array indexed by sky pixel
        # and incident-energy bin; the histogram retains the axes and units.
        model_values = np.asarray(fitted_image.contents, dtype=float)
        self._validate_imaging_values(fitted_image, model_values, detector_response)

        # Forward-fold the fitted sky through the interval response.
        # The helper calls COSIpy for each pixel response, then combines each
        # response with the fitted image to return source counts in CDS bins.
        source = self._forward_fold_image(
            fitted_image, model_values, detector_response, interval_orientation,
            earth_occ=earth_occ, nside_scatt_map=nside_scatt_map,
        )
        # Removing the leading Time axis leaves the CDS axes for one interval.
        expected_axes = list(self.observed_hist.axes)[1:]

        # ------------------------------------------------------------------
        # Scale the simulated background for the same time interval
        # ------------------------------------------------------------------
        # Select the template in the same time interval as the source response.
        background_values = temporal_background.contents[time_bin_index]
        # Convert only this selected slice when the template is sparse.
        if hasattr(background_values, "todense"):
            background_values = background_values.todense()
        background_values = np.asarray(background_values, dtype=float)
        self._validate_temporal_background_values(background_values)

        # Scale each CDS bin by the normalization obtained from the fit.
        background = Histogram(
            expected_axes,
            contents=background_normalization * background_values,
            copy_contents=False,
        )
        # ------------------------------------------------------------------
        # Add source and background and return the component histograms
        # ------------------------------------------------------------------
        # Add the source and background counts in corresponding CDS bins.
        total = Histogram(
            expected_axes,
            contents=np.asarray(source.contents) + background.contents,
            copy_contents=False,
        )
        return {"source": source, "background": background, "total": total}

    # ----------------------------------------------------------------------
    # Calculation helpers
    # ----------------------------------------------------------------------

    @staticmethod
    def _calculate_background_normalization(temporal_background, fitted_background):
        """Derive and check the scale between temporal and fitted backgrounds."""
        # The fitted background has only CDS axes, for example (Em, Phi, PsiChi).
        # Collect their names as strings in a tuple; Time is not among them.
        cds_labels = tuple(str(label) for label in fitted_background.axes.labels)

        # histpy.project keeps the named CDS axes and sums over the omitted Time
        # axis: template[i] = sum_t temporal_background[t, i]. The asterisk
        # passes each axis name separately; to_dense changes storage, not counts.
        template = temporal_background.project(*cds_labels).to_dense(copy=False)

        # .contents extracts the counts; np.asarray makes floating-point arrays
        # that can be summed and compared element by element with NumPy.
        template_values = np.asarray(template.contents, dtype=float)
        fitted_values = np.asarray(fitted_background.contents, dtype=float)
        TemporalPredictionBuilder._validate_background_normalization_counts(
            template_values, fitted_values,
        )

        # Derive one scale from totals: alpha = sum_i fitted[i] / sum_i template[i].
        # This is a background scale from the fit, not a normalization to data.
        background_normalization = float(fitted_values.sum() / template_values.sum())

        # Matching totals alone is insufficient: check every CDS bin against
        # alpha * template[i], allowing the specified relative/absolute tolerance.
        TemporalPredictionBuilder._validate_background_scale(
            template_values, fitted_values, background_normalization,
        )
        # The caller stores this scale and later applies it to each time bin.

        return background_normalization

    @staticmethod
    # Key helper  for the parametric point-source path
    def _fold_parametric_point_source(
        source, 
        data, 
        detector_response, 
        interval, 
        fitted_energy_axis, 
        nside_scatt_map,
    ):
        """Use COSIpy to fold one fitted point source for the selected interval.

        COSIpy constructs the response for the fitted source position and the
        interval spacecraft history. It integrates the fitted spectrum over
        incident-energy bins and returns predicted counts on the CDS axes.
        """
        from cosipy.response import BinnedInstrumentResponse, BinnedThreeMLPointSourceResponse

        # Wrap the full detector response for the measured CDS axes.
        instrument = BinnedInstrumentResponse(detector_response, data)

        # Configure COSIpy with the selected history and fitted energy bins.
        point_response = BinnedThreeMLPointSourceResponse(
            data=data, 
            instrument_response=instrument, 
            sc_history=interval,
            energy_axis=fitted_energy_axis,
            nside=nside_scatt_map or 2 * data.axes["PsiChi"].nside,
        )
        # COSIpy folds this fitted source through the interval response.
        point_response.set_source(source)

        return point_response.expectation()

    @staticmethod
    def _parametric_extended_flux_map(source, extended_sky_axis, fitted_energy_axis):
        """Evaluate one fitted extended source as a sky-by-energy flux map.

        COSIpy evaluates the spatial model on the fitted sky grid and
        integrates the spectrum over the fitted incident-energy bins. The
        caller combines these maps before folding them through the response.
        """
        from astromodels import Function3D
        from cosipy.response.functions import get_integrated_extended_model

        TemporalPredictionBuilder._validate_extended_source_shape(source)
        flux = get_integrated_extended_model(source, extended_sky_axis, fitted_energy_axis)

        # Extract the sky-pixel by energy values for NumPy accumulation.
        flux_values = np.asarray(flux.contents, dtype=float)
        TemporalPredictionBuilder._validate_extended_flux(flux, flux_values)

        return flux, flux_values

    def _forward_fold_image(
        self, 
        fitted_image, 
        model_values, 
        detector_response, 
        interval_orientation,
        *, 
        earth_occ, 
        nside_scatt_map,
    ) -> Histogram:
        """Predict interval source counts by folding a fitted image through COSIpy.

        For each nonzero sky pixel, COSIpy calculates the detector response
        using the interval spacecraft history. Multiply that response by the
        image and the pixel solid angle, sum over incident energy, and add the
        contributions of all sky pixels. The result has the fitted CDS axes.

        S[t, i] = sum over p and E of I[p, E] * R[t, p, E, i] * pixel_area.
        """
        import astropy.units as u

        # Start an empty CDS-shaped array; each sky pixel will contribute to it.
        source_values = np.zeros(self.integrated_expected_hist.shape, dtype=float)

        # The actual source axes are obtained from the first nonzero pixel response.
        source_axes = None
        
        # histpy returns the solid angle of one sky pixel from the image axis.
        pixel_area = fitted_image.axes["lb"].pixarea()

        # Each iteration handles one sky pixel and all its incident-energy bins.
        for pixel_index, pixel_values in enumerate(model_values):
            # A pixel with zero intensity in every energy bin contributes nothing.
            if not np.any(pixel_values):
                continue

            # COSIpy builds the response for this sky pixel and interval from
            # the full detector response and the selected spacecraft history.
            # It returns a histogram over incident energy and measured CDS bins.
            pixel_response = detector_response.get_point_source_response_per_image_pixel(
                pixel_index,
                interval_orientation,
                coordsys="galactic",
                nside_image=fitted_image.axes["lb"].nside,
                nside_scatt_map=nside_scatt_map,
                earth_occ=earth_occ,
            )
            # Check that this response uses the fitted energy and CDS grids.
            pixel_axes = pixel_response.axes
            self._validate_pixel_response_axes(pixel_axes, fitted_image)

            # Combine image and response units with the pixel solid angle;
            # Astropy checks that the product converts to dimensionless counts.
            conversion = (fitted_image.unit * pixel_response.unit * pixel_area).to_value(
                u.dimensionless_unscaled
            )
            # NumPy multiplies the image values by the response and sums over
            # their first axes (incident energy). += accumulates sky pixels,
            # leaving one predicted source count for each measured CDS bin.
            source_values += conversion * np.tensordot(
                pixel_values, np.asarray(pixel_response.contents), axes=(0, 0)
            )
            # The response axes after incident energy are the measured CDS axes.
            source_axes = pixel_axes[1:]

        # An all-zero image has no pixel response; use the integrated CDS axes.
        if source_axes is None:
            source_axes = self.integrated_expected_hist.axes
        # Return the accumulated source prediction as a CDS histogram.
        return Histogram(source_axes, contents=source_values, copy_contents=False)

    # ----------------------------------------------------------------------
    # Validation helpers
    # ----------------------------------------------------------------------

    def _validate_parametric_inputs(
        self, 
        fitted_image, 
        fitted_model, 
        fitted_energy_axis, 
        extended_sky_axis,
        temporal_background, 
        fitted_background, 
        background_normalization, 
        earth_occ,
    ):
        """Check the fitted model, response axes and background inputs."""
        from astromodels import Model
        from astromodels.sources import ExtendedSource

        # Check the model and the response axes required by its sources.
        if fitted_image is not None or not isinstance(fitted_model, Model):
            raise ValueError("Provide one fitted image or one astromodels Model.")
        if fitted_energy_axis is None or fitted_background is None:
            raise ValueError("A parametric model requires the fitted energy axis and background.")
        if extended_sky_axis is None and any(
            isinstance(source, ExtendedSource) for source in fitted_model.sources.values()
        ):
            raise ValueError("Extended sources require the fitted sky axis.")
        if background_normalization is not None:
            raise ValueError("For a parametric model provide fitted_background, not a scale.")
        if not earth_occ:
            raise ValueError("Parametric point-source folding currently requires earth_occ=True.")
        if temporal_background.axes != self.observed_hist.axes:
            raise ValueError("Temporal background and observed data must have matching axes.")
        if fitted_background.axes != self.integrated_expected_hist.axes:
            raise ValueError("Fitted background and integrated model axes must match.")

    @staticmethod
    def _validate_parametric_response_axes(
        fitted_model, extended_sky_axis, fitted_energy_axis, detector_response,
    ):
        """Check response axes and return the sky grid for extended sources.

        A point-source-only model does not need an extended sky grid and
        returns None. An extended model uses a Galactic RING image grid whose
        coordinates must match the fitted response grid.
        """
        from astropy.coordinates import Galactic
        from astromodels.sources import ExtendedSource
        from histpy import HealpixAxis

        if fitted_energy_axis != detector_response.axes["Ei"]:
            raise ValueError("Fitted and detector response energy bins do not match.")
        if any(isinstance(source, ExtendedSource) for source in fitted_model.sources.values()):
            if extended_sky_axis is None:
                raise ValueError("Extended sources require the fitted sky axis.")
            # The output grid is also used later to build the fitted flux map.
            image_axis = HealpixAxis(nside=extended_sky_axis.nside, coordsys="galactic",
                                     scheme="ring", label="lb")
            indices = np.arange(image_axis.npix)
            if (not isinstance(extended_sky_axis.coordsys, Galactic)
                    or not np.allclose(image_axis.pix2ang(indices), extended_sky_axis.pix2ang(indices))):
                raise ValueError("The fitted sky axis must use the same Galactic RING pixelization.")
            return image_axis
        return None

    @staticmethod
    def _validate_observed_axes(observed, integrated_expected):
        """Check that Time is first and the observed and fitted CDS axes match."""
        if "Time" not in observed.axes.labels:
            raise ValueError("Temporal analysis requires observed D(t,i) with a Time axis.")
        expected_labels = ("Time", *(str(label) for label in integrated_expected.axes.labels))
        if tuple(str(label) for label in observed.axes.labels) != expected_labels:
            raise ValueError("Observed D(t,i) must have Time first and match the integrated model axes.")
        for label in integrated_expected.axes.labels:
            if observed.axes[str(label)] != integrated_expected.axes[str(label)]:
                raise ValueError(f"Observed D(t,i) and integrated M have incompatible {label} bins.")

    @staticmethod
    def _validate_background_normalization_counts(template_values, fitted_values):
        """Check that the two background arrays can be used to derive a scale."""
        if (not np.all(np.isfinite(template_values)) or np.any(template_values < 0)
                or not np.all(np.isfinite(fitted_values)) or np.any(fitted_values < 0)
                or template_values.sum() <= 0):
            raise ValueError("Background counts must be finite, non-negative, and non-empty.")

    @staticmethod
    def _validate_background_scale(template_values, fitted_values, background_normalization):
        """Check that the fitted background matches one scaled template bin by bin."""
        if not np.allclose(fitted_values, template_values * background_normalization,
                           rtol=1e-8, atol=1e-12):
            raise ValueError("Fitted background is not a single scaled temporal template.")

    def _validate_time_range(self, start, stop, bins_per_group):
        """Check the selected range and grouping, returning the resolved stop."""
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
        return stop

    def _validate_timestamps(self, start_time, stop_time):
        """Validate scalar timestamps against the original observation edges."""
        from astropy.time import Time

        times = []
        for name, value in (("start_time", start_time), ("stop_time", stop_time)):
            if not isinstance(value, (str, Time)):
                raise TypeError(f"{name} must be a UTC timestamp string or astropy Time.")
            try:
                timestamp = Time(value, scale="utc") if isinstance(value, str) else value
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} is not a valid UTC timestamp.") from exc
            if not timestamp.isscalar or np.ma.is_masked(timestamp.unix):
                raise ValueError(f"{name} must be a finite scalar timestamp.")
            if not np.isfinite(timestamp.unix):
                raise ValueError(f"{name} must be a finite scalar timestamp.")
            times.append(timestamp)

        start_time, stop_time = times
        if start_time > stop_time:
            raise ValueError("start_time must not be later than stop_time.")
        edges = Time(
            self.observed_hist.axes["Time"].edges.to_value("s"),
            format="unix", scale="utc",
        )
        if start_time < edges[0] or stop_time >= edges[-1]:
            raise ValueError("Timestamps must fall within the observed Time bins.")
        return start_time, stop_time

    @staticmethod
    def _validate_parametric_source(source):
        """Check that one fitted source uses a supported type and spectrum."""
        from astromodels.sources import PointSource, ExtendedSource
        from cosipy.threeml.util import to_linear_polarization

        if not isinstance(source, (PointSource, ExtendedSource)):
            raise TypeError("Only point and extended sources are supported.")
        if not hasattr(source.spectrum, "main"):
            raise ValueError("Only single-component, unpolarized spectra are supported.")
        polarization = to_linear_polarization(source.spectrum.main.polarization)
        if polarization.degree.value != 0:
            raise ValueError("Polarized sources are not supported by temporal folding.")

    def _validate_parametric_source_axes(self, expected):
        """Check the predicted source histogram against the fitted CDS axes."""
        if expected.axes != self.integrated_expected_hist.axes:
            raise ValueError("Source expectation and fitted CDS axes do not match.")

    @staticmethod
    def _validate_extended_source_shape(source):
        """Reject unsupported three-dimensional extended spatial models."""
        from astromodels import Function3D

        if isinstance(source.spatial_shape, Function3D):
            raise ValueError("Three-dimensional spatial models are not yet supported.")

    @staticmethod
    def _validate_extended_flux(flux, flux_values):
        """Check the values and units of an extended-source flux map."""
        if not np.all(np.isfinite(flux_values)) or np.any(flux_values < 0):
            raise ValueError("The fitted sky image contains invalid values.")
        if flux.unit is None:
            raise ValueError("The fitted sky image must have physical units.")

    def _validate_imaging_time_bin_inputs(
        self, time_bin_index, fitted_image, temporal_background, background_normalization,
    ):
        """Check the selected interval, background and fitted image axes."""
        if isinstance(time_bin_index, bool) or not isinstance(time_bin_index, (int, np.integer)):
            raise TypeError("time_bin_index must be an integer.")
        if not 0 <= time_bin_index < self.observed_hist.axes["Time"].nbins:
            raise IndexError("time_bin_index is outside the available Time bins.")
        if temporal_background.axes != self.observed_hist.axes:
            raise ValueError("Temporal background and observed data must have matching axes.")
        if not np.isfinite(background_normalization) or background_normalization < 0:
            raise ValueError("Background normalization must be finite and non-negative.")
        if tuple(fitted_image.axes.labels) != ("lb", "Ei"):
            raise ValueError("The fitted sky image must have axes (lb, Ei).")

    @staticmethod
    def _validate_imaging_values(fitted_image, model_values, detector_response):
        """Check image counts, units and incident-energy bins before folding."""
        if not np.all(np.isfinite(model_values)) or np.any(model_values < 0):
            raise ValueError("The fitted sky image contains invalid values.")
        if fitted_image.axes["Ei"] != detector_response.axes["Ei"]:
            raise ValueError("Fitted sky and detector response energy bins do not match.")
        if fitted_image.unit is None:
            raise ValueError("The fitted sky image must have physical units.")

    @staticmethod
    def _validate_temporal_background_values(background_values):
        """Check the selected background counts before applying the fit scale."""
        if not np.all(np.isfinite(background_values)) or np.any(background_values < 0):
            raise ValueError("Temporal background contains invalid counts.")

    def _validate_pixel_response_axes(self, pixel_axes, fitted_image):
        """Check the response energy and measured-data axes for one sky pixel."""
        if pixel_axes["Ei"] != fitted_image.axes["Ei"]:
            raise ValueError("Pixel response and fitted sky energy bins do not match.")
        if pixel_axes[1:] != self.integrated_expected_hist.axes:
            raise ValueError("Interval source response and fitted CDS axes do not match.")

    # ----------------------------------------------------------------------
    # Storage helper
    # ----------------------------------------------------------------------

    @staticmethod
    def _densify_observed_counts(histogram: Histogram) -> Histogram:
        """Return a dense copy, preserving integer counts when applicable.

        The constructor currently calls this helper for dense inputs, which
        are copied. Its sparse branch also supports compact integer dtypes when
        the helper is called directly with non-negative integer counts.
        """
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

