# Using the residual-analysis classes

This guide explains how to compare observed and fitted COSI histograms with
`ResidualAnalysis`, and how to make plots from those results with
`PlottingResiduals`. The examples use the disk-scale products in notebook 06,
but the classes can be used with another matching pair of `histpy.Histogram`
objects.

## What the two classes do

`ResidualAnalysis` prepares a common valid-bin mask, makes projections of the
observed and expected histograms, calculates residual histograms, and reports
fit-versus-observed statistics. It does not fit the model.

`PlottingResiduals` receives a `ResidualAnalysis` object and uses its
projections and residual calculations to make figures. It does not require a
second copy of the input histograms.

The intended order is:

1. Load the observed and expected histograms.
2. Create one `ResidualAnalysis` object from that pair.
3. Inspect the mask report and request statistics or residuals.
4. Create a `PlottingResiduals` object and request figures.

## Inputs and bin matching

The constructor needs two `histpy.Histogram` objects:

- `observed`: counts measured in the data.
- `expected`: fitted model counts, in the same bins as `observed`.

They must have identical axis definitions, axis order, and shape. The class
checks these requirements and rejects non-finite or negative histogram
contents. The original input histograms are kept; masking is applied to copies
used internally for projection.

### The optional artificial expected-count floor

`expected_floor` is an optional third constructor argument. Use it only when
you know that the expected histogram was prepared by replacing zero values
with a particular artificial floor. For example, notebook 03 replaced zeros
with `1e-12`, so notebook 06 passes that value explicitly:

```python
# Import the classes and load the matching observed and expected histograms first.
from src.residuals import ResidualAnalysis

# Tell the mask about the artificial floor used when these products were made.
residual_analysis = ResidualAnalysis(
    observed_disk_scale,
    expected_disk_scale,
    expected_floor=1e-12,
)
```

The mask excludes bins whose expected value is at or below the supplied floor,
provided those bins contain no observed counts. If any observed counts are in
those bins, construction raises a `ValueError` instead of silently discarding
them. If `expected_floor` is omitted, the class only excludes values at the
float64 numerical-zero threshold. Do not pass a small value just because it
looks negligible: it must be the known floor used to prepare that expected
product.

After construction, inspect the decision with:

```python
# Show how many full-CDS bins the mask retained and excluded.
residual_analysis.mask_report
```

The report includes total, valid, and excluded bin counts, observed counts in
excluded bins, the supplied floor, and minimum expected values before and
after masking.

### Internal preparation methods

`mask_function()` builds the full-CDS mask and its report. It is called during
initialization, so users normally inspect `mask` or `mask_report` rather than
calling it themselves. `split_psichi_function()` converts the HEALPix
`PsiChi` coordinate into the current rectangular Chi and Psi histograms. It
is used by `projected_data()` and is also normally not called directly.

## `ResidualAnalysis` methods

### `projection_function(*axis_names)`

Projects both masked histograms onto the requested existing axes and returns a
dictionary with `"observed"` and `"expected"` histograms. For example,
`projection_function("Phi")` returns the pair projected onto Phi. Both
histograms are projected in the same way, so the counts being compared remain
aligned.

### `projected_data()`

Builds and caches the projections used by the current workflow. It returns a
dictionary keyed by projection ID:

- `Phi_projection`
- `Chi_projection`
- `Psi_projection`
- `PsiChi_projection`
- `Phi_Chi_projection`
- `Phi_Psi_projection`
- `Phi_PsiChi_projection`

The one-dimensional Chi and Psi products are split from the `PsiChi` HEALPix
axis using the class's current 5-degree coordinate binning. The two-dimensional
Phi-Chi and Phi-Psi products are formed from the Phi-by-PsiChi projection.

### `residual_function(projection_id, residual_type="raw")`

Returns a `Histogram` for the selected projection and residual definition:

- `raw`: \(D_i - M_i\), the observed-minus-expected count difference.
- `pearson`: \((D_i - M_i)/\sqrt{M_i}\).
- `fractional`: \((D_i - M_i)/M_i\).

Here, \(D_i\) is the projected observed count and \(M_i\) is the projected
expected count. Pearson and fractional residuals are only defined where the
projected expectation is positive; other bins are stored as `NaN`. The mask is
applied to the full histograms before projection. In particular, projected
residuals are calculated from aggregated observed and expected counts, not by
projecting already-calculated residuals.

Example:

```python
# Calculate Pearson residuals for the Phi projection.
phi_pearson = residual_analysis.residual_function(
    "Phi_projection",
    residual_type="pearson",
)
```

### `pearson_poisson_reference(projection_id, n_simulations=1000, bins=15, seed=None, phi_bin_index=None)`

Builds a fixed-fit Poisson reference for the Pearson residual histogram. Each
realization is drawn in full CDS space, then masked and projected using the
same analysis binning. The result contains common histogram edges, the mean
simulated density, and pointwise 95% intervals across independent simulated
histograms. The full simulated CDS datasets are not retained. This reference
does not refit the model and is not a calibrated goodness-of-fit test.

### `prepare_pearson_poisson_references(projection_bins, n_simulations=1000, seed=None, phi_bin_indices=None)`

This lower-level method draws each full-CDS realization once, projects it into
every requested view, and caches the small reference summaries. It is available
for direct use, but normal plotting does not require calling it: configure
`PlottingResiduals` with the simulation options, and it prepares references
automatically the first time a Pearson plot is requested.

```python
# This lower-level call is optional; PlottingResiduals can prepare references automatically.
residual_analysis.prepare_pearson_poisson_references(
    {
        "Phi_projection": 15,
        "Chi_projection": 15,
        "Psi_projection": 15,
        "Phi_Chi_projection": 40,
        "Phi_Psi_projection": 40,
        "PsiChi_projection": 40,
        "Phi_PsiChi_projection": 40,
    },
    n_simulations=1000,
    seed=42,
    phi_bin_indices={"Phi_PsiChi_projection": 30},
)
```

### Residual statistics tables

`statistics_table(residual_type)` creates one compact global summary for one
residual type. It uses only valid full-CDS bins; it does not add rows for
projections or individual Phi slices.

Every table reports the number of valid bins, mean, median, standard deviation,
RMS, minimum, and maximum for its residual type. The Pearson table additionally
reports the percentages with \(|R|>2\) and \(|R|>3\), Pearson chi-square, and
Poisson deviance. Variance is not listed separately because it is the square
of the standard deviation. The per-bin Poisson uncertainty
\(\sigma_i=\sqrt{M_i}\) is shown in the observed-versus-expected count plots;
collapsing it to one table value would hide its bin-by-bin variation. The
Pearson chi-square and deviance are descriptive here: the table does not
assign degrees of freedom or a goodness-of-fit probability. A running mean is
a visual smoothing curve and is not included in these global summaries.

```python
# Display one compact full-CDS table for each residual definition.
display(residual_analysis.statistics_table("raw"))
display(residual_analysis.statistics_table("pearson"))
display(residual_analysis.statistics_table("fractional"))
```

## `PlottingResiduals`

Create the plotting helper from the existing analysis object. Set simulation
options once here; the first Pearson plot prepares references for the available
projections, then later Pearson plots reuse the cache. Raw and fractional plots
do not run simulations. If no seed is provided, the plotter generates one and
reuses it for that instance.

```python
# Reuse the analysis object so plots use its projections and residuals.
from src.plot_residuals import PlottingResiduals

residual_plots = PlottingResiduals(
    residual_analysis,
    n_simulations=1000,
    random_seed=42,
    phi_bin_index=30,
)
```

### `plot(projection_id, residual_type="raw", phi_bin_index=None, running_mean_window=None, n_simulations=None, random_seed=None, show_gaussian_reference=None)`

Returns a Matplotlib `Figure` and also calls `plt.show()`. One-dimensional
plots include observed and expected counts with a nominal Poisson band of
\(\pm\sqrt{M_i}\). Pearson and fractional plots also include a residual
distribution and descriptive statistics. Pearson plots show reference levels
at 0, ±1, ±2, and ±3. By default, Pearson plots retain the theoretical standard
normal reference. Set `n_simulations` on the constructor (or on an individual
`plot` call) to use the fixed-fit Poisson reference instead; the Gaussian can still be overlaid with
`show_gaussian_reference=True`. Simulated references show the observed
histogram, mean simulated density, and a 95% pointwise interval. This is not a
calibrated goodness-of-fit test because the model is not refitted. The 2D and
HEALPix variants follow the same behavior for their residual distributions.

Use the projection IDs from `projected_data()`:

- `Phi_projection`, `Chi_projection`, or `Psi_projection` for one-dimensional
  plots.
- `Phi_Chi_projection` or `Phi_Psi_projection` for paired rectangular 2D
  residual maps.
- `PsiChi_projection` for a HEALPix view integrated over Phi.
- `Phi_PsiChi_projection` for a HEALPix view at one selected Phi bin.

For example:

```python
# Plot fractional residuals across the one-dimensional Phi projection.
phi_fractional_figure = residual_plots.plot(
    "Phi_projection",
    residual_type="fractional",
)
```

### Selecting a Phi slice in the HEALPix view

For `Phi_PsiChi_projection`, `phi_bin_index` is the zero-based bin index, not
an angle in degrees. The plot title reports the corresponding Phi-bin center.
Check the axis centers before choosing the index:

```python
# Inspect the Phi bin centers so the selected slice is explicit.
phi_axis = residual_analysis.projected_data()[
    "Phi_PsiChi_projection"
]["observed"].axes["Phi"]
phi_centers_deg = phi_axis.centers.to_value("deg")
phi_centers_deg
```

Then pass the desired index:

```python
# Plot one Phi slice; replace 30 with the index you want to inspect.
phi_slice_figure = residual_plots.plot(
    "Phi_PsiChi_projection",
    residual_type="pearson",
    phi_bin_index=30,
)
```

`phi_bin_index` is only accepted for `Phi_PsiChi_projection`; it is not used
for `PsiChi_projection`, which is integrated over Phi.

### Optional running mean

For a one-dimensional Pearson plot, pass `running_mean_window` as a positive
integer number of bins. The running mean is centered; at the first and last
bins the window is truncated to the available bins and does not wrap around.
It is not available for raw or fractional residuals, or for 2D/HEALPix plots.

```python
# Overlay a centered 5-bin running mean on the Phi Pearson residuals.
phi_pearson_figure = residual_plots.plot(
    "Phi_projection",
    residual_type="pearson",
    running_mean_window=5,
)
```

## A compact notebook workflow

Once the two histograms have been loaded, the central workflow is:

```python
# Create the analysis using the floor documented for this data product.
residual_analysis = ResidualAnalysis(
    observed_disk_scale,
    expected_disk_scale,
    expected_floor=1e-12,
)

# Check the full-CDS mask and report each residual type separately.
display(residual_analysis.mask_report)
display(residual_analysis.statistics_table("raw"))
display(residual_analysis.statistics_table("pearson"))
display(residual_analysis.statistics_table("fractional"))

# Make a plot from the same analysis and projection products.
residual_plots = PlottingResiduals(residual_analysis)
phi_pearson_figure = residual_plots.plot(
    "Phi_projection",
    residual_type="pearson",
)
```

Use `expected_floor=1e-12` only for products prepared with that specific
replacement. For another data product, check its preparation step and pass its
documented floor, or omit the argument if no artificial floor was applied.
