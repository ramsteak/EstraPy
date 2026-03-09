---
title: Rebin
parent: Data Manipulation
grand_parent: Syntax
nav_order: 3
permalink: /commands/data-manipulation/rebin/
math: katex
---

# Rebin

The `rebin` command bins data into fixed intervals and averages all points within each bin. This is a **destructive operation** that reduces noise but permanently replaces original data.

(Note that the original data file remains unchanged; only the in-memory dataset is modified.)

## Basic Usage

```sh
rebin <range> <--interval <step> | --number <n>> [options]
```

where:

- `<range>` defines the binning limits
- Either `--interval` or `--number` specifies bin size (exactly one required)

## Command Options

| Option | Description |
|--------|-------------|
| `<range>` | Range for rebinning (required). Must have finite bounds. See [range specification]({{ "/commands/general-syntax#range-specification" | relative_url }}) for details. |
| `--interval <step>` | Fixed bin width (mutually exclusive with `--number`). Must include units. |
| `--number <n>` | Number of bins to create (mutually exclusive with `--interval`). |
| `--axis <axis>` <br> `-x <axis>` | Axis column for binning (auto-inferred from range units if omitted). |
| `--domain <domain>` | Domain in which to rebin (default: `reciprocal`). Options: `reciprocal`, `fourier`. |
| `--fix-points` | Interpolate bin averages back to exact bin centers (preserves target axis values). |

## Binning Method

The command:

1. Creates bin boundaries at half-intervals: `[range[0] - step/2, ..., range[1] + step/2]`
2. Assigns each data point to a bin
3. Computes the mean of all points in each bin (all columns)
4. Optionally interpolates to exact bin centers if `--fix-points` is set

## Fix Points Option

- **Default (no `--fix-points`):** Bin centers are the mean axis values within each bin (may not align exactly with target grid)
- **With `--fix-points`:** After averaging, cubic spline interpolation repositions points to exact bin centers

Use `--fix-points` when you need bin centers at precise positions (e.g., for subsequent operations requiring exact alignment between files).

## Examples

```sh
# Rebin energy axis to 1 eV intervals
rebin 8000eV 9000eV --interval 1eV

# Create exactly 100 bins in k-space
rebin 2k 14k --number 100

# Rebin Fourier domain with fixed bin centers
rebin 0A 5A --interval 0.1A --domain fourier --fix-points

# Explicit axis specification
rebin 0 1000 --interval 2 --axis E --domain reciprocal
```

## Behavior

Rebinning:

- **Reduces noise** by averaging multiple points within each bin
- **Downsamples** high-resolution data for faster processing
- **Removes outliers** if most points in a bin are consistent
- **Smooths** irregular step sizes into uniform intervals

{: .warning }
**Destructive operation:** Original data is replaced with binned averages. The original data file remains unchanged; only the in-memory dataset is modified. Data points outside the extended range `[range[0] - step/2, range[1] + step/2]` are discarded.

## Use Cases

- **Noise reduction:** Average out high-frequency noise while preserving overall shape
- **Downsampling:** Reduce point count from oversampled scans
- **Uniformity:** Convert variable-spacing scans to fixed intervals
- **Pre-processing for averaging:** Align multiple scans to identical grids before merging

## Tips and Best Practices

1. **Choose appropriate bin size:**

   - Too small → minimal noise reduction
   - Too large → loss of spectral features
   - Typical: 0.5–2 eV for energy, 0.05–0.2 k⁻¹ for k-space

2. **Check data spacing:** Bin width should be larger than original step size to achieve averaging

3. **Use for noise, not upsampling:** Rebinning cannot increase resolution; use `interpolate` for upsampling (interpolation does not add information, only resamples existing data)

4. **Edge effects:** Data near range boundaries may be lost if bins extend outside the data range

## Comparison with Interpolate

| Feature | `rebin` | `interpolate` |
|---------|---------|---------------|
| Method | Binning + averaging | Cubic spline fitting |
| Noise reduction | Yes (averaging reduces noise) | No |
| Outlier handling | Robust (majority vote per bin) | Sensitive (fits through all points) |
| Use case | Noise reduction + downsampling | Resampling to new grid |

**See also:**

- [Data Manipulation overview]({{ "/commands/data-manipulation" | relative_url }})
- [Interpolate]({{ "/commands/data-manipulation/interpolate" | relative_url }})
- [General syntax]({{ "/commands/general-syntax" | relative_url }})
