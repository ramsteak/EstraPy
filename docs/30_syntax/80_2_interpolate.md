---
title: Interpolate
parent: Data Manipulation
grand_parent: Syntax
nav_order: 2
permalink: /commands/data-manipulation/interpolate/
math: katex
---

# Interpolate

The `interpolate` command resamples data onto a new uniform grid using cubic spline interpolation. This is a **destructive operation** — the original axis and data values are replaced.

(Note that the original data file remains unchanged; only the in-memory dataset is modified.)

## Basic Usage

```sh
interpolate <range> <--interval <step> | --number <n>> [options]
```

where:

- `<range>` defines the new axis limits
- Either `--interval` or `--number` specifies grid spacing (exactly one required)

## Command Options

| Option | Description |
|--------|-------------|
| `<range>` | Range for the new interpolated axis (required). See [range specification]({{ "/commands/general-syntax#range-specification" | relative_url }}) for details. |
| `--interval <step>` | Fixed step size for the new axis (mutually exclusive with `--number`). Must include units. |
| `--number <n>` | Number of evenly-spaced points in the new axis (mutually exclusive with `--interval`). |
| `--axis <axis>` | Axis column for interpolation (auto-inferred from range units if omitted). |
| `--domain <domain>` | Domain in which to interpolate (auto-inferred if omitted). Options: `reciprocal`, `fourier`. |

## Interpolation Method

EstraPy uses **cubic spline interpolation** (degree 3) via `scipy.interpolate.make_interp_spline`. This provides:

- Smooth, continuous curves through existing data points
- C² continuity (smooth first and second derivatives)
- Minimal oscillation between points

## Examples

```sh
# Interpolate to 0.5 eV steps over 8000-9000 eV
interpolate 8000eV 9000eV --interval 0.5eV

# Interpolate to exactly 500 points in k-space
interpolate 3k 15k --number 500

# Fine r-space grid with 0.01 Å spacing
interpolate 0A 6A --interval 0.01A --domain fourier

# Explicit axis specification
interpolate 0 1000 --number 2000 --axis E --domain reciprocal
```

## Behavior

The command:

1. Creates a new uniform axis within `[range[0], range[1]]` with specified spacing or count
2. Interpolates all data columns onto this new axis using cubic splines
3. Replaces the domain's data with interpolated values

{: .warning }
**Destructive operation:** Original axis values and data are replaced with interpolated results. The original data file remains unchanged; only the in-memory dataset is modified. Interpolation cannot add information — it only resamples existing data, and may introduce noise or artifacts if the original data is sparse or noisy.

## Use Cases

- **Uniform grids:** Convert irregularly-spaced data to fixed intervals
- **Alignment:** Resample multiple scans to identical grids before averaging
- **Upsampling:** Increase point density for smoother plots (cosmetic only)
- **Downsampling:** Reduce point count while preserving overall shape

## Tips and Best Practices

1. **Avoid excessive upsampling:** Interpolation does not improve resolution or add real information
2. **Preserve Nyquist limits:** When downsampling, ensure the new interval captures all spectral features
3. **Check endpoint behavior:** Cubic splines may oscillate near range boundaries if data is noisy
4. **Pre-filter noise:** Consider using `rebin` instead if your goal is noise reduction

## Comparison with Rebin

| Feature | `interpolate` | `rebin` |
|---------|---------------|---------|
| Method | Cubic spline fitting | Binning + averaging |
| Noise reduction | No | Yes (averages reduce noise) |
| Speed | Moderate | Fast |
| Use case | Resampling to new grid | Noise reduction + downsampling |

**See also:**

- [Data Manipulation overview]({{ "/commands/data-manipulation" | relative_url }})
- [Rebin]({{ "/commands/data-manipulation/rebin" | relative_url }})
- [General syntax]({{ "/commands/general-syntax" | relative_url }})
