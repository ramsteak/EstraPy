---
title: Data Manipulation
parent: Syntax
nav_order: 60
permalink: /commands/data-manipulation/
has_children: true
math: katex
---

# Data Manipulation

Data manipulation commands modify the structure or organization of XAS datasets. These operations include resampling, filtering, merging, and extracting subsets of data to prepare it for analysis or visualization.

EstraPy provides six data manipulation commands:

1. **[`cut`]({{ "/commands/data-manipulation/cut" | relative_url }})** - Extracts a subset of data within a specified range
2. **[`interpolate`]({{ "/commands/data-manipulation/interpolate" | relative_url }})** - Resamples data onto a new uniform grid via interpolation
3. **[`rebin`]({{ "/commands/data-manipulation/rebin" | relative_url }})** - Bins data into fixed intervals and averages within bins
4. **[`average`]({{ "/commands/data-manipulation/average" | relative_url }})** - Merges multiple scans into averaged spectra
5. **[`noise`]({{ "/commands/data-manipulation/noise" | relative_url }})** - Estimates noise levels from even-odd point differences
6. **[`filter`]({{ "/commands/data-manipulation/filter" | relative_url }})** - Removes pages that do not satisfy dataset-level constraints

## Destructive vs. Non-Destructive Operations

{: .warning }
Most data manipulation commands are **destructive**: they replace original data and cannot be undone. Always verify your parameters before executing.

(Note that the original data file remains unchanged; only the in-memory dataset is modified. You can reload the file to restore original data if needed.)

- **Destructive:** `cut`, `interpolate`, `rebin`, `average`, `filter` — Original working dataset is permanently modified
- **Non-destructive:** `noise` — Adds a new column without altering existing data

## When to Use Data Manipulation

### Cut

Use `cut` to:

- Remove noisy or unusable regions at the beginning or end of a scan
- Focus analysis on a specific energy or k-range
- Trim data before merging files with different scan ranges

### Interpolate

Use `interpolate` to:

- Resample data onto a finer or coarser grid
- Align multiple scans with different step sizes before averaging
- Prepare data for algorithms requiring uniform spacing

### Rebin

Use `rebin` to:

- Reduce noise by averaging neighboring points
- Downsample high-resolution data for faster processing
- Create uniform bins from irregularly spaced data

### Average

Use `average` to:

- Merge multiple scans of the same sample to improve signal-to-noise
- Combine replicate measurements
- Group scans by metadata (temperature, concentration, etc.)

### Noise

Use `noise` to:

- Assess data quality and identify noisy regions
- Weight data points in fitting routines
- Compare noise levels across different experimental conditions

### Filter

Use `filter` to:

- Remove scans that do not fully cover a required axis range
- Enforce consistent coverage before averaging or fitting
- Exclude truncated pages from batch processing

## Domain Considerations

- `cut`, `interpolate`, `rebin` can operate in **any domain** (reciprocal, Fourier)
- `average` operates on a **specified domain** (default: reciprocal)
- `noise` operates in the **reciprocal domain** only
- `filter` supports inferred or explicit domain selection, depending on the range/axis used

## See Also

- **[Cut]({{ "/commands/data-manipulation/cut" | relative_url }})** - Extract data subsets
- **[Interpolate]({{ "/commands/data-manipulation/interpolate" | relative_url }})** - Resample onto new grid
- **[Rebin]({{ "/commands/data-manipulation/rebin" | relative_url }})** - Bin and average
- **[Average]({{ "/commands/data-manipulation/average" | relative_url }})** - Merge multiple scans
- **[Noise]({{ "/commands/data-manipulation/noise" | relative_url }})** - Estimate noise levels
- **[Filter]({{ "/commands/data-manipulation/filter" | relative_url }})** - Remove pages that do not meet coverage requirements

---

**Next:** Choose a command to learn about specific options and usage.
