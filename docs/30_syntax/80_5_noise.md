---
title: Noise
parent: Data Manipulation
grand_parent: Syntax
nav_order: 5
permalink: /commands/data-manipulation/noise/
math: katex
---

# Noise

The `noise` command estimates noise levels in XAS data by analyzing the difference between even and odd data points. Unlike other manipulation commands, this is **non-destructive** — it creates a new column without altering existing data.

## Basic Usage

```sh
noise [options]
```

All options are optional; defaults typically work well for most XAS data.

## Command Options

| Option | Description |
|--------|-------------|
| `--xaxiscol <axis>` | Axis column used for even-odd pairing (default: `E`). |
| `--yaxiscol <data>` | Data column to analyze for noise (default: `a`). |

## Noise Estimation Method

The command uses the **even-odd difference method**:

1. Pairs consecutive data points: (point 0, point 1), (point 2, point 3), ...
2. Computes a regression-based difference between even and odd subsequences
3. Estimates noise standard deviation from these differences

This method is robust because:

- True signal varies smoothly and cancels out in even-odd differences
- High-frequency noise (point-to-point fluctuations) is preserved
- It works well even with correlated noise


## Output

The command creates a new data column named `s<yaxiscol>`:

- If `--yaxiscol a` (default): creates column `sa`
- If `--yaxiscol chi`: creates column `schi`
- If `--yaxiscol mu`: creates column `smu`

This column contains the estimated standard deviation (noise level) at each point.

## Examples

```sh
# Estimate noise in default absorption column 'a'
noise

# Estimate noise in chi column
noise --yaxiscol chi

# Use k-axis for even-odd pairing
noise --xaxiscol k --yaxiscol chi

# Estimate noise in normalized mu
noise --yaxiscol mu
```

## Use Cases

- **Data quality assessment:** Identify noisy regions in scans
- **Weighting for fitting:** Use noise estimates to weight data points in EXAFS fitting
- **Experiment comparison:** Compare noise levels across different measurement conditions
- **Outlier detection:** Identify regions where signal-to-noise ratio is poor
- **Visualization:** Plot noise alongside data to assess quality

## Tips and Best Practices

1. **Run early in workflow:** Estimate noise before normalization or background removal to assess raw data quality

2. **Multiple estimates:** Run `noise` on different columns (`a`, `chi`, `mu`) to understand how processing affects uncertainty

3. **Visual inspection:** Always plot noise alongside data:
   ```sh
   noise
   plot E:a --figure 1
   plot E:sa --figure 1  # Plot absorption with noise overlay
   ```

**See also:**

- [Data Manipulation overview]({{ "/commands/data-manipulation" | relative_url }})
- [General syntax]({{ "/commands/general-syntax" | relative_url }})
