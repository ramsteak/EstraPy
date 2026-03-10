---
title: Background Removal
parent: Syntax
nav_order: 41
permalink: /commands/background-removal/
has_children: false
math: katex
---

# Background Removal

The `background` command removes a model background from the EXAFS signal ($$\chi(k)$$) and stores the computed background as a new column.
It supports three fitting methods: **polynomial**, **spline**, and **Fourier**.

After background removal, the `chi` column is updated to subtract the modeled background:

$$\chi_{\text{corrected}}(k) = \chi(k) - \text{background}(k)$$

## Basic Usage

```sh
background <method> [range] [method-options]
```

where:

- `<method>` is one of: `polynomial`, `spline`, or `fourier`
- `[range]` is an optional k-space range (default: full range)
- `[method-options]` are method-specific parameters

## Command Options

| Option | Description |
|--------|-------------|
| `[range]` | Optional range in k-space where the background model is fit. Accepts `k` units (default: `0k` to `+∞`).  See [range specification]({{ "/commands/general-syntax#range-specification" | relative_url }}) for details. |

## Background Methods

### Polynomial

Fits a single polynomial across the selected k range.

```sh
background polynomial [range] [options]
```

| Option | Description |
|--------|-------------|
| `--degree <n>` <br> `-d <n>` | Polynomial degree (default: `3`) |
| `--linear` <br> `-l` | Use degree 1 |
| `--quadratic` <br> `-q` | Use degree 2 |
| `--cubic` <br> `-c` | Use degree 3 (default) |
| `--kweight <value>` | k-weighting exponent (default: `2.0`). The fit is performed on k-weighted data to emphasize higher-k features. |

**When to use:** For smooth, quasi-monotonic backgrounds across the entire scan range.

### Spline

Fits a piecewise spline with nodal control points. More flexible than polynomial for irregular background shapes.

```sh
background spline [range] --nodes <node1> [<node2> ...] --degrees <deg1> [<deg2> ...]  [options]
```

| Option | Description |
|--------|-------------|
| `--nodes <n1> [<n2> ...]` | Knot positions or number of equidistant knots (required). If a single dimensionless number is given, that many equidistant knots are generated in the selected range. Otherwise, provide explicit position(s) in k units. |
| `--degrees <d1> [<d2> ...]` | Spline degree(s) for each piece (required). If one node and one degree given, a single spline with that degree is fit. |
| `--kweight <value>` | k-weighting exponent (default: `2.0`). |
| `--fixed-points <k,y> [<k,y> ...]` | Force the spline to pass through specific `(k, χ)` points (optional). Format: `k_value,y_value` (comma-separated). |
| `--continuity <n>` | Continuity order at knots (default: `1`). Use `-1` for full continuity (up to degree-1). |

**When to use:** For backgrounds with multiple features or variable slope, when polynomial fitting is insufficient.

### Fourier

Uses Fourier filtering to isolate low-frequency background. The method transforms to r-space, applies a window to exclude the signal region, then inverse-Fourier transforms.

```sh
background fourier [range] [options]
```

| Option | Description |
|--------|-------------|
| `--rmax <distance>` | Radius cutoff in real space (default: `1.0 Å`). Signals beyond this distance are removed, leaving only long-range background. |
| `--kweight <value>` | k-weighting exponent (default: `2.0`). |
| `--forward-pad <value>` | Padding before the fitted range in k-space (default: `0.1 k`). Suppresses ringing. |
| `--forward-width <value>` | Width of the edge window in k-space (default: `1.0 k`). Controls transition steepness. |
| `--backward-pad <value>` | Padding beyond the cutoff in r-space (default: `0.1 Å`). |
| `--backward-width <value>` | Width of the edge window in r-space (default: `0.2 Å`). |
| `--epsilon <value>` | Small regularization constant (default: `1e-30`) to avoid division by zero in k-weighting. |

**When to use:** When you need automatic, physically-motivated separation of signal from background based on real-space locality.

## Examples

```sh
# Simple cubic polynomial background
background polynomial

# Quadratic polynomial fit in the 3-14 k range
background polynomial 3k 14k --quadratic

# Spline with 5 equidistant knots
background spline --nodes 5 --degrees 3

# Spline with explicit knots at 3.5 and 8.5
background spline --nodes 3.5k 8.5k --degrees 3 3 3

# Fourier background with 2 Å cutoff
background fourier --rmax 2.0A

# Combined: polynomial for range 0-3, Fourier for the scan region
background polynomial 0k 3k --linear
background fourier 3k 15k --rmax 1.5A
```

## Results

After execution, the command:

- creates a `bkg` column containing the computed background
- updates `chi` to `chi - bkg` (EXAFS signal with background removed)

**See also:**

- [Data cleaning]({{ "/commands/data-cleaning" | relative_url }})
- [General syntax]({{ "/commands/general-syntax" | relative_url }})
