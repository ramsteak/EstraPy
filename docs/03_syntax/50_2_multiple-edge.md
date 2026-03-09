---
title: Multiple Edges
parent: Data cleaning
grand_parent: Syntax
nav_order: 2
permalink: /commands/multiple-edges/
math: katex
---

# Multiple Edges

The `multiedge` command removes a secondary edge-like contribution from a data column by building a model function and subtracting it.

In practice, `multiedge`:

1. computes a compensation curve from the selected edge function
2. stores it as a new column called `edge_<kind>`
3. updates the target column by subtracting that curve

## Basic Usage

```sh
multiedge <kind> <a> <b> <c> [options]
```

where:

- `<kind>` selects the edge function
- `<a>` is the function amplitude
- `<b>` is the edge position (number with unit)
- `<c>` controls width/shape (number with unit)

## Command Options

| Option | Description |
|--------|-------------|
| `--axis <axis>` <br> `-x <axis>` | Axis used to evaluate the function. If omitted, inferred from units of `b` and `c`. |
| `--column <col>` <br> `-c <col>` | Data column to correct (default: `a`). |
| `--reparam` <br> `-r` | Disable reparameterized scaling and use native function scaling. |

## Edge Function Types

The following function kinds are available:

| Kind | Description | Accepted aliases |
|------|-------------|------------------|
| `atan` | Arctangent step-like model | `arctangent`, `cauchy`, `lorentzian` |
| `tanh` | Hyperbolic-tangent (logistic-like) model | `hyperbolictangent`, `logistic`, `sigmoid` |
| `erf` | Error-function step model | `errorfunction`, `normal`, `gaussian` |
| `exp` | One-sided exponential model | `exponential`, `onesidedexponential` |
| `lin` | Linear ramp model | `linear`, `ramp`, `triangle`, `triangular`, `sawtooth` |

Notes:

- Function names are matched fuzzily (minimum 3 characters).
- `b` and `c` accept `eV` or `k` units, depending on the axis used for evaluation.

## Examples

```sh
# Arctangent compensation on default column 'a'
multiedge atan 0.25 8400eV 50eV

# Exponential compensation using k-axis units
multiedge exp 0.12 7.5k 0.8k

# Correct a different data column
multiedge erf 0.18 8350eV 35eV --column mu

# Linear ramp with explicit axis
multiedge lin 0.10 6.0k 1.2k --axis k
```

## Results

For each processed page in the reciprocal domain, the command:

- creates `edge_<kind>` with the computed compensation curve
- subtracts this curve from the selected target column

This removes the modeled multiple-edge contribution while keeping the correction curve available for inspection.

**See also:**

- [Data cleaning overview]({{ "/commands/data-cleaning" | relative_url }})
- [Glitch Removal]({{ "/commands/glitch-removal" | relative_url }})
