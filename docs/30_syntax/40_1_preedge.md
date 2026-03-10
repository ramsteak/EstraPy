---
title: Pre-edge Correction
parent: Normalization
grand_parent: Syntax
nav_order: 1
permalink: /commands/normalization/preedge/
math: katex
---

# Pre-edge Correction

The `preedge` command removes the pre-edge background from the absorption signal and determines the absorption edge energy ($$E_0$$).

## Basic Usage

```sh
preedge <range> [options]
```

Fit a polynomial to the absorption signal before the edge and subtract it to establish a zero baseline.

**Examples:**
```sh
preedge .. -50eV --linear

preedge 8000eV 8300eV --degree 1
```

## Command options

| Option | Description |
|--------|-------------|
| `<range>` | Energy range for fitting the pre-edge polynomial (required). See [range]({{ "/commands/general-syntax#range-specification" | relative_url }}) for details. |


### Polynomial degree options

<small>These options specify the degree of the polynomial used to fit the pre-edge background, and are mutually exclusive.</small>


| Option | Description |
|-------|-------------|
| `--degree <n>` <br> `--deg <n>` | Polynomial degree (default: 1) |
| `--constant` <br> `-C` | Use constant (degree 0) |
| `--linear` <br> `-l` | Use linear fit (degree 1, default) |
| `--quadratic` <br> `-q` | Use quadratic fit (degree 2) |
| `--cubic` <br> `-c` | Use cubic fit (degree 3) |

## Results

The command subtracts the extrapolated fitted polynomial from the absorption signal, and updates the `a` column with the corrected absorption. It also defines the `pre` column, containing the fitted polynomial values.

## Tips and Best Practices

1. **Range selection:** 
   - Start as far from the edge as your data allows
   - End 20-100 eV before the edge
   - Avoid including any XANES features

2. **Polynomial degree:** 
   - Linear (degree 1) works for most cases
   - Use constant (degree 0) only if baseline is flat
   - Use quadratic (degree 2) for strongly curved backgrounds
   - Avoid higher degrees unless absolutely necessary

**See also:**

- [Post-edge Correction]({{ "/commands/normalization/postedge" | relative_url }}) - Next normalization step
- [Normalization]({{ "/commands/normalization/normalization" | relative_url }}) - Final normalization step
- [Normalization Overview]({{ "/commands/normalization" | relative_url }}) - Complete workflow
