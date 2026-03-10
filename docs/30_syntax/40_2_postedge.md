---
title: Post-edge Correction
parent: Normalization
grand_parent: Syntax
nav_order: 2
permalink: /commands/normalization/postedge/
math: katex
---

# Post-edge Correction

The `postedge` command removes the post-edge background from the absorption signal and determines the edge step ($$J_{0}$$).

## Basic Usage

```sh
postedge <range> [options]
```

Fits a polynomial to the absorption signal after the edge and removes it from the data to determine the absorption step.

**Examples:**
```sh
postedge 50eV .. --linear

postedge 8300eV 9000eV --degree 3
```

## Command options

| Option | Description |
|--------|-------------|
| `<range>` | Energy range for fitting the post-edge polynomial (required). See [range]({{ "/commands/general-syntax#range-specification" | 
relative_url }}) for details. |

### Polynomial degree options

<small>These options specify the degree of the polynomial used to fit the post-edge background, and are mutually exclusive.</small>


| Option | Description |
|-------|-------------|
| `--degree <n>` <br> `--deg <n>` | Polynomial degree (default: 2) |
| `--constant` <br> `-C` | Use constant (degree 0) |
| `--linear` <br> `-l` | Use linear fit (degree 1) |
| `--quadratic` <br> `-q` | Use quadratic fit (degree 2, default) |
| `--cubic` <br> `-c` | Use cubic fit (degree 3) |

### Fitting axis options

<small>These options specify which column to use for fitting the post-edge background, and are mutually exclusive.</small>

| Option | Description |
|-------|-------------|
| `--xaxis <col>` | Use specified column for fitting (default: `E`) |
| `--E-axis` <br> `-E` | Use energy column for fitting (default) |
| `--k-axis` <br> `-k` | Use k column for fitting |
| `--e-axis` <br> `-e` | Use relative energy (E - E0) for fitting |

### Mode options

<small>These options specify how the post-edge background is removed from the data, and are mutually exclusive.</small>

| Option | Description |
|-------|-------------|
| `--mode <mode>` | Post-edge correction mode (default: `division`) |
| `--subtraction` <br> `--subtract` <br> `--sub` <br> `-s` | Subtract fitted polynomial from data |
| `--division` <br> `--divide` <br> `--div` <br> `-d` | Divide data by fitted polynomial (default) |

## Results

The command fits a polynomial to the specified post-edge region, and uses it to correct the absorption signal. The `a` column is updated with the corrected absorption, and a new `post` column is defined containing the fitted polynomial values. The edge step ($$J_{0}$$) is determined from the value of the fitted polynomial at the edge energy ($$E_0$$) and stored in the `J0` variable.

The **subtraction** mode removes the fitted polynomial from the absorption signal, such that the corrected absorption is given by:

$$a_{\text{corrected}} = a_{\text{original}} - \text{post} + J_{0}$$

The **division** mode divides the absorption signal by the fitted polynomial, such that the corrected absorption is given by:

$$a_{\text{corrected}} = \frac{a_{\text{original}}}{\text{post}} \cdot J_{0}$$

The command does not perform normalization by the edge step, it only determines its value. Normalization is performed in the subsequent `normalize` step.
This could seem counterintuitive, but it allows for a consistent workflow between the two modes, and gives users more control over the normalization process.

## Tips and Best Practices

1. **Range selection:**
   - Start 50-200 eV after the edge
   - End as far from the edge as your data allows
   - Avoid including any early EXAFS features

2. **Polynomial degree:**
   - Quadratic (degree 2) works for most cases
   - Constant (degree 0) is suited for XANES analysis, where the post-edge background may be severely distorted by early EXAFS features
   - Use linear (degree 1) only if background is nearly linear
   - Use cubic (degree 3) for strongly curved backgrounds
   - Avoid higher degrees unless absolutely necessary

**See also:**

- [Pre-edge Correction]({{ "/commands/normalization/preedge" | relative_url }}) - Previous normalization step
- [Normalization]({{ "/commands/normalization/normalization" | relative_url }}) - Final normalization step
- [Normalization Overview]({{ "/commands/normalization" | relative_url }}) - Complete workflow
