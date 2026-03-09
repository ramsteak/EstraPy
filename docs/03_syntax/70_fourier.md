---
title: Fourier Transform
parent: Syntax
nav_order: 50
permalink: /commands/fourier-transform/
has_children: false
math: katex
---

# Fourier Transform

The `fourier` command transforms EXAFS data from k-space to R-space (radial distance space) via Fourier transform, enabling analysis of the local coordination environment around the absorbing atom.

The transform is applied to k-weighted chi data with optional apodization:

$$F(R) = \int k^n \cdot \chi(k) \cdot w(k) \cdot e^{2ikR} dk$$

where $$n$$ is the k-weight, $$w(k)$$ is an apodization window, and the integral is over the specified k-range.

## Basic Usage

```sh
fourier <k-range> [options]
```

where:

- `<k-range>` is a mandatory two-value range in k-space (required)
- `[options]` control k-weighting, window function, and r-axis spacing

## Command Options

| Option | Description |
|--------|-------------|
| `<k-range>` | k-space range for the transform (required). Both limits must be specified; use `..` for open bounds within the available data. Accepts `k` units. See [range specification]({{ "/commands/general-syntax#range-specification" | relative_url }}) for details. |
| `--kweight <n>` <br> `-k <n>` | k-weighting exponent (default: `0.0`). Common values: `1` (linear), `2` (quadratic), `3` (cubic). |
| `--apodizer <func>` <br> `-a <func>` | Window function for apodization (default: `hanning`). See [Apodization Functions](#apodization-functions) for choices. |
| `--parameter <p>` <br> `-p <p>` | Shape parameter for window functions that support it (default: `3.0`). |
| `--width <distance>` <br> `-w <distance>` | Ramp width for the apodization window edges (default: half the k-range span). Specified in `Å`. |
| `--maxR <distance>` | Maximum R value for the output r-axis (default: auto-computed from k-range). Specified in `Å`. |
| `--dr <distance>` <br> `--spacing <distance>` | Fixed r-axis spacing (default: Nyquist-optimal). Specified in `Å`. |

## Apodization Functions

Apodization windows reduce spectral ringing from the sharp k-range cutoffs. EstraPy supports:

| Function | Description | Parameter use |
|----------|-------------|----------------|
| `rectangular` | No apodization (sharp cutoff) | - |
| `hanning` | Cosine-based window (default) | - |
| `hamming` | Variant of Hanning with reduced sidelobe | - |
| `welch` | Parabolic window | - |
| `sine` | Pure sine window | - |
| `blackman` | Very low sidelobe window | - |
| `gaussian` | Gaussian shape with tunable width | See definition [below](#parameter-definition). |
| `exponential` | Exponential decay with tunable rate | See definition [below](#parameter-definition). |

Window names are matched fuzzily (minimum 3 characters required).

### Window definitions

#### Rectangular

No apodization is applied (sharp cutoff):

$$w(r) = \begin{cases} 1 & r \leq 1 \\ 0 & r > 1 \end{cases}$$

#### Hanning

Cosine-based smooth window:

$$w(r) = \frac{1}{2} + \frac{1}{2}\cos(\pi r)$$

#### Hamming

Variant of Hanning with reduced first sidelobe:

$$w(r) = 0.54 - 0.46\cos(2\pi r)$$

#### Welch

Parabolic window:

$$w(r) = 1 - r^2$$

#### Sine

Pure sine taper:

$$w(r) = \cos\left(\frac{\pi r}{2}\right)$$

#### Blackman

Very low sidelobe window with excellent stopband attenuation:

$$w(r) = 0.42 - 0.5\cos(2\pi r) + 0.08\cos(4\pi r)$$

#### Gaussian

Gaussian shape with tunable width controlled by parameter $$p$$ (default: 3.0):

$$w(r) = \exp(-r^2 p)$$

Higher $$p$$ values create narrower windows with sharper cutoffs.

#### Exponential

Exponential decay with tunable rate controlled by parameter $$p$$ (default: 3.0):

$$w(r) = \exp(-rp)$$

Higher $$p$$ values create faster decay rates.

### Parameter definition

For `gaussian` and `exponential` windows, the `--parameter` option controls the shape:

- **Gaussian**: Controls the width; higher values ($$p \geq 5$$) create sharper transitions
- **Exponential**: Controls the decay rate; higher values create steeper roll-off

The normalized distance $$r$$ ranges from 0 (at the flat region edge) to 1 (at the window outer edge).

## R-axis Specification

The r-axis (output radial distance grid) can be specified via:

| Case | r-axis behavior |
|------|-----------------|
| No options | **Nyquist-optimal**: spacing and extent determined from k-range via $$\Delta R = \pi / (2 \Delta k_{\max})$$ |
| `--maxR <Rmax>` only | Nyquist spacing computed, axis spans $$[0, R_{\max}]$$ |
| `--dr <spacing>` only | Fixed spacing, extent computed from k-range |
| Both `--maxR` and `--dr` | Fixed spacing and extent: axis spans $$[0, R_{\max}]$$ with steps of `--dr` |

## Results

After executing `fourier`, a new **Fourier domain** is created for each page containing:

- **`r`** - Radial distance axis (in Ångströms)
- **`f`** - Fourier transform magnitude

The magnitude is typically plotted to identify coordination shells.

## Examples

```sh
# Basic Fourier transform with Hanning window
fourier 3k 15k

# Quadratic k-weighted transform (emphasizes higher-k)
fourier 2k 14k --kweight 2

# Custom window with narrow ramp (sharp edges)
fourier 3k 14k --apodizer kaiser --parameter 5 --width 0.5A

# Fine r-space sampling up to 6 Ångströms
fourier 2k 16k --maxR 6A --dr 0.01A

# Linear k-weighted with Blackman window
fourier 3k 13k --kweight 1 --apodizer blackman
```

## Tips and Best Practices

1. **k-range selection:**
   - Start where the data quality is good (typically 2–3 k)
   - End where noise dominates (often 12–15 k, depending on material and temperature)
   - Wider ranges improve r-resolution per Nyquist

2. **k-weight choice:**
   - `k=1`: Emphasizes low-k structure
   - `k=2`: Default choice, balanced emphasis
   - `k=3`: Emphasizes high-k (farther-distance) features

3. **r-axis:**
   - Nyquist default provides good efficiency
   - Reduce `--maxR` to focus on near-neighbor shells
   - Increase `--dr` for smoother curves at the cost of larger output (does not generate new information beyond Nyquist)

**See also:**

- [Background Removal]({{ "/commands/background-removal" | relative_url }})
- [General syntax]({{ "/commands/general-syntax" | relative_url }})
