---
title: Background removal
parent: Commands
nav_order: 40
permalink: /commands/background-removal
math: katex
---

# Background removal

XAS analysis requires removing background contributions and normalizing the signal. This is performed in different steps, represented by different commands. In EstraPy, the background removal is represented by the subtraction of the background before the edge, called the preedge, the removal of the background after the edge, called the postedge, and removing contributions to the signal due to free, non-interacting electrons. These three steps are performed using the [`preedge`](#preedge), [`postedge`](#postedge) and [`background`](#background) commands.

## PreEdge

The `preedge` command estimates the preedge contribution from the signal in the `x` column, stores it in the `pre` column, and updates the signal with the corrected values. The preedge is modelled as a polynomial of degree n, usually linear, in the given range, and extrapolates the estimate across the entire data range. The range can be specified in either eV or k.

```sh
preedge <range> [--options]
```

|Argument|Explanation|
|--|--|
|<span class="nowrap">`--constant` / `-C`</span>|Models the preedge as a polynomial of order 0 (a constant contribution)|
|<span class="nowrap">`--linear` / `-l`</span>|Models the preedge as a polynomial of order 1 (a linear contribution)|
|<span class="nowrap">`--quadratic` / `-q`</span>|Models the preedge as a polynomial of order 2|
|<span class="nowrap">`--cubic` / `-c`</span>|Models the preedge as a polynomial of order 3|
|<span class="nowrap">`--polynomial` / `-p` `<degree>`</span>|Models the preedge as a polynomial of order `degree`|

## PostEdge

The `postedge` command estimates the postedge contribution from the signal in the `x` column, stores it in the `post` column, and updates the signal with the corrected values. The postedge is modelled as a polynomial of degree n in the given range, and this estimation is extrapolated across the entire data range. The range can be specified in either eV or k.
The command also defines the variable `J0` for each file, defined to be the value of the postedge estimated at $$E = E_{0}$$.

```sh
postedge <range> [--options]
```

|Argument|Explanation|
|--|--|
|<span class="nowrap">`--constant` / `-C`</span>|Models the postedge as a polynomial of order 0 (a constant contribution).|
|<span class="nowrap">`--linear` / `-l`</span>|Models the postedge as a polynomial of order 1 (a linear contribution).|
|<span class="nowrap">`--quadratic` / `-q`</span>|Models the postedge as a polynomial of order 2.|
|<span class="nowrap">`--cubic` / `-c`</span>|Models the postedge as a polynomial of order 3.|
|<span class="nowrap">`--polynomial` / `-p` `<degree>`</span>|Models the postedge as a polynomial of order `degree`.|
|<span class="nowrap">`--divide` / `-d`</span>|Corrects the data by dividing the `x` column by the postedge.|
|<span class="nowrap">`--subtract` / `-s`</span>|Corrects the data by subtracting the postedge from the `x` column.|
|<span class="nowrap">`--energy` / `-e`</span>|Performs the polynomial regression on $$\mu(E)$$, in energy space. If neither this flag nor `--wavevector` is specified, the regression space is inferred from the range.|
|<span class="nowrap">`--wavevector` / `-k`</span>|Performs the polynomial regression on $$\mu(k)$$, in wavevector space. If neither this flag nor `--energy` is specified, the regression space is inferred from the range.|

## Background

The `background` command extracts the background contribution from the signal in the `x` column, and stores it in the `bkg` column. This background estimated is then subtracted from the signal, yielding the corrected data. The command supports multiple methods for background estimation.

```sh
background <mode> [--options]
```

### Constant

Removes a constant contribution, by default equal to 1, from the data. This is useful if no background correction is needed, but the result of normalization has mean value of 1.

```sh
background constant [--options]
```

|Argument|Explanation|
|--|--|
|<span class="nowrap">`--value` / `-v` `<value>`</span>|Subtracts the given `value` from the signal. Default is 1.0.|

### BSpline

Models the background using B-splines, automatically determining the optimal smoothing. Uses `scipy.interpolate.UnivariateSpline`. The range can be specified in eV or k.

```sh
background bspline <range> [--options]
```

|Argument|Explanation|
|--|--|
|<span class="nowrap">`--kweight` / `-k` `<value>`</span>|Performs the operation on data weighed by the specified factor: $$k^{n}\cdot\mu(k)$$. By default, does not weigh the data (equal to a kweight of 0).|

### Fourier

Models the background as low-frequency Fourier contributions. This method performs a continuous Fourier transform on the signal in k-space, filters out the high frequencies uses a Hann window (up to Rmax), and then performs the inverse transform to estimate the background.

```sh
background fourier <Rmax> [--options]
```

|Argument|Explanation|
|--|--|
|<span class="nowrap">`--kweight` / `-k` `<value>`</span>|Performs the operation on data weighed by the specified factor: $$k^{n}\cdot\mu(k)$$. By default, does not weigh the data (equal to a kweight of 0).|
|<span class="nowrap">`--iterations` / `-i`</span>|The number of times to iterate the method. By default, applies the background removal three times.|

### Smoothing

Models the background as a smoothed version of the raw data. The method uses the lowess algorithm to smooth the data.

```sh
background smoothing <range> [--options]
```

|Argument|Explanation|
|--|--|
|<span class="nowrap">`--kweight` / `-k` `<value>`</span>|Performs the operation on data weighed by the specified factor: $$k^{n}\cdot\mu(k)$$. By default, does not weigh the data (equal to a kweight of 0).|
|<span class="nowrap">`--iterations` / `-i`</span>|The number of times to iterate the method. By default, applies the background removal once.|
|<span class="nowrap">`--fraction` / `-f`</span>|The fraction of the total datapoints to use for the smoothing. Must be between 0 and 1. By default uses 30% of the datapoints (0.3)|

## Examples

### PreEdge example

```sh
preedge .. -80eV -l
```

This snippet subtracts a linear preedge from the `x` column, estimated over the region from $$-\infty$$ to -80eV relative to $$E_{0}$$ (determined in a previous `edgeenergy` step).

### PostEdge example

```sh
postedge 3k .. -cde
```

This snippet divides the data by a cubic polinomial, fitted over $$\mu(E)$$ in the region from 3k to $$\infty$$, to model the postedge contribution of the `x` column.

### Background example

```sh
background fourier 1.1A -k2
```

This snippet estimates the background from the lower frequencies of the fourier transform, up to 1.1A, with a k-weight of 2. The process is performed three times (the default value). The background contribution is then subtracted from the signal.
