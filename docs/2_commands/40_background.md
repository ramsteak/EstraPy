---
title: Background removal
parent: Commands
nav_order: 4
permalink: /commands/background-removal
math: katex
---

# Background removal

XAS analysis requires removal of the background contributions to the signal and signal normalization. This operation is performed in multiple consecutive steps, represented by different commands. In EstraPy, the background removal is represented by the subtraction of the background before the edge, called the preedge, the removal of the background after the edge, called the postedge, and removing contributions to the signal due to free, non-interacting electrons. These three steps are performed by three different commands, namely [`preedge`](#preedge), [`postedge`](#postedge) and [`background`](#background).

## PreEdge

The `preedge` command extracts the preedge contribution from the signal in the `x` column, assigning it to the `pre` column, and modifies the signal column with the corrected value. The preedge is modelled as a polynomial of degree n in the given range, and this estimation is extended to the entire data range. The range can be specified in either eV or k.

```sh
preedge <range> [--options]
```

|Argument|Explanation|
|--|--|
|<div class="nowrap">`--constant` / `-C`</div>|Models the preedge as a polynomial of order 0 (a constant contribution)|
|<div class="nowrap">`--linear` / `-l`</div>|Models the preedge as a polynomial of order 1 (a linear contribution)|
|<div class="nowrap">`--quadratic` / `-q`</div>|Models the preedge as a polynomial of order 2|
|<div class="nowrap">`--cubic` / `-c`</div>|Models the preedge as a polynomial of order 3|
|<div class="nowrap">`--polynomial` / `-p` `<degree>`</div>|Models the preedge as a polynomial of order `degree`|

## PostEdge

The `postedge` command extracts the postedge contribution from the signal in the `x` column, assigning it to the `post` column, and modifies the signal column with the corrected value. The postedge is modelled as a polynomial of degree n in the given range, and this estimation is extended to the entire data range. The range can be specified in either eV or k.
The command also declares the variable `J0` for each file, defined to be the value of the postedge at $$E = E_{0}$$.

```sh
postedge <range> [--options]
```

|Argument|Explanation|
|--|--|
|<div class="nowrap">`--constant` / `-C`</div>|Models the postedge as a polynomial of order 0 (a constant contribution).|
|<div class="nowrap">`--linear` / `-l`</div>|Models the postedge as a polynomial of order 1 (a linear contribution).|
|<div class="nowrap">`--quadratic` / `-q`</div>|Models the postedge as a polynomial of order 2.|
|<div class="nowrap">`--cubic` / `-c`</div>|Models the postedge as a polynomial of order 3.|
|<div class="nowrap">`--polynomial` / `-p` `<degree>`</div>|Models the postedge as a polynomial of order `degree`.|
|<div class="nowrap">`--divide` / `-d`</div>|Corrects the data by dividing the `x` column by the postedge.|
|<div class="nowrap">`--subtract` / `-s`</div>|Corrects the data by subtracting the postedge from the `x` column.|
|<div class="nowrap">`--energy` / `-e`</div>|Performs the polynomial regression on $$\mu(E)$$, in energy space. If neither this flag nor `--wavevector` is specified, the regression space is inferred from the range.|
|<div class="nowrap">`--wavevector` / `-k`</div>|Performs the polynomial regression on $$\mu(k)$$, in wavevector space If neither this flag nor `--energy` is specified, the regression space is inferred from the range.|

## Background

The `background` command extracts the background contribution from the signal in the `x` column, and stores it in the `bkg` column. This contribution is then subtracted from the signal. The command supports multiple methods for background estimation.

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
|<div class="nowrap">`--value` / `-v` `<value>`</div>|Subtracts the given `value` from the signal. Default is 1.0.|

### BSpline

Models the background using bsplines, automatically determining the optimal smoothing. Uses scipy.interpolate.UnivariateSpline. The range can be specified in eV or k.

```sh
background bspline <range> [--options]
```

|Argument|Explanation|
|--|--|
|<div class="nowrap">`--kweight` / `-k` `<value>`</div>|Performs the operation on data weighed by the specified factor: $$k^{n}\cdot\mu(k)$$. By default, does not weigh the data (equal to a kweight of 0).|

### Fourier

Models the background as low-frequency Fourier contributions. This method performs a continuous Fourier transform onto a limited R-space, and uses a Hann window to select only the lower frequencies, up to Rmax. Then performs the inverse transform to calculate the background.

```sh
background fourier <Rmax> [--options]
```

|Argument|Explanation|
|--|--|
|<div class="nowrap">`--kweight` / `-k` `<value>`</div>|Performs the operation on data weighed by the specified factor: $$k^{n}\cdot\mu(k)$$. By default, does not weigh the data (equal to a kweight of 0).|
|<div class="nowrap">`--iterations` / `-i`</div>|The number of times to iterate the method. By default, applies the background removal three times.|

### Smoothing

Models the background as a smoothed version of the raw data. The method uses the lowess algorithm to smooth the data.

```sh
background smoothing <range> [--options]
```

|Argument|Explanation|
|--|--|
|<div class="nowrap">`--kweight` / `-k` `<value>`</div>|Performs the operation on data weighed by the specified factor: $$k^{n}\cdot\mu(k)$$. By default, does not weigh the data (equal to a kweight of 0).|
|<div class="nowrap">`--iterations` / `-i`</div>|The number of times to iterate the method. By default, applies the background removal once.|
|<div class="nowrap">`--fraction` / `-f`</div>|The fraction of the total datapoints to use for the smoothing. Must be between 0 and 1. By default uses 30% of the datapoints (0.3)|

## Examples

### PreEdge example

```sh
preedge .. -80eV -l
```

This snippet subtracts a linear preedge from the `x` column, selecting the region from $$-\infty$$ to -80eV relative to the $$E_{0}$$ determined in a previous `edgeenergy` step.

### PostEdge example

```sh
postedge 3k .. -cde
```

This snippet divides the data by a cubic polinomial, fitted over $$\mu(E)$$ that models the postedge of the `x` column, selecting the region from 3k to $$\infty$$

### Background example

```sh
background fourier 1.1A -k2
```

This snippet estimates the background from the lower frequencies of the fourier transform, up to 1.1A, with a k-weight of 2. The process is performed three times (the default value). The background contribution is then subtracted from the signal.
