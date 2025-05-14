---
title: Fourier transform
parent: Commands
nav_order: 50
permalink: /commands/fourier-transform
math: katex
---

# Fourier transform

In order to interpret EXAFS spectra, it is useful to analyze the fourier-transform of the data. The `fourier` command calculates the transform, adding the calculated result to the fourier domain of the file.

The command requires the input and output ranges of the data. The input range is taken from the real domain, and can be expressed in eV or k, while the output range is in the fourier domain, thus can only be expressed in Angstrom.
While the input range is expressed as `(start) (end)`, the output Fourier range is expressed as `(end) (spacing)`, as the beginning of the Fourier range is by default set to 0.

The command adds a `win` column, containing the window function, to the real domain, and creates the fourier domain containing the `R` index column and the `f` complex valued column.

```sh
fourier <range> <output> [--options]
```

|Argument|Explanation|
|--|--|
|`<range>`|The range considered to perform the fourier transform. See [Number and unit specification]({{ "/commands/general-syntax#number-and-unit-specification" | relative_url }}) for the range syntax explanation.|
|<span class="nowrap">`--kweight` / `-k` `<value>`</span>|Performs the operation on data weighed by the specified factor: $$k^{n}\cdot\chi(k)$$. By default, does not weigh the data (equal to a kweight of 0).|
|<span class="nowrap">`--apodizer` / `-a` `<apodizer>`</span>|The apodizing window shape. By default, uses a Hann window. The available windows are listed in the [related section](#apodizer)|
|<span class="nowrap">`--width` / `-w` `<value>`</span>|The width of the ramps of the window. See the [related section](#apodizer) for a visual explanation of this parameter.|
|<span class="nowrap">`--method` `<method>`</span>|<div><p>Uses a different method to perform the fourier transform.</p><ul><li>`dft` is the default method, and uses matrix multiplication to calculate the transform</li><li>`finuft` uses the finufft.nufft1d3 method</li><li>`fft` uses the numpy rfft method, which requires the data to be uniformly spaced.</li></ul></div>|

## Apodizer

Each apodizing window is defined by its first three letters. Some apodizers require an additional parameter, such as the exponential window and the gaussian window. This parameter is included in the window definition, separated by `:`.

|Window|Description|
|--|--|
|`han`|Hann window|
|`sin`|Sine window|
|`rec`|Rectangular window|
|`bar`|Bartlett window, also called triangular window.|
|`wel`|Welch window|
|`exp:p`|Exponential window. The parameter modifies the window as $$\exp(- p\cdot t)$$|
|`gau:p`|Gaussian window. The parameter modifies the window as $$\exp(-p \cdot x^2)$$|

## Example

```sh
fourier 3k 12k 6A 0.01A -a hann -k2 -w1
```

This snippet calculates the fourier transform of the $$k^2$$-weighed data between 3k and 12k, from 0A to 6A with a spacing of 0.01A using a hann window of width 1k.

## Phase manipulation

To perform manipulations on the phase of the fourier-transform data, the `phase` command can be used. This command does not modify the intensity of the fourier data, but only its phase.

### Phase alignment

```sh
phase align
```

This command aligns all the spectra to minimize the distance to the first spectrum. The result is that, in the complex space, all spectra get rotate to align themselves to the first spectrum.

### Phase correction

```sh
phase correct
```

This command estimates a linear contribution from each spectrum, and corrects it. The removed contribution is set to the column `pcorr`. This operation is analoguous to NMR phase correction.
