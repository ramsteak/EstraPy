---
title: Glitch removal
parent: Commands
nav_order: 45
permalink: /commands/glitch-removal
math: katex
---

# Deglitching and Multiple Excitation Removal

In X-ray Absorption Spectroscopy (XAS), glitches—isolated high or low intensity spikes—can arise from detector noise, electronic interference, or artifacts like monochromator diffraction peaks. These artifacts distort the signal and must be removed prior to Fourier transforms or fitting routines.

Additionally, multiple excitations within the scanned energy range (common in some elements) can introduce discontinuities in the data. The deglitch command in EstraPy is designed to correct these issues.

## Deglitch Command

The `deglitch` command handles the removal of spurious datapoints, visible in the acquired signals as very high (or low) intensity regions.

```sh
deglitch <range> [--options] <finder> [--options] <remover> [--options]
```

The command is modular, allowing users to specify:

- a range to apply the glitch removal,
- a glitch finder method to detect outliers,
- a glitch remover method to correct or interpolate affected regions.

Multiple finder and remover options can be combined to create a custom deglitching workflow.
The deglitch command presents a variety of options, allowing to combine the functionality of glitch-finding and glitch-removal methods.

### Main options

|Argument|Explanation|
|--|--|
|<span class="nowrap">`--column` / `-c` `<column>`</span>|Selects which column to process for glitch detection.|

### Finder options

|Finder|Description|Required arguments|
|--|--|--|
|`force`|Flags the entire selected range as glitched, without analysis.|-|
|`variance`|Flags outliers based on standard deviation over subranges.|`--width`, `--pvalue`|
|`smooth`|Applies LOWESS smoothing and flags the deviations.|`--fraction`, `--width`, `--pvalue`|
|`polynomial`|Fits a polynomial, then flags deviations from the fit.|`--degree`, `--width`, `--pvalue`|
|`even`|Compares even vs. odd indexed data via regression, then flags deviations.|`--pvalue`|

All methods flag a value as a glitch if its deviation exceeds the specified statistical threshold (`--pvalue`), relative to the calculated standard deviation.

|Argument|Explanation|
|--|--|
|<span class="nowrap">`--width` / `-w` `<value>`</span>|Number of subranges used to compute the median standard deviation. Default is 10.|
|<span class="nowrap">`--fraction` / `-f` `<value>`</span>|The fraction of points to use for LOWESS smoothing (`smooth`). Default is 0.3.|
|<span class="nowrap">`--degree` / `-d` `<value>`</span>|The degree of the polynomial to use for the `polynomial` finder. Default is 1 (linear).|
|<span class="nowrap">`--pvalue` / `-p` `<value>`</span>|Statistical cutoff (e.g. 0.01 = 99% confidence) for flagging a glitch. Default is 0.001 (99.9% confidence).|

### Removal method options

After glitches are identified, a remover method is applied to correct or eliminate the affected data points.

|Remover|Description|Required arguments|
|--|--|--|
|`remove`|Removes the identified glitch from the data, without replacing it.|-|
|`base`|Replaces glitches using a baseline estimated during the finder phase. Only works when `--column` is set to `x`, and the finder supports baseline estimation.|-|
|`smooth`|Fits the non-glitched data using LOWESS regression, then interpolates over the glitch regions.|`--fraction`|

|Argument|Explanation|
|--|--|
|<span class="nowrap">`--fraction` / `-f` `<value>`</span>|The fraction of points to use for LOWESS smoothing (`smooth`). Default is 0.05.|
|<span class="nowrap">`--noise` / `-n`</span>|Adds Gaussian noise to the interpolated segment to avoid visibly perfect smoothing.|

## Multiple Excitation correction

Within some datasets, the data might contain secondary excitations, which change the shape of the data resulting in artifacts in the Fourier transform and poor fitting.
EstraPy offers a command to manually remove these contributions, which can be modelled as either a simple function or as a shifted version of the data.

```sh
multiedge [--axis] <function> <parameters>
```

|Argument|Explanation|
|--|--|
|`--energy` / `-E`|Calculates the function on the energy axis. This is the default behavior.|
|`--relenergy` / `-e`|Calculates the function on the relative energy axis.|
|`--kvector` / `-k`|Calculates the function on the k-wavevector axis.|

EstraPy offers the following functions:

|Function|Description|
|--|--|
|<span class="nowrap">`atan <b> <a> <c>`|Models the excitation as an arctangent.|
|<span class="nowrap">`erf <b> <a> <c>`|Models the excitation as an error-function.|
|<span class="nowrap">`exp <b> <a> <c>`|Models the excitation as an exponential.|

All the step functions have been modified, so that `a` describes the total height of the curve,
and `c` describes the distance from `b` where the curve evaluates to 10% (or 90%) of `a`.
We can thus identify the following notable points:

|$$x$$|$$\text{atan}(x)$$|$$\text{erf}(x)$$|$$\text{exp}(x)$$|
|:--:|:--:|:--:|:--:|
|$$-\infty$$|$$0$$   |$$0$$   |$$0$$   |
|$$b-c$$    |$$0.1a$$|$$0.1a$$|$$0$$   |
|$$b$$      |$$0.5a$$|$$0.5a$$|$$0$$   |
|$$b+c$$    |$$0.9a$$|$$0.9a$$|$$0.9a$$|
|$$\infty$$ |$$a$$   |$$a$$   |$$a$$   |

The exact mathematical description of each function is described below.

|Function|Mathematical expression|
|--|:--:|
|`atan`|$$\dfrac{a}{\pi}\cdot\arctan{(\tan(0.4\pi)\cdot\dfrac{x-b}{c}) + 0.5a}$$|
|`erf`|$$\dfrac{a}{2}\cdot(1+\text{erf}{(\text{erf}^{-1}{(0.8)\cdot\dfrac{x-b}{c}})})$$|
|`exp`|$$a\cdot(1 - \exp(\log(0.1)\cdot\frac{x-b}{c}))$$|
