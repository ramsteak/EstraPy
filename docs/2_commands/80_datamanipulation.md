---
title: Data manipulations
parent: Commands
nav_order: 80
permalink: /commands/data-manipulation
math: katex
---

# Data manipulation

EstraPy provides core data manipulation commands to prepare and reshape XAS datasets before analysis. This page covers three essential operations: cutting, rebinning, and smoothing.

## Cut

The `cut` command restricts the dataset to a given range, discarding all values outside the specified bounds.
It modifies the entire data table, removing the values outside the specified range.
See [Number and unit specification]({{ "/commands/general-syntax#number-and-unit-specification" | relative_url }}) for the range syntax explanation.

```sh
cut <range>
```

The bounds may be specified in any valid unit. EstraPy automatically infers the domain over which to cut (real domain: eV, k, fourier domain: A). If the unit is not specified, the default domain is assumed to be the energy axis (`E`, eV).

### Example

```sh
cut 5000eV 8000eV
```

This cuts the dataset to include only data between 5000eV and 8000eV.

---

## Smooth

The `smooth` command applies LOWESS smoothing to a selected column in the dataset. This helps reduce noise while preserving trends in the signal.

```sh
smooth [column] [axis] [--options]
```

|Argument|Explanation|
|--|--|
|<span class="nowrap">`[column]`</span>|Column to smooth. Default is `x`.|
|<span class="nowrap">`[axis]`</span>|Axis to use as the x-coordinate for the smoothing. Default is `E`.|
|<span class="nowrap">`--window` / `-w` `<value>`</span>|Width of the window (in number of points) to use for the LOWESS smoothing.|

### Example

```sh
smooth x E --window 15
```

Smooths the `x` column over the energy axis using a 15-point LOWESS window.

---

## Rebin

The `rebin` command redistributes and averages data across a new set of bins with specified interval or number of points. This is useful for uniform sampling or to match data resolution for comparative analysis.
The command only averages the data, and does not create new datapoints. Bins that do not refer to any datapoints are discarded.
The commands modifies the entire table in-place.

```sh
rebin <range> <interval>
rebin <range> --number <points>
```

You can specify the range explicitly with units, and either provide a fixed interval or specify the number of desired points.

|Argument|Explanation|
|--|--|
|<span class="nowrap">`<range>` </span>|The range to rebin within. See [Number and unit specification]({{ "/commands/general-syntax#number-and-unit-specification" | relative_url }}) for the range syntax explanation.|
|<span class="nowrap">`<interval>`</span>|The spacing between output points. If the interval unit is n (e.g. 15n), is equivalent to `--number 15`|
|<span class="nowrap">`--number` / `-n` `<value>`</span>|Number of output points to generate. Cannot be use with `interval`.|

### Examples

```sh
rebin 5000eV 8000eV 0.5eV
```

Rebins the dataset between 5000 and 8000 eV with a spacing of 0.5 eV.
