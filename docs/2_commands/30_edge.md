---
title: Edge detection
parent: Commands
nav_order: 3
permalink: /commands/edge-detection
math: katex
---

# Edge detection

$$E_{0}$$ estimation is a crucial step in XAS data analysis. The commands `align` and `edgeenergy` perform this estimation. The two commands share the same estimation logic, but the former is geared towards spectra alignment, while the second solely calculates the $$E_{0}$$ for each file.

## Align

The `align` command calculates the $$E_{0}$$ on the `ref` column, representing $$\mu_{ref}(E)$$, and shifts each spectrum in order to align the file to the correct $$E_{0}$$ value for the selected element.

```sh
align <method> [--options]
```

|Argument|Explanation|
|--|--|
|`method`|Defines the method used in the $$E_{0}$$ estimation. The method syntax is defined [below](#method)|
|`--E0` / `-E` `<energy>`|The tabulated $$E_{0}$$ value for the analyzed edge. The energy can be specified either by the value, such as `14652eV`, or as an element edge, such as `Pd.K`. If needed, the tabulated edge can be shifted, e.g. `Pd.K+1.2eV`|
|`--dE0` / `-d` `<value>`|The range around which to search for the $$E_{0}$$.|
|`--search` / `-s` `<energy>`|The energy around which to search for the $$E_{0}$$. If not specified, searches around `--E0`. The energy value follows the same syntax as `--E0`.|

## EdgeEnergy

The `edgeenergy` command calculates the $$E_{0}$$ on the `x` column, that is on $$\mu_{exp}(E)$$. Afterwards, the command also calculates the indices `e`, representing the relative energy from the determined $$E_{0}$$ value, and `k`, representing the wavevector. The `k` column also contains negative values, as this column is calculated with a symmetrical square root function. This is only for mathematical convenience.

```sh
edgeenergy <method> [--options]
```

|Argument|Explanation|
|--|--|
|`method`|Defines the method used in the $$E_{0}$$ estimation. The method syntax is defined [below](#method)|
|`--E0` / `-E` `<energy>`|The tabulated $$E_{0}$$ value for the analyzed edge. The energy can be specified either by the value, such as `14652eV`, or as an element edge, such as `Pd.K`. If needed, the tabulated edge can be shifted, e.g. `Pd.K+1.2eV`|
|`--dE0` / `-d` `<value>`|The range around which to search for the $$E_{0}$$.|

## Method

Methods for both commands can be constructed from a series of operations, defined in the table below. The operations must end with a terminal method, which estimates a single number.

|Operation|n|Explanation|
|--|--|--|
|`c`|&#10004;|Cuts the current data range to the range defined as `E0`-`dE0`~`E0`+`dE0`. If `n` is specified, expands the range by `n`$$\cdot$$`dE0` on both sides to provide a buffer for other operations.|
|`p`|&#10004;|Performs polynomial regression of order `n` on the current data.|
|`s`|&#10004;|Smooths the data, with a window of size `n`.|
|`d`|&#10004;|Calculates the `n`-th derivative of the data.|
|`i`|&#10004;|Interpolates the data with a spline of order `n`.|
|`M`|&#10008;|Terminal method, calculates the maximum of the current data.|
|`m`|&#10008;|Terminal method, calculates the minimum of the current data.|
|`Z`|&#10008;|Terminal method, calculates the zero of the current data.|
|`S`|&#10008;|Terminal method, bypasses all instructions and sets the E0 value to the given `E0`.|

Alternatively, the method can be one of the following options, which are simply aliases for some common methods.

|Alias|Operations|
|--|--|
|`set`|`S`|
|`fitderivative`|`c1.s5.d1.p3.M`|
|`fitpolynomial`|`c1.p3.d1.M`|
|`fitmaximum`|`c1.p3.M`|
|`interpderivative`|`c1.s5.d1.i3.M`|
|`maximum`|`c1.i3.M`|
