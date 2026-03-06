---
title: Spectra alignment
parent: Syntax
nav_order: 30
permalink: /commands/alignment
math: katex
---

# Spectra alignment

Reference edge alignment is a fundamental preprocessing step used to correct for energy shifts caused by instabilities in the beamline optics or monochromator calibration. By aligning the absorption edge to a known reference value, we can ensure that the spectra are accurately calibrated in energy, making it possible to compare spectra from different measurements or samples.

The alignment process is performed with the `align` command, which estimates the reference shift, relative to the `ref` column, and shifts the pages to align to the tabulated reference edge.

## Basic usage

```sh
align <method> [--options]
```

Aligns all pages to the given reference edge, using the specified method for $$E_{0}$$ estimation. The method syntax is defined [below](#method).

**Examples:**

```sh
align shift Cu.K-10eV Cu.K+20eV --res 10.meV --shift 6eV -E Cu.K
```

Aligns all pages by minimizing the relative distance across spectra, as compared to the average of the spectra, in a 30 eV window around the tabulated Cu K edge.

## Command options

The `align` command supports two methods for edge detection, and each method has its own set of options.

The two methods are defined by two subcommands, namely `calc` and `shift`

### Shift method

The shift method estimates the reference shift by minimizing the relative distance across spectra, defined as the L2 norm of the difference between each spectrum and the average of all spectra in a specified energy window.
This method is particularly useful when the spectra are expected to be similar, and it can effectively align the spectra by finding the optimal shift that minimizes the overall difference.

```sh
align shift <range> [--options]
```

| Option | Description |
|--------|-------------|
| `<range>` | A pair of energy values that define the window to consider for the alignment, in eV. The range can be specified as either absolute values (e.g. 2000eV 2100eV) or as a tabulated edge with an optional shift (e.g. `Cu.K-10eV Cu.K+20eV`). By default uses the full spectrum (not recommended). See [range]({{ "/commands/general-syntax#range-specification" | relative_url }}) for details. |
| `--resolution <value>` <br> `--res <value>` | The energy resolution to use for the alignment, in eV. This option is used to determine the step size for the energy grid when calculating the average spectrum. By default, it is set to 100meV |
| `--shift <value>` <br> `-s <value>` | The amount to shift the spectra by, to find the best position for alignment. By default it is set to 5eV. |
| `--derivative <order>` <br> `-d <order>` | The order of the derivative to apply to the spectra before calculating the average spectrum. This can help to enhance features in the spectra and improve the alignment. By default, no derivative is applied. |
| `--energy <value>` <br> `--E0 <value>` <br> `-E <value>` | The reference edge energy to set in the metadata after alignment. Note that the `shift` method does not align to a fixed position, but ensures that the spectra are aligned relative to each other. |

## Results

After running the `align` command, the spectra will be shifted to align to the reference edge. The metadata of each page will be updated with the new reference edge energy, which can be specified with the `--energy` option.

The alignment result is stored as a plottable result, which can be visualized using the `plot result` command (see [here]{{ "/commands/plotting" | relative_url }})

### Histogram plot

```sh
plot result align.histogram
```

Plots a histogram of the shifts applied to the spectra, which can help to visualize the distribution of shifts and identify any outliers.

### Shifts plot

```sh
plot result align.shifts
```

Generates a scatter plot of the applied shifts for each spectrum, with the shift as the y-axis and the spectrum index as the x-axis. This can help to identify any trends or patterns in the shifts across the spectra.

### L2Norms plot

```sh
plot result align.L2Norms
```

Plots the value of the L2 norms for each spectrum, which can help to evaluate the quality of the alignment and identify any spectra that may still be misaligned.

### Spectra plot

```sh
plot result align.spectra
```

Plots the spectra before and after alignment, with the original spectra in blue and the aligned spectra in orange. This can help to visually assess the effectiveness of the alignment process.

### Plot

```sh
plot result align
```

Plots all the above plots in a single figure, allowing for a comprehensive evaluation of the alignment results.

**See also:**

- [Edge energy estimation]({{ "/commands/edge-energy" | relative_url }})
