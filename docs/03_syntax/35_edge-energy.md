---
title: Edge energy estimation
parent: Syntax
nav_order: 35
permalink: /commands/edge-energy
math: katex
---

# Edge energy estimation

The $$E_0$$ value defines the "threshold energy" where a photoelectron is ejected from the atom, marking the point where the kinetic energy of the electron is effectively zero. Accurate estimation establishes the starting point for the k-space transformation. If $E_0$ is incorrectly assigned, the entire $k$-scale is shifted, which can lead to misinterpretation of the EXAFS oscillations and errors in the derived structural parameters.

The `edge` command estimates the edge energy for each page, by applying a specified method to the spectra. The estimated edge energy is stored in the metadata of each page, and can be used for further analysis or alignment.

Furthermore, the columns `e` and `k` are calculated based on the estimated edge energy, and can be used for plotting or further analysis. The `e` column represents the energy relative to the edge energy, while the `k` column represents the photoelectron wave vector.

## Basic usage

```sh
edge <method> [--options]
```

Estimates the edge energy for each page, using the specified method for $$E_{0}$$ estimation. The method syntax is defined [below](#method).

**Examples:**

```sh
edge set Cu.K
```

Sets the edge energy to the tabulated value for the Cu K edge.

```sh
edge shift Pd.K-40eV Pd.K+40eV --res 10meV --shift 6eV 
```

Estimates the edge energy by minimizing the relative distance between the spectrum and the reference spectrum, in a 80 eV window around the tabulated Pd K edge with a resolution of 10 meV and a shift of 6 eV.

## Command options

The `edge` command supports two methods for edge energy estimation, and each method has its own set of options.

The two methods are defined by two subcommands, namely `set` and `shift`.

### Set method

The set method simply assigns the edge energy to a specified value, which can be either an absolute energy value or a tabulated edge with an optional shift.

```sh
edge set <value>
```

The command takes no additional options, and the edge energy is set to the specified value for all pages.

### Shift method

The shift method estimates the edge energy by minimizing the relative distance, defined as the L2 norm of the difference between the spectrum and a reference spectrum, in a specified energy window.

```sh
edge shift <range> [--options]
```

| Option | Description |
|--------|-------------|
| `<range>` | A pair of energy values that define the window to consider for the edge energy estimation, in eV. The range can be specified as either absolute values (e.g. 2000eV 2100eV) or as a tabulated edge with an optional shift (e.g. `Cu.K-10eV Cu.K+20eV`). By default uses the full spectrum (not recommended). See [range]({{ "/commands/general-syntax#range-specification" | relative_url }}) for details. |
| `--resolution <value>` <br> `--res <value>` | The energy resolution to use for the edge energy estimation, in eV. This option is used to determine the step size for the energy grid when calculating the reference spectrum. By default, it is set to 100meV |
| `--shift <value>` <br> `-s <value>` | The amount to shift the spectrum by, to find the best position for edge energy estimation. By default it is set to 5eV. |
| `--derivative <order>` <br> `-d <order>` | The order of the derivative to apply to the spectrum before calculating the reference spectrum. This can help to enhance features in the spectrum and improve the edge energy estimation. By default, no derivative is applied. |

## Results

After running the `edge` command, the edge energy will be estimated for each page, and the metadata of each page will be updated with the new edge energy value, stored in the `E0` variable. The `e` and `k` columns will also be calculated based on the estimated edge energy, and can be used for plotting or further analysis.

**See also:**

- [Alignment]({{ "/commands/alignment" | relative_url }})
