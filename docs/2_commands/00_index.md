---
title: Commands
nav_order: 3
permalink: /commands/
---

# Commands

EstraPy reads instructions from an input file, which defines the operations to perform.
These operations include loading experimental data, applying transformations, generating plots, and exporting results.
Each instruction is written as a *command*, which follows a simple and consistent syntax.

Commands are executed in the order they appear in the file, and many support multiple options for customization.
The syntax and usage of each command are detailed in the sections below.

When a column is modified it is first copied, with a numbered suffix in ascending order, in order to not lose any data. For example, the `align` command modifies the `E` axis of the data, first copying the old values into `E_0`.

Some commands, such as the `fourier` command, transform the data into a space that is entirely different from the original. This is represented in EstraPy as a different *domain*. Two domains exist in EstraPy, the *real* domain, where the acquired signal exists, and the *fourier* domain, where the result of the fourier transform exists.

## Data storage

EstraPy stores processed data in tables, where each column is either an *index* or *data* column.  
Each column is uniquely identified by its name.

Commands can read, create, and modify these columns depending on their purpose.  
When a column is modified, EstraPy automatically copies the original version using a numbered suffix (e.g. `a_0`, `a_1`, etc.) to prevent data loss.

For example, the `align` command modifies the `E` axis of the data, but first backs up the original values as `E_0`.

### Domains

Some commands, such as [`fourier`]({{ "/commands/fourier-transform" | relative_url }}), operate in a space fundamentally different from the original signal.

EstraPy represents this using the concept of *domains*. Currently, two domains are supported:

- **real**: The default domain where the original, acquired signal resides, with energy (`E`, eV) and wavevector (`k`, Å⁻¹) axes.
- **fourier**: The distance domain, produced by applying a Fourier transform, with distance (`R`, Å) axis.

Most commands apply only to data in the real domain, but some explicitly operate on or generate data in the Fourier domain. Other commands, such as `rebin` and `plot`, can operate on any domain depending on the specified range and/or columns.
