---
title: Syntax
nav_order: 4
permalink: /commands/
has_children: true
---

# Syntax

EstraPy reads instructions from an input file, which defines the operations to perform on synchrotron data.
These operations include loading experimental data, applying transformations, generating plots, and exporting results.

Commands are executed sequentially in the order they appear in the file, and many support multiple options for customization.

## How Commands Work

Each command follows a simple, shell-like syntax:

```sh
command-name [arguments] [options] [subcommands]
```

Commands can span multiple lines for readability - continuation lines must start with at least one space.

For example:

```sh
preedge -150eV -25eV --linear

edge set Cu.K
```

## Data Structure

EstraPy organizes data into **pages** and **domains**:

- **Pages** - Represent individual data files or averaged results
- **Domains** - Different data spaces (reciprocal, fourier, discriminant analysis)
- **Columns** - Individual data arrays within a domain

### Column Versioning

When a command modifies a column, EstraPy automatically preserves the original by creating a numbered backup:

```txt
E_0  →  (after modification)  →  E_0, E_1
a_0  →  (after modification)  →  a_0, a_1
```

**You don't need to worry about version numbers** - commands automatically use the latest version when you reference a column by its base name (e.g., `E` always refers to the most recent `E_N`).

This ensures that most commands **do not lose data** during processing.
Some commands, like `cut`, `average`, `rebin` and `interpolate` need to modify the length of the data and the original data is lost.

### Domains

EstraPy currently supports two domains:

| Domain | Description | Axes |
|--------|-------------|------|
| **reciprocal** | Original signal space | Energy (`E`, eV), wavevector (`k`, Å⁻¹), absorption (`mu`), fine structure (`chi`), ... |
| **fourier** | Distance space after Fourier transform | Distance (`R`, Å), Fourier transform magnitude (`f`), ... |

Most commands operate in the reciprocal domain by default. Commands like [`fourier`]({{ "/commands/fourier-transform" | relative_url }}) explicitly transform data between the reciprocal and fourier domains.

### Column Types

Each column is classified as either:

- **Axis** - Independent variable (e.g., `E`, `k`, `R`)
- **Data** - Dependent variable (e.g., `mu`, `chi`, `f`)
- **Noise** - Uncertainty or noise estimates (starting with `s`, e.g., `schi`)

Some commands require you to specify which domain, axis, and columns to use. Others (like `preedge`) have sensible defaults and work with the only logical choices (`reciprocal` domain, `E` axis, `a` column).

## Getting Started

For practical examples, see the [Tutorials]({{ "/tutorials" | relative_url }}) section.

---

**Next:** [General Syntax]({{ "/commands/general-syntax" | relative_url }}) - Learn the input file structure and syntax rules
