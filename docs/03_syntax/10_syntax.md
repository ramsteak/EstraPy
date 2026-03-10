---
title: General Syntax
parent: Syntax
nav_order: 10
permalink: /commands/general-syntax/
math: katex
---

# General Syntax

All EstraPy input files (`.estra`) follow a consistent syntax. The file is parsed before execution, and any syntax errors are reported immediately.

## File Structure

An input file consists of four sections, in order:

1. **Version declaration** (required)
2. **Directives** (optional)
3. **Comments** (optional, anywhere)
4. **Commands** (main content)

---

## Version Declaration

**Every input file must start with a version declaration:**

```sh
# version: {{ site.version }}
```

This ensures compatibility between your input file and the installed EstraPy version:

- If the file requires a **newer version**, EstraPy will prompt you to update
- If the file uses an **older version**, EstraPy will attempt to upgrade the syntax automatically

---

## Directives

Directives are special instructions processed at parse time (before execution). They control the working environment and are prefixed with `%`.

Directives must appear after the version declaration and before any commands.

### Clear

```sh
% clear
```

Clears the output folder before execution, removing all files and directories from previous runs. This ensures:

- No confusion from leftover output files
- A clean slate for each analysis

**Example:**

```sh
# version: {{ site.version }}
% clear

filein data/scan_001.dat
<...>
```

### Archive

```sh
% archive
```

Creates a compressed `.zip` archive of all output files at the end of the analysis run. The archive includes the entire output folder, containing:

- All generated plots
- Exported data files
- Log files
- Any other outputs created during the run

as well as a copy of the input file and any other accessed files, such as fitting models.

The archive name is generated with the input filename, the project [`title`](#title) and the current time.

**Example:**

`analysis.estra`

```sh
# version: {{ site.version }}
% clear
% archive
% title "Cu foil experiment"

filein data/scan_001.dat --energy E --intensities I0 I1 I2

plot E:mu

save results.csv --columns E a
# Creates: analysis_Cu_foil_experiment_YYYYMMDD_HHMMSS.zip
```

**Note:** Using `% archive` automatically implies `% clear` to ensure only files from the current run are archived.

### Title

```sh
% title "<project name>"
```

Sets a descriptive title for the project. This title is used to name the archive file when `% archive` is enabled.

<!-- ### Define

```
% define <variable> <value>
```

**Note:** This directive is currently not functional in version 2.0.0 and is reserved for future use.

Creates a variable that will be substituted throughout the file wherever `${variable}` or `%variable%` appears.

**Planned use cases:**
- Store commonly used values (k-weights, normalization factors)
- Use in both commands and plot labels
- Make files easier to maintain and modify

**Example (future):**
```
# version: {{ site.version }}
% define kweight 2
% define sample "Cu foil"

preedge --kweight ${kweight}
plot --title "${sample} - k^${kweight} weighted"
```
-->

---

## Comments

Lines beginning with `#` are treated as comments and ignored during parsing:

```sh
# This is a comment
filein data/scan_001.dat  # Comments can also appear after commands

# Comments are useful for:
# - Explaining complex operations
# - Documenting parameter choices
# - Temporarily disabling commands
```

---

## Commands

Commands are the main content of the file and perform all data processing operations.

### Basic Syntax

Commands follow shell-command syntax:

```sh
command_name [arguments] [--option value]
```

**Example:**

```sh
preedge .. -80eV --linear
```

### Multi-line Commands

Commands can span multiple lines for readability. **All continuation lines must start with at least one space:**

```sh
plot --fig 1:1.2
     --xlabel "$Wavevector\ [Å^{-1}]$"
     --ylabel "$Signal\ intensity$"
     --title  "$\chi^{exp}(k)$"
```

### Multi-line Rules

❌ **Incorrect** - continuation line doesn't start with space:

```sh
plot --fig 1:1.2
--xlabel "$Wavevector\ [Å^{-1}]$"
```

❌ **Incorrect** - different number of spaces on continuation lines:

```sh
plot --fig 1:1.2
    --xlabel "$Wavevector\ [Å^{-1}]$"
     --ylabel "$Signal\ intensity$"
      --title  "$\chi^{exp}(k)$"
```

✅ **Correct** - all continuation lines start with the same number of spaces:

```sh
plot --fig 1:1.2
    --xlabel "$Wavevector\ [Å^{-1}]$"
    --ylabel "$Signal\ intensity$"
    --title  "$\chi^{exp}(k)$"
```

---

## Numbers and Units

Many commands accept numeric arguments with units. EstraPy supports automatic unit conversion and axis inference.

### Supported Units

| Number | Meaning | Typical use |
|------|---------|-------------|
| `10eV` | electronvolts (eV) | Energy axis |
| `-10eV` \| `+10eV` | relative electronvolts (eV) | Relative energy axis |
| `2k` | reciprocal distance (Å⁻¹) | Wavevector |
| `5.3A` | distance (Å) | Distance (Fourier domain) |

**SI prefixes are supported:** `15keV` = `15000eV`

### Range Specification

Ranges can combine explicit numbers with special aliases:

| Syntax | Meaning |
|--------|---------|
| `.. ..` | $$-\infty$$ to $$\infty$$ (full range) |
| `.. 10eV` | $$-\infty$$ to 10 eV |
| `-10eV ..` | -10 eV to $$\infty$$ |
| `3k 12k` | 3 Å⁻¹ to 12 Å⁻¹ |

**Examples:**

```sh
# Remove preedge from the beginning up to 80 eV below edge
preedge .. -80eV --linear

# Fourier transform over full k range
fourier .. .. <...>
```

### Automatic Axis Inference

When you specify a unit, EstraPy tries to automatically determine which axis to use:

```sh
# These commands infer the correct axis from the units:
rebin 0k 15k --interval 0.05k      # Uses k axis (wavevector)
```

To do so, all values must be of the same type (e.g. all in eV or all in k). Mixing units will cause an error, and EstraPy will prompt you to specify the axis explicitly:

```sh
# This will cause an error due to mixed units:
rebin 0k 1500eV --interval 0.05k

# ✅ Correct - explicitly specify axis
rebin 0k 1500eV --interval 0.05k --axis k  
```

---

## Column References

When referencing columns in commands, **prefer using the base name** without version numbers:

```sh
plot k:chi    # ✅ Correct - uses latest chi_N and k_N
plot k:chi_0  # ❌ Avoid - hardcodes specific version.
```

EstraPy automatically resolves base names to their latest versions.
Using base names allows you to plot older versions of columns, to show the evolution of the data processing steps.
Note that destructive commands, such as `rebin` and `cut`, will modify the entire history, so even old versions of columns will reflect the changes.

### Columns in Different Domains

Some commands can operate on any domain, depending on the columns you specify:

```sh
# These work in reciprocal domain (E, k axes):
plot E:mu
plot k:chi

# These work in fourier domain (R axis):
plot r:abs(f)
plot r:real(f)
```

---

<!--
## TODO:
Complete Example
---
-->
