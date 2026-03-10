---
title: Glitch Removal
parent: Data cleaning
grand_parent: Syntax
nav_order: 1
permalink: /commands/glitch-removal/
math: katex
---

# Glitch Removal

The `deglitch` command detects and corrects local artifacts (glitches) in reciprocal-domain data.
It is composed of two parts:

1. a **finder** subcommand, used to flag glitch points
2. a **remover** subcommand, used to apply the correction

## Basic Usage

```sh
deglitch [range] <finder ...> <remover ...>
```

where:

- `[range]` is an optional range of two numbers
- `<finder ...>` chooses how glitches are detected
- `<remover ...>` chooses how flagged points are handled

If `[range]` is omitted, the command operates on the full available axis range.

## Command Options

| Option | Description |
|--------|-------------|
| `[range]` | Optional two-value range where detection is performed. See [range]({{ "/commands/general-syntax#range-specification" | relative_url }}) for syntax details. |
| `--kweight <value>` <br> `-k <value>` | Non-negative k-weight parameter (default: `0.0`). |

## Finder Subcommands

The finder decides which points are marked as glitches.

| Finder | Description |
|--------|-------------|
| `force` | Selects the whole given region as glitch. |
| `polynomial` | Fits a polynomial in the selected region, then flags points whose residual is above a threshold from `--pvalue`. |
| `point <position>` | Flags a single point within the selected region (closest sample to `position`). |
| `even` | Estimates noise from even/odd differences and flags points above the noise threshold. |

### Finder-specific options

#### polynomial

```sh
deglitch [range] polynomial [options] <remover ...>
```

| Option | Description |
|--------|-------------|
| `--degree <n>` <br> `-d <n>` | Polynomial degree (required). |
| `--constant` <br> `-C` | Degree `0`. |
| `--linear` <br> `-l` | Degree `1`. |
| `--quadratic` <br> `-q` | Degree `2`. |
| `--cubic` <br> `-c` | Degree `3`. |
| `--pvalue <value>` <br> `-p <value>` | Statistical threshold for glitch detection (default: `0.0002`). |
| `--axis <axis>` | Axis used for fitting (auto-inferred if omitted). |
| `--column <col>` <br> `--col <col>` | Data column used for fitting (default: `I0`). |

#### point

```sh
deglitch [range] point <position> <remover ...>
```

`<position>` accepts `eV` or `k` units and flags the closest sample point.

This finder can be used to remove a known bad point. The range can be omitted, as the position is sufficient to identify the target point.

```sh
deglitch point 8333eV remove
```

#### even

```sh
deglitch [range] even [options] <remover ...>
```

| Option | Description |
|--------|-------------|
| `--column <col>` <br> `--col <col>` | Data column used to estimate noise (default: `I0`). |
| `--pvalue <value>` <br> `-p <value>` | Statistical threshold for glitch detection (default: `0.0002`). |
| `--median-window <n>` | Median window size control (default: `1`). |

## Remover Subcommands

The remover applies the correction to points flagged by the finder.

| Remover | Description |
|---------|-------------|
| `remove` | Removes all flagged points from the dataset. |
| `interpolate` | Replaces flagged points by interpolation over neighboring non-glitch points. |

## Examples

```sh
# Force-remove all points in a narrow region
deglitch 8.2k 8.4k force remove

# Polynomial finder with interpolation
deglitch -50eV 250eV polynomial --linear --pvalue 0.001 interpolate

# Remove a single known bad point
deglitch point 8333eV remove

# Even/odd noise-based detection with interpolation
deglitch even --pvalue 0.0005 --median-window 2 interpolate
```

## Notes

- `remove` is irreversible because flagged rows are deleted.
- `interpolate` preserves row count and updates data columns with interpolated values.
- If a finder flags too many points, revise the range and threshold before applying removal.

**See also:**

- [Data cleaning overview]({{ "/commands/data-cleaning" | relative_url }})
- [General syntax]({{ "/commands/general-syntax" | relative_url }})
