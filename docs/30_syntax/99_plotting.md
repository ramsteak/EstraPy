---
title: Plotting
parent: Syntax
nav_order: 99
permalink: /commands/plotting/
math: katex
---

# Plotting

The `plot` command draws data in two ways:

1. **Expression pair mode**: plot `x:y` expressions such as `E:mu` or `k:k*chi`
2. **Result mode**: plot a result object produced by a previous command (currently implemented for `align`)

## Basic Usage

```sh
# 1) Expression pair mode
plot <x_expr:y_expr> [options]

# 2) Result mode
plot result <result_name>[.<plot_kind>] [options]
```

## Expression Pair Mode (`x:y`)

When `kind` contains a colon, EstraPy parses both sides as expressions and plots them for each page.

```sh
plot E:mu
plot k:k*chi
```

### Supported expressions

- Column names: `E`, `k`, `mu`, `chi`, ...
- Arithmetic expressions: `k*chi`, `chi*k^2`, `E-E0`, ...
- Expressions are evaluated per page using available columns/metadata.

### Examples

```sh
# Basic spectra
plot E:mu

# k-weighted EXAFS
plot k:k*chi

# Restrict view and add labels
plot k:k*chi --xlim 2 14 --xlabel "k [Å⁻¹]" --ylabel "k·χ(k)"

# Color by metadata variable
plot E:mu --colorby temperature

# Place on figure/axis and style
plot E:mu --figure 1.1 --dashed --linewidth 2 --marker o
```

## Result Mode (`plot result ...`)

Result mode plots callback outputs stored in `context.results` by previous commands.

```sh
plot result <result_name>
plot result <result_name>.<plot_kind>
```

- `<result_name>` is usually the command name (or its output alias, if set).
- `<plot_kind>` selects one of the `plot_*` callbacks exposed by that result.

## Common Plot Options

The command supports figure placement, labels, limits, grid, and style options.

### Figure and layout

| Option | Description |
|--------|-------------|
| `--figure`, `--fig`, `--ax` | Target figure/axis position. If omitted, creates a new non-numbered figure. |
| `--figsize <W>x<H>` | Figure size in inches, e.g. `--figsize 8x6`. |

### Axis labels and limits

| Option | Description |
|--------|-------------|
| `--xlabel <text>` | Set x-axis label |
| `--ylabel <text>` | Set y-axis label |
| `--title <text>` | Set axis title |
| `--suptitle <text>` | Set figure title |
| `--xlim <min> <max>` | Set x-axis limits |
| `--ylim <min> <max>` | Set y-axis limits |

### Grid and legend

| Option | Description |
|--------|-------------|
| `--grid [true\|false]` | Enable/disable full grid |
| `--xgrid [true\|false]` | Enable/disable x grid |
| `--ygrid [true\|false]` | Enable/disable y grid |
| `--legend [true\|false]` | Enable/disable legend |
| `--legendname <text>` | Legend title |

### Styling and color

| Option | Description |
|--------|-------------|
| `--colorby <expr-or-template>` | Variable used to color traces (categorical or continuous) |
| `--color <cmap-or-color-spec>` | Colormap or color spec |
| `--alpha <value>` | Transparency |
| `--linestyle` / `--solid` / `--dashed` / `--dotted` / `--dashdot` / `--noline` | Line style |
| `--linewidth <value>` | Line width |
| `--marker <value>` | Marker style |
| `--markersize <value>` | Marker size |
| `--markeredgecolor <color>` | Marker edge color |
| `--markerfacecolor <color>` | Marker face color |

## Notes

- `plot` without a `kind` can still be used to apply axis/figure settings, but data-style options require a plot kind.
- Expression mode plots one trace per page.
- Result mode depends on callback methods provided by the producing command result.

**See also:**

- [Alignment]({{ "/commands/alignment" | relative_url }})
- [General syntax]({{ "/commands/general-syntax" | relative_url }})
