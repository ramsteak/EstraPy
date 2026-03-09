---
title: File Saving
parent: Syntax
nav_order: 70
permalink: /commands/file-saving/
has_children: false
math: katex
---

# File Saving

The `save` command exports processed data to text files.
It supports two modes:

1. **`columns`** - export selected expressions for each page into separate files
2. **`table`** - export one shared axis with one column per page into a single table file

## Basic Usage

```sh
save columns --path <path-template> --columns <expr1> [<expr2> ...] [options]

save table --path <path-template> --axis <axis-expr> --column <data-expr> [options]
```

## Modes Overview

| Mode | Output pattern | Typical use |
|------|----------------|-------------|
| `columns` | One file per page | Save processed columns (e.g. `k`, `chi`, expressions) |
| `table` | One file per target path, with one column per page | Build comparison tables for plotting/analysis outside EstraPy |

## Save columns

Exports one file for each page.
Each file contains the requested expression columns evaluated on that page.

```sh
save columns --path <path-template> --columns <expr1> [<expr2> ...] [options]
```

### Options

| Option | Description |
|--------|-------------|
| `--path <path-template>` <br> `-p <path-template>` | Output path template (required). Supports metadata placeholders like `{.fn}`. |
| `--columns <expr1> [<expr2> ...]` <br> `-c ...` | Expressions to export as columns (required). |
| `--select <expr>` | Boolean expression used to filter rows before writing. |
| `--domain <domain>` <br> `-d <domain>` | Domain used to resolve expressions (default: `reciprocal`). |

### Behavior

- Expressions in `--columns` are evaluated per row.
- `--select` keeps only rows where the expression is `true`.
- One file is generated per page (after path template expansion).
- Header includes EstraPy version, project name, original filename, analysis date, and metadata.

### Example

```sh
save columns --path "processed/{.fn}_chi.dat" --columns k chi --domain reciprocal

save columns --path "processed/{.fn}_windowed.dat" --columns k "chi*k^2" --select "k > 3 and k < 14"
```

## Save table

Exports one table file per resolved path.
Inside each table, first column is the selected axis expression, and additional columns are one data series per page.

```sh
save table --path <path-template> --axis <axis-expr> --column <data-expr> [options]
```

### Options

| Option | Description |
|--------|-------------|
| `--path <path-template>` <br> `-p <path-template>` | Output path template (required). Pages resolving to the same path are grouped in one table. |
| `--axis <expr>` <br> `-a <expr>` | Axis expression (required). Must be identical across grouped pages. |
| `--column <expr>` <br> `-c <expr>` | Data expression (required). One output column per page. |
| `--select <expr>` | Boolean expression used to filter rows before writing. |
| `--domain <domain>` <br> `-d <domain>` | Domain used to resolve expressions. If omitted, inferred from axis expression. |

### Behavior

- Pages with the same resolved `--path` are merged into one output table.
- Column names are page names.
- If axis arrays differ between grouped pages, EstraPy raises an error (`Axis data mismatch ...`).
- Header includes axis/column expressions and list of original files.

### Example

```sh
save table --path "tables/chi_vs_k.dat" --axis k --column chi --domain reciprocal

save table --path "tables/chi_k2.dat" --axis k --column k^2*chi --select 2<=k<=14
```

## Path Templates

`--path` supports metadata placeholders (template replacement), for example:

- `{.fn}` → original file name
- `{sample}` → metadata field named `sample`
- `{.f}_{.g}` → generated page name and group ID

Example:

```sh
save columns --path "out/{sample}_{.fn}_norm.dat" --columns E mu
```

## Notes and Best Practices

- Use explicit expressions in `--columns`, `--axis`, and `--column` to make exported files self-documenting.
- For `save table`, align data first (`cut`, `interpolate`, or `rebin`) to avoid axis mismatch errors.
- Use `--select` to export only the physically relevant interval (e.g. EXAFS region).
- Exports are plain text with fixed-width numeric formatting.

## See Also

- [Data Manipulation]({{ "/commands/data-manipulation" | relative_url }})
- [General syntax]({{ "/commands/general-syntax" | relative_url }})
