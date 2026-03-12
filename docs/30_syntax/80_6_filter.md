---
title: Filter
parent: Data Manipulation
grand_parent: Syntax
nav_order: 6
permalink: /commands/data-manipulation/filter/
math: katex
---

# Filter

The `filter` command removes pages that do not satisfy a dataset-level condition. It is useful for keeping only scans that fully cover a required axis interval. This is a **destructive operation** — pages that fail the condition are removed from the working dataset.

(Note that the original data files remain unchanged; only the in-memory dataset is modified.)

## Basic Usage

```sh
filter enoughaxis <range> [options]
```

where:

- `enoughaxis` checks whether each page axis spans the full requested range
- `<range>` is a two-value range that must be fully covered by each page

## Command Options

| Option | Description |
|--------|-------------|
| `enoughaxis` | Subcommand that keeps only pages whose axis range fully covers `<range>`. |
| `<range>` | Required range to validate against each page axis. See [range specification](/commands/general-syntax#range-specification) for details. |
| `--axis <axis>` | Axis column to test (auto-inferred from range units if omitted). |
| `--domain <domain>` | Domain in which to evaluate axis coverage (auto-inferred if omitted). Options: `reciprocal`, `fourier`. |

## Coverage Rule (`enoughaxis`)

For each page, EstraPy computes axis limits:

- `axmin = min(axis)`
- `axmax = max(axis)`

The page is **kept** only if:

$$
axmin \le range_{start} \quad \text{and} \quad axmax \ge range_{end}
$$

If either condition fails, the page is removed.

## Range and Unit Inference

If `--axis` and `--domain` are omitted, EstraPy infers them from range units:

- `eV` or `k` units → reciprocal domain
- `Å` units → Fourier domain
- Axis is selected from unit type (`E`, `k`, or `r`)

## Examples

```sh
# Keep only pages with full coverage from 8000 to 9000 eV
filter enoughaxis 8000eV 9000eV

# Keep only pages covering EXAFS k-range 3 to 14
filter enoughaxis 3k 14k

# Keep only pages covering R-space interval 1.5 to 4.5 Å
filter enoughaxis 1.5A 4.5A

# Explicit axis/domain
filter enoughaxis 0 1000 --axis E --domain reciprocal
```

## Behavior

The command:

1. Evaluates each page independently
2. Checks axis minimum and maximum against the requested range
3. Removes pages that do not fully cover the interval
4. Keeps only pages with sufficient axis extent

{: .warning }
**Destructive operation:** Pages that fail the filter are removed from the workspace dataset and cannot be recovered without reloading data.
The original files remain unchanged; only the in-memory dataset is modified.

## Use Cases

- **Pre-screening for batch processing:** Remove partial scans before normalization, fitting, or averaging
- **Coverage consistency:** Ensure all remaining pages support the same axis window
- **Quality control:** Exclude truncated scans from downstream analysis
- **Domain-specific workflows:** Enforce valid ranges in reciprocal or Fourier space

## Tips and Best Practices

1. **Filter before averaging:** Use `filter enoughaxis` before `average` so groups do not mix full and truncated scans

2. **Specify range in physical units:** Prefer `eV`, `k`, or `Å` values so axis/domain inference is explicit and robust

3. **Use explicit axis/domain when needed:** If your dataset uses custom axis names, pass `--axis` and `--domain` directly

4. **Validate expected data loss:** If many pages are removed, reduce the requested range or inspect raw scan limits first

**See also:**

- [Data Manipulation overview]({{ "/commands/data-manipulation" | relative_url }})
- [Average]({{ "/commands/data-manipulation/average" | relative_url }})
- [Cut]({{ "/commands/data-manipulation/cut" | relative_url }})
- [General syntax]({{ "/commands/general-syntax" | relative_url }})
