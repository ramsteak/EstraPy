---
title: Data normalization
parent: Normalization
grand_parent: Syntax
nav_order: 3
permalink: /commands/normalization/normalization/
math: katex
---

# Data Normalization

The `normalize` command performs the final normalization of the absorption signal after pre-edge and post-edge corrections have been applied. It scales the absorption to set the edge step to 1, resulting in a normalized spectrum.

## Basic Usage

```sh
normalize [options]
```

Divides by the given factor (default: edge step, `J0`) to set the absorption step to 1.

**Examples:**
```sh
normalize

normalize --factor J0

normalize --factor 2e5
```

## Command options

| Option | Description |
|--------|-------------|
| `--factor <value>` | Normalization factor (default: edge step, `J0`). Can be a column name or a numeric value. |

## Results

The command divides the absorption values in the `a` column by the specified normalization factor, resulting in a normalized absorption spectrum where the edge step is set to 1.

The normalized absorption is stored in the `mu` column, and the EXAFS-corrected signal is stored in the `chi` column.

For clarity, the new columns are defined as follows:

$$\mu = \frac{a}{\text{factor}}$$
$$\chi = \frac{a}{\text{factor}} - 1$$

**See also:**

- [Pre-edge Correction]({{ "/commands/normalization/preedge" | relative_url }}) - First normalization step
- [Post-edge Correction]({{ "/commands/normalization/postedge" | relative_url }}) - Second normalization step
- [Normalization Overview]({{ "/commands/normalization" | relative_url }}) - Complete workflow
