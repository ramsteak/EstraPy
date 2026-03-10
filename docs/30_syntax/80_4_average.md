---
title: Average
parent: Data Manipulation
grand_parent: Syntax
nav_order: 4
permalink: /commands/data-manipulation/average/
math: katex
---

# Average

The `average` command merges multiple data pages (scans) into averaged spectra based on metadata grouping. This is a **destructive operation** — all original pages are replaced with averaged results.

(Note that the original data file remains unchanged; only the in-memory dataset is modified.)

## Basic Usage

```sh
average [options]
```

All options are optional; without `--groupby`, all pages are averaged into a single spectrum.

## Command Options

| Option | Description |
|--------|-------------|
| `--groupby <key1> [<key2> ...]` | Metadata keys to group by before averaging (default: none — average all pages together). |
| `--minsize <n>` | Minimum number of scans required to form a group (default: `1`). Groups smaller than this are discarded. |
| `--maxsize <n>` | Maximum number of scans per group (default: unlimited). Larger groups are split into subgroups. |
| `--domain <domain>` | Domain in which to perform averaging (default: `reciprocal`). Options: `reciprocal`, `fourier`. |
| `--axis <axis>` | Axis along which to average (default: auto-selected for domain). |

## Grouping Logic

1. **Group by metadata:** Pages sharing identical values for all `--groupby` keys are grouped together
2. **Split large groups:** If a group exceeds `--maxsize`, it is divided into subgroups of size ≤ `--maxsize`
3. **Discard small groups:** Final groups smaller than `--minsize` are discarded, except:
   - If splitting creates a last subgroup smaller than `--minsize`, it is merged with the previous subgroup (even if this exceeds `--maxsize`)

## Averaging Method

For each group:

- All data rows with matching indices are averaged across pages
- Metadata is merged:
  - Numerical metadata values are averaged
  - Non-numerical metadata uses the first value in the group
  - Special metadata variables are added (see below)

## Generated Metadata

The averaged page includes new metadata variables:

| Variable | Description |
|----------|-------------|
| `.navg` | Number of scans averaged |
| `.g` | Group ID (sequential numbering) |
| `.sg` | Subgroup ID (if group was split due to `--maxsize`) |
| `.f` | Generated page name |
| `.fn` | Same as `.f` |
| `.f1`, `.f2`, ... | Parts of the page name split by `_` |
| `.g1`, `.g2`, ... | Grouping key values |

## Page Naming

Averaged pages are named by concatenating groupby values with `_`:

- If groupby values exist: `value1_value2_value3`
- If no groupby specified: `spectrum_<groupid>`

## Examples

```sh
# Average all scans into a single spectrum
average

# Group by sample name, average each sample separately
average --groupby sample

# Group by temperature and composition
average --groupby temperature composition

# Require at least 3 scans per group, max 10 scans per average
average --groupby sample --minsize 3 --maxsize 10

# Average in Fourier domain
average --domain fourier --groupby phase
```

## Behavior

The command:

1. Groups all pages based on `--groupby` keys
2. Splits groups exceeding `--maxsize`
3. Averages data within each group
4. Replaces all pages with the averaged results

{: .warning }
**Destructive operation:** All original pages are replaced with averaged groups. The original data file remains unchanged; only the in-memory dataset is modified. The workspace will contain only the averaged spectra after execution.

## Use Cases

- **Improve SNR:** Merge replicate scans to reduce random noise
- **Sample series:** Average multiple measurements per sample, grouped by sample ID
- **Temperature/pressure series:** Group scans by experimental condition
- **Quality control:** Discard unreliable single scans using `--minsize`

## Tips and Best Practices

1. **Align first:** Use `cut`, `interpolate`, or `rebin` to ensure all scans have identical axes before averaging

2. **Group carefully:** Choose metadata keys that uniquely identify experimental conditions:

   ```sh
   # Good: temperature is controlled variable
   average --groupby temperature
   
   # Better: multiple grouping keys for complex experiments
   average --groupby sample_id temperature pressure
   ```

3. **Check metadata:** Use consistent metadata naming across scans:
   - `temp`, `T`, `temperature` are treated as different keys
   - Verify metadata with the file inspection commands before averaging

4. **Control group size:**
   - `--maxsize` limits memory usage and prevents over-averaging of drifting data
   - `--minsize` ensures statistical validity (typically 2–3 minimum)

5. **Preserve originals:** Save original data before averaging if you may need individual scans later

## Output Structure

After averaging:

- Original page count may be reduced significantly
- Each new page represents one averaged group
- Metadata header includes list of original filenames
- Column structure matches the original (same axes and data columns)

**See also:**

- [Data Manipulation overview]({{ "/commands/data-manipulation" | relative_url }})
- [General syntax]({{ "/commands/general-syntax" | relative_url }})
