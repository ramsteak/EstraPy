---
title: Pre-edge Correction
parent: Normalization
grand_parent: Commands
nav_order: 1
permalink: /commands/normalization/preedge/
math: katex
---

# Pre-edge Correction

The `preedge` command removes the pre-edge background from the absorption signal and determines the absorption edge energy ($$E_0$$).

## Basic Usage

```sh
preedge <range_start> <range_end> [options]
```

Fit a polynomial to the absorption signal before the edge and subtract it to establish a zero baseline.

## Command Arguments

### Range (Required)

Specify the energy range for fitting the pre-edge polynomial:

```sh
preedge .. -50eV      # From beginning to 50 eV below edge
preedge 8000eV 8300eV # Absolute energy range
preedge .. -100eV     # From beginning to 100 eV below edge
```

Range can be specified in:
- **Absolute energy:** `8000eV 8300eV` (requires energy axis)
- **Relative to edge:** `-200eV -50eV` (relative to initial E0 estimate)
- **Unbounded:** Use `..` for -∞ or +∞

The upper bound is typically 30-100 eV below the edge to avoid XANES features.

## Command Options

### Polynomial Degree

| Option | Alias | Description |
|--------|-------|-------------|
| `--degree <n>` | `--deg <n>` | Polynomial degree (default: 1) |
| `--constant` | `-C` | Use constant (degree 0) |
| `--linear` | `-l` | Use linear fit (degree 1, default) |
| `--quadratic` | `-q` | Use quadratic fit (degree 2) |
| `--cubic` | `-c` | Use cubic fit (degree 3) |

**Example:**
```sh
preedge .. -50eV --linear      # Linear background (default)
preedge .. -50eV --quadratic   # Quadratic background
preedge .. -50eV --degree 0    # Constant offset
```

### Edge Energy Determination

| Option | Description |
|--------|-------------|
| `--edge <energy>` | Manually specify edge energy E0 |
| `--edgemethod <method>` | Method for finding E0 (default: auto) |

**Edge determination methods:**
- **Auto** (default) - Automatically detect edge from derivative
- **Manual** - Use `--edge` to specify E0 explicitly

**Examples:**
```sh
preedge .. -50eV                    # Auto-detect E0
preedge .. -50eV --edge 8979eV      # Use Cu K-edge energy
preedge .. -50eV --edge 24350eV     # Use Pd K-edge energy
```

### Normalization Range

| Option | Description |
|--------|-------------|
| `--normrange <start> <end>` | Energy range for edge jump calculation |

Specify the range over which to calculate the edge jump for normalization. This is typically 30-50 eV above the edge.

**Example:**
```sh
preedge .. -50eV --normrange +30eV +50eV
```

## Output

The command:
1. Fits a polynomial to the specified pre-edge range
2. Subtracts the polynomial from the absorption signal
3. Determines the edge energy $$E_0$$ (or uses provided value)
4. Creates derived axes ($$e$$, $$k$$) based on $$E_0$$
5. Sets metadata variables

**Columns modified:**
- `a` - New version with pre-edge background removed

**Columns created:**
- `e` - Relative energy ($$e = E - E_0$$)
- `k` - Wavevector ($$k = \sqrt{2m_e(E-E_0)/\hbar^2}$$)

**Metadata set:**
- `E0` - Absorption edge energy (eV)

## Edge Energy (E0)

The absorption edge energy $$E_0$$ is a critical parameter that determines:
- The zero point for relative energy ($$e = E - E_0$$)
- The conversion to wavevector space ($$k$$)
- The position of the edge jump

**Auto-detection:** By default, EstraPy finds $$E_0$$ by locating the maximum of the first derivative of the absorption spectrum.

**Manual specification:** For difficult spectra or specific analysis requirements, you can specify $$E_0$$ manually:
```sh
preedge .. -50eV --edge 8979eV  # Cu K-edge
```

## Derived Axes

After determining $$E_0$$, the command creates two new axis columns:

### Relative Energy (e)

$$e = E - E_0$$

Energy relative to the absorption edge, useful for:
- Aligning multiple spectra
- Comparing near-edge features
- Specifying energy ranges relative to the edge

### Wavevector (k)

$$k = \sqrt{\frac{2m_e(E - E_0)}{\hbar^2}}$$

The photoelectron wavevector in Å⁻¹, essential for:
- EXAFS analysis
- Fourier transforms
- K-space plotting and fitting

## Examples

### Standard Pre-edge Correction

```sh
# Linear fit from beginning to 50 eV below edge
preedge .. -50eV

# Equivalent explicit form
preedge .. -50eV --linear
```

### Manual Edge Energy

```sh
# Specify Cu K-edge energy
preedge .. -50eV --edge 8979eV

# Specify Pd K-edge energy  
preedge .. -100eV --edge 24350eV
```

### Quadratic Background

```sh
# Quadratic fit for curved backgrounds
preedge .. -80eV --quadratic
```

### Specific Energy Range

```sh
# Fit between 8000 and 8300 eV
preedge 8000eV 8300eV --linear
```

### Constant Offset Removal

```sh
# Remove constant baseline only
preedge .. -50eV --constant
```

## Typical Workflow

```sh
# 1. Import data
filein data/*.dat -E energy -t I0 I1

# 2. Remove pre-edge (this command)
preedge .. -50eV

# 3. Remove post-edge
postedge 150eV ..

# 4. Normalize
normalize
```

## Tips and Best Practices

1. **Range selection:** 
   - Start as far from the edge as your data allows
   - End 30-100 eV before the edge
   - Avoid including any XANES features

2. **Polynomial degree:** 
   - Linear (degree 1) works for most cases
   - Use constant (degree 0) only if baseline is flat
   - Use quadratic (degree 2) for strongly curved backgrounds
   - Avoid higher degrees unless absolutely necessary

3. **Edge energy:**
   - Auto-detection works well for clean, well-defined edges
   - Use manual specification for:
     - Multiple edges in the spectrum
     - Noisy data
     - Weak or delayed edges
     - When you need exact literature values

4. **Verify the correction:** Plot the result to ensure the pre-edge region is flat:

   ```sh
   plot a vs E
   ```

5. **Check E0 detection:** The automatically detected $$E_0$$ should be at the inflection point of the edge. If not, specify it manually.

## Common Edge Energies

For reference, some common K-edge energies:

| Element | E0 (eV) | Element | E0 (eV) |
|---------|---------|---------|---------|
| Cu | 8979 | Ni | 8333 |
| Zn | 9659 | Co | 7709 |
| Fe | 7112 | Mn | 6539 |
| Pd | 24350 | Pt | 11564 |
| Au | 11919 | Ag | 25514 |

## Error Messages

**"Invalid range"** - Check that:

- Range uses consistent units
- Range falls within your data
- Upper bound is below the edge

**"Cannot determine E0"** - The edge detection failed. Try:

- Manually specifying E0 with `--edge`
- Checking data quality
- Ensuring the edge is in the data range

**"Column 'a' not found"** - Ensure data has been imported with the absorption signal

## Physical Background

The pre-edge region contains:

- **Baseline drift** - Instrumental or systematic offset
- **Other edge contributions** - Absorption from lower-energy edges
- **Background absorption** - Smooth atomic absorption below the edge

Removing this background ensures:

- Zero baseline before the edge
- Accurate edge jump determination
- Clean EXAFS oscillations after normalization

The subtraction gives:

$$a_{corrected}(E) = a_{raw}(E) - P(E)$$

where $$P(E)$$ is the fitted polynomial.

## See Also

- [Post-edge Correction]({{ "/commands/normalization/postedge" | relative_url }}) - Next normalization step
- [Normalize]({{ "/commands/normalization/normalize" | relative_url }}) - Final normalization step
- [Normalization Overview]({{ "/commands/normalization" | relative_url }}) - Complete workflow
- [File Input]({{ "/commands/file-input" | relative_url }}) - Importing XAS data