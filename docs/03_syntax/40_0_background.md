---
title: Normalization
parent: Commands
nav_order: 3
permalink: /commands/normalization/
has_children: true
math: katex
---

# Normalization

Normalization is a critical step in XAS data analysis that removes systematic variations and isolates the oscillatory EXAFS signal. EstraPy provides three commands that work together to perform complete XAS normalization:

1. **[`preedge`]({{ "/commands/normalization/preedge" | relative_url }})** - Removes the pre-edge background
2. **[`postedge`]({{ "/commands/normalization/postedge" | relative_url }})** - Removes the post-edge background (atomic absorption)
3. **[`normalize`]({{ "/commands/normalization/normalize" | relative_url }})** - Normalizes the signal to the edge jump

## Normalization Workflow

The typical normalization workflow follows these steps in order:

```sh
# 1. Remove pre-edge background
preedge .. -50eV

# 2. Remove post-edge atomic absorption
postedge 150eV ..

# 3. Normalize to edge jump
normalize
```

After normalization, you'll have:

- **`mu`** - Normalized absorption coefficient ($$\mu(E)/\mu_0(E_0)$$)
- **`chi`** - EXAFS oscillations ($$\chi(E) = \mu(E)/\mu_0(E_0) - 1$$)

## What Each Command Does

### Pre-edge Correction

The `preedge` command fits a polynomial to the absorption before the edge and subtracts it. This removes:

- Instrumental baseline drift
- Absorption from other edges at lower energies
- Linear or curved backgrounds

**Result:** Clean absorption spectrum with zero baseline before the edge.

### Post-edge Correction

The `postedge` command fits a polynomial to the absorption well above the edge and removes the smooth atomic absorption ($$\mu_0(E)$$). This can be done by:

- **Subtraction:** $$a(E) - \mu_0(E) + J_0$$
- **Division:** $$a(E) / \mu_0(E) \times J_0$$

where $$J_0$$ is the edge jump height (stored as metadata variable `J0`).

**Result:** Isolated EXAFS oscillations on top of a constant baseline.

### Jump Normalization

The `normalize` command divides the signal by the edge jump $$J_0$$ to create standardized spectra:

- **`mu`** = $$a(E) / J_0$$ (normalized absorption)
- **`chi`** = $$a(E) / J_0 - 1$$ (EXAFS oscillations)

**Result:** Dimensionless absorption coefficient and EXAFS signal ready for Fourier analysis.

## Common Patterns

### Standard XAS Normalization

```sh
# Complete normalization workflow
preedge .. -50eV           # Fit pre-edge up to 50 eV below edge
postedge 150eV .. --div    # Fit post-edge from 150 eV above, divide
normalize                  # Create mu and chi
```

### K-weighted Normalization

For fitting the post-edge in k-space (useful for distant post-edge regions):

```sh
preedge .. -50eV
postedge 3k 12k --k-axis --kweight 2
normalize
```

### Alternative Post-edge Range

Using relative energy ($$e = E - E_0$$):

```sh
preedge .. -50eV
postedge +150eV +1000eV -e
normalize
```

## Metadata Variables

The normalization commands set important metadata variables:

| Variable | Set by | Description |
|----------|--------|-------------|
| `E0` | `preedge` | Absorption edge energy (eV) |
| `J0` | `postedge` | Edge jump height |

These variables are stored per-page and can be used in subsequent commands, plots, and exports.

## Domain and Column Requirements

All normalization commands operate in the **reciprocal domain** and require:

- **Axes:** `E` (energy), `e` (relative energy), `k` (wavevector)
- **Data:** `a` (absorption signal)
- **Metadata:** `E0` (edge energy)

The edge energy `E0` must be set by the `edge` command for normalization to work properly.

## See Also

- **[Pre-edge Correction]({{ "/commands/normalization/preedge" | relative_url }})** - Detailed `preedge` documentation
- **[Post-edge Correction]({{ "/commands/normalization/postedge" | relative_url }})** - Detailed `postedge` documentation
- **[Normalize]({{ "/commands/normalization/normalize" | relative_url }})** - Detailed `normalize` documentation
- **[Fourier Transform]({{ "/commands/fourier" | relative_url }})** - Next step after normalization

---

**Next:** Choose a specific normalization command to learn more about its options and usage.