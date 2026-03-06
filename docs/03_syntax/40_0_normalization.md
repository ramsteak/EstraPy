---
title: Normalization
parent: Syntax
nav_order: 40
permalink: /commands/normalization/
has_children: true
math: katex
---

# Normalization

Normalization is a critical step in XAS data analysis that removes systematic variations and isolates a normalized XANES edge and oscillatory EXAFS signal. EstraPy provides three commands that work together to perform complete XAS normalization:

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
normalize --factor J0
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

**Result:** Spectrum corrected for atomic absorption, with an edge jump standardized to $$J_0$$.

### Jump Normalization

The `normalize` command divides the signal by the edge jump $$J_0$$ to create standardized spectra:

- **`mu`** = $$a(E) / J_0$$ (normalized absorption)
- **`chi`** = $$a(E) / J_0 - 1$$ (EXAFS oscillations)

**Result:** Dimensionless absorption coefficient and EXAFS signal ready for Fourier analysis.

Note that the `a` column is not modified by `normalize` and retains the original absorption signal. The normalized `mu` and `chi` columns are created as new columns in the dataset.

## Domain and Column Requirements

All normalization commands operate in the **reciprocal domain** and require:

- **Axes:** `E` (energy), `e` (relative energy), `k` (wavevector)
- **Data:** `a` (absorption signal)
- **Metadata:** `E0` (edge energy)

The edge energy `E0` must be set by the `edge` command for normalization to work properly.

## See Also

- **[Pre-edge Correction]({{ "/commands/normalization/preedge" | relative_url }})** - Detailed `preedge` documentation
- **[Post-edge Correction]({{ "/commands/normalization/postedge" | relative_url }})** - Detailed `postedge` documentation
- **[Normalize]({{ "/commands/normalization/normalization" | relative_url }})** - Detailed `normalize` documentation

---

**Next:** Choose a specific normalization command to learn more about its options and usage.
