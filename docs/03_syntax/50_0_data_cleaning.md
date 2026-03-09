---
title: Data cleaning
parent: Syntax
nav_order: 40
permalink: /commands/data-cleaning/
has_children: true
math: katex
---

# Data cleaning

Data cleaning removes non-physical signal features that can bias normalization, Fourier transforms and fitting.
In EstraPy, this section focuses on two operations:

1. **[glitch removal]({{ "/commands/glitch-removal" | relative_url }})** - Detects and corrects isolated spikes or drops (glitches)
2. **[multiple edges]({{ "/commands/multiple-edges" | relative_url }})** - Corrects secondary edge-like contributions in the scanned range

## When to Use Data Cleaning

Apply data cleaning when your spectra show one or more of the following:

- Sudden, narrow spikes inconsistent with neighboring points
- Sharp drops caused by detector or monochromator artifacts
- Extra step-like features from secondary excitations
- Distortions that create artificial oscillations after background removal

Data cleaning is typically performed after setting the edge and before final normalization/Fourier analysis.

## Typical Workflow

In a typical workflow, you would first perform data cleaning 

```sh
# 1. Remove isolated glitches
glitch-removal ...

# 2. Correct secondary edge contributions
multiple-edges ...

# 3. Continue with normalization / Fourier workflow
preedge ...
postedge ...
normalize ...
```

## See Also

- **[Glitch Removal]({{ "/commands/glitch-removal" | relative_url }})** - Detect and repair local artifacts
- **[Multiple Edges]({{ "/commands/multiple-edges" | relative_url }})** - Correct secondary edge-like features

---

**Next:** Start with [Glitch Removal]({{ "/commands/glitch-removal" | relative_url }}) for local spike cleanup
