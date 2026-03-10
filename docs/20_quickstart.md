---
title: Quick Start Guide
nav_order: 3
permalink: /quickstart/
---

# Quick Start Guide

This guide will walk you through running your first EstraPy analysis in just a few minutes.

## Prerequisites

Make sure you have [installed EstraPy]({{ '/installation' | relative_url }}) and verified the installation with `estrapy --version`.

## Your First Analysis

### Step 1: Get the Example Files

Clone or download the EstraPy repository to access the example files:

```sh
git clone https://github.com/ramsteak/EstraPy
cd EstraPy/examples
```

Alternatively, download the [examples folder](https://github.com/ramsteak/EstraPy/tree/main/examples) directly from GitHub.

The examples folder contains several complete analysis workflows, each with:

- Input `.estra` files defining the analysis
- Sample data files from synchrotron experiments
- Expected output for verification

### Step 2: Explore the Example Files

List the available examples:

```sh
ls
```

You'll see folders for different types of analyses. Let's start with a basic example. Navigate to one of the example directories:

```sh
cd basic_analysis  # or whichever example you want to try
```

Each example contains:

- `*.estra` - The input file with analysis commands
- `data/` - Sample synchrotron data files
- `README.md` - Description of the analysis (if available)

### Step 3: Examine the Input File

Open the `.estra` file to see the analysis workflow:

```sh
cat example.estra
```

Input files use a simple, readable syntax. Here's what a typical file looks like:

```sh
# version: {{ site.version }}
# Each file starts with the version of the EstraPy syntax being used
% title "Basic XANES Analysis"

# Load data from synchrotron experiment
filein AD09.xdi --dir AD09data -E BraggEnergy -t I0_eh1 I0_eh2

# Set the energy of the E0 to the K edge of Manganese
edge set Mn.K

# Correct the baseline and normalize the jump
preedge -200eV -60eV --linear
postedge +150eV .. --divide --quadratic --energy
normalize --factor J0

# Export the processed XANES data to a CSV file named AD09.csv
save AD09.csv --columns E mu

# Export the processed EXAFS data to a CSV file as column k, chi, and k*chi
save AD09.csv --columns k chi k*chi
```

Don't worry about understanding every command yet - we'll cover the [syntax]({{ '/syntax' | relative_url }}) in detail later.

### Step 4: Run the Analysis

Execute the analysis with:

```sh
estrapy example.estra
```

EstraPy will:

1. Read, parse and validate your input file
2. Execute commands sequentially
3. Process the data files
4. Generate plots
5. Save output files
6. Display a log of all operations

You should see output similar to:

```log
[INF] EstraPy - XAS data analysis tool
[INF] (c) 2024 Marco Stecca
[INF] Version {{ site.version }}
[INF] Imported 1 file in 99.37 ms (122.6kB).
[INF] Set E0 to 6539.0eV for all pages.
[INF] Completed spline background calculation for all pages.
```

### Step 5: Check the Output

After the analysis completes, check your directory for output files:

```sh
ls
```

A new folder should have been created with the results of your analysis.
Inside, you'll find:

- Generated plots (`.png`, `.pdf`, or other image formats)
- Exported data files (`.csv`, `.txt`, etc.)
- The log file with analysis details

## Try More Examples

The examples folder contains several analysis types to explore:

- **Basic analysis** - Simple data import, processing, and plotting
- **Peak fitting** - Identify and fit peaks in spectroscopy data
- **Batch processing** - Analyze multiple files at once
- **Advanced plotting** - Custom visualization options

Navigate to each example directory and run the provided `.estra` files to see different capabilities.

## Understanding the Workflow

Every EstraPy analysis follows this pattern:

1. **Write** an input `.estra` file with your analysis steps
2. **Run** the analysis: `estrapy run yourfile.estra`
3. **Review** the output files and plots

The command-line interface provides several options:

```sh
estrapy input.estra          # Run an analysis
estrapy input.estra --verbose # Show detailed output
```

<!--
TODO: List all CLI commands such as 
  estrapy * 
  estrapy ?
-->

## Creating Your Own Analysis

Now that you've run an example, you can create your own analysis:

1. **Prepare your data files** - Ensure your synchrotron data is in a supported format
2. **Create an input file** - Write a new `.estra` file with your analysis steps
3. **Run the analysis** - Execute with `estrapy run yourfile.estra`

For details on writing input files, see the [Input File Syntax]({{ '/syntax' | relative_url }}) documentation.

## Common Commands Overview

Here are some essential commands you'll use in your `.estra` files:

| Command | Purpose |
|---------|---------|
| `import <file>` | Load data from a file |
| `plot <x> vs <y>` | Create a plot |
| `export <file>` | Save processed data |
| `baseline_correction` | Remove baseline signal |
| `normalize` | Normalize data values |

## Getting Help

- **Syntax questions** - See [Input File Syntax]({{ '/syntax' | relative_url }})
- **Detailed tutorials** - Check out [Tutorials]({{ '/tutorials' | relative_url }})
- **Troubleshooting** - Visit [Troubleshooting]({{ '/troubleshooting' | relative_url }})
- **Report issues** - Open an [issue on GitHub](https://github.com/ramsteak/EstraPy/issues)

## Next Steps

Now that you've run your first analysis:

1. Explore more examples in the repository
2. Read the [Input File Syntax]({{ '/syntax' | relative_url }}) guide
3. Try modifying an example to analyze your own data
4. Check out [Tutorials]({{ '/tutorials' | relative_url }}) for specific workflows

---

Ready to dive deeper? Continue to the [Input File Syntax]({{ '/syntax' | relative_url }}) documentation.
