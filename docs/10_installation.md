---
title: Installation
nav_order: 2
permalink: /installation/
---

# Installation

## Requirements

- Python 3.11 or higher
- pip or uv package manager

## Quick Install

The easiest way to install EstraPy is using pip:

```sh
pip install estrapy
```

Or, if you're using uv:

```sh
uv pip install estrapy
```

## Verify Installation

To test if the installation was successful, run:

```sh
estrapy --version
```

You should see:

```sh
EstraPy 2.0.0
```

## Virtual Environment (Recommended)

It's recommended to install EstraPy in a virtual environment to avoid dependency conflicts:

### Using venv

```sh
# Create a virtual environment
python -m venv estrapy-env

# Activate it (Windows)
estrapy-env\Scripts\activate

# Activate it (Linux/macOS)
source estrapy-env/bin/activate

# Install EstraPy
pip install estrapy
```

### Using uv

```sh
# Create and activate virtual environment with uv
uv venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install EstraPy
uv pip install estrapy
```

If you are using a virtual environment, remember to activate it each time before using EstraPy.

## Manual Installation

### From GitHub Releases

Download the latest `.whl` file from the [releases page](https://github.com/ramsteak/EstraPy/releases) and install it:

```sh
pip install estrapy-2.0.0-py3-none-any.whl
```

### Building from Source

If you want to build from source or contribute to development:

**Using GitHub CLI:**

```sh
gh repo clone ramsteak/EstraPy
cd EstraPy
python -m build
pip install dist/estrapy-2.0.0-py3-none-any.whl
```

**Using git:**

```sh
git clone https://github.com/ramsteak/EstraPy
cd EstraPy
python -m build
pip install dist/estrapy-2.0.0-py3-none-any.whl
```

**Note:** Building from source requires the `build` package:

```sh
pip install build
```

## Updating EstraPy

To update to the latest version:

```sh
pip install --upgrade estrapy
```

Or with uv:

```sh
uv pip install --upgrade estrapy
```

## Dependencies

EstraPy automatically installs its required dependencies during installation. Key dependencies include:

- NumPy - Numerical computing
- Matplotlib - Plotting and visualization
- SciPy - Scientific computing

For a complete list of dependencies, see the [requirements](https://github.com/ramsteak/EstraPy/blob/main/pyproject.toml) in the repository.

## Next Steps

Once installed, proceed to the [Quick Start Guide]({{ '/quickstart' | relative_url }}) to run your first analysis.
