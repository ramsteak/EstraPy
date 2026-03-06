---
title: Home
layout: home
nav_order: 1
description: "EstraPy is a synchrotron data analysis framework, built in Python 3."
permalink: /
---

# EstraPy
{: .fs-9 }

EstraPy is a synchrotron data analysis framework, built in Python 3.
{: .fs-6 .fw-300 }

Synchrotron facilities produce X-ray diffraction and spectroscopy data used in materials science, crystallography, and structural biology research. EstraPy provides a command-line framework for analyzing this data using simple, readable input files.

[Get Started]({{ "/installation" | relative_url }}){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[Examples](https://github.com/ramsteak/EstraPy/tree/main/example){: .btn .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/ramsteak/EstraPy){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## Key Features

- **Simple Input Syntax** - Write analysis workflows in plain text `.estra` files
- **Comprehensive Data Handling** - Import, process, and export synchrotron data formats
- **Automated Plotting** - Generate publication-ready visualizations
- **Command-Line Workflow** - Execute analysis scripts with a single command
- **Extensible** - Modular design for custom analysis pipelines

---

## Quick Start

**Requirements:** Python 3.11+

Install EstraPy and run your first analysis:

````bash
pip install estrapy
estrapy analysis.estra
````

See the [Quick Start Guide]({{ "/quickstart" | relative_url }}) for a complete walkthrough.

---

## Documentation

### [Installation]({{ '/installation' | relative_url }})

Install EstraPy and set up your environment

### [Quick Start Guide]({{ '/quickstart' | relative_url }})

Get up and running in minutes

### [Input File Syntax]({{ '/syntax' | relative_url }})

Learn the .estra file format and commands

### [Tutorials]({{ '/tutorials' | relative_url }})

Step-by-step guides for common workflows

### [Troubleshooting]({{ '/troubleshooting' | relative_url }})

Common issues and solutions

---

## About the Project

EstraPy is &copy; 2024-{{ "now" | date: "%Y" }} by [Marco Stecca](https://github.com/ramsteak).

### License

EstraPy is distributed under an [MIT license](https://github.com/ramsteak/EstraPy/blob/main/LICENSE).

### Contributing

To report issues with the program, create a new [issue](https://github.com/ramsteak/EstraPy/issues) on GitHub. Please include:
- Your input `.estra` file
- Relevant data files
- The output log file
- Python version and operating system

For feature requests and general discussion, visit the [Discussions](https://github.com/ramsteak/EstraPy/discussions) page.
