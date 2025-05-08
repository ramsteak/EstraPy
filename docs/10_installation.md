---
title: Installation
nav_order: 2
permalink: /installation/
---

# Installation

To download the latest release of EstraPy, navigate to the [releases](https://github.com/ramsteak/EstraPy2/releases) section of the [repository](https://github.com/ramsteak/EstraPy2/) and download the latest release. Install with your python package manager, such as:

```sh
py -m pip install estrapy-1.0.0-py3-none-any.whl
```

Alternatively, clone the github repository and build from source:

```sh
gh repo clone ramsteak/EstraPy2
python -m build
```

or

```sh
git clone https://github.com/ramsteak/EstraPy
python -m build
```

If you install it in a virtual environment, remember to activate it before using EstraPy.

To test if the installation was successful, run

```sh
$ estrapy --version

EstraPy 1.0.0
```

The program will display the installed version and exit.
