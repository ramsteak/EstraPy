---
permalink: /python-installation/
---

# Python Installation

This small guide will help you install Python, which is required to run EstraPy.

## Windows

- Head to the [Python downloads page](https://www.python.org/downloads/)

- Download the Python Install Manager for Windows

You will be prompted with multiple options, these are the recommended ones:

```txt
Windows is not configured to allow paths longer than 260 characters.

Python and some other apps can exceed this limit, but it requires changing a
system-wide setting, which may need an administrator to approve, and will
require a reboot. Some packages may fail to install without long path support
enabled.
Update setting now? [y/N]
```

Input `y` to enable long path support

```txt
The global shortcuts directory is not configured.

Configuring this enables commands like python3.14.exe to run from your
terminal, but is not needed for the python or py commands (for example, py
-V:3.14).

We can add the directory (C:\Users\WDAGUtilityAccount\AppData\Local\Python\bin)
to PATH now, but you will need to restart your terminal to use it. The entry
will be removed if you run py uninstall --purge, or else you can remove it
manually when uninstalling Python.
Add commands directory to your PATH now? [y/N]
```

This is not required, you can input `n`