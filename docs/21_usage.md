---
title: Command-Line Usage
nav_order: 4
permalink: /usage/
---

# Command-Line Usage

EstraPy is invoked from the command line using the `estrapy` command. It reads an `.estra` input file describing the analysis workflow and produces output files in the specified directory.

## Synopsis

```sh
estrapy [options] [inputfile]
```

## Positional Arguments

### `inputfile`

Path to the `.estra` input file that defines the analysis workflow.

```sh
estrapy analysis.estra
```

If `inputfile` is omitted, EstraPy starts in **interactive mode**. Each command is executed as soon as Enter is pressed, giving immediate feedback:

```sh
estrapy
```

In interactive mode, a command that spans multiple lines can be continued by ending each intermediate line with a backslash (`\`). The command is executed once a line is entered without a trailing backslash:

```
filein mydata.xdi --dir data/ \
    -E BraggEnergy -t I0_eh1 I0_eh2
```

## Options

### `-h`, `--help`

Print the help message and exit.

```sh
estrapy --help
```

---

### `-V`, `--version`

Print the current version of EstraPy and exit.

```sh
estrapy --version
```

Expected output:

```txt
EstraPy {{ site.version }}
```

---

### `-o OUTPUTDIR`, `--outputdir OUTPUTDIR`

Directory where all output files (CSV exports, plots, logs, etc.) will be saved.

```sh
estrapy analysis.estra -o results/
```

**Default behavior:**

- When an `inputfile` is provided, outputs are saved in a folder with the **same name as the input file** (without extension), located next to it. For example, `analysis.estra` → `analysis/`.
- When running in interactive mode (no input file), outputs are saved in the **current working directory**.

---

### `--cwd CWD`

Override the working directory used to resolve **relative paths** that appear inside the `.estra` script (e.g., paths to data files).

```sh
estrapy analysis.estra --cwd /data/experiment01
```

This does not change the shell's current directory — it only affects how EstraPy interprets relative paths written in the input file.

**Default:** the actual current working directory of the shell session.

This option is useful when running EstraPy from a script or CI pipeline where the shell's working directory differs from where the data files are stored.

---

### `-v`, `--verbose`

Enable verbose output. Sets the logging level to **debug**, printing detailed information about each processing step.

```sh
estrapy analysis.estra --verbose
```

Useful for inspecting intermediate results and diagnosing unexpected behavior.

---

### `--debug`

Enable debug mode. This implies `--verbose` and additionally changes certain execution behaviors to make debugging easier:

- **Disables multiprocessing** — all tasks run sequentially in the main process, so Python tracebacks are cleaner and easier to follow.
- Enables detailed timing output (implies `--timings`).

```sh
estrapy analysis.estra --debug
```

{: .note }
`--debug` mode may be significantly slower than normal execution for analyses with many files, since multiprocessing is disabled.

---

### `--timings`

Print detailed timing information for each analysis step. Useful for profiling performance bottlenecks in large workflows.

```sh
estrapy analysis.estra --timings
```

This flag is implied by `--debug`.

---

### `--vars [NAME=VALUE ...]`

Define one or more **variables** to inject into the script at runtime. Variables are specified as `NAME=VALUE` pairs.

```sh
estrapy analysis.estra --vars THRESHOLD=0.5 ELEMENT=Fe
```

Multiple variables can be passed in a single `--vars` invocation:

```sh
estrapy analysis.estra --vars EDGE=Fe.K RANGE=200
```

Inside the `.estra` script, these variables can be referenced by name to parameterize the analysis without editing the file. This makes it straightforward to run the same script with different inputs from a shell loop or batch script:

```sh
for SAMPLE in S1 S2 S3; do
    estrapy analysis.estra --vars SAMPLE=$SAMPLE -o output/$SAMPLE
done
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| `0`  | Success — analysis completed without errors. |
| `1`  | Error — analysis was aborted due to a script or runtime error. |

---

## Examples

Run an analysis and save results to a specific folder:

```sh
estrapy experiment.estra -o results/experiment
```

Run with verbose output and custom working directory:

```sh
estrapy experiment.estra --verbose --cwd /data/synchrotron
```

Pass runtime variables to a parameterized script:

```sh
estrapy template.estra --vars ELEMENT=Pd EDGE=K -o pd_analysis/
```

Start an interactive session with outputs directed to a specific folder:

```sh
estrapy -o results/
```
