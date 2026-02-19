---
title: File Input
parent: Syntax
nav_order: 2
permalink: /commands/file-input/
math: katex
---

# File Input

EstraPy uses the `filein` command to import data files. The command reads specified columns from data files, applies necessary transformations, and handles metadata.

## Basic Usage

```sh
filein <files> [options]
```

Import a single file or multiple files, optionally using glob patterns.

**Examples:**

```sh
# Import specified file names
filein data/scan_001.dat data/scan_002.dat -E energy -t I0 I1

# Import multiple files with glob pattern
filein data/*.dat -E energy -t I0 I1
```

## Command Options

### File Selection

| Option | Description |
|--------|-------------|
| `<file>` | Path to the file(s). Supports glob patterns (e.g., `*.dat`) |
| `--dir <directory>` | Base directory for relative paths |

### Axis Import Options

| Option | Description |
|--------|-------------|
| `--energy <col>` <br> `-E <col>` | Column for energy values |
| `--wavevector <col>` <br> `-k <col>` | Column for wavenumber values |
| `--rspace <col>` <br> `-r <col>` | Column for reciprocal space values |
| `--qvector <col>` <br> `-q <col>` | Column for momentum transfer values |

Only one of the above options can be used to specify the x-axis column. If multiple are specified, an error will be raised.
Energy, wavevector and qvector values are in the `reciprocal` domain, while rspace values are in the `fourier` domain. The data must match the specified domain, otherwise an error will be raised.

### Signal Import Options

#### Reciprocal Space Options

| Option | Description |
|--------|-------------|
| `--beamintensity <col>` <br> `-I0 <col>` | Column for beam intensity (I0) values |
| `--sampleintensity <col>` <br> `-I1 <col>` | Column for sample intensity (I1) values |
| `--referenceintensity <col>` <br> `-I2 <col>` | Column for reference intensity (I2) values |
| `--samplefluorescence <col>` <br> `-If1 <col>` | Column for sample fluorescence (If) values |
| `--transmission [col col]` <br> `-t [col col]` | Two columns for transmission values. If the columns are not specified, uses -log(I1 / I0) from intensities. |
| `--referencetransmission [col col]` <br> `-rt [col col]` | Two columns for reference transmission values. If the columns are not specified, uses -log(I2 / I1) from intensities. |
| `--fluorescence [col col]` <br> `-f [col col]` | Two columns for fluorescence values. If the columns are not specified, uses If / I0 from intensities. |
| `--intensities <col> <col> ...` <br> `-I <col> <col> ...` | Multiple columns (from 1 to 4) for intensity values. They are treated as beam, sample, reference and fluorescence intensities, in this order. Note that either `--transmission` or `--fluorescence` must be set. |
| `--xanes <col>` <br> `--mu <col>` | Option for directly importing xanes values. |
| `--exafs <col>` <br> `--chi <col>` | Option for directly importing exafs values. |
| `--kchi <col>` | Option for directly importing k*chi values. |
| `--k2chi <col>` | Option for directly importing k^2*chi values. |

#### Fourier Space Options

| Option | Description |
|--------|-------------|
| `--fouriermagnitude <col>` <br> `--fm <col>` | Column for Fourier magnitude values |
| `--fourierphase <col>` <br> `--fp <col>` | Column for Fourier phase values |
| `--fourierreal <col>` <br> `--fr <col>` | Column for Fourier real values |
| `--fourierimaginary <col>` <br> `--fi <col>` | Column for Fourier imaginary values |
| `--fouriercartesian <col> <col>` <br> `--fc <col> <col>` | Two columns for Fourier real and imaginary values. |
| `--fouriereulerian <col> <col>` <br> `--fe <col> <col>` | Two columns for Fourier magnitude and phase values. |

#### Error Options

All previous single-column data options allow to specify an error column. This is achieved by appending `error` to the option name, or prepending an `s` to the short option name. For example:

| Option | Description |
|--------|-------------|
| `--beamintensityerror <col>` <br> `-sI0 <col>` | Column for beam intensity (I0) error values |
| `--xaneserror <col>` <br> `--smu <col>` | Column for xanes error values |

Note that this list is not exhaustive, and the error column can be specified for any of the single-column data options by following the same pattern.

### Other Options

| Option | Description |
|--------|-------------|
| `--shift` | Shift the x-axis values by a constant amount. The value is added to the column. |
| `--sortby <var>` | Sort the files by the specified variable (e.g. alphabetically by file name or by date of creation) |

### Importer Format Options

EstraPy supports various import options, to adapt to different tabular data formats. These are specified with the `--format` option, which accepts two parameters:

| Option | Description |
|--------|-------------|
| `--format decimal <char>` | Decimal separator character (e.g., `.` or `,`) |
| `--format separator <char>` | Column separator character (e.g., `\t`, `,`, `;`) |
| `--format comment <char>` | Comment character to ignore lines (e.g., `#`) |
| `--format skip <n>` | Number of lines to skip at the beginning of the file |
| `--format leadingheader <n>` | Used to better adapt to header line parameter (see [below](#leading-header)) |

#### Decimal Separator

The `decimal` option allows you to specify the character used as a decimal separator in the data files. This is particularly useful for files that use a comma (`,`) instead of a period (`.`) as the decimal separator, which is common in some locales. By default, a period (`.`) is used as the decimal separator.

#### Column Separator

The `separator` option specifies the character that separates columns in the data files. Common separators include tabs (`\t`), commas (`,`), and semicolons (`;`). By default, whitespace is used as the column separator.

#### Comment Character

The `comment` option allows you to specify a character that indicates the start of a comment line. Lines starting with this character will be ignored during import. This is useful for files that contain metadata or comments interspersed with data. By default, `#` is used as the comment character.

#### Skip Lines

The `skip` option allows you to specify a number of lines to skip at the beginning of the file. This is useful for files that do not use a comment character but have a fixed number of header lines. If the `comment` option is not specified, the `skip` option can be used to ignore header lines. By default, no lines are skipped.

#### Leading Header

The `leadingheader` option is used to specify if the leading column name in the header line should be ignored.
This distinguishes the two cases:

- With leading header: `#H energy I0 I1` (the first column name is ignored)
- Without leading header: `#energy I0 I1` (the first column name is stripped of the comment character and used as the name of the first column)

## Column Specification

Columns can be referenced by:

- **Column number** - Starting from 1 (e.g., `1`, `2`, `3`)
- **Column name** - From file header (e.g., `energy`, `I0`)

**Column ranges and combinations:**

Columns can be specified in lists, separated by commas, and inclusive ranges, specified with `..`. For example:

- Range: `col1..col5` (columns 1-5)
- List: `col1,col3,col5` (the columns 1, 3, and 5)
- Mixed: `col1..col3,col5..col7` (columns 1, 2, 3, 5, 6, and 7)

Multiple columns are summed together when specified.

## File Metadata

EstraPy automatically extracts metadata from filenames and headers, and sets them as variables. Some commands, namely `save` and `plot`, can use these variables for dynamic file naming and labeling.

It also attempts to extract timestamps from the header of the file, and adds them as variables. Note that the timestamp may not be fully supported yet, and may not work for all file formats.

| Variable | Description |
|----------|-------------|
| `.fn` | Filename with extension |
| `.f` | Filename without extension |
| `.fe` | File extension (with dot) |
| `.fp` | Full file path |
| `.fs` | File size in bytes |
| `.fa` | File access time (timestamp) |
| `.fc` | File creation time (timestamp) |
| `.fm` | File modification time (timestamp) |
| `.f1` .. `.f<N>` | File name split by `_` |
| `.fn1` .. `.fn<N>` | File name split by `_` from the end |
| `.fd` | Immediate parent directory name |
| `.fd0` .. `.fd<N>` | Parent directory names (0 is immediate parent) |
| `.h<line>.<column>` | Header line and column (separated by spaces) |
| `.t` | Timestamp extracted from header |
| `.ts` | Timestamp extracted from header as seconds since epoch |
| `.d` | Measurement duration extracted from header |
| `.ds` | Measurement duration extracted from header in seconds |

When applicable, EstraPy parses the metadata values to their appropriate types (e.g., numbers).
For example, if the filename is `/data/sample_300K.dat`, the variables would be:

| Variable | Value |
|----------|-------|
| `.fn` | "sample_300K_001.dat" |
| `.f` | "sample_300K_001" |
| `.fe` | ".dat" |
| `.f1` = `.fn3` | "sample" |
| `.f2` = `.fn2` | "300K" |
| `.f3` = `.fn1` | 1 |
| `.fd` = `.fd0` | "data" |

Additional metadata can be set with `--var <name> <value>`.
You can also specify an already-defined variable as the value, such as
`--var temperature .h1.2` to set the variable `temperature` to the value of the header in line 1, column 2.

## Example

```sh
# Import XAS data files
filein data/*.xy
       --energy E
       --intensities I0 I1 I2 If1..If5
       --fluorescence
       --reftransmittance

# Import with added metadata
filein data/sample_001.dat
       --energy E
       --transmission I0 I1 
       --var sample "Cu foil"
       --var temperature 300
```

**See also:**

- [General Syntax]({{ "/commands/general-syntax" | relative_url }}) - Input file syntax rules
- [Data Processing]({{ "/commands/processing" | relative_url }}) - Commands for processing imported data