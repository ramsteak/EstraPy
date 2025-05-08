---
title: File input
parent: Commands
nav_order: 2
permalink: /commands/file-input
math: katex
---

# File input

File input in EstraPy is done with the `filein` command, and optionally with the `batchin` command.
The command reads from a data file, importing the selected columns and performing the necessary calculations.
It also handles metadata and additional variables.

## Usage

```sh
filein <file> [options]
```

The `batchin` command sets the default arguments for the file input, and takes the same optional arguments as the `filein` command.

|Argument|Explanation|
|--|--|
|`file`|The path that points to the file. Can be either an absolute path, or a relative path from the working directory. Glob patterns are supported.|
|`--dir` `<directory>`|Uses this directory to import the files from, if the path is relative, instead of the current working directory.|
|`--batch` / `-b`|Signals to the `filein` command to use the arguments defined by all previous `batchin` commands.|
|`--xaxiscolumn` / `-x` `<column>` |Sets the column containing the x axis. By default, it is equivalent to `--index <column>`. To modify the behavior, you can either use one of the other axis flags, or use e.g. `-x <column> -E`, equivalent to `-E <column>`|
|`--index` `[column]`|Imports the given axis column as a raw index. If no column is specified, uses the column given by `--xaxiscolumn`.|
|`--energy` / `-E` `[column]`|Imports the given axis column as energy in electronvolts. If no column is specified, uses the column given by `--xaxiscolumn`.|
|`--kvector` / `-k` `[column]`|Imports the given axis column as a k wavevector. If no column is specified, uses the column given by `--xaxiscolumn`.|
|`--rdistance` / `-R` `[column]`|Imports the given axis column as a distance in Angstroms. If no column is specified, uses the column given by `--xaxiscolumn`.|
|`--qvector` / `-q` `[column]`|Imports the given axis column as a q wavevector. If no column is specified, uses the column given by `--xaxiscolumn`.|
|`--shift` `<float>`|Shifts the x axis index by the specified amount.|
|`--intensities` / `-I` `<column> <column> <column> <column>`|Imports the specified columns as intensities, importing between 1 and 4 columns. They are, in order, $$I_0$$, $$I_1$$, $$I_2$$, $$I_f$$, imported as the columns `I0` `I1` `I2` `If`. If `--intensities` is not specified, the columns are inferred from the other signal options.|
|`--transmission` / `-t` `[column] [column]`|Calculates $$\mu_{exp}$$ as $$\log_{10}{\frac{I_0}{I_1}}$$, imported as the column `x`.|
|`--fluorescence` / `-f` `[column] [column]`|Calculates $$\mu_{exp}$$ as $$\frac{I_f}{I_0}$$, imported as the column `x`.|
|`--intensity` / `-i` `<column>`|Imports the signal value directly, importing it as the column `x`.|
|`--reftransmittance` / `-T` `[column] [column]`|Calculates $$\mu_{ref}$$ as $$\log_{10}{\frac{I_1}{I_2}}$$, imported as the column `ref`.|
|`--refabsorption` / `-A` `<column>`|Imports the signal value directly, importing it as the column `ref`.|
|`--var` `<name> <value>`|Adds the given variable to the file metadata, useful in other commands. The automatic variable definition is defined in the [metadata](#metadata) section. The value can be an auto-defined variable, such as .h1.1, wich will be resolved to the imported value.|

### Columns

To specify the column to be imported, you can either use the column number (counting from 1 up to the number of columns), or the column name. The column name is automatically imported line before the start of the data. If the length of the headers is different from the actual number of columns, the headers are automatically expanded by adding column names of the form `col1`, `col2` etc, where the number matches the column number, or cut.
Some options, namely `--transmission` `--fluorescence` `--intensity` `--reftransmittance` `--refabsorption`, can import multiple columns by adding them together. To import more than one column, you can specify a contiguous column range, such as `col1..col5`, separate columns, such as `col1,col3`, or a mix of the two, such as `col1..col3,col5..col7,col9`.

## Metadata

The file input automatically defines some variables from the file headers and the file name, which some commands can use. The program will attempt to parse the value as a number. If this fails, the value will be interpreted as a string. All variables defined with the `--var` option are added to the metadata of the file.

|Variable name|Description|
|--|--|
|`.f`|The entire file name|
|`.fn`|The name of the file, without the extension.|
|`.fe`|The extension of the file, with the leading dot.|
|`.i`|The index of the imported file, within a single `filein` execution, indexed from one.|
|`.n`|The global index of the imported file, indexed from one.|
|`.f1`|The filename, without extension, is split at each `_`. Each section is referred by its number, indexed from one.|
|`.h1.1`|The header of the file is split line by line and at each space. Each section is referred by its line and position.|
|`.st`|The signal type. Can be `fl` (fluorescence),`tr` (transmittance),`i` (intensity),`o` (other)|
|`.rt`|The reference signal type. Can be `fl` (fluorescence),`tr` (transmittance),`i` (intensity),`o` (other)|
|`name`|Imported from header lines like `#U name value`|
|`E0`|Defined by the `edgeenergy` command.|
|`rE0`|Defined by the `align` command.|
|`J0`|Defined by the `postedge` command.|

## Example

For the palladium example, the file input would look like this:

```sh
batchin -E energyc -I dif_ic0 dif_ic1 dif_ic2 x_ch1_roi1..x_ch13_roi1
filein data/*.xy -b -f -T
```

This snippet reads all the .xy files from the data folder. It then imports the following columns:

|Argument|Resulting column|Explanation|
|--|--|--|
|`-I dif_ic0 .. .. ..`|$$I_0$$: `I0`|Imports the $$I_0$$ intensity|
|`-I .. dif_ic1 .. ..`|$$I_1$$: `I1`|Imports the $$I_1$$ intensity|
|`-I .. .. dif_ic2 ..`|$$I_2$$: `I2`|Imports the $$I_2$$ reference intensity|
|`-I .. .. .. x_ch1_roi1..x_ch13_roi1`| $$I_f$$: `If` |Imports the $$I_f$$ fluorescence intensity|
|`-f`|$$\mu(E)$$: `x`|Calculates the experimental fluorescence signal, as $$\frac{I_f}{I_0}$$ |
|`-T`|$$\mu_{ref}(E)$$: `ref`|Calculates the experimental reference transmission signal, as $$\log_{10}{\frac{I_0}{I_1}}$$|
