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

## FileIn

```sh
filein <file> [options]
```

The `batchin` command sets the default arguments for the file input, and takes the same optional arguments as the `filein` command.

|Argument|Explanation|
|--|--|
|<span class="nowrap">`file`</span>|The path that points to the file. Can be either an absolute path, or a relative path from the working directory. Glob patterns are supported.|
|<span class="nowrap">`--dir` `<directory>`</span>|Uses this directory to import the files from, if the path is relative, instead of the current working directory.|
|<span class="nowrap">`--batch` / `-b`</span>|Signals to the `filein` command to use the arguments defined by all previous `batchin` commands.|
|<span class="nowrap">`--xaxiscolumn` / `-x` `<column>` </span>|Sets the column containing the x axis. By default, it is equivalent to `--index <column>`. To modify the behavior, you can either use one of the other axis flags, or use e.g. `-x <column> -E`, equivalent to `-E <column>`|
|<span class="nowrap">`--index` `[column]`</span>|Imports the given axis column as a raw index. If no column is specified, uses the column given by `--xaxiscolumn`.|
|<span class="nowrap">`--energy` / `-E` `[column]`</span>|Imports the given axis column as energy in electronvolts. If no column is specified, uses the column given by `--xaxiscolumn`.|
|<span class="nowrap">`--kvector` / `-k` `[column]`</span>|Imports the given axis column as a k wavevector. If no column is specified, uses the column given by `--xaxiscolumn`.|
|<span class="nowrap">`--rdistance` / `-R` `[column]`</span>|Imports the given axis column as a distance in Angstroms. If no column is specified, uses the column given by `--xaxiscolumn`.|
|<span class="nowrap">`--qvector` / `-q` `[column]`</span>|Imports the given axis column as a q wavevector. If no column is specified, uses the column given by `--xaxiscolumn`.|
|<span class="nowrap">`--shift` `<float>`</span>|Shifts the x axis index by the specified amount.|
|<span class="nowrap">`--intensities` / `-I` `<column> <column> <column> <column>`</span>|Imports the specified columns as intensities, importing between 1 and 4 columns. They are, in order, $$I_0$$, $$I_1$$, $$I_2$$, $$I_f$$, imported as the columns `I0` `I1` `I2` `If`. If `--intensities` is not specified, the columns are inferred from the other signal options.|
|<span class="nowrap">`--transmission` / `-t` `[column] [column]`</span>|Calculates the signal $$\mu_{exp}$$ as $$\log_{10}{\dfrac{I_0}{I_1}}$$, importing it as the column `x`.|
|<span class="nowrap">`--fluorescence` / `-f` `[column] [column]`</span>|Calculates the signal $$\mu_{exp}$$ as $$\dfrac{I_f}{I_0}$$, importing it as the column `x`.|
|<span class="nowrap">`--intensity` / `-i` `<column>`</span>|Reads the signal $$\mu_{exp}$$ directly, importing it as the column `x`.|
|<span class="nowrap">`--reftransmittance` / `-T` `[column] [column]`</span>|Calculates the reference signal $$\mu_{ref}$$ as $$\log_{10}{\dfrac{I_1}{I_2}}$$, importing it as the column `ref`.|
|<span class="nowrap">`--refabsorption` / `-A` `<column>`</span>|Imports the reference signal $$\mu_{ref}$$ directly, importing it as the column `ref`.|
|<span class="nowrap">`--var` `<name> <value>`</span>|Adds the given variable to the file metadata, useful in other commands. The automatic variable definition is defined in the [metadata](#metadata) section. The value can be an auto-defined variable, such as .h1.1, wich will be resolved to the imported value.|

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
|<span class="nowrap">`-I dif_ic0 .. .. ..`</span>|$$I_0$$: `I0`|Imports the $$I_0$$ intensity|
|<span class="nowrap">`-I .. dif_ic1 .. ..`</span>|$$I_1$$: `I1`|Imports the $$I_1$$ intensity|
|<span class="nowrap">`-I .. .. dif_ic2 ..`</span>|$$I_2$$: `I2`|Imports the $$I_2$$ reference intensity|
|<span class="nowrap">`-I .. .. .. x_ch1_roi1..x_ch13_roi1`</span>| $$I_f$$: `If` |Imports the $$I_f$$ fluorescence intensity|
|<span class="nowrap">`-f`|$$\mu(E)$$: `x`</span>|Calculates the experimental fluorescence signal, as $$\dfrac{I_f}{I_0}$$ |
|<span class="nowrap">`-T`|$$\mu_{ref}(E)$$: `ref`</span>|Calculates the experimental reference transmission signal, as $$\log_{10}{\dfrac{I_0}{I_1}}$$|
