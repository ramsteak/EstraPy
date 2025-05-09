---
title: File saving
parent: Commands
nav_order: 7
permalink: /commands/file-saving
math: katex
---

# Saving

The `save` command exports the data from the program to files on disk. The command has a few options for file saving.

```sh
save <filename> <mode> [--options]
```

The file saving modes are outlined below.

## Batch

```sh
save <filename> batch <column>
```

|Argument|Explanation|
|--|--|
|`column`|The column is specified in the same manner as for plotting, specifying the x:y columns to be selected.|

The result of this command is a file, with the specified name at the location relative to the output directory. The file will contain pairs of columns, two for each file, named `x_file1`, `y_file1`, `x_file2`, `y_file2`.

## Aligned

```sh
save <filename> aligned <column> --align <range>
```

|Argument|Explanation|
|--|--|
|<span class="nowrap">`column`</span>|The column is specified in the same manner as for plotting, specifying the x:y columns to be selected.|
|<span class="nowrap">`--align` / `-a` `range`</span>|Contains the interpolation output range, with spacing. The bounds cannot be `..`, but `:.` and `.:` are allowed.|

This command is similar to the `save batch` command, but instead of having pairs of columns, the required data is interpolated over the given range and spacing. The output file has the given x range as the first column, and the other columns are the interpolated y values. The header of each y column is the file name.

## Columns

```sh
save <filename> columns <column> [column, ...]
```

|Argument|Explanation|
|--|--|
|`column`|The column is specified in the same manner as for plotting, specifying the x:y columns to be selected.|

This command performs one export per datum, yielding one file each. It aggregates the specified columns in a single file. To save each file with a different file name you can use variables, including them in braces. For example, the filename `{.fn}_norm.dat` will save each file with its original filename (`{.fn}`), appending `_norm.dat`, and `out_{.n}.xy` will output each file with in order of import (`{.n}`). In the case of filename collisions, the filename will be modified by appending a progressive number after the extension. The collision is only checked within one call to `save columns`, so successive calls might overwrite eachother.
