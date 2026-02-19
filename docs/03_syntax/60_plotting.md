---
title: Plotting
parent: Commands
nav_order: 60
permalink: /commands/plotting
math: katex
---

# Plotting

```sh
plot [data] [--options]
```

|Argument|Explanation|
|--|--|
|<span class="nowrap">`data` `<column>`</span>|Selects the data to plot. The format is outlined in the [column](#column) section.|
|<span class="nowrap">`--xlabel` `<text>`</span>|Sets the x-axis label. You can use $$\LaTeX$$ formatting with \$text\$|
|<span class="nowrap">`--ylabel` `<text>`</span>|Sets the y-axis label. You can use $$\LaTeX$$ formatting with \$text\$|
|<span class="nowrap">`--title` `<text>`</span>|Sets the title of the plot. You can use $$\LaTeX$$ formatting with \$text\$|
|<span class="nowrap">`--xlim` `<range>`</span>|Sets the x plotting range.|
|<span class="nowrap">`--ylim` `<range>`</span>|Sets the y plotting range.|
|<span class="nowrap">`--vshift` `<value>`</span>|Draws each consecutive plot vertically shifted by the given amount, relative to the previous. Default is 0, does not shift.|
|<span class="nowrap">`--colorby` `<value>`</span>|Chooses a variable to color the data by, such as `--colorby .fn` or `--colorby E0`. See the [metadata]( {{"/commands/file-input#Metadata" | relative_url }}) section for the default variable names.|
|<span class="nowrap">`--figure` `<figure>`</span>|Selects a specific figure to plot within. You can specify the figure number and the subplot position as such: `3:1.2` will plot the data in the figure number 3, in the subplot at position 1,2. By default creates a new figure.|
|<span class="nowrap">`--color` `<color>`</span>|Colors the spectra with the given color or colormap. The colormap names are defined by [matplotlib](https://matplotlib.org/stable/users/explain/colors/colormaps.html). You can specify a linear colormap as a sequence of colors, such as `--color red yellow green blue`.|
|<span class="nowrap">`--alpha` `<value>`</span>|Specify the opacity of the plots. Must be between 0 (transparent) and 1(fully visible). Default is 1.|
|<span class="nowrap">`--show`</span>|If set, immediately shows the selected figure, and waits for it to close. By default, all figures are shown at the end of the program.|
|<span class="nowrap">`--linewidth` `<value>`</span>|Specify the width of the lines. Default is 1.|
|<span class="nowrap">`--xxthick`</span>|Convenience method to set the line width. Equivalent to `--linewidth 8`|
|<span class="nowrap">`--xthick`</span>|Convenience method to set the line width. Equivalent to `--linewidth 4`|
|<span class="nowrap">`--thick`</span>|Convenience method to set the line width. Equivalent to `--linewidth 2`|
|<span class="nowrap">`--thin`</span>|Convenience method to set the line width. Equivalent to `--linewidth 0.5`|
|<span class="nowrap">`--xthin`</span>|Convenience method to set the line width. Equivalent to `--linewidth 0.25`|
|<span class="nowrap">`--xxthin`</span>|Convenience method to set the line width. Equivalent to `--linewidth 0.125`|
|<span class="nowrap">`--linestyle` `<value>`</span>|Specify the line style of the spectra. Default is `solid`. Can be either a named style(`solid`,`dotted`,`dashed`,`dashdot`) or a sequence of numbers separated by `.`, indicating the length of each drawn section.|
|<span class="nowrap">`--solid`</span>|Convenience method to set the line style. Equivalent to `--linestyle solid`|
|<span class="nowrap">`--dotted`</span>|Convenience method to set the line style. Equivalent to `--linestyle dotted`|
|<span class="nowrap">`--dashed`</span>|Convenience method to set the line style. Equivalent to `--linestyle dashed`|
|<span class="nowrap">`--dashdot`</span>|Convenience method to set the line style. Equivalent to `--linestyle dashdot`|

## Column

When plotting, you can specify the data columns and apply transformations directly using a concise syntax. The basic syntax is as follows:

```sh
axis:column
```

where `axis` represents the x axis, such as `E`, `e`, `k` or `R`, and `column` represents the y values, such as `a`, `mu`, `x`, `pre` and `I0`.

### Transformations

You can apply operations to both the x and y axes by chaining operators with a dot (`.`). Each side of the `:` (x-axis and y-axis) supports independent transformation chains.

To specify y operations it is necessary to include the x axis.

|Operator|Requires argument|x-axis|y-axis|Explanation|
|--|:--:|:--:|:--:|--|
|`r`|<span class="text-red-200">&#10007;</span>|<span class="text-green-000">&#10003;</span>|<span class="text-green-000">&#10003;</span>|Takes the real part of the column.|
|`i`|<span class="text-red-200">&#10007;</span>|<span class="text-green-000">&#10003;</span>|<span class="text-green-000">&#10003;</span>|Takes the imaginary part of the column.|
|`a`|<span class="text-red-200">&#10007;</span>|<span class="text-green-000">&#10003;</span>|<span class="text-green-000">&#10003;</span>|Takes the absolute value of the column.|
|`p`|<span class="text-red-200">&#10007;</span>|<span class="text-green-000">&#10003;</span>|<span class="text-green-000">&#10003;</span>|Calculates the complex phase of the column.|
|`s`|<span class="text-green-000">&#10003;</span>|<span class="text-red-200">&#10007;</span>|<span class="text-green-000">&#10003;</span>|Smooths the column. The number identifies the window width.|
|`d`|<span class="text-green-000">&#10003;</span>|<span class="text-red-200">&#10007;</span>|<span class="text-green-000">&#10003;</span>|Calculates the n-th derivative of the column, with respect to the x axis.|
|`k`|<span class="text-green-000">&#10003;</span>|<span class="text-red-200">&#10007;</span>|<span class="text-green-000">&#10003;</span>|Weighs the y axis by the x axis with the specified power.|
|`w`|<span class="text-green-000">&#10003;</span>|<span class="text-red-200">&#10007;</span>|<span class="text-green-000">&#10003;</span>|Subtracts the constant value of 1, then weighs the y axis by the x axis, then adds 1.|
|`W`|<span class="text-green-000">&#10003;</span>|<span class="text-red-200">&#10007;</span>|<span class="text-green-000">&#10003;</span>|Subtracts the average of the data, then weighs the y axis by the x axis, then adds back the average.|

### Examples

|Example|Explanation|
|--|--|
|<span class="nowrap">`plot E:d1.x`</span>|Plots the first derivative of $$\chi(E)$$.|
|<span class="nowrap">`plot k:k2.x`</span>|Plots the data weighed by $$k^{2}$$: $$k^{2}\cdot\chi(k)$$|
|<span class="nowrap">`plot R:a.f`</span>|Plots the absolute value of the fourier transform against the R axis.|
|<span class="nowrap">`plot r.f:a.f`</span>|Plots the imaginary value of the fourier transform against its real value.|
