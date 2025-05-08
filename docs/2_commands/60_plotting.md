---
title: Plotting
parent: Commands
nav_order: 6
permalink: /commands/plotting
math: katex
---

# Plotting

```sh
plot [data] [--options]
```

|Argument|Explanation|
|--|--|
|`data` `<column>`|Selects the data to plot. The format is outlined in the [column](#column) section.|
|`--xlabel` `<text>`|Sets the x-axis label. You can use $$\LaTeX$$ formatting with \$text\$|
|`--ylabel` `<text>`|Sets the y-axis label. You can use $$\LaTeX$$ formatting with \$text\$|
|`--title` `<text>`|Sets the title of the plot. You can use $$\LaTeX$$ formatting with \$text\$|
|`--xlim` `<range>`|Sets the x plotting range. If `..`, calculates the limits based on the data.|
|`--ylim` `<range>`|Sets the y plotting range. If `..`, calculates the limits based on the data.|
|`--colorby` `<value>`|Chooses a variable to color the data by.|
|`--figure` `<figure>`|Selects a specific figure to plot within. You can specify the figure number and the subplot position as such: `3:1.2` will plot the data in the figure number 3, in the subplot at position 1,2. By default creates a new figure.|
|`--color` `<color>`|Colors the spectra with the given color or colormap. The colormaps are defined by matplotlib. You can specify a linear colormap as a sequence of colors, such as `red blue`.|
|`--alpha` `<value>`|Specify the opacity of the plots. Must be between 0 (transparent) and 1(fully visible). Default is 1.|
|`--linewidth` `<value>`|Specify the width of the spectra. Default is 1.|
|`--show`|If set, immediately shows the selected figure, and waits for it to close. By default, all figures are shown at the end of the program.|

## Column

To select a column, you can specify x:y, where x is a valid axis and y any valid column.
