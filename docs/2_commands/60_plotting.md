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
|<span class="nowrap">`data` `<column>`</span>|Selects the data to plot. The format is outlined in the [column](#column) section.|
|<span class="nowrap">`--xlabel` `<text>`</span>|Sets the x-axis label. You can use $$\LaTeX$$ formatting with \$text\$|
|<span class="nowrap">`--ylabel` `<text>`</span>|Sets the y-axis label. You can use $$\LaTeX$$ formatting with \$text\$|
|<span class="nowrap">`--title` `<text>`</span>|Sets the title of the plot. You can use $$\LaTeX$$ formatting with \$text\$|
|<span class="nowrap">`--xlim` `<range>`</span>|Sets the x plotting range. If `..`, calculates the limits based on the data.|
|<span class="nowrap">`--ylim` `<range>`</span>|Sets the y plotting range. If `..`, calculates the limits based on the data.|
|<span class="nowrap">`--colorby` `<value>`</span>|Chooses a variable to color the data by.|
|<span class="nowrap">`--figure` `<figure>`</span>|Selects a specific figure to plot within. You can specify the figure number and the subplot position as such: `3:1.2` will plot the data in the figure number 3, in the subplot at position 1,2. By default creates a new figure.|
|<span class="nowrap">`--color` `<color>`</span>|Colors the spectra with the given color or colormap. The colormaps are defined by matplotlib. You can specify a linear colormap as a sequence of colors, such as `red blue`.|
|<span class="nowrap">`--alpha` `<value>`</span>|Specify the opacity of the plots. Must be between 0 (transparent) and 1(fully visible). Default is 1.|
|<span class="nowrap">`--linewidth` `<value>`</span>|Specify the width of the spectra. Default is 1.|
|<span class="nowrap">`--show`</span>|If set, immediately shows the selected figure, and waits for it to close. By default, all figures are shown at the end of the program.|

## Column

To select a column, you can specify x:y, where x is a valid axis and y any valid column.
