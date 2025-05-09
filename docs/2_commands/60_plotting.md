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
|<div class="nowrap">`data` `<column>`</div>|Selects the data to plot. The format is outlined in the [column](#column) section.|
|<div class="nowrap">`--xlabel` `<text>`</div>|Sets the x-axis label. You can use $$\LaTeX$$ formatting with \$text\$|
|<div class="nowrap">`--ylabel` `<text>`</div>|Sets the y-axis label. You can use $$\LaTeX$$ formatting with \$text\$|
|<div class="nowrap">`--title` `<text>`</div>|Sets the title of the plot. You can use $$\LaTeX$$ formatting with \$text\$|
|<div class="nowrap">`--xlim` `<range>`</div>|Sets the x plotting range. If `..`, calculates the limits based on the data.|
|<div class="nowrap">`--ylim` `<range>`</div>|Sets the y plotting range. If `..`, calculates the limits based on the data.|
|<div class="nowrap">`--colorby` `<value>`</div>|Chooses a variable to color the data by.|
|<div class="nowrap">`--figure` `<figure>`</div>|Selects a specific figure to plot within. You can specify the figure number and the subplot position as such: `3:1.2` will plot the data in the figure number 3, in the subplot at position 1,2. By default creates a new figure.|
|<div class="nowrap">`--color` `<color>`</div>|Colors the spectra with the given color or colormap. The colormaps are defined by matplotlib. You can specify a linear colormap as a sequence of colors, such as `red blue`.|
|<div class="nowrap">`--alpha` `<value>`</div>|Specify the opacity of the plots. Must be between 0 (transparent) and 1(fully visible). Default is 1.|
|<div class="nowrap">`--linewidth` `<value>`</div>|Specify the width of the spectra. Default is 1.|
|<div class="nowrap">`--show`</div>|If set, immediately shows the selected figure, and waits for it to close. By default, all figures are shown at the end of the program.|

## Column

To select a column, you can specify x:y, where x is a valid axis and y any valid column.
