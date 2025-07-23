---
title: Averaging
parent: Commands
nav_order: 50
permalink: /commands/averaging
math: katex
---

# Averaging

The average command averages multiple spectra together. If one or a series of variables is given, the data is divided into groups, and the average is calculated for each group. The group name is given by the sequence of variables, separated by "_".

```sh
average [--options]
```

|Argument|Explanation|
|--|--|
|<span class="nowrap">`--by` `<variable>`</span>|The variables to average by. The name of the averaged data is given by the sequence of variable values, separated by "_".|
|<span class="nowrap">`--axis` / `-a` `<variable>`</span>|The axis to interpolate onto. By default, uses the default axis of the datum.|

Each column is interpolated to the axis of the first file, and is then averaged together. Furthermore, the resulting data page shows the standard deviation and variance across files, onto columns starting by `s` and `v`.
