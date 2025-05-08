---
title: File input
parent: Commands
nav_order: 2
---

# FileIn

File input in EstraPy is done with the `filein` command, and optionally with the `batchin` command.
The command reads from a data file, importing the selected columns and performing the necessary calculations.
It also handles metadata and additional variables.

## Usage

```sh
filein <file> --arguments
```

The `batchin` command sets the default arguments for the file input, and takes the same optional arguments as the `filein` command.

## Example

For the palladium example, the file input would look like this:

```sh
batchin -E energyc -I dif_ic0 dif_ic1 dif_ic2 x_ch1_roi1..x_ch13_roi1
filein data/*.xy -b -f -T
```

This snippet reads all the .xy files from the data folder. It then imports the following columns:

|Argument|Resulting column||
|--|--|--|
|-I dif_ic0 .. .. ..|$I_0$|Imports the $I_0$ intensity|
|-I .. dif_ic1 .. ..|$I_1$|Imports the $I_1$ intensity|
|-I .. .. dif_ic2 ..|$I_2$|Imports the $I_2$ reference intensity|
|-I .. .. .. x_ch1_roi1..x_ch13_roi1| $I_f$ |Imports the $I_f$ fluorescence intensity|
|-f|$\mu(E)$|Calculates the experimental fluorescence signal, as $\frac{I_f}{I_0}$ |
|-T|$\mu_{ref}(E)$|Calculates the experimental reference transmission signal, as $\log_{10}\frac{I_0}{I_1}$|
