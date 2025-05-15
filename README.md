# EstraPy

EstraPy is a synchrotron analysis tool, built in Python 3.
It uses input files, written in a simple to read syntax, to perform analysis of synchrotron data files. The program supports the import, main data handling, plotting and output data.
EstraPy is run by the command line, and executes the commands one by one.

To download the latest release of EstraPy, navigate to the [releases](https://github.com/ramsteak/EstraPy/releases) section of the [repository](https://github.com/ramsteak/EstraPy/) and download the latest release. Install with your python package manager, such as:

```sh
py -m pip install estrapy-1.0.0-py3-none-any.whl
```

Alternatively, you clone the github repository and build from source:

```sh
gh repo clone ramsteak/EstraPy
python -m build
```

To test if the installation was successful, run

```sh
$ estrapy --version

EstraPy 1.0.0
```

## About the project

EstraPy is &copy; 2024-{{ "now" | date: "%Y" }} by [Marco Stecca](https://github.com/ramsteak).

### License

EstraPy is distributed by an [MIT license](https://github.com/ramsteak/EstraPy/blob/main/LICENSE).

### Contributing

To report issues with the program, you can create a new [issue](https://github.com/ramsteak/EstraPy/issues) in the github section. Be sure to upload both your input file and the relevant data files, along with the output log file.
