import colorlog
import logging
import os
import sys

from argparse import ArgumentParser
from pathlib import Path
from tomllib import loads as toml_loads

from . import __version__

from .commands._context import Paths, Context, Options
from .parser import parse_version, parse_directives, parse_commands


def main():
    parser = ArgumentParser(
        "estrapy", description="Analyze XAS data files from an instruction file."
    )

    parser.add_argument("inputfile", nargs="?", help="The instruction file.")
    parser.add_argument(
        "outputdir", nargs="?", help="The output directory. If empty, creates one."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Makes the output more complete."
    )
    parser.add_argument(
        "--legacy", "-l", help="Runs the legacy version of the program."
    )
    parser.add_argument(
        "--version", "-V", action="version", version=f"EstraPy {__version__}"
    )
    parser.add_argument(
        "--no-color", action="store_true", help="Removes color from the log."
    )
    parser.add_argument(
        "--no-update",
        action="store_true",
        help="Does not update the input file if an old version is detected.",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Does not read or store from the cache."
    )

    args = parser.parse_args()

    # --- Set context ----------------------------------------------------------

    currentdir = Path(os.getcwd())

    userdir = Path.home() / ".estrapy"
    userdir.mkdir(exist_ok=True)

    configfile = userdir / "config.toml"
    configfile.touch(exist_ok=True)

    match args.inputfile, args.outputdir:
        case ("-", _o):
            # Interactive
            raise NotImplementedError("Interactive mode is not implemented.")

        case (None, None):
            # Search the current directory for input files
            raise NotImplementedError("Automatic discovery mode is not implemented.")

        case (_i, None):
            inputfile = Path(_i)
            outputdir = inputfile.parent / inputfile.name.removesuffix(inputfile.suffix)

        case (None, _o):
            raise RuntimeError("Impossible input combination.")

        case (_i, _o):
            inputfile = Path(_i)
            outputdir = Path(_o)

    if not inputfile.exists():
        raise FileNotFoundError("The specified file does not exist.")

    outputdir.mkdir(exist_ok=True)
    logfile = outputdir / (inputfile.name.removesuffix(inputfile.suffix) + ".log")

    paths = Paths(
        inputfile,
        currentdir,
        inputfile.parent,
        outputdir,
        userdir,
        logfile,
        configfile,
    )
    config = toml_loads(configfile.read_text("utf-8"))

    inputfiledata = inputfile.read_text("utf-8")
    directives = parse_directives(inputfiledata)

    options = Options(
        False,
        args.verbose,
        parse_version(inputfiledata),
        not args.no_color,
        not args.no_update,
        not args.no_cache,
        {},
    )
    context = Context(paths, config, options, directives)

    # --- Logging --------------------------------------------------------------

    rootlogger = colorlog.getLogger()
    rootlogger.setLevel(logging.DEBUG)

    # Rename all levels to align in the log file
    [
        logging.addLevelName(v, n)
        for n, v in (("DBG", 10), ("INF", 20), ("WRN", 30), ("ERR", 40), ("FTL", 50))
    ]
    log_colors = {
        "DBG": "light_black",
        "INF": "white",
        "WRN": "yellow",
        "ERR": "red",
        "FTL": "bold_red",
    }

    fmt = logging.Formatter("[%(levelname)s] <%(name)s> %(message)s")
    flh = logging.FileHandler(logfile, encoding="utf-8")
    flh.setLevel(logging.DEBUG)
    flh.setFormatter(fmt)
    rootlogger.addHandler(flh)

    if options.logcolor:
        cfm = colorlog.ColoredFormatter(
            "%(log_color)s[%(levelname)s] <%(name)s> %(message)s", log_colors=log_colors
        )
        sth = colorlog.StreamHandler(sys.stdout)
        sth.setLevel(logging.DEBUG if options.verbose else logging.WARNING)
        sth.setFormatter(cfm)
    else:
        sth = logging.StreamHandler(sys.stdout)
        sth.setLevel(logging.DEBUG if options.verbose else logging.WARNING)
        sth.setFormatter(fmt)
    rootlogger.addHandler(sth)

    log = logging.getLogger("estrapy")

    # --------------------------------------------------------------------------

    log.info(f" EstraPy version {__version__} ".center(64, "-"))
    log.debug(f"Input file is {paths.inputfile}")
    log.debug(f"Working directory is {paths.workdir}")
    log.debug(f"Output directory is {paths.outputdir}")

    # --- Parse input file -----------------------------------------------------

    commands = parse_commands(inputfiledata, context)

    for executor, commandargs in commands:
        res = executor.execute(commandargs, context)

    pass
