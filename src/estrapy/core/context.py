from dataclasses import dataclass, field
from logging import Logger
from pathlib import Path
from typing import Any

from .timers import TimerCollection
from .datastore import DataStore


@dataclass(slots=True)
class Paths:
    # inputfile is None if not provided.
    # If None, reads from stdin.
    inputfile: Path | None

    # Directory where the outputs will be. Has the same name as the input file.
    # If inputfile is None, outputdir is the current directory.
    outputdir: Path

    # Current directory when the program is started.
    currentdir: Path

    # Working directory. All relative paths are relative to this.
    workingdir: Path

    # Log file path. If None, only logs to console.
    logfile: Path | None

    # Output file path. Contains the output of the program as a report.
    outfile: Path | None


@dataclass(slots=True)
class Options:
    # Whether to run in debug mode.
    # verbose -> log debug messages to the log file.
    # debug -> changes some execution behavior for easier debugging (e.g. do not use multiprocessing).
    verbose: bool = False
    debug: bool = False

    # Whether interactive mode is enabled.
    interactive: bool = False


@dataclass(slots=True)
class Context:
    # Paths object containing all relevant paths.
    paths: Paths

    # Project name, set from the input file name. If inputfile is None, projectname is the current directory name.
    projectname: str

    # Timers object containing all relevant timers.
    timers: TimerCollection

    # Options object containing all relevant options.
    options: Options

    # Logger object for logging.
    logger: Logger

    # Variables defined in the script.
    vars: dict[str, Any] = field(default_factory=dict[str, Any])

    # DataStore object containing all loaded data files.
    datastore: DataStore = field(default_factory=DataStore)


@dataclass(slots=True)
class ParseContext:
    # Logger object for logging.
    logger: Logger
