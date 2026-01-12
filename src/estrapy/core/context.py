from dataclasses import dataclass, field
from logging import Logger
from pathlib import Path
from typing import Any, Callable
from datetime import datetime
from lark import Lark

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .timers import TimerCollection
from .datastore import DataStore

from .grammar.axisindexpos import AxisIndexPosition


@dataclass(slots=True)
class Directive: ...

@dataclass(slots=True)
class CommandArguments: ...

@dataclass(slots=True)
class CommandResult: ...


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

    # Additional paths that have been needed during execution, as (source, archive path) tuples.
    additional_paths: dict[Path,Path] = field(default_factory=dict[Path,Path])


@dataclass(slots=True)
class Options:
    # Whether to run in debug mode.
    # verbose -> log debug messages to the log file.
    # debug -> changes some execution behavior for easier debugging (e.g. do not use multiprocessing).
    verbose: bool = False
    debug: bool = False
    timings: bool = False
    archive: bool = False

    # Whether interactive mode is enabled.
    interactive: bool = False


@dataclass(slots=True)
class ParseContext:
    # Paths object containing all relevant paths.
    paths: Paths

    # Timers object containing all relevant timers.
    timers: TimerCollection

    # Logger object for logging.
    logger: Logger

    # Grammar parser
    parser: Lark


# Plotting context
@dataclass(slots=True)
class AxisSpecification:
    pos:  AxisIndexPosition
    callbacks: list[Callable[[Axes, Figure], Any]] = field(default_factory=list[Any])

@dataclass(slots=True)
class FigureSpecification:
    figsize: tuple[float, float] | None = None
    # Index the axes with (row, column) tuples, without the size specified
    axes: dict[tuple[int, int], AxisSpecification] = field(default_factory=dict[tuple[int, int], AxisSpecification])

@dataclass(slots=True)
class PlotContext:
    # Hold all figure settings
    numberedfigures: dict[int, FigureSpecification] = field(default_factory=dict[int, FigureSpecification])
    nonnumberedfigures: list[FigureSpecification] = field(default_factory=list[FigureSpecification])
    

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

    # Parser object for parsing commands
    parser: Lark

    # Project title, set from a directive. Used to comment the zip output.
    projecttitle: str = ''

    # Variables defined in the script.
    vars: dict[str, Any] = field(default_factory=dict[str, Any])

    # Datetime when the program started (used for output, does not change during
    # execution and context creation is accurate enough).
    starttime: datetime = field(default_factory=datetime.now)

    # DataStore object containing all loaded data files.
    datastore: DataStore = field(default_factory=DataStore)

    # Results of the executed commands
    results: dict[str, CommandResult] = field(default_factory=dict[str, CommandResult])

    # overwrite __repr__ to avoid printing the entire context in debug logs
    def __repr__(self) -> str:
        return f'Context(projectname={self.projectname}, paths={self.paths}, options={self.options}, ...)'

    # Plotting context
    plotcontext: PlotContext = field(default_factory=PlotContext)


from typing import NamedTuple
from lark import Token, Tree
from dataclasses import dataclass
from typing import Self, Generic, TypeVar

from .context import ParseContext, Context

_A = TypeVar('_A', bound=CommandArguments, covariant=True)
_R = TypeVar('_R', bound=CommandResult, covariant=True)

@dataclass(slots=True)
class Command(Generic[_A, _R]):
    line: int
    name: str
    args: _A
    outname : str | None = None

    @classmethod
    def parse(cls, commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext) -> Self: ...

    def execute(self, context: Context) -> _R: ...


class Script(NamedTuple):
    directives: list[Directive]
    commands: list[Command[CommandArguments, CommandResult]]
