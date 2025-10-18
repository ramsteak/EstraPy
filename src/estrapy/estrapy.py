from time import perf_counter_ns

# Get time at the start of the program
_program_start_time = perf_counter_ns()

import argparse  # noqa: E402
import colorlog  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402

from io import StringIO  # noqa: E402
from pathlib import Path  # noqa: E402
from lark.exceptions import VisitError  # noqa: E402

from . import __version__, __version_tuple__  # noqa: E402
from .grammar import file_parser, transformer  # noqa: E402
from .core.context import Context, Paths, Options  # noqa: E402

from .core.timers import TimerCollection  # noqa: E402
from .dispatcher import execute_script  # noqa: E402

from .core.errors import ParseError  # noqa: E402

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

LEVELS: list[tuple[str, int, str]] = [
    ("DBG", logging.DEBUG, "light_black"),
    ("INF", logging.INFO, "white"),
    ("WRN", logging.WARNING, "yellow"),
    ("ERR", logging.ERROR, "red"),
    ("FTL", logging.CRITICAL, "bold_red"),
]


def init_logging(log_file: Path | None = None, debug: bool = False) -> logging.Logger:
    # Check the logfile folder exists
    log_level = logging.DEBUG if debug else logging.INFO

    handlers: list[logging.Handler] = []

    for name, level, _ in LEVELS:
        logging.addLevelName(level, name)
    log_colors = {name: color for name, _, color in LEVELS}

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            "[%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s[%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors=log_colors,
    )
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)

    logger = logging.getLogger('estrapy')
    logger.setLevel(log_level)
    for handler in handlers:
        logger.addHandler(handler)

    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze XAS data files from an instruction file.",
        epilog="(c) 2024 Marco Stecca",
    )
    parser.add_argument(
        "inputfile",
        type=Path,
        nargs="?",
        default=None,
        help="Path to the input .estra file. If not provided, reads from stdin.",
    )
    parser.add_argument(
        "-o",
        "--outputdir",
        type=Path,
        default=None,
        help="Directory where to save the outputs. Defaults to a folder with the same name as the input file, or the current directory if reading from stdin.",
    )
    parser.add_argument(
        '--cwd',
        type=Path,
        default=None,
        help="Set the current working directory for relative paths in the script. Defaults to the actual current working directory.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (debug level).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode. Implies --verbose. Changes some execution behavior for easier debugging (e.g. do not use multiprocessing).",
    )
    parser.add_argument(
        "--vars",
        type=str,
        nargs="*",
        default=[],
        help="Variables to define in the script, in the form NAME=VALUE. Multiple variables can be defined by repeating this argument.",
    )

    return parser.parse_args()


def initialize_context(args: argparse.Namespace, timers: TimerCollection) -> Context:
    # Initialize context with default values
    context = Context(
        paths=Paths(
            inputfile=None,
            currentdir=Path.cwd(),
            workingdir=Path.cwd(),
            outputdir=Path.cwd(),
            logfile=None,
            outfile=StringIO(),
        ),
        timers=timers,
        projectname=Path.cwd().name,
        options=Options(
            # If debug flag is set, also set verbose to True
            verbose=args.verbose or args.debug,
            debug = args.debug
        ),
        logger=logging.getLogger("estrapy"),
    )

    if args.cwd is not None:
        if args.cwd.exists() and args.cwd.is_dir():
            context.paths.workingdir = args.cwd.resolve()
        else:
            raise FileNotFoundError(
                f"Current working directory '{args.cwd}' does not exist or is not a directory."
            )

    # Set input file and projectname in context
    inputfile: Path | None = args.inputfile
    if inputfile is not None:
        inputfile = inputfile.resolve()
        if inputfile.exists() and inputfile.is_file():
            context.paths.inputfile = inputfile
            context.paths.workingdir = inputfile.parent
            context.projectname = inputfile.stem
        else:
            raise FileNotFoundError(
                f"Input file '{args.inputfile}' does not exist or is not a file."
            )
        context.options.interactive = False
    else:
        context.projectname = context.paths.workingdir.name
        context.options.interactive = True

    # Set output directory in context and create it if it doesn't exist
    outputdir: Path | None = args.outputdir
    match outputdir, inputfile, context.options.interactive:
        case None, Path(), False:  # input file provided, no output dir provided
            context.paths.outputdir = inputfile.with_suffix("").resolve()
        case outdir, Path(), False:
            context.paths.outputdir = outdir.resolve()
        case None, None, True:  # interactive mode, no output dir provided
            context.paths.outputdir = context.paths.workingdir
        case outdir, None, True:  # interactive mode, output dir provided
            context.paths.outputdir = outdir.resolve()
        case _:
            raise RuntimeError("Unreachable state when setting output directory.")
    context.paths.outputdir.mkdir(parents=True, exist_ok=True)

    # Set variables in context
    for var in args.vars:
        if "=" not in var:
            raise ValueError(f"Variable '{var}' is not in the form NAME=VALUE.")
        name, value = var.split("=", 1)
        context.vars[name] = value

    # Set log and output files in context
    context.paths.logfile = context.paths.outputdir / "estrapy.log"
    context.paths.outfile = context.paths.outputdir / (context.projectname + ".out")

    # Create or clear the output file
    if context.paths.outfile is not None:  # type: ignore
        context.paths.outfile.touch(exist_ok=True)
        context.paths.outfile.write_text("")

    return context


def main() -> None:
    timers = TimerCollection()
    timers.start("", already_started_at=_program_start_time)
    timers.stop("imports", already_started_at=_program_start_time)

    args = parse_args()
    context = initialize_context(args, timers)

    # Initialize logging
    context.logger = init_logging(context.paths.logfile, context.options.verbose)
    log = context.logger

    # Log information about the program

    log.info("EstraPy - XAS data analysis tool")
    log.info("(c) 2024 Marco Stecca")
    log.info(f"Version {__version__}")
    log.debug(f"Time to load imports: {(timers["imports"]) / 1e6:.2f} ms")

    # Parse the input file
    assert (
        context.paths.inputfile is not None\
    ), "Interactive mode (no input file) is not implemented yet."

    with timers.time("parsing"):
        # Lark command parsing gives an error if all commands do not end in \n,
        # so we append \n to the input file so the last command has at least one.
        input_file_data = context.paths.inputfile.read_text() + "\n"

        # Check that the input file version is lower or equal to the program version
        first_line = input_file_data.partition("\n")[0]
        version_match = re.match(
            r"^#\s*version\s*:?\s*(?:(\d+)\.(\d+)(?:\.(\d+))?)\s*$",
            first_line,
            re.IGNORECASE,
        )
        if not version_match:
            raise ValueError(
                "The first line of the input file must specify the version in the form '# version X.Y[.Z]'."
            )

        file_version = tuple(
            int(x) if x is not None else 0 for x in version_match.groups()
        )
        if file_version > __version_tuple__[: len(file_version)]:
            raise ValueError(
                f"The input file version {'.'.join(map(str, file_version))} is higher than the program version {__version__}. Please update the program."
            )

        parsed_tree = file_parser.parse(input_file_data)  # type: ignore

        # Transform the parse tree into a more manageable structure
        try:
            t_tree = transformer.transform(parsed_tree)
        except VisitError as ve:
        # On transformation error, check if it's a CommandSyntaxError and re-raise it with input file context
            if isinstance(ve.orig_exc, ParseError):
                token = ve.orig_exc.token
                # If the token is None, raise the original exception
                if token is None:
                    raise ve.orig_exc from ve

                # Otherwise, log the error with some context from the input file
                line = token.line
                col, endcol = token.column, token.end_column
                if line is None:
                    raise ve.orig_exc from ve
                if col is None:
                    raise ve.orig_exc from ve
                if endcol is None:
                    endcol = col + 1

                all_lines = input_file_data.splitlines()
                message_lines:list[str] = []
                message_lines.append(f"Error in line {line}, column {col}:")
                message_lines.extend(all_lines[max(0, line-5) : line])
                message_lines.append(' ' * (col-1) + '^' * (endcol - col) + f"\n\n{ve.orig_exc}")

                raise ParseError("\n".join(message_lines), token) from ve
            raise ve from ve
    log.debug(f"Time to parse the input file: {(timers["parsing"]) / 1e6:.2f} ms")

    with timers.time("execution"):
        # Execute the commands in the transformed tree

        execute_script(t_tree, context)
    log.debug(f"Time to execute the script: {(timers["execution"]) / 1e6:.2f} ms")

    # End of the program

    timers.stop()
    log.debug(f"Total execution time: {timers.get_ms(""):.2f} ms")

    for line in context.timers.table_format("ms").splitlines():
        log.debug(line)


def entry_point() -> None:
    try:
        main()
    except Exception as e:
        if logging.getLogger('estrapy').isEnabledFor(logging.DEBUG):
            import traceback
            traceback.print_exc()
        else:
            import sys
            print(f"Fatal error: {e}", file=sys.stderr)
            exit(1)


if __name__ == "__main__":
    entry_point()
