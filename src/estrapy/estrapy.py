from time import perf_counter_ns

# Get time at the start of the program
_program_start_time = perf_counter_ns()

import argparse  # noqa: E402
import colorlog  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402

from pathlib import Path  # noqa: E402
from lark.exceptions import VisitError, UnexpectedToken  # noqa: E402
from dataclasses import dataclass  # noqa: E402

from . import __version__, __version_tuple__, copyright, banner  # noqa: E402
from .dispatcher import execute_script  # noqa: E402
from .core.grammar.estrapyparser import file_parser  # noqa: E402
from .transformer import EstraTransformer
from .core.context import Context, Paths, Options, ParseContext  # noqa: E402
from .core.errors import CommandError, EstraCommandErrorContextManager  # noqa: E402
from .core.timers import TimerCollection  # noqa: E402

global_LOGGING_LEVEL = logging.INFO

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

LEVELS: list[tuple[str, int, str]] = [
    ('DBG', logging.DEBUG, 'light_black'),
    ('INF', logging.INFO, 'white'),
    ('WRN', logging.WARNING, 'yellow'),
    ('ERR', logging.ERROR, 'red'),
    ('FTL', logging.CRITICAL, 'bold_red'),
]

def init_logging(log_file: Path | None = None, debug: bool = False) -> logging.Logger:
    # Check the logfile folder exists
    if debug:
        log_level = logging.DEBUG
        global global_LOGGING_LEVEL
        global_LOGGING_LEVEL = logging.DEBUG
    else:
        log_level = logging.INFO

    handlers: list[logging.Handler] = []

    for name, level, _ in LEVELS:
        logging.addLevelName(level, name)
    log_colors = {name: color for name, _, color in LEVELS}

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('[%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s[%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors=log_colors,
    )
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)

    logger = logging.getLogger('estrapy')
    logger.setLevel(logging.DEBUG)
    for handler in handlers:
        logger.addHandler(handler)

    return logger

VERSION_RE = re.compile(r'^#\s*version\s*:?\s*(?:(\d+)\.(\d+)(?:\.(\d+))?)\s*$', re.IGNORECASE)
def parse_version_line(line: str) -> tuple[int, ...]:
    """Parse a version line of the form '# version X.Y[.Z]' and return a tuple of integers.
    Raises ValueError if the line is not in the correct format."""
    match = VERSION_RE.match(line)
    if not match:
        raise ValueError(
            "The version line must be in the form '# version X.Y[.Z]'."
        )
    return tuple(int(x) if x is not None else 0 for x in match.groups())

@dataclass(slots=True)
class ArgumentConfig:
    inputfile: str | None
    outputdir: Path | None
    cwd: Path | None
    verbose: bool
    debug: bool
    timings: bool
    vars: dict[str, str]

def parse_args() -> ArgumentConfig:
    parser = argparse.ArgumentParser(
        description='Analyze XAS data files from an instruction file.',
        epilog=copyright,
    )
    parser.add_argument(
        'inputfile',
        type=str,
        nargs='?',
        default=None,
        help='Path to the input .estra file. If not provided, reads from stdin.',
    )
    parser.add_argument(
        '-o',
        '--outputdir',
        type=Path,
        default=None,
        help='Directory where to save the outputs. Defaults to a folder with the same name as the input file, or the current directory if reading from stdin.',
    )
    parser.add_argument(
        '--cwd',
        type=Path,
        default=None,
        help='Set the current working directory for relative paths in the script. Defaults to the actual current working directory.',
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Enable verbose logging (debug level).',
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode. Implies --verbose. Changes some execution behavior for easier debugging (e.g. do not use multiprocessing).',
    )
    parser.add_argument(
        '--timings',
        action='store_true',
        help='Enable detailed timing logging. Implied by --debug.',
    )
    parser.add_argument(
        '--vars',
        type=str,
        nargs='*',
        default=[],
        help='Variables to define in the script, in the form NAME=VALUE. Multiple variables can be defined by repeating this argument.',
    )
    parser.add_argument(
        '--version',
        '-V',
        action='version',
        version=f'EstraPy version {__version__}',
        help='Show the program version and exit.',
    )

    args = parser.parse_args()
    return ArgumentConfig(
        inputfile=args.inputfile,
        outputdir=args.outputdir,
        cwd=args.cwd,
        verbose=args.verbose,
        debug=args.debug,
        timings=args.timings,
        vars=args.vars,
    )


def initialize_context(args: ArgumentConfig, timers: TimerCollection) -> Context:
    # Initialize context with default values
    context = Context(
        paths=Paths(
            inputfile=None,
            currentdir=Path.cwd(),
            workingdir=Path.cwd(),
            outputdir=Path.cwd(),
            logfile=None,
            outfile=None,
        ),
        timers=timers,
        projectname=Path.cwd().name,
        options=Options(
            # If debug flag is set, also set verbose to True
            verbose=args.verbose or args.debug,
            debug=args.debug,
            timings=args.timings or args.debug,
        ),
        logger=logging.getLogger('estrapy'),
        parser=file_parser,
    )

    if args.cwd is not None:
        if args.cwd.exists() and args.cwd.is_dir():
            context.paths.workingdir = args.cwd.resolve()
        else:
            raise FileNotFoundError(f"Current working directory '{args.cwd}' does not exist or is not a directory.")

    # Set input file and projectname in context
    inputfile_str: str | None = args.inputfile

    match inputfile_str:
        case None:
            context.options.interactive = True
            inputfile = None
        case "-":
            import sys
            inputfile = sys.stdin
        case "?":
            # Discover input files in the current working directory and ask the user to select one
            discovered_files = discover_input_files(context.paths.workingdir)
            if not discovered_files:
                raise FileNotFoundError(f'No input files found in directory {context.paths.workingdir} for interactive selection.')
            selected_file = select_input_file([str(f.relative_to(context.paths.workingdir)) for f in discovered_files])
            inputfile = context.paths.workingdir / selected_file
            context.paths.inputfile = inputfile.resolve()
            context.paths.workingdir = inputfile.parent
            context.projectname = inputfile.stem
        case "*":
            raise NotImplementedError('Batch processing of all input files in a directory is not implemented yet.')
        case str(inputfile):
            inputfile = Path(inputfile).resolve()
            if not inputfile.exists() or not inputfile.is_file():
                raise FileNotFoundError(f"Input file '{inputfile}' does not exist or is not a file.")
            context.paths.inputfile = inputfile
            context.paths.workingdir = inputfile.parent
            context.projectname = inputfile.stem

    # Set output directory in context and create it if it doesn't exist
    outputdir: Path | None = args.outputdir
    match outputdir, inputfile, context.options.interactive:
        case None, Path(), False:  # input file provided, no output dir provided
            context.paths.outputdir = inputfile.with_suffix('').resolve()
        case outdir, Path(), False if outdir is not None:
            context.paths.outputdir = outdir.resolve()
        case None, None, True:  # interactive mode, no output dir provided
            context.paths.outputdir = context.paths.workingdir
        case outdir, None, True if outdir is not None:  # interactive mode, output dir provided
            context.paths.outputdir = outdir.resolve()
        case _:
            raise RuntimeError('Unreachable state when setting output directory.')
    context.paths.outputdir.mkdir(parents=True, exist_ok=True)

    # Set variables in context
    for var in args.vars:
        if '=' not in var:
            raise ValueError(f"Variable '{var}' is not in the form NAME=VALUE.")
        name, value = var.split('=', 1)
        context.vars[name] = value

    # Set log and output files in context
    context.paths.logfile = context.paths.outputdir / 'estrapy.log'
    context.paths.outfile = context.paths.outputdir / (context.projectname + '.out')

    return context

def discover_input_files(directory: Path) -> list[Path]:
    """Discover valid input files in the given directory and its subdirectories."""
    # A valid input file is a non-empty file with extension .estra, .inp, .estrapy
    # that starts with a version comment.
    def is_valid_file(path: Path) -> bool:
        if not path.is_file():
            return False
        if path.stat().st_size == 0:
            return False
        if path.suffix.lower() not in ('.estra', '.inp', '.estrapy'):
            return False
        
        # Filter by files that start with a version comment
        try:
            with path.open('r', encoding='utf-8') as f:
                first_line = f.readline()
                return VERSION_RE.match(first_line) is not None
        except Exception:
            return False
    
    return [p for p in directory.rglob('*.*') if is_valid_file(p)]


def select_input_file(files: list[str]) -> str:
    import inquirer # pyright: ignore[reportMissingTypeStubs] # 

    questions = [
        inquirer.List(
            'inputfile',
            message='Multiple input files found. Please select one to process',
            choices=files,
        )
    ]
    answers = inquirer.prompt(questions) # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    if answers is None or 'inputfile' not in answers:
        raise RuntimeError('No input file selected.')
    return str(answers['inputfile']) # pyright: ignore[reportUnknownArgumentType]

def main() -> None:
    timers = TimerCollection()
    timers.start('', already_started_at=_program_start_time)
    timers.stop('imports', already_started_at=_program_start_time)

    args = parse_args()
    context = initialize_context(args, timers)

    # Initialize logging
    context.logger = init_logging(context.paths.logfile, context.options.verbose)
    log = context.logger

    # Log information about the program
    log.info('EstraPy - XAS data analysis tool')
    log.info('(c) 2024 Marco Stecca')
    log.info(f'Version {__version__}')
    log.debug(f"Time to load imports: {(timers["imports"]) / 1e6:.2f} ms")

    # Parse the input file
    match context.options.interactive, context.paths.inputfile:
        case True, _:
            estrapy_interactive_mode(context, timers)
        case False, Path():
            estrapy_file_mode(context, timers)
        case _:
            raise RuntimeError('Unreachable state when determining execution mode.')

def estrapy_interactive_mode(context: Context, timers: TimerCollection) -> None:
    parsecontext = ParseContext(context.paths, timers, context.logger.getChild('parser'), context.parser)
    transformer = EstraTransformer(parsecontext)
    
    previous = ""
    try:    
        while True:
            prompt = '...    ' if previous else ' > '
            line = input(prompt) + "\n"
            if previous:
                line = "    " + line  # Preserve indentation for continued lines
            
            if line.strip() in ("exit", "quit", "q"):
                print('Exiting interactive mode.')
                break

            if line.strip().endswith('\\'):
                previous += line.strip()[:-1] + '\n'
                continue

            
            line = previous + line
            previous = ""

            try:
                parsed_tree = file_parser.parse(line) # pyright: ignore[reportUnknownMemberType]
                script = transformer.transform(parsed_tree)
                execute_script(script, context)
            except UnexpectedToken as e:
                print(f'Syntax error: {e}')
            except CommandParseError as e:
                print(f'Parse error: {e}')
            except VisitError as e:
                print(f'Error during execution: {e}')
    except (KeyboardInterrupt, EOFError):
        print('\nExiting interactive mode.')
            


def estrapy_file_mode(context: Context, timers: TimerCollection) -> None:
    log = context.logger
    with timers.time('parsing'):
        # Lark command parsing gives an error if all commands do not end in \n,
        # so we append \n to the input file so the last command has at least one.
        assert context.paths.inputfile is not None, "Input file must be set in file mode."
        input_file_data = context.paths.inputfile.read_text(encoding='utf-8') + '\n'

        # Check that the input file version is lower or equal to the program version
        first_line = input_file_data.partition('\n')[0]
        file_version = parse_version_line(first_line)
        if file_version > __version_tuple__[: len(file_version)]:
            raise ValueError(
                f"The input file version {'.'.join(map(str, file_version))} is higher than the program version {__version__}. Please update the program."
            )
        
        parsed_tree = file_parser.parse(input_file_data) # pyright: ignore[reportUnknownMemberType]

        # Transform the parse tree into a more manageable structure

        parsecontext = ParseContext(context.paths, timers, context.logger.getChild('parser'), context.parser)

        with EstraCommandErrorContextManager(input_file_data):
            transformer = EstraTransformer(parsecontext)
            script = transformer.transform(parsed_tree)
        
    log.debug(f"Time to parse the input file: {(timers["parsing"]) / 1e6:.2f} ms")

    with timers.time('execution'):
        # Execute the commands in the transformed tree

        execute_script(script, context)
    log.debug(f"Time to execute the script: {(timers["execution"]) / 1e6:.2f} ms")

    # Show the plots if any were created
    if context.plotcontext.nonnumberedfigures or context.plotcontext.numberedfigures:
        for fign, figspec in context.plotcontext.numberedfigures.items():
            from .commands.plot.show import realize_figure
            fig = realize_figure(figspec)
            fig.savefig(context.paths.outputdir / f'figure_{fign}.png', dpi=300) # pyright: ignore[reportUnknownMemberType]
            fig.show()
        for ufign, figspec in enumerate(context.plotcontext.nonnumberedfigures):
            from .commands.plot.show import realize_figure
            fig = realize_figure(figspec)
            fig.savefig(context.paths.outputdir / f'figure_u{ufign}.png', dpi=300) # pyright: ignore[reportUnknownMemberType]
            fig.show()
        from matplotlib import pyplot as plt
        plt.show()
    # End of the program

    timers.stop()
    log.debug(f"Total execution time: {timers.get_ms(""):.2f} ms")

    if context.options.timings:
        log.info('Timing summary:')
        for line in context.timers.table_format('ms').splitlines():
            log.info(line)
    
    # Close all logging handlers
    handlers = context.logger.handlers[:]
    for handler in handlers:
        handler.close()
        context.logger.removeHandler(handler)
    
    # Archive output directory if requested
    if context.options.archive:
        import zipfile
        timestr = "_" + context.starttime.strftime('%Y%m%d_%H%M%S')
        titlestr = ("_" + context.projecttitle.replace(' ', '_')) if context.projecttitle else ""
        archive_path = context.paths.outputdir.parent / f'{context.paths.outputdir.stem}{titlestr}{timestr}.zip'

        # In the root of the zip file, we want the output directory itself, the estra input file
        # and (in a future version) # TODO a copy of the imported data files, as pre-parsed
        # (possibly as a data.h5 / data.parquet or similar)

        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for file in context.paths.outputdir.rglob('*'):
                archive.write(
                    file,
                    arcname=file.relative_to(context.paths.outputdir.parent),
                )
            archive.write(
                context.paths.inputfile,
                arcname=context.paths.inputfile.name,
            )
            for path, archivepath in context.paths.additional_paths.items():
                archive.write(path, arcname=archivepath)


def entry_point() -> None:
    try:
        main()
    except Exception as e:
        if global_LOGGING_LEVEL == logging.DEBUG:
            raise
        else:
            import sys

            print(f'Fatal error [{e.__class__.__name__}]: {e}', file=sys.stderr)
            for note in e.__notes__:
                print(note, file=sys.stderr)
            exit(1)
    except KeyboardInterrupt:
        import sys
        
        print('\nExecution interrupted by user.', file=sys.stderr)
        exit(1)

if __name__ == '__main__':
    entry_point()
