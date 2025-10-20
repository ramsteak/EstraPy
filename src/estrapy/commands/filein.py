import re

from dataclasses import dataclass, field
from io import StringIO
from itertools import zip_longest
from pathlib import Path
from lark import Tree, Token
from typing import TypeAlias, Sequence, Callable
from types import EllipsisType
from tqdm.std import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from os import cpu_count

from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core.errors import CommandSyntaxError, ExecutionError
from ..core.grammarclasses import CommandArguments, Command, CommandMetadata
from ..core.number import Number, parse_number, Unit
from ..core.context import Context, ParseContext
from ..core.datastore import FileMetadata, DataDomain, Domain, Column, ColumnType, DataFile
from ..core.misc import peek, Bag, fmt

import pandas as pd
import numpy as np

Column_id: TypeAlias = str | int
ColumnRange: TypeAlias = tuple[Column_id, Column_id]
# A column descriptor can be a single column, a range of columns, or a list of either
# 1..3,7..9 -> [(1,3),(7,9)]
# A,C,E     -> ['A','C','E']
# 1,3,5..7  -> [1,3,(5,7)]
# 1..5      -> [(1,5)]
# A         -> ['A']
ColumnDescriptor: TypeAlias = list[Column_id | ColumnRange]

# An imported column is calculated from other columns. The first element is a
# tuple of imported column names to be used to calculate it, and the second is
# the function to use for the calculation.


@dataclass(slots=True)
class ImporterOptions:
    separator: str | None = None
    comment_prefix: str = '#'
    skip_rows: int = 0
    decimal: str | None = None

    re_sep: re.Pattern[str] | None = None


Expr: TypeAlias = Callable[[pd.DataFrame], pd.Series]


@dataclass(slots=True)
class CommandArguments_filein(CommandArguments):
    filenames: list[str]
    directory: Path

    # Variable to sort the data by
    sortby: str

    # Variables to be defined on the imported data
    vars: dict[str, str | Number | int] = field(default_factory=dict[str, str | Number | int])

    # Options for the importer
    importeroptions: ImporterOptions = field(default_factory=ImporterOptions)

    # What columns are to be imported. The tuple stores the new column name as
    # first element and the data column name (or range of columns) as second element.
    columns: list[tuple[Column, Domain, ColumnDescriptor]] = field(
        default_factory=list[tuple[Column, Domain, ColumnDescriptor]]
    )

    # Signals to be calculated from the imported columns. The
    signals: list[tuple[Column, Domain, Expr]] = field(default_factory=list[tuple[Column, Domain, Expr]])


@dataclass(slots=True)
class FileInOptions:
    # ------------------------- File importing options -------------------------
    # <filenames> [<filenames> ...]
    filenames: list[str] = field(default_factory=list[str])
    # --dir <directory>
    directory: Token | EllipsisType = ...
    # ------------------------ Column specification options ------------------------
    # # ---------------------- Axis columns ----------------------
    reciprocal_axis_flag__: bool = False
    reciprocal_intensity_flag__: bool = False
    # --energy <column> | -E <column>
    energy: Token | EllipsisType = ...
    # --wavevector <column> | -k <column>
    wavevector: Token | EllipsisType = ...
    # --rspace <column> | -r <column>
    rspace: Token | EllipsisType = ...
    # --qvector <column> | -q <column>
    qvector: Token | EllipsisType = ...
    # # ------------------------ Data columns ----------------------
    # # ----------------------- Raw intensities -----------------------
    # --beamintensity <columns> | --I0 <columns>
    reciprocal_sample_signal_mode__: str | None = None
    reciprocal_reference_signal_mode__: str | None = None
    beamintensity: Token | EllipsisType = ...
    # --sampleintensity <columns> | --I1 <columns>
    sampleintensity: Token | EllipsisType = ...
    # --samplefluorescence <columns> | --If <columns>
    samplefluorescence: Token | EllipsisType = ...
    # --referenceintensity <columns> | --I2 <columns>
    referenceintensity: Token | EllipsisType = ...
    # --xanes <columns> | --mu <columns>
    xanes: Token | EllipsisType = ...
    # --exafs <columns> | --chi <columns>
    exafs: Token | EllipsisType = ...
    # --fouriermagnitude <columns> | --fm <columns>
    fouriermagnitude: Token | EllipsisType = ...
    # --fourierphase <columns> | --fp <columns>
    fourierphase: Token | EllipsisType = ...
    # --fourierreal <columns> | --fr <columns>
    fourierreal: Token | EllipsisType = ...
    # --fourierimaginary <columns> | --fi <columns>
    fourierimaginary: Token | EllipsisType = ...
    # # # ---------------------- Compound columns ----------------------
    # --intensities <columns> [<columns> ...] | -I <columns> [<columns> ...]
    # Up to 4 columns can be specified, in order I0, I1, I2, If. To skip, use a dash (-).
    # # directly sets beamintensity, sampleintensity, referenceintensity, samplefluorescence
    # --fluorescence <columns> <column> | -f <columns> <column> | --fluorescence / -f (can be a flag)
    # fluorescence is calculated as samplefluorescence / referenceintensity
    fluorescence: Token | EllipsisType = ...
    # --transmission <columns> <column> | -t <columns> <column> | --transmission / -t (can be a flag)
    # transmission is calculated as log(beamintensity / sampleintensity)
    transmission: Token | EllipsisType = ...
    # --referencetransmission <columns> <column> | --rt <columns> <column> | --referencetransmission / -T (can be a flag)
    # referencetransmission is calculated as log(referenceintensity / sampleintensity)
    referencetransmission: Token | EllipsisType = ...
    # --fourier <columns> <columns> | --f <columns> <columns> automatic detection of type (real/imaginary or magnitude/phase)
    # if the first column contains only positive values, it is assumed to be magnitude/phase, otherwise real/imaginary
    fourier: list[Token] | EllipsisType = ...
    # # ---------------------- Error columns ----------------------
    # --beamintensityerror <columns> | --sI0 <columns>
    beamintensityerror: Token | EllipsisType = ...
    # --sampleintensityerror <columns> | --sI1 <columns>
    sampleintensityerror: Token | EllipsisType = ...
    # --samplefluorescenceerror <columns> | --sIf <columns>
    samplefluorescenceerror: Token | EllipsisType = ...
    # --referenceintensityerror <columns> | --sI2 <columns>
    referenceintensityerror: Token | EllipsisType = ...
    # --xaneserror <columns> | --smu <columns>
    xaneserror: Token | EllipsisType = ...
    # --exafserror <columns> | --schi <columns>
    exafserror: Token | EllipsisType = ...
    # --fouriermagnitudeerror <columns> | --sfm
    fouriermagnitudeerror: Token | EllipsisType = ...
    # --fourierphaseerror <columns> | --sfp
    fourierphaseerror: Token | EllipsisType = ...
    # --fourierrealerror <columns> | --sfr
    fourierrealerror: Token | EllipsisType = ...
    # --fourierimaginaryerror <columns> | --sfi
    fourierimaginaryerror: Token | EllipsisType = ...
    # # ---------------------- Other columns ----------------------
    # --shift <value> | -s <value>
    shift: Number | EllipsisType = ...
    # --var <name> <value>
    vars: dict[str, str | Number | int] = field(default_factory=dict[str, str | Number | int])

    # ---------------------- Importer options ----------------------
    importeroptions = ImporterOptions()

    sortby: str | EllipsisType = ...  # Default sort by filename (.fn)

    # TODO: check if all required options are used and actually set and checked in the code


def _assert_option_not_assigned(options: FileInOptions, option: str, token: Token) -> bool:
    if getattr(options, option) is ...:
        return True
    raise CommandSyntaxError(f'The {token.value} option can only be specified once.', token)


def _assert_option_required(options: FileInOptions, option: str, token: Token) -> bool:
    if getattr(options, option) is not ...:
        return True
    raise CommandSyntaxError(f'The {token.value} option requires {option}.', token)


def _assert_option_not_assigned_set(
    options: FileInOptions, option: str, optiontoken: Token, valuetoken: Token | list[Token]
):
    if getattr(options, option) is not ...:
        raise CommandSyntaxError(f'The {optiontoken.value} option can only be specified once.', optiontoken)
    setattr(options, option, valuetoken)


def _assert_flag_not_assigned_set(options: FileInOptions, option: str, optiontoken: Token, value: bool | str = True):
    if bool(getattr(options, option)) is not False:
        raise CommandSyntaxError(f'The {optiontoken.value} option can only be specified once.', optiontoken)
    setattr(options, option, value)


def parse_filein_command(
    cmd: Token, args: Sequence[Token | Tree[Token]], parsecontext: ParseContext
) -> Command[CommandArguments_filein]:
    linenumber = cmd.line if cmd.line else 0
    options = get_filein_options(args, parsecontext)
    commandargs = get_filein_command(linenumber, options, parsecontext)
    metadata = CommandMetadata(chainable=False, requires_global_context=True, cpu_bound=False)
    return Command[CommandArguments_filein](linenumber, 'filein', commandargs, metadata)


def parse_column_descriptor(t: Token) -> ColumnDescriptor:
    s = str(t.value)
    desc: ColumnDescriptor = []
    for part in s.split(','):
        match part.split('..'):
            case [str(one)]:
                desc.append(int(one) if one.isdigit() else one)
            case [str(start), str(end)]:
                desc.append((int(start) if start.isdigit() else start, int(end) if end.isdigit() else end))
            case _:
                raise CommandSyntaxError('Invalid column descriptor.', t)
    return desc


def expand_column_descriptor(desc: ColumnDescriptor, columns: list[str]) -> list[str]:
    res: list[str] = []
    for item in desc:
        match item:
            case str(column):
                res.append(column)
            case int(index):
                res.append(columns[index])
            case [str(start), str(stop)]:
                res.extend((start, stop))
            case [str(start), int(stop)]:
                res.extend((start, columns[stop]))
            case [int(start), str(stop)]:
                res.extend((columns[start], stop))
            case [int(start), int(stop)]:
                res.extend((columns[start], columns[stop]))
    return res


def get_filein_command(line: int, options: FileInOptions, parsecontext: ParseContext) -> CommandArguments_filein:
    cmd = CommandArguments_filein(
        filenames=options.filenames,
        # Check if the directory is absolute. If not, make it relative to the working directory.
        directory=Path(options.directory.value) if options.directory is not ... else parsecontext.paths.workingdir,
        sortby=options.sortby if options.sortby is not ... else '.fn',
        vars=options.vars,
        importeroptions=options.importeroptions,
    )

    # Only one reciprocal axis can be specified, thanks to reciprocal_axis_flag__
    if options.energy is not ...:
        columndescriptor = parse_column_descriptor(options.energy)
        column = Column(name='E', unit=Unit.EV, type=ColumnType.AXIS)
        cmd.columns.append((column, Domain.RECIPROCAL, columndescriptor))
    elif options.wavevector is not ...:
        columndescriptor = parse_column_descriptor(options.wavevector)
        column = Column(name='k', unit=Unit.K, type=ColumnType.AXIS)
        cmd.columns.append((column, Domain.RECIPROCAL, columndescriptor))
    elif options.qvector is not ...:
        columndescriptor = parse_column_descriptor(options.qvector)
        column = Column(name='q', unit=Unit.K, type=ColumnType.AXIS)
        cmd.columns.append((column, Domain.RECIPROCAL, columndescriptor))

    if options.rspace is not ...:
        columndescriptor = parse_column_descriptor(options.rspace)
        column = Column(name='R', unit=Unit.A, type=ColumnType.AXIS)
        cmd.columns.append((column, Domain.FOURIER, columndescriptor))

    # Check reciprocal domain columns
    if options.beamintensity is not ...:
        columndescriptor = parse_column_descriptor(options.beamintensity)
        column = Column(name='I0', unit=None, type=ColumnType.DATA)
        cmd.columns.append((column, Domain.RECIPROCAL, columndescriptor))
    if options.sampleintensity is not ...:
        columndescriptor = parse_column_descriptor(options.sampleintensity)
        column = Column(name='I1', unit=None, type=ColumnType.DATA)
        cmd.columns.append((column, Domain.RECIPROCAL, columndescriptor))
    if options.samplefluorescence is not ...:
        columndescriptor = parse_column_descriptor(options.samplefluorescence)
        column = Column(name='If', unit=None, type=ColumnType.DATA)
        cmd.columns.append((column, Domain.RECIPROCAL, columndescriptor))
    if options.referenceintensity is not ...:
        columndescriptor = parse_column_descriptor(options.referenceintensity)
        column = Column(name='I2', unit=None, type=ColumnType.DATA)
        cmd.columns.append((column, Domain.RECIPROCAL, columndescriptor))

    if options.beamintensityerror is not ...:
        columndescriptor = parse_column_descriptor(options.beamintensityerror)
        column = Column(name='sI0', unit=None, type=ColumnType.ERROR)
        cmd.columns.append((column, Domain.RECIPROCAL, columndescriptor))
    if options.sampleintensityerror is not ...:
        columndescriptor = parse_column_descriptor(options.sampleintensityerror)
        column = Column(name='sI1', unit=None, type=ColumnType.ERROR)
        cmd.columns.append((column, Domain.RECIPROCAL, columndescriptor))
    if options.samplefluorescenceerror is not ...:
        columndescriptor = parse_column_descriptor(options.samplefluorescenceerror)
        column = Column(name='sIf', unit=None, type=ColumnType.ERROR)
        cmd.columns.append((column, Domain.RECIPROCAL, columndescriptor))
    if options.referenceintensityerror is not ...:
        columndescriptor = parse_column_descriptor(options.referenceintensityerror)
        column = Column(name='sI2', unit=None, type=ColumnType.ERROR)
        cmd.columns.append((column, Domain.RECIPROCAL, columndescriptor))

    if options.xanes is not ...:
        columndescriptor = parse_column_descriptor(options.xanes)
        column = Column(name='mu', unit=None, type=ColumnType.DATA)
        cmd.columns.append((column, Domain.RECIPROCAL, columndescriptor))
    if options.exafs is not ...:
        columndescriptor = parse_column_descriptor(options.exafs)
        column = Column(name='chi', unit=None, type=ColumnType.DATA)
        cmd.columns.append((column, Domain.RECIPROCAL, columndescriptor))

    if options.xaneserror is not ...:
        columndescriptor = parse_column_descriptor(options.xaneserror)
        column = Column(name='mu', unit=None, type=ColumnType.ERROR)
        cmd.columns.append((column, Domain.RECIPROCAL, columndescriptor))
    if options.exafserror is not ...:
        columndescriptor = parse_column_descriptor(options.exafserror)
        column = Column(name='chi', unit=None, type=ColumnType.ERROR)
        cmd.columns.append((column, Domain.RECIPROCAL, columndescriptor))

    if options.fourierreal is not ... and options.fourierimaginary is not ...:
        columndescriptor = parse_column_descriptor(options.fourierreal)
        column = Column(name='fourierreal', unit=None, type=ColumnType.TEMP)
        cmd.columns.append((column, Domain.FOURIER, columndescriptor))
        columndescriptor = parse_column_descriptor(options.fourierimaginary)
        column = Column(name='fourierimaginary', unit=None, type=ColumnType.TEMP)
        cmd.columns.append((column, Domain.FOURIER, columndescriptor))

    if options.fouriermagnitude is not ...:
        columndescriptor = parse_column_descriptor(options.fouriermagnitude)
        column = Column(name='fouriermagnitude', unit=None, type=ColumnType.TEMP)
        cmd.columns.append((column, Domain.FOURIER, columndescriptor))
    if options.fourierphase is not ...:
        columndescriptor = parse_column_descriptor(options.fourierphase)
        column = Column(name='fourierphase', unit=None, type=ColumnType.TEMP)
        cmd.columns.append((column, Domain.FOURIER, columndescriptor))

    if options.fourierrealerror is not ...:
        columndescriptor = parse_column_descriptor(options.fourierrealerror)
        column = Column(name='fourierrealerror', unit=None, type=ColumnType.TEMP)
        cmd.columns.append((column, Domain.FOURIER, columndescriptor))
    if options.fourierimaginaryerror is not ...:
        columndescriptor = parse_column_descriptor(options.fourierimaginaryerror)
        column = Column(name='fourierimaginaryerror', unit=None, type=ColumnType.TEMP)
        cmd.columns.append((column, Domain.FOURIER, columndescriptor))

    if options.fouriermagnitudeerror is not ...:
        columndescriptor = parse_column_descriptor(options.fouriermagnitudeerror)
        column = Column(name='fouriermagnitudeerror', unit=None, type=ColumnType.TEMP)
        cmd.columns.append((column, Domain.FOURIER, columndescriptor))
    if options.fourierphaseerror is not ...:
        columndescriptor = parse_column_descriptor(options.fourierphaseerror)
        column = Column(name='fourierphaseerror', unit=None, type=ColumnType.TEMP)
        cmd.columns.append((column, Domain.FOURIER, columndescriptor))

    # Setup all signals and relative functions
    match options.reciprocal_sample_signal_mode__:
        case 'calc_fluorescence':
            importer: Expr = lambda df: (df['If'] / df['I0']).rename('a')  # noqa: E731
            column = Column(name='a', unit=None, type=ColumnType.DATA, calc=importer)
            cmd.signals.append((column, Domain.RECIPROCAL, importer))
        case 'raw_fluorescence':
            assert options.fluorescence is not ..., 'Invalid program state: #MPylXTU7xl'
            columndescriptor = parse_column_descriptor(options.fluorescence)
            column = Column(name='a', unit=None, type=ColumnType.DATA)
            cmd.columns.append((column, Domain.RECIPROCAL, columndescriptor))
        case 'calc_transmission':
            importer: Expr = lambda df: (np.log10(df['I0'] / df['I1'])).rename('a')  # type: ignore  # noqa: E731
            column = Column(name='a', unit=None, type=ColumnType.DATA, calc=importer)
            cmd.signals.append((column, Domain.RECIPROCAL, importer))
        case 'raw_transmission':
            assert options.transmission is not ..., 'Invalid program state: #qGBvcHeGhS'
            columndescriptor = parse_column_descriptor(options.transmission)
            column = Column(name='a', unit=None, type=ColumnType.DATA)
            cmd.columns.append((column, Domain.RECIPROCAL, columndescriptor))
        case _:
            assert False, 'Invalid program state: #k5Ay58oM6V'

    match options.reciprocal_reference_signal_mode__:
        case 'calc_referencetransmission':
            importer: Expr = lambda df: (np.log10(df['I1'] / df['I2'])).rename('ref')  # type: ignore  # noqa: E731
            column = Column(name='ref', unit=None, type=ColumnType.DATA, calc=importer)
            cmd.signals.append((column, Domain.RECIPROCAL, importer))
        case 'raw_referencetransmission':
            assert options.referencetransmission is not ..., 'Invalid program state: #Xvlkkp7Wyz'
            columndescriptor = parse_column_descriptor(options.referencetransmission)
            column = Column(name='ref', unit=None, type=ColumnType.DATA)
            cmd.columns.append((column, Domain.RECIPROCAL, columndescriptor))
        case _:
            assert False, 'Invalid program state: #Cxcis9JxTS'

    # Add complex number recovery (from real/imaginary or magnitude/phase) if fourier columns are specified
    if options.fourierreal is not ... and options.fourierimaginary is not ...:
        # real/imaginary specified as pure columns. Calculate f = real + i*imaginary
        importer: Expr = lambda df: (df['fourierreal'] + 1j * df['fourierimaginary']).rename('f')  # noqa: E731
        column = Column(name='fourier', unit=None, type=ColumnType.DATA, calc=importer)
        cmd.signals.append((column, Domain.FOURIER, importer))
    elif options.fouriermagnitude is not ... and options.fourierphase is not ...:
        # magnitude/phase specified as pure columns. Calculate f = magnitude * (cos(phase) + i * sin(phase))
        # sin/cos is faster than exp
        importer: Expr = lambda df: (
            df['fouriermagnitude'] * (np.cos(df['fourierphase']) + 1j * np.sin(df['fourierphase']))
        ).rename('f')  # noqa: E731
        column = Column(name='fourier', unit=None, type=ColumnType.DATA, calc=importer)
        cmd.signals.append((column, Domain.FOURIER, importer))

    # TODO: fourier errors

    # Compile importer options
    if cmd.importeroptions.separator is None:
        cmd.importeroptions.separator = r'\s+'
    else:
        cmd.importeroptions.re_sep = re.compile(cmd.importeroptions.separator)
        cmd.importeroptions.separator = (
            ' '  # If a custom separator is given, the regex will be used to edit the separator to a single space
        )

    if cmd.importeroptions.decimal is None:
        cmd.importeroptions.decimal = '.'
    return cmd


def get_filein_options(args: Sequence[Token | Tree[Token]], parsecontext: ParseContext) -> FileInOptions:
    options = FileInOptions()
    for arg in args:
        match arg:
            case Token('STRING', str(path)):
                options.filenames.append(path)
            case Tree('option', [Token('OPTION', '--dir') as t, Token('STRING', str()) as o]):
                _assert_option_not_assigned_set(options, 'directory', t, o)
            case Tree(
                'option', [Token('OPTION', '--energy' | '-E') as t, Token('STRING' | 'INTEGER', str() | int()) as o]
            ):
                _assert_flag_not_assigned_set(options, 'reciprocal_axis_flag__', t)
                _assert_option_not_assigned_set(options, 'energy', t, o)
            case Tree(
                'option', [Token('OPTION', '--wavevector' | '-k') as t, Token('STRING' | 'INTEGER', str() | int()) as o]
            ):
                _assert_flag_not_assigned_set(options, 'reciprocal_axis_flag__', t)
                _assert_option_not_assigned_set(options, 'wavevector', t, o)
            case Tree(
                'option', [Token('OPTION', '--rspace' | '-r') as t, Token('STRING' | 'INTEGER', str() | int()) as o]
            ):
                _assert_option_not_assigned_set(options, 'rspace', t, o)
            case Tree(
                'option', [Token('OPTION', '--qvector' | '-q') as t, Token('STRING' | 'INTEGER', str() | int()) as o]
            ):
                _assert_flag_not_assigned_set(options, 'reciprocal_axis_flag__', t)
                _assert_option_not_assigned_set(options, 'qvector', t, o)
            case Tree(
                'option',
                [Token('OPTION', '--beamintensity' | '--I0') as t, Token('STRING' | 'INTEGER', str() | int()) as o],
            ):
                _assert_option_not_assigned_set(options, 'beamintensity', t, o)
            case Tree(
                'option',
                [Token('OPTION', '--sampleintensity' | '--I1') as t, Token('STRING' | 'INTEGER', str() | int()) as o],
            ):
                _assert_option_not_assigned_set(options, 'sampleintensity', t, o)
            case Tree(
                'option',
                [
                    Token('OPTION', '--samplefluorescence' | '--If') as t,
                    Token('STRING' | 'INTEGER', str() | int()) as o,
                ],
            ):
                _assert_option_not_assigned_set(options, 'samplefluorescence', t, o)
            case Tree(
                'option',
                [
                    Token('OPTION', '--referenceintensity' | '--I2') as t,
                    Token('STRING' | 'INTEGER', str() | int()) as o,
                ],
            ):
                _assert_option_not_assigned_set(options, 'referenceintensity', t, o)
            case Tree('option', [Token('OPTION', '--fluorescence' | '-f') as t]):
                # No columns specified -> use as flag to import as fluorescence
                _assert_flag_not_assigned_set(options, 'reciprocal_sample_signal_mode__', t, 'calc_fluorescence')
            case Tree(
                'option',
                [Token('OPTION', '--fluorescence' | '-f') as t, Token('STRING' | 'INTEGER', str() | int()) as o],
            ):
                # Only one column is set -> import as fluorescence signal (not intensity)
                _assert_flag_not_assigned_set(options, 'reciprocal_sample_signal_mode__', t, 'raw_fluorescence')
                _assert_option_not_assigned_set(options, 'fluorescence', t, o)
            case Tree(
                'option',
                [
                    Token('OPTION', '--fluorescence' | '-f') as t,
                    Token('STRING' | 'INTEGER', str() | int()) as o1,
                    Token('STRING' | 'INTEGER', str() | int()) as o2,
                ],
            ):
                # Two columns are set -> import as fluorescence and beam intensities
                _assert_flag_not_assigned_set(options, 'reciprocal_sample_signal_mode__', t, 'calc_fluorescence')
                _assert_option_not_assigned_set(options, 'samplefluorescence', t, o1)
                _assert_option_not_assigned_set(options, 'beamintensity', t, o2)
            case Tree('option', [Token('OPTION', '--transmission' | '-t') as t]):
                # No columns specified -> use as flag to import as transmission
                _assert_flag_not_assigned_set(options, 'reciprocal_sample_signal_mode__', t, 'calc_transmission')
            case Tree(
                'option',
                [Token('OPTION', '--transmission' | '-t') as t, Token('STRING' | 'INTEGER', str() | int()) as o],
            ):
                # One column -> import as transmission signal (not intensity)
                _assert_flag_not_assigned_set(options, 'reciprocal_sample_signal_mode__', t, 'raw_transmission')
                _assert_option_not_assigned_set(options, 'transmission', t, o)
            case Tree(
                'option',
                [
                    Token('OPTION', '--transmission' | '-t') as t,
                    Token('STRING' | 'INTEGER', str() | int()) as o1,
                    Token('STRING' | 'INTEGER', str() | int()) as o2,
                ],
            ):
                # Two columns -> import as beam and sample intensities
                _assert_flag_not_assigned_set(options, 'reciprocal_sample_signal_mode__', t, 'calc_transmission')
                _assert_option_not_assigned_set(options, 'beamintensity', t, o1)
                _assert_option_not_assigned_set(options, 'sampleintensity', t, o2)
            case Tree('option', [Token('OPTION', '--referencetransmission' | '--rt' | '-T') as t]):
                # No columns specified -> use as flag to import as referencetransmission
                _assert_flag_not_assigned_set(
                    options, 'reciprocal_reference_signal_mode__', t, 'calc_referencetransmission'
                )
            case Tree(
                'option',
                [
                    Token('OPTION', '--referencetransmission' | '--rt' | '-T') as t,
                    Token('STRING' | 'INTEGER', str() | int()) as o,
                ],
            ):
                # One column -> import as referencetransmission signal (not intensity)
                _assert_flag_not_assigned_set(
                    options, 'reciprocal_reference_signal_mode__', t, 'raw_referencetransmission'
                )
                _assert_option_not_assigned_set(options, 'referencetransmission', t, [o])
            case Tree(
                'option',
                [
                    Token('OPTION', '--referencetransmission' | '--rt' | '-T') as t,
                    Token('STRING' | 'INTEGER', str() | int()) as o1,
                    Token('STRING' | 'INTEGER', str() | int()) as o2,
                ],
            ):
                # Two columns -> import as reference and sample intensities
                _assert_flag_not_assigned_set(
                    options, 'reciprocal_reference_signal_mode__', t, 'calc_referencetransmission'
                )
                _assert_option_not_assigned_set(options, 'referenceintensity', t, o1)
                _assert_option_not_assigned_set(options, 'sampleintensity', t, o2)
            case Tree('option', [Token('OPTION', '--intensities' | '-I') as t, *opts]):
                if (not all(isinstance(o, Token) and o.type in ('STRING', 'INTEGER') for o in opts)) or len(opts) > 4:
                    raise CommandSyntaxError('The -I/--intensities option requires up to 4 string arguments.', t)
                # Filter only STRING tokens to make pylance happy
                opts = [o for o in opts if isinstance(o, Token) and o.type in ('STRING', 'INTEGER')]
                # Up to 4 columns can be specified, in order I0, I1, I2, If.
                # To skip one, use a dash (-).
                # Check if each column to be assigned (not a dash) has not been assigned yet
                for o, name in zip(
                    opts, ['beamintensity', 'sampleintensity', 'referenceintensity', 'samplefluorescence']
                ):
                    if o.value == '-':
                        continue
                    _assert_option_not_assigned_set(options, name, t, o)
            case Tree(
                'option', [Token('OPTION', '--xanes' | '--mu') as t, Token('STRING' | 'INTEGER', str() | int()) as o]
            ):
                _assert_option_not_assigned_set(options, 'xanes', t, o)
            case Tree(
                'option', [Token('OPTION', '--exafs' | '--chi') as t, Token('STRING' | 'INTEGER', str() | int()) as o]
            ):
                _assert_option_not_assigned_set(options, 'exafs', t, o)

            case Tree(
                'option',
                [Token('OPTION', '--fouriermagnitude' | '--fm') as t, Token('STRING' | 'INTEGER', str() | int()) as o],
            ):
                _assert_option_not_assigned_set(options, 'fouriermagnitude', t, o)
            case Tree(
                'option',
                [Token('OPTION', '--fourierphase' | '--fp') as t, Token('STRING' | 'INTEGER', str() | int()) as o],
            ):
                _assert_option_not_assigned_set(options, 'fourierphase', t, o)
            case Tree(
                'option',
                [Token('OPTION', '--fourierreal' | '--fr') as t, Token('STRING' | 'INTEGER', str() | int()) as o],
            ):
                _assert_option_not_assigned_set(options, 'fourierreal', t, o)
            case Tree(
                'option',
                [Token('OPTION', '--fourierimaginary' | '--fi') as t, Token('STRING' | 'INTEGER', str() | int()) as o],
            ):
                _assert_option_not_assigned_set(options, 'fourierimaginary', t, o)
            case Tree(
                'option',
                [
                    Token('OPTION', '--fouriercartesian' | '--fc') as t,
                    Token('STRING' | 'INTEGER', str() | int()) as o1,
                    Token('STRING' | 'INTEGER', str() | int()) as o2,
                ],
            ):
                _assert_option_not_assigned(options, 'fouriermagnitude', t)
                _assert_option_not_assigned(options, 'fourierphase', t)
                _assert_option_not_assigned_set(options, 'fourierreal', t, o1)
                _assert_option_not_assigned_set(options, 'fourierimaginary', t, o2)
            case Tree(
                'option',
                [
                    Token('OPTION', '--fouriereulerian' | '--fe') as t,
                    Token('STRING' | 'INTEGER', str() | int()) as o1,
                    Token('STRING' | 'INTEGER', str() | int()) as o2,
                ],
            ):
                _assert_option_not_assigned(options, 'fourierreal', t)
                _assert_option_not_assigned(options, 'fourierimaginary', t)
                _assert_option_not_assigned_set(options, 'fouriermagnitude', t, o1)
                _assert_option_not_assigned_set(options, 'fourierphase', t, o2)
            case Tree(
                'option',
                [
                    Token('OPTION', '--fourier' | '--f') as t,
                    Token('STRING' | 'INTEGER', str() | int()) as o1,
                    Token('STRING' | 'INTEGER', str() | int()) as o2,
                ],
            ):
                _assert_option_not_assigned_set(options, 'fourier', t, [o1, o2])
            case Tree(
                'option',
                [
                    Token('OPTION', '--beamintensityerror' | '--sI0') as t,
                    Token('STRING' | 'INTEGER', str() | int()) as o,
                ],
            ):
                _assert_option_not_assigned_set(options, 'beamintensityerror', t, o)
            case Tree(
                'option',
                [
                    Token('OPTION', '--sampleintensityerror' | '--sI1') as t,
                    Token('STRING' | 'INTEGER', str() | int()) as o,
                ],
            ):
                _assert_option_not_assigned_set(options, 'sampleintensityerror', t, o)
            case Tree(
                'option',
                [
                    Token('OPTION', '--samplefluorescenceerror' | '--sIf') as t,
                    Token('STRING' | 'INTEGER', str() | int()) as o,
                ],
            ):
                _assert_option_not_assigned_set(options, 'samplefluorescenceerror', t, o)
            case Tree(
                'option',
                [
                    Token('OPTION', '--referenceintensityerror' | '--sI2') as t,
                    Token('STRING' | 'INTEGER', str() | int()) as o,
                ],
            ):
                _assert_option_not_assigned_set(options, 'referenceintensityerror', t, o)
            case Tree(
                'option',
                [Token('OPTION', '--xaneserror' | '--smu') as t, Token('STRING' | 'INTEGER', str() | int()) as o],
            ):
                _assert_option_not_assigned_set(options, 'xaneserror', t, o)
            case Tree(
                'option',
                [Token('OPTION', '--exafserror' | '--schi') as t, Token('STRING' | 'INTEGER', str() | int()) as o],
            ):
                _assert_option_not_assigned_set(options, 'exafserror', t, o)
            case Tree(
                'option',
                [
                    Token('OPTION', '--fouriermagnitudeerror' | '--sfm') as t,
                    Token('STRING' | 'INTEGER', str() | int()) as o,
                ],
            ):
                _assert_option_not_assigned_set(options, 'fouriermagnitudeerror', t, o)
            case Tree(
                'option',
                [
                    Token('OPTION', '--fourierphaseerror' | '--sfp') as t,
                    Token('STRING' | 'INTEGER', str() | int()) as o,
                ],
            ):
                _assert_option_not_assigned_set(options, 'fourierphaseerror', t, o)
            case Tree(
                'option',
                [Token('OPTION', '--fourierrealerror' | '--sfr') as t, Token('STRING' | 'INTEGER', str() | int()) as o],
            ):
                _assert_option_not_assigned_set(options, 'fourierrealerror', t, o)
            case Tree(
                'option',
                [
                    Token('OPTION', '--fourierimaginaryerror' | '--sfi') as t,
                    Token('STRING' | 'INTEGER', str() | int()) as o,
                ],
            ):
                _assert_option_not_assigned_set(options, 'fourierimaginaryerror', t, o)
            case Tree('option', [Token('OPTION', '--shift') as t, Token('FLOAT', str()) as o]):
                try:
                    shift = parse_number(o.value)
                except ValueError:
                    raise CommandSyntaxError('The shift option requires a numeric argument.', o)
                # Shift can only have eV as unit
                if shift.unit is None:
                    shift = Number(shift.value, Unit.EV)
                elif shift.unit != Unit.EV:
                    raise CommandSyntaxError('The shift option only supports eV as unit.', o)
                _assert_option_not_assigned_set(options, 'shift', t, o)
            case Tree('option', [Token('OPTION', '--var') as t, Token('STRING', str(name)) as n, Token('STRING') as v]):
                if name in options.vars:
                    raise CommandSyntaxError(f"The variable '{name}' has already been defined.", n)
                options.vars[name] = str(v.value)
            case Tree(
                'option', [Token('OPTION', '--var') as t, Token('STRING', str(name)) as n, Token('INTEGER') as v]
            ):
                if name in options.vars:
                    raise CommandSyntaxError(f"The variable '{name}' has already been defined.", n)
                options.vars[name] = int(v.value)
            case Tree('option', [Token('OPTION', '--var') as t, Token('STRING', str(name)) as n, Token('NUMBER') as v]):
                if name in options.vars:
                    raise CommandSyntaxError(f"The variable '{name}' has already been defined.", n)
                options.vars[name] = parse_number(v.value)
            case Tree('option', [Token('OPTION', '--sortby') as t, Token('STRING', str(name)) as n]):
                _assert_option_not_assigned_set(options, 'sortby', t, n.value)
            case Tree('option', [Token('OPTION', '--format'), Token('STRING', str(kind)) as k, Token() as v]):
                match kind.lower(), v:
                    case 'decimal' | 'dec', Token('STRING', str(dec)) if len(dec) == 1:
                        options.importeroptions.decimal = dec
                    case 'decimal' | 'dec', _:
                        raise CommandSyntaxError('The decimal option requires a single character as argument.', v)
                    case 'separator' | 'sep', Token('STRING', str(sep)):
                        options.importeroptions.separator = sep
                    case 'comment', Token('STRING', str(cmt)) if len(cmt) == 1:
                        options.importeroptions.comment_prefix = cmt
                    case 'comment' | 'com', _:
                        raise CommandSyntaxError(
                            'The comment prefix option requires a single character as argument.', v
                        )
                    case 'skip', Token('INTEGER', str(skip)):
                        options.importeroptions.skip_rows = int(skip)
                    case 'skip', _:
                        raise CommandSyntaxError('The skip option requires an integer argument.', v)
                    case opt, _:
                        raise CommandSyntaxError(f"Unknown format option '{opt}'.", k)

            case Tree('option', [Token('OPTION', str(unknown)) as t, *_]):
                raise CommandSyntaxError(f'Unknown option {unknown} or wrong arguments for filein command.', t)
            case _:
                raise CommandSyntaxError('Wrong arguments for filein command.')

    # Check required options
    if len(options.filenames) == 0:
        raise CommandSyntaxError('At least one filename must be specified for the filein command.')
    if options.reciprocal_axis_flag__ is False and options.rspace is ...:
        raise CommandSyntaxError(
            'At least one axis column must be specified for the filein command (energy, wavevector, rspace, qvector).'
        )

    # Fourier real requires imaginary, phase requires magnitude
    if options.fourierreal is not ...:
        _assert_option_required(options, 'fourierimaginary', options.fourierreal)
    if options.fourierimaginary is not ...:
        _assert_option_required(options, 'fourierreal', options.fourierimaginary)

    if options.fouriermagnitude is not ...:
        _assert_option_required(options, 'fourierphase', options.fouriermagnitude)
    if options.fourierphase is not ...:
        _assert_option_required(options, 'fouriermagnitude', options.fourierphase)

    # If the error column is specified, the signal column is required
    if options.beamintensityerror is not ...:
        _assert_option_required(options, 'beamintensity', options.beamintensityerror)
    if options.sampleintensityerror is not ...:
        _assert_option_required(options, 'sampleintensity', options.sampleintensityerror)
    if options.samplefluorescenceerror is not ...:
        _assert_option_required(options, 'samplefluorescence', options.samplefluorescenceerror)
    if options.referenceintensityerror is not ...:
        _assert_option_required(options, 'referenceintensity', options.referenceintensityerror)
    if options.xaneserror is not ...:
        _assert_option_required(options, 'xanes', options.xaneserror)
    if options.exafserror is not ...:
        _assert_option_required(options, 'exafs', options.exafserror)
    if options.fouriermagnitudeerror is not ...:
        _assert_option_required(options, 'fouriermagnitude', options.fouriermagnitudeerror)
    if options.fourierphaseerror is not ...:
        _assert_option_required(options, 'fourierphase', options.fourierphaseerror)
    if options.fourierrealerror is not ...:
        _assert_option_required(options, 'fourierreal', options.fourierrealerror)
    if options.fourierimaginaryerror is not ...:
        _assert_option_required(options, 'fourierimaginary', options.fourierimaginaryerror)

    # Return options
    return options


def execute_filein_command(command: CommandArguments_filein, context: Context) -> None:
    log = context.logger.getChild('filein')
    with context.timers.time('execution/filein'):
        # Get filename list from the given folder and filenames
        directory = command.directory or context.paths.workingdir
        if not directory.is_absolute():
            directory = (context.paths.workingdir / directory).resolve()

        files = [f for fs in command.filenames for f in directory.glob(fs)]

        # If no files found, raise an error
        if not files:
            raise ExecutionError(f"No files found in directory '{directory}' matching the specified filenames.")

        imported_files: list[DataFile] = []
        if context.options.debug or len(files) <= 8:
            # Single-threaded file reading, better for debugging and small number of files
            with logging_redirect_tqdm([context.logger]):
                for f in tqdm(files, desc='Importing files', unit='file', leave=False):
                    data = read_file(f, command, context)
                    _f = (
                        data.meta.path.relative_to(command.directory)
                        if data.meta.path.is_relative_to(command.directory)
                        else data.meta.path
                    )
                    log.debug(f"Imported file '{_f}'")

                    imported_files.append(data)
        else:
            # Multithreaded file reading
            workers = min(32, (cpu_count() or 4) + 4)
            with logging_redirect_tqdm([context.logger]), ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(read_file, f, command, context): f for f in files}

                for future in tqdm(
                    as_completed(futures), total=len(futures), desc='Importing files', unit='file', leave=False
                ):
                    data = future.result()
                    _f = (
                        data.meta.path.relative_to(command.directory)
                        if data.meta.path.is_relative_to(command.directory)
                        else data.meta.path
                    )
                    log.debug(f"Imported file '{_f}'")
                    imported_files.append(data)

        # Sort by the given sortby variable
        imported_files.sort(key=lambda df: df.meta[command.sortby])

        # Set the variable '.n' for each file (file index in the current import session)
        # Set the variable '.N' for each file (file index across all import sessions)
        previous_file_count = len(context.datastore.files)
        for n, df in enumerate(imported_files, start=1):
            df.meta['.n'] = n
            df.meta['.N'] = previous_file_count + n

    # Store imported files in the context
    context.datastore.files.update({df.meta.name: df for df in imported_files})  # to check the error
    _total_size = sum(f.meta['.fs'] for f in imported_files)
    log.debug(
        f"Imported {len(imported_files)} files in {context.timers.get_ms('execution/filein'):0.2f} ms ({fmt.human(_total_size, "B", format="0.1f", order=1024)})."
    )


def guess_type(s: str) -> str | Number | int:
    # Try to guess if the string is an integer, a number with unit, or just a string
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return parse_number(s)
    except ValueError:
        pass
    return s


def parse_header_vars(header: list[str]) -> dict[str, str | Number | int]:
    # Parse header lines for variable definitions. Each header position is defined
    # as a variable, in the format .h<line>.<position>, starting from 1.
    return {
        f'.h{linenum}.{pos}': guess_type(item)
        for linenum, line in enumerate(header, start=1)
        for pos, item in enumerate(line.split(), start=1)
    }


def parse_filename_vars(file: Path) -> dict[str, str | int | Number]:
    vars: dict[str, str | int | Number] = {
        f'.f{i+1}': guess_type(part) for i, part in enumerate(file.stem.split('_'), start=1)
    }
    vars['.fn'] = file.name
    vars['.f'] = file.stem
    vars['.fe'] = file.suffix[1:] if file.suffix.startswith('.') else file.suffix
    vars['.fd'] = file.parent.name
    vars['.fp'] = str(file)
    stat = file.stat()
    vars['.fs'] = stat.st_size
    vars['.ta'] = int(stat.st_atime)
    vars['.tc'] = int(stat.st_birthtime)
    vars['.tm'] = int(stat.st_mtime)
    return vars


def normalize_csv(s: str, re_sep: re.Pattern[str] | None) -> str:
    if re_sep is None:
        return s
    return re_sep.sub(' ', s)


def read_file(file: Path, command: CommandArguments_filein, context: Context) -> DataFile:
    # Open the file, and read lines until the first non-comment line.
    # The last comment line is the header of the file.
    fileheader: list[str] = []
    # If the first character is a comment prefix, read until the first non-comment
    # line and store it as file header.
    with open(file, 'r', encoding='utf-8') as f:
        if command.importeroptions.skip_rows > 0:
            for _ in range(command.importeroptions.skip_rows):
                f.readline()

        if peek(f) == command.importeroptions.comment_prefix:
            line = '#'  # Initialize line to enter the loop. Useless assignment, warning suppression.
            while peek(f) == command.importeroptions.comment_prefix:
                line = f.readline()
                fileheader.append(line.strip())
            # Skip the leading comment (we know line is defined here, because we made sure the line is a comment)
            header = line.strip().split()[1:]
        else:
            header = []

        # Now we replace separator characters with spaces, and read the data using pandas.
        normfiledata = normalize_csv(f.read(), command.importeroptions.re_sep)
        dat = pd.read_csv(  # type: ignore
            StringIO(normfiledata),
            sep=command.importeroptions.separator,
            header=None,
            decimal=command.importeroptions.decimal or '.',
            engine='c',
            dtype=float,
        )
        dat.columns = [h or c for h, c in zip_longest(header, dat.columns)][: len(dat.columns)]

    # Parse file header and name variables
    file_variables = parse_header_vars(fileheader)
    file_variables |= parse_filename_vars(file)

    # Read command variables. If some values are equal to other variable names,
    # set them to the same value.
    for var, val in command.vars.items():
        if isinstance(val, str) and val in file_variables:
            file_variables[var] = file_variables[val]
        else:
            file_variables[var] = val

    # Get the specified columns and store them in their respective domains
    extractedcolumns = Bag[Domain, tuple[Column, pd.Series]].from_iter(
        (
            domain,
            (column, dat[expand_column_descriptor(descriptor, dat.columns.to_list())].mean(axis=1).rename(column.name)),
        )
        for column, domain, descriptor in command.columns
    )
    signalcolumns = Bag[Domain, tuple[Column, Expr]].from_iter(
        (domain, (column, expr)) for column, domain, expr in command.signals
    )

    # Get data from extracted columns
    newdata = {
        domain: DataDomain([c for c, _ in columns], pd.DataFrame(c for _, c in columns).T)
        for domain, columns in extractedcolumns.groups()
    }
    # Compute and store the new signals in their respective domains
    for domain, newcolexps in signalcolumns.groups():
        if domain not in newdata:
            raise ExecutionError(f'Cannot compute signal in domain {domain} without any columns.')
        newdata[domain].columns.extend(c for c, _ in newcolexps)
        newdata[domain].data = newdata[domain].data.assign(**{c.name: e for c, e in newcolexps})  # type: ignore

    return DataFile(
        FileMetadata(
            path=file,
            name=file.name,
            _dict=file_variables,
        ),
        newdata,
    )
