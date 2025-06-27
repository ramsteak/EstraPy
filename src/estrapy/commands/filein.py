from __future__ import annotations

import os
import pandas as pd
import numpy as np

from io import StringIO
from logging import getLogger
from pathlib import Path
from typing import NamedTuple, Sequence
from enum import Enum

from ._exceptions import FileParsingException, ArgumentException
from ._context import AxisType, Context, Data, MetaData, SignalType, Column, Domain, DataColType
from ._handler import CommandHandler, Token, CommandResult
from ._misc import attempt_parse_number
from ._parser import CommandParser, InputFileParsingException


def _read_file_m1(file: Path) -> tuple[CommandResult, pd.DataFrame, dict[str, str | int | float]]:
    # Expect the data file to start with a "#"
    filecontent = file.read_text()
    if filecontent[0] != "#":
        raise FileParsingException

    lines = filecontent.splitlines()
    # Gets the index of the first line that does not start with #
    id_firstdataline = (
        i for i, line in enumerate(lines) if not line.lstrip().startswith("#")
    ).__next__()

    metadatalines = [line.split() for line in lines[: id_firstdataline - 1]]
    headers = lines[id_firstdataline - 1].split()[1:]

    firstdataline = lines[id_firstdataline]
    
    # Check if comma -> ,\s+
    if "," in firstdataline:
        if "." in firstdataline:
            # Assume , is column separator and . is decimal separator
            sep,engine = r",\s+","python"
    else:
        sep,engine = r"\s+",None

    try:
        data = pd.read_csv(
            StringIO("\n".join(lines[id_firstdataline:])), sep=sep, header=None, engine=engine,
        )
    except pd.errors.ParserError:
        raise FileParsingException

    datalen, headlen = len(data.columns), len(headers)

    if datalen < headlen:
        headers = headers[:datalen]
        warn = f"Header and data length mismatch: {headlen} {datalen}. Header was cut to match."
    elif datalen > headlen:
        headers.extend(f"col{i+1}" for i in range(headlen, datalen))
        warn = f"Header and data length mismatch: {headlen} {datalen}. Header was extended to match."
    else:
        warn = None

    data.columns = headers

    # --- Parse metadata -------------------------------------------------------
    metavars: dict[str, str | int | float] = {}

    # Add file header metadata. The split is done at every space.
    metavars.update(
        {
            f".h{ln}.{pn}": attempt_parse_number(item)
            for ln, line in enumerate(metadatalines, 1)
            for pn, item in enumerate(line, 1)
        }
    )

    # If a line starts with #U then what follows is a varname and a value
    for line in metadatalines:
        if not line:
            continue
        if line[0] == "#U" and len(line) >= 3:
            vname, vval = line[1], attempt_parse_number(line[2])
            if vname.startswith("."):
                raise NameError("Variables cannot start with dot.")
            metavars[vname] = vval

    return CommandResult(True, warning=warn), data, metavars


_file_read_methods = [_read_file_m1]


def _get_column_index(_columns: Sequence[str], column: str | int) -> int:
    if isinstance(column, int):
        return column
    if column.isdigit():
        return int(column) - 1
    if column in _columns:
        return _columns.index(column)
    raise IndexError(f"Column {column} not found.")


def _get_column_range(
    _columns: Sequence[str], cola: str | int, colb: str | int | None = None
) -> list[int]:
    if colb is None:
        return [_get_column_index(_columns, cola)]
    return [
        *range(_get_column_index(_columns, cola), _get_column_index(_columns, colb) + 1)
    ]


def _parse_column_range(_columns: Sequence[str], range: str) -> list[int]:
    return [
        col
        for interval in range.split(",")
        for col in _get_column_range(_columns, *interval.split(".."))
    ]

def get_filepaths(file:str, dir:Path) -> list[Path]:
    if os.path.isabs(file):
        _p = Path(file)
        _d = Path((_p.drive) + "/")
        _r = _p.relative_to(_d)
        return [*_d.glob(_r.name)]
    else:
        return [*dir.glob(file)]

def _read_file(file:Path, log_fname:str|None=None) -> tuple[CommandResult, pd.DataFrame, dict[str, str | int | float]]:
    log_fname = log_fname or file.name
    for method in _file_read_methods:
        return method(file)
    else:
        raise RuntimeError(f"Unrecognized file format: {log_fname}")

def read_file(file: Path, args: Args_FileIn) -> Data:
    """Reads the given file and extracts data and metadata.
    Attempts all methods before failing."""
    log = getLogger("filein")

    if args.directory is not None:
        _log_fname = file.relative_to(args.directory)
    else:
        _log_fname = file

    log.debug(f"Reading file: {_log_fname}")

    res, filedat, mdat = _read_file(file)

    if res.warning is not None:
        log.warning(f"{_log_fname}: {res.warning}")
    
    # Add file name metadata. The split is done at every underscore.
    mdat.update(
        {
            f".f{n}": attempt_parse_number(e)
            for n, e in enumerate(file.name.removesuffix(file.suffix).split("_"), 1)
        }
    )
    mdat[".f"] = file.name
    mdat[".fn"] = file.name.removesuffix(file.suffix)
    mdat[".fe"] = file.suffix

    # Add all the given variables from the command
    for vname, vval in args.vars.items():
        if vname.startswith("."):
            raise NameError("Variables cannot start with dot.")
        if isinstance(vval, str) and vval.startswith("."):
            # If the variable value starts with a period, check if it is a header
            # or filename var, and set the predefined value to a pretty $name
            mdat[vname] = mdat.get(vval, vval)
        else:
            mdat[vname] = vval

    name = file.name.removesuffix(file.suffix)
    signaltype = args.signaltype[0] if args.signaltype is not None else None
    refsigtype = args.refsigtype[0] if args.refsigtype is not None else None
    metadata = MetaData(signaltype, refsigtype, name, file, mdat)

    headercolumns = list(filedat.columns)
    dat = Data(metadata=metadata)

    # Import the main axis column
    xaxiscol = _get_column_index(headercolumns, args.xaxiscol)
    xcolumn = filedat.iloc[:, xaxiscol]
    match args.axis:
        case AxisType.INDEX:
            signaldomain = Domain.REAL
        case AxisType.ENERGY:
            dat.add_col("E", xcolumn, Column("eV", False, AxisType.ENERGY), Domain.REAL)
            signaldomain = Domain.REAL
            dat.datums[signaldomain].set_default_axis("E")
        case AxisType.RELENERGY:
            dat.add_col("e", xcolumn, Column("eV", True, AxisType.RELENERGY), Domain.REAL)
            signaldomain = Domain.REAL
            dat.datums[signaldomain].set_default_axis("e")
        case AxisType.KVECTOR:
            dat.add_col("k", xcolumn, Column("k", None, AxisType.KVECTOR), Domain.REAL)
            signaldomain = Domain.REAL
            dat.datums[signaldomain].set_default_axis("k")
        case AxisType.DISTANCE:
            dat.add_col("R", xcolumn, Column("A", None, AxisType.DISTANCE), Domain.FOURIER)
            signaldomain = Domain.FOURIER
            dat.datums[signaldomain].set_default_axis("R")
        case AxisType.QVECTOR:
            dat.add_col("q", xcolumn, Column("q", None, AxisType.QVECTOR), Domain.REAL)
            signaldomain = Domain.REAL
            dat.datums[signaldomain].set_default_axis("q")
    
    # Import the intensities
    for cname, (crange, ctype) in args.columns.items():
        dat.add_col(
            cname,
            filedat.iloc[:,_parse_column_range(headercolumns, crange)].sum(axis=1),
            ctype,
            signaldomain
        )

    if args.signaltype is not None:
        stype, scols = args.signaltype
        match stype:
            case SignalType.INTENSITY:
                scolumn = dat.get_col(scols[0])
                dat.add_col("a", scolumn, Column(None, None, SignalType.INTENSITY), signaldomain)
            case SignalType.TRANSMITTANCE:
                scolumn = np.log10(dat.get_col(scols[0]) / dat.get_col(scols[1]))
                dat.add_col("a", scolumn, Column(None, None, SignalType.TRANSMITTANCE), signaldomain)
            case SignalType.FLUORESCENCE:
                scolumn = dat.get_col(scols[0]) / dat.get_col(scols[1])
                dat.add_col("a", scolumn, Column(None, None, SignalType.FLUORESCENCE), signaldomain)

    if args.refsigtype is not None:
        rtype, rcols = args.refsigtype
        match rtype:
            case SignalType.INTENSITY:
                scolumn = dat.get_col(rcols[0])
                dat.add_col("ref", scolumn, Column(None, None, SignalType.INTENSITY), signaldomain)
            case SignalType.TRANSMITTANCE:
                scolumn = np.log10(dat.get_col(rcols[0]) / dat.get_col(rcols[1]))
                dat.add_col("ref", scolumn, Column(None, None, SignalType.TRANSMITTANCE), signaldomain)
            case SignalType.FLUORESCENCE:
                scolumn = dat.get_col(rcols[0]) / dat.get_col(rcols[1])
                dat.add_col("ref", scolumn, Column(None, None, SignalType.FLUORESCENCE), signaldomain)

    # Sort the axis
    axis = dat.datums[signaldomain].default_axis
    if axis is not None:
        df = dat.datums[signaldomain].df
        df.sort_values(axis, inplace=True)
        df = df.groupby(axis, as_index=False).mean()
        dat.datums[signaldomain].df = df
    return dat


class Args_FileIn(NamedTuple):
    filepos: str
    directory: str | None

    axis: AxisType
    xaxiscol: str
    shift: float

    columns: dict[str, tuple[str, Column]]

    signaltype: tuple[SignalType, tuple[str, ...]]
    refsigtype: tuple[SignalType, tuple[str, ...]] | None

    vars: dict[str, str | int | float]


class BatchIn(CommandHandler):
    @staticmethod
    def parse(tokens: list[Token], context: Context) -> None:
        if "batcharguments" not in context.options.other:
            context.options.other["batcharguments"] = list[list[Token]]()
        context.options.other["batcharguments"].append(tokens)

    @staticmethod
    def execute(args: None, context: Context) -> CommandResult:
        return CommandResult(True)

    @staticmethod
    def undo(args: NamedTuple, context: Context) -> CommandResult:
        # TODO:
        raise NotImplementedError


class FileIn(CommandHandler):
    @staticmethod
    def parse(tokens: list[Token], context: Context) -> Args_FileIn:
        parser = CommandParser("filein", description="Imports the data from files.")
        parser.add_argument(
            "filepos", help="The file to import. Can be a glob pattern."
        )
        parser.add_argument(
            "--dir",
            help="The directory to import the files from, instead of the current workdir.",
        )
        parser.add_argument(
            "--batch", "-b", action="store_true", help="Uses the batch arguments"
        )

        parser.add_argument("--xaxiscolumn", "-X", help="Column containing the x-axis.")

        groupx = parser.add_mutually_exclusive_group()
        groupx.add_argument(
            "--index",
            nargs="?",
            type=str,
            default=None,
            const=True,
            help="Sets the x axis to a numeric index.",
        )
        groupx.add_argument(
            "--energy",
            "-E",
            nargs="?",
            type=str,
            default=None,
            const=True,
            help="Reads the x-axis column as energy.",
        )
        groupx.add_argument(
            "--kvector",
            "-k",
            nargs="?",
            type=str,
            default=None,
            const=True,
            help="Reads the x-axis column as k-vector.",
        )
        groupx.add_argument(
            "--rdistance",
            "-R",
            nargs="?",
            type=str,
            default=None,
            const=True,
            help="Reads the x-axis column as radial distance.",
        )
        groupx.add_argument(
            "--qvector",
            "-q",
            nargs="?",
            type=str,
            default=None,
            const=True,
            help="Reads the x-axis column as q-vector.",
        )

        parser.add_argument(
            "--shift",
            type=float,
            default=0,
            help="Shifts the data by the specified amount.",
        )

        parser.add_argument("--intensities","-I",nargs="+",help="Imports raw intensities, in order: I0, I1, I2, If. If not specified, determines columns from data import commands. To skip one, use '-'",)

        groupd = parser.add_mutually_exclusive_group()
        groupd.add_argument("--transmission","-t",nargs="*",help="Reads transmission data, and calculates as log(#1/#2).",)
        groupd.add_argument("--fluorescence","-f",nargs="*",help="Reads fluorescence data, and calculates as #1/#2.",)
        groupd.add_argument("--intensity", "-i", help="Reads signal, as is.")
        groupd.add_argument("--mu", "-m", help="Imports the specified column as a XANES signal.")
        groupd.add_argument("--chi", "-x", help="Imports the specified column as an EXAFS signal.")

        groupr = parser.add_mutually_exclusive_group()
        groupr.add_argument("--reftransmittance","-T",nargs="*",help="Reads transmission reference data, and calculates as log(#1/#2).",)
        groupr.add_argument("--refabsorption", "-A", help="Reads reference signal, as is.")

        parser.add_argument("--var", action="append", nargs=2)

        # Preparse the current arguments. If batch is used, reparse the arguments
        # parsing the batch arguments first, to update the namespace.
        args = parser.parse(tokens)
        if args.batch:
            if "batcharguments" not in context.options.other:
                raise ArgumentException("No batch arguments specified.")
            namespace = None
            for batcharg in context.options.other["batcharguments"]:
                batcharg: list[Token]
                namespace = parser.parse([Token("", -1, -1), *batcharg], namespace)
            args = parser.parse(tokens, namespace)

        # --- Detect the input x axis type -------------------------------------

        match args.xaxiscolumn, args.index, args.energy, args.kvector, args.rdistance, args.qvector:
            case [None, None, None, None, None, None]:
                raise InputFileParsingException("No x axis specified.")
            case [col, None, None, None, None, None]:
                xaxiscol, xaxis = col, AxisType.INDEX
            case [None, col, None, None, None, None]:
                xaxiscol, xaxis = col, AxisType.INDEX
            case [col, True, None, None, None, None]:
                xaxiscol, xaxis = col, AxisType.INDEX
            case [None, None, col, None, None, None]:
                xaxiscol, xaxis = col, AxisType.ENERGY
            case [col, None, True, None, None, None]:
                xaxiscol, xaxis = col, AxisType.ENERGY
            case [None, None, None, col, None, None]:
                xaxiscol, xaxis = col, AxisType.KVECTOR
            case [col, None, None, True, None, None]:
                xaxiscol, xaxis = col, AxisType.KVECTOR
            case [None, None, None, None, col, None]:
                xaxiscol, xaxis = col, AxisType.DISTANCE
            case [col, None, None, None, True, None]:
                xaxiscol, xaxis = col, AxisType.DISTANCE
            case [None, None, None, None, None, col]:
                xaxiscol, xaxis = col, AxisType.QVECTOR
            case [col, None, None, None, None, True]:
                xaxiscol, xaxis = col, AxisType.QVECTOR
            case _:
                raise InputFileParsingException("Invalid x axis type combination.")

        # --- Intensities, transmission, absorption, fluorescence are only valid
        # if the x axis is ENERGY. If else, use intensity ----------------------

        cols: dict[str, tuple[str, Column]] = {}

        match xaxis, args.transmission, args.fluorescence, args.intensities, args.intensity, args.mu, args.chi:
            case [AxisType.ENERGY, None, None, None, None, None, None]:
                # None of transmission, fluorescence, intensities, intensity specified.
                raise InputFileParsingException("No y axis specified.")
            case [AxisType.ENERGY, [str(cI)], None, None, None, None, None]:
                ytype = None
                cols["a"] = cI, Column(None, None, DataColType.ALPHA)
            case [AxisType.ENERGY, [str(cI0), str(cI1)], None, None, None, None, None]:
                # --transmission I0 I1
                ytype = SignalType.TRANSMITTANCE, ("I0", "I1")
                cols["I0"] = cI0, Column(None, None, SignalType.INTENSITY)
                cols["I1"] = cI1, Column(None, None, SignalType.INTENSITY)
            case [AxisType.ENERGY, None, [str(cI)], None, None, None, None]:
                ytype = SignalType.FLUORESCENCE, ("I0", "I1")
                cols["a"] = cI, Column(None, None, DataColType.ALPHA)
            case [AxisType.ENERGY, None, [str(cIf), str(cI0)], None, None, None, None]:
                # --fluorescence If I0
                ytype = SignalType.FLUORESCENCE, ("If", "I0")
                cols["I0"] = cI0, Column(None, None, SignalType.INTENSITY)
                cols["If"] = cIf, Column(None, None, SignalType.INTENSITY)
            case [AxisType.ENERGY, None, None, [str(cI)], None, None, None]:
                # --intensities I
                ytype = SignalType.INTENSITY, ("I",)
                cols["I"] = cI, Column(None, None, SignalType.INTENSITY)
            case [AxisType.ENERGY, None, None, _, None, None, None]:
                # --intensities I0 I1
                raise InputFileParsingException(
                    "No y signal mode specified. Specify either transmission, fluorescence or intensity."
                )
            case [AxisType.ENERGY, [], None, [str(cI0), str(cI1)], None, None, None]:
                # --transmission --intensities I0 I1
                ytype = SignalType.TRANSMITTANCE, ("I0", "I1")
                cols["I0"] = cI0, Column(None, None, SignalType.INTENSITY)
                cols["I1"] = cI1, Column(None, None, SignalType.INTENSITY)
            case [AxisType.ENERGY, None, [], [str(cI0), str(cI1)], None, None, None]:
                # --fluorescence --intensities I0 I1
                raise InputFileParsingException(
                    "Fluorescence intensity (If) is not specified."
                )
            case [AxisType.ENERGY, [], None, [str(cI0), str(cI1), str(cI2)], None, None, None]:
                # --transmission --intensities I0 I1 I2
                ytype = SignalType.TRANSMITTANCE, ("I0", "I1")
                cols["I0"] = cI0, Column(None, None, SignalType.INTENSITY)
                cols["I1"] = cI1, Column(None, None, SignalType.INTENSITY)
                cols["I2"] = cI2, Column(None, None, SignalType.INTENSITY)
            case [AxisType.ENERGY, None, [], [str(cI0), str(cI1), str(cI2)], None, None, None]:
                # --fluorescence --intensities I0 I1 I2
                raise InputFileParsingException(
                    "Fluorescence intensity (If) is not specified."
                )
            case [AxisType.ENERGY,[],None,[str(cI0), str(cI1), str(cI2), str(cIf)],None, None, None]:
                # --transmission --intensities I0 I1 I2 If
                ytype = SignalType.TRANSMITTANCE, ("I0", "I1")
                cols["I0"] = cI0, Column(None, None, SignalType.INTENSITY)
                cols["I1"] = cI1, Column(None, None, SignalType.INTENSITY)
                cols["I2"] = cI2, Column(None, None, SignalType.INTENSITY)
                cols["If"] = cIf, Column(None, None, SignalType.INTENSITY)
            case [AxisType.ENERGY,None,[],[str(cI0), str(cI1), str(cI2), str(cIf)],None, None, None]:
                # --fluorescence --intensities I0 I1 I2 If
                ytype = SignalType.FLUORESCENCE, ("If", "I0")
                cols["I0"] = cI0, Column(None, None, SignalType.INTENSITY)
                cols["I1"] = cI1, Column(None, None, SignalType.INTENSITY)
                cols["I2"] = cI2, Column(None, None, SignalType.INTENSITY)
                cols["If"] = cIf, Column(None, None, SignalType.INTENSITY)
            case [AxisType.ENERGY, None, None, None, str(cI), None, None]:
                # --intensity I
                ytype = SignalType.INTENSITY, (cI,)
                cols["I"] = cI, Column(None, None, SignalType.INTENSITY)
            case [AxisType.ENERGY | AxisType.RELENERGY | AxisType.KVECTOR, None, None, None, None, str(cMu), None]:
                ytype = None
                cols["mu"] = cMu, Column(None, None, DataColType.MU)
            case [AxisType.ENERGY | AxisType.RELENERGY | AxisType.KVECTOR, None, None, None, None, None, str(cX)]:
                ytype = None
                cols["x"] = cX, Column(None, None, DataColType.CHI)
            case [AxisType.ENERGY, _, _, _, _, _, _]:
                raise InputFileParsingException("Too many signal specifications.")
            case _:
                raise InputFileParsingException(
                    "Only energy x-axis supports this type of data input."
                )

        match xaxis, args.reftransmittance, args.refabsorption:
            case [AxisType.ENERGY, None, None]:
                rtype = None
            case [AxisType.ENERGY, [], None]:
                if "I1" not in cols:
                    raise InputFileParsingException("No I1 column specified.")
                if "I2" not in cols:
                    raise InputFileParsingException("No I2 column specified.")
                rtype = SignalType.TRANSMITTANCE, ("I1", "I2")
            case [AxisType.ENERGY, [str(rI1), str(rI2)], None]:
                cols["rI1"] = rI1, Column(None, None, SignalType.INTENSITY)
                cols["rI2"] = rI2, Column(None, None, SignalType.INTENSITY)
                rtype = SignalType.TRANSMITTANCE, ("rI1", "rI2")
            case [AxisType.ENERGY, None, []]:
                raise InputFileParsingException("No reference signal mode specified.")
            case [AxisType.ENERGY, None, str(rI)]:
                cols["Ir"] = rI, Column(None, None, SignalType.INTENSITY)
                rtype = SignalType.INTENSITY, ("Ir",)
            case [AxisType.ENERGY, _, _]:
                raise InputFileParsingException(
                    "Too many reference signal specifications."
                )
            case _:
                raise InputFileParsingException(
                    "Only energy x-axis supports this type of data input."
                )

        if args.var is not None:
            vars = {
                str(k): attempt_parse_number(v) if str(v).isdecimal() else str(v)
                for k, v in args.var
            }
        else:
            vars = {}

        return Args_FileIn(
            args.filepos,
            args.dir,
            xaxis,
            xaxiscol,
            args.shift,
            cols,
            ytype, # type: ignore
            rtype,
            vars,
        )

    @staticmethod
    def execute(args: Args_FileIn, context: Context) -> CommandResult:
        log = getLogger("filein")

        # Check if the path is relative or absolute.
        relativeto = (
            Path(args.directory)
            if args.directory is not None
            else context.paths.workdir
        )
        files = get_filepaths(args.filepos, relativeto)
        # if os.path.isabs(args.filepos):
        #     _p = Path(args.filepos)
        #     _d = Path((_p.drive) + "/")
        #     _r = _p.relative_to(_d)
        #     files = [*_d.glob(_r.name)]
        #     # if _p.parent.name == "**":
        #     #     files = [*_p.parent.parent.rglob(_p.name)]
        #     # else:
        #     #     files = [*_p.parent.glob(_p.name)]
        # else:
        #     files = [*relativeto.glob(args.filepos)]

        if not files:
            raise FileNotFoundError(
                f"The specified file does not exist: {args.filepos}"
            )

        filecount = len(context.data.data)
        for i, file in enumerate(files, 1):
            data = read_file(file, args)
            context.data.add_data(data)

            data.meta.vars[".n"] = filecount + i
            data.meta.vars[".i"] = i
            log.info(f"Imported {data.meta.name}")

        return CommandResult(True)

    @staticmethod
    def undo(args: Args_FileIn, context: Context) -> CommandResult:
        # TODO:
        raise NotImplementedError

class TestType(Enum):
    ENERGYCOLUMN = "energycolumn"
    GROUPS = "groups"

class Args_Test(NamedTuple):
    filepos: str
    testtype:TestType
    columns:str|None=None

class FileTest(CommandHandler):
    @staticmethod
    def parse(tokens: list[Token], context: Context) -> Args_Test:
        parser = CommandParser("test", description="Tests the file columns.")
        parser.add_argument("filename", help="The file to use for the test.")
        
        subparsers = parser.add_subparsers(dest="method")
        subparsers.add_parser("energy")
        subparsers.add_parser("groups")


        args = parser.parse(tokens)

        match args.method:
            case "energy":
                return Args_Test(args.filename, TestType.ENERGYCOLUMN, None)
            case "groups":
                return Args_Test(args.filename, TestType.GROUPS, None)
            case t: raise ValueError(f"Unknown test type: {t}.")

    @staticmethod
    def execute(args: Args_Test, context: Context) -> CommandResult:
        log = getLogger("test")

        file = get_filepaths(args.filepos, context.paths.workdir)[0]
        res, filedat, mdat = _read_file(file)

        match args.testtype:
            case TestType.ENERGYCOLUMN:
                # Calculate the difference between successive points. Energy
                # columns are strictly monotonic, so should be 0.
                diff = (filedat.diff() <= 0).sum()
                
            case TestType.GROUPS:
                pass

        return CommandResult(True)

    @staticmethod
    def undo(args: NamedTuple, context: Context) -> CommandResult:
        # TODO:
        raise NotImplementedError
