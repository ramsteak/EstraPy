from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
import re

from typing import NamedTuple
from logging import getLogger
from dataclasses import dataclass
from scipy.interpolate import interp1d
from pathlib import Path


from ._context import Context, MetaData
from ._handler import CommandHandler, Token, CommandResult
from ._numberunit import NumberUnit, parse_range, NumberUnitRange, actualize_range

from ._parser import CommandParser

from .plot import ColKind, parse_column, get_column_


def save_df_to_dat(filename:str|Path, df:pd.DataFrame,*, index:bool=False, head:str|None=None, mode:str="wt"):
    colwidth = 16
    decformat = np.vectorize(lambda x:format(x, "<10.6f").ljust(colwidth))
    sdecformat = np.vectorize(lambda x:format(x, "<+10.6f").ljust(colwidth))
    sciformat = np.vectorize(lambda x:format(x, "<10.6e").ljust(colwidth))
    ssciformat = np.vectorize(lambda x:format(x, "<+10.6e").ljust(colwidth))

    def formatcol(x:npt.NDArray) -> npt.NDArray[np.str_]:
        sci = ((x != 0)&(np.abs(np.log(np.abs(x + np.finfo(x.dtype).eps))) >= 4)).any()
        sgn = (x < 0).any()
        match bool(sci),bool(sgn):
            case False, False:
                fmt = decformat
            case False, True :
                fmt = sdecformat
            case True , False:
                fmt = sciformat
            case True , True :
                fmt = ssciformat
        return fmt(x)

    if index:
        headers = [str(df.index.name).ljust(colwidth)] + [str(c).ljust(colwidth) for c in df.columns]
        columns = [formatcol(df.index.to_numpy())] + [formatcol(df.loc[:, col].to_numpy()) for col in df.columns]
    else:
        headers = [str(c).ljust(colwidth) for c in df.columns]
        columns = [formatcol(df.loc[:, col].to_numpy()) for col in df.columns]

    stacked = np.column_stack(columns)

    with open(filename, mode) as fw:
        if head:
            fw.write(head + "\n")
        fw.write(f"#L {"".join(headers).rstrip()}\n")
        fw.writelines(f"   {"".join(l).rstrip()}\n" for l in stacked)

def get_var_headers(meta:MetaData) -> str:
    r = re.compile(r"\.h(\d+)\.(\d+)")
    _head:list[list[str]] = []
    for varname, val in meta.vars.items():
        m = r.match(varname)
        if m is None: continue

        _line,_pos = m.groups()
        line = int(_line) - 1
        pos = int(_pos) - 1

        while len(_head) <= line:
            _head.append([])
        while len(_head[line]) <= pos:
            _head[line].append("")
        _head[line][pos] = str(val)
    
    head = [" ".join(line) for line in _head]

    if ".st" in meta:
        head.append(f"#V signal {meta.get(".st")}")
    if ".rt" in meta:
        head.append(f"#V reference {meta.get(".rt")}")
    if "E0" in meta:
        head.append(f"#V E0 {meta.get("E0")}")
    if "rE0" in meta:
        head.append(f"#V refE0 {meta.get("rE0")}")

    for vname, v in meta.vars.items():
        if vname.startswith("."):
            continue
        head.append(f"#V {vname} {v}")
    
    return "\n".join(head)


@dataclass(slots=True, frozen=True)
class Mode:
    filename: str

@dataclass(slots=True, frozen=True)
class Each(Mode):
    columns: list[ColKind]

@dataclass(slots=True, frozen=True)
class Batch(Mode):
    column: ColKind

@dataclass(slots=True, frozen=True)
class Aligned(Mode):
    align: NumberUnitRange
    column: ColKind

class Args_Save(NamedTuple):
    mode: Mode


class Save(CommandHandler):
    @staticmethod
    def parse(tokens: list[Token], context: Context) -> Args_Save:
        parser = CommandParser("save", description="Saves the data to a file.")
        parser.add_argument("filename")

        subparsers = parser.add_subparsers(dest="mode")
        aligned = subparsers.add_parser("aligned")
        aligned.add_argument("column")
        aligned.add_argument("--align", "-a", nargs=3, required=True)

        batch = subparsers.add_parser("batch")
        batch.add_argument("column")

        columns = subparsers.add_parser("columns")
        columns.add_argument("columns", nargs="+")
        
        args = parser.parse(tokens)

        match args.mode:
            case "batch":
                col = parse_column(args.column)
                if col.xcol is None:
                    raise ValueError("No x column specified.")
                mode = Batch(args.filename, col)
                
            case "aligned":
                col = parse_column(args.column)
                if col.xcol is None:
                    raise ValueError("No x column specified.")

                if col.xop:
                    raise ValueError("The output x column cannot be modified.")
                
                bounds = parse_range(*args.align)
                if bounds.lower.value == -np.inf:
                    raise ValueError("Alignment range cannot be ..")
                if bounds.upper.value == np.inf:
                    raise ValueError("Alignment range cannot be ..")
                    
                mode = Aligned(args.filename, bounds, col)
            case "columns":
                cols = [parse_column(c) for c in args.columns]
                mode = Each(args.filename, cols)

        
        return Args_Save(mode)

    @staticmethod
    def execute(args: Args_Save, context: Context) -> CommandResult:
        log = getLogger("save")
        
        match args.mode:
            case Batch(fname, column):
                outfile = context.paths.outputdir / fname
                cols:list[pd.Series] = []
                for data in context.data:
                    x,y = get_column_(column, data)
                    X = pd.Series(x, name=f"{column.xcol}_{data.meta.name}")
                    cols.append(X)
                    Y = pd.Series(y, name=f"{column.ycol}_{data.meta.name}")
                    cols.append(Y)
                df = pd.concat(cols, axis=1)
                save_df_to_dat(outfile, df, index=False)
            case Aligned(fname, align, column):
                outfile = context.paths.outputdir / fname
                if column.xcol is None:
                    raise RuntimeError("No x column specified.")
                domain = context.data.data[0]._get_col_domain(column.xcol)
                if domain != align.domain:
                    raise RuntimeError("Invalid range selection: range does not match given abscissa.")
                unit = context.data.data[0].datums[domain].cols[column.xcol].unit
                bounds = actualize_range(align, [data.get_col_(column.xcol) for data in context.data], unit)
                
                match bounds.inter:
                    case NumberUnit(i, _, _):
                        newx = np.arange(bounds.lower.value, bounds.upper.value, i)
                    case int(n):
                        newx = np.linspace(bounds.lower.value, bounds.upper.value, n)
                    case _:
                        raise RuntimeError("Unknown error. #23794618")
                
                out:list[pd.Series] = []
                x = pd.Index(newx, name = column.xcol)
                for data in context.data:
                    X,Y = get_column_(column, data)
                    y = interp1d(X,Y,"cubic")(newx)
                    out.append(pd.Series(y, x, name=data.meta.name))
                df = pd.concat(out, axis=1)
                save_df_to_dat(outfile, df, index=True)
                    

            case Each(fname, columns):
                outnames:dict[str,int] = {}
                for data in context.data:
                    filename = fname
                    for m in reversed([*re.finditer(r"\{([^{}]*)\}", fname)]):
                        val = data.meta.get(m.group(1))
                        filename = filename[:m.start()] + str(val) + filename[m.end():]
                    outfile = context.paths.outputdir / filename
                    if outfile.name in outnames:
                        n = outnames[outfile.name]
                        newname = outfile.name.removesuffix(outfile.suffix) + f"_{n}" + outfile.suffix
                        outfile = context.paths.outputdir / newname
                        outnames[outfile.name] = n + 1
                    else:
                        outnames[outfile.name] = 1

                    cols = [pd.Series(get_column_(c, data)[1], name=c.ydef) for c in columns]
                    df = pd.concat(cols, axis=1)
                    head = get_var_headers(data.meta)
                    save_df_to_dat(outfile, df, index=False, head=head)
    
        return CommandResult(True)

    @staticmethod
    def undo(args: NamedTuple, context: Context) -> CommandResult:
        # TODO:
        raise NotImplementedError
