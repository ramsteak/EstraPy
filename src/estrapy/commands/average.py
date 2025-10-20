from __future__ import annotations

import numpy as np

from logging import getLogger
from typing import NamedTuple, Iterable, TypeVar
from functools import reduce
from scipy.interpolate import interp1d

from ._context import Context, Data, DataStore, MetaData, SignalType, AxisType
from ._handler import CommandHandler, Token, CommandResult
from ._parser import CommandParser

_T = TypeVar("_T")

def eqor(s:Iterable[_T], *, default:_T = ...) -> _T:
    _it = iter(s)
    try: s0 = next(_it)
    except StopIteration:
        if default is ...: raise
        return default
    for e in _it:
        if e != s0:
            if default is ...: raise ValueError("All elements are not equal.")
            return default
    return s0

class Args_Average(NamedTuple):
    groupvars: list[str]
    axis: str | None

class Average(CommandHandler):
    @staticmethod
    def parse(tokens: list[Token], context: Context) -> Args_Average:
        parser = CommandParser("average", description="Averages data.")
        parser.add_argument("--by", "-b", nargs="+")
        parser.add_argument("--axis", "-a")

        args = parser.parse(tokens)

        return Args_Average(groupvars=args.by or [], axis=args.axis)

    @staticmethod
    def execute(args: Args_Average, context: Context) -> CommandResult:
        log = getLogger("average")

        bins:dict[tuple, list[Data]] = {}
        for outdata in context.data:
            var = tuple(outdata.meta.get(v, default=None) for v in args.groupvars)
            if var not in bins: bins[var] = []
            bins[var].append(outdata)
        
        newdata = DataStore()
        for var,datas in bins.items():

            # Define metadata for the new data
            ivars:set[tuple[str, str|int|float]] = reduce(set.intersection, [{*data.meta.vars.items()} for data in datas])
            avars:set[tuple[str, str|int|float]] = reduce(set.union, [{*data.meta.vars.items()} for data in datas])
            mvars:dict[str, list[int|float]] = {}
            for k,v in avars.difference(ivars):
                if k.startswith("."): continue
                if isinstance(v, str): continue
                if k not in mvars: mvars[k] = []
                mvars[k].append(v)
            vvars: dict[str, int|float] = {k:sum(vs)/len(vs) for k,vs in mvars.items()}
            nvars = dict(ivars) | vvars

            signaltype = eqor([data.meta.signaltype for data in datas], default = SignalType.OTHER)
            refsigtype = eqor([data.meta.refsigtype for data in datas], default = SignalType.OTHER)
            name = "_".join(str(v) for v in var)
            path = context.paths.workdir
            refE0 = eqor([data.meta.refE0 for data in datas], default = None)

            run = {"mean": datas}
            
            E0s = [data.meta.E0 for data in datas]
            if all(E is not None for E in E0s):
                E0 = sum(E0s) / len(E0s) # type: ignore
            else: E0 = None

            # Add the {.fn} variable as the "_".join of the variables
            nvars[".fn"] = name
            outmeta = MetaData(signaltype, refsigtype, name, path, nvars, run, refE0, E0)
            outdata = Data(outmeta)

            # TODO: Temo che al momento non sia possibile fare le cose bene.
            # Ciò richiederebbe di definire un modo per convertire da `E` a `e`
            # e `k` univoco, definendo a necessità `E0` per ciascuno spettro mediato.
            # Al momento faccio le cose "male", prendendo come nuovi assi gli 
            # assi del primo (entro i range degli altri)

            datums = set(d for _d in datas for d in _d.datums)
            for datum in datums:
                firstdata, *_ = [data.datums[datum] for data in datas]
                # Get the first data axes and add them to data
                for colname, colkind in firstdata.cols.items():
                    if isinstance(colkind.axis, AxisType):
                        outdata.add_col(colname, firstdata.df[colname], colkind, domain=datum)
                
                if args.axis is None:
                    daxisname, daxis = firstdata.default_axis, firstdata.get_default_axis()
                else:
                    daxisname, daxis = args.axis, firstdata.df[args.axis]

                for colname, colkind in firstdata.cols.items():
                    # Skip the columns that are already present
                    if colname in outdata.datums[datum].cols: continue
                    # For each column, get all other columns
                    interp:list[np.ndarray] = [interp1d(*data.get_xy_(daxisname, colname), # type: ignore
                                  kind="cubic",
                                  fill_value="extrapolate")(daxis) # type: ignore
                        for data in datas]
                    newcol = np.mean(interp, axis=0)
                    varcol = np.var(interp, axis=0)
                    stdcol = np.std(interp, axis=0)
                    outdata.add_col(colname, newcol, colkind, domain=datum)
                    outdata.add_col("v"+colname, varcol, colkind, domain=datum)
                    outdata.add_col("s"+colname, stdcol, colkind, domain=datum)
            newdata.add_data(outdata)
        
        context.data = newdata

        return CommandResult(True)

    @staticmethod
    def undo(args: NamedTuple, context: Context) -> CommandResult:
        # TODO:
        raise NotImplementedError
