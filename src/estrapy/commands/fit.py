from __future__ import annotations

from typing import NamedTuple
from logging import getLogger

from ._context import Context, Domain, range_to_index, DataColType, Column
from ._handler import CommandHandler, Token, CommandResult
from ._numberunit import NumberUnit, parse_range, NumberUnitRange, actualize_range
from ._parser import CommandParser


class Shell(NamedTuple):
    feff_file: str
    

class Args_Fit(NamedTuple):
    bounds: NumberUnitRange


class Fit(CommandHandler):
    @staticmethod
    def parse(tokens: list[Token], context: Context) -> Args_Fit:
        parser = CommandParser("fit", description="Fits the experimental data to a number of shells, with given parameters.")
        parser.add_argument("range", nargs=2)
        parser.add_argument("--mode", "-m", choices=["k", "r"], default="k", help="Fitting mode: k-space or r-space. Default is k-space.")
        parser.add_argument("--dir", "-d", type=str, default="./feff", help="Directory where the FEFF files are located. Default is in a separate feff folder.")
        parser.add_argument("--param", action="append", nargs="+", help="Define a fitting parameter. Usage: --param name initial [min max step]. Can be used multiple times.")
        parser.add_argument("--shell", action="append", nargs="+", help="Define a fitting shell. Usage: --shell feff_file -param1 param1 -param2 param2 ... Can be used multiple times.")

        args = parser.parse(tokens)

        return Args_Fit()

    @staticmethod
    def execute(args: Args_Fit, context: Context) -> CommandResult:
        log = getLogger("fit")

        domain = args.bounds.domain or Domain.RECIPROCAL
        _axes = [data.get_col_(data.datums[domain].default_axis) for data in context.data] # type: ignore
        range = actualize_range(args.bounds, _axes, "eV")
        
        for data in context.data:
            idx = range_to_index(data, range)
            data.datums[domain].df = data.datums[domain].df.loc[idx, :]
                
            log.debug(f"{data.meta.name}: Cut data in the range {range.lower.value:0.3f}{range.lower.unit} ~ {range.upper.value:0.3f}{range.upper.unit}") # type: ignore

        return CommandResult(True)

    @staticmethod
    def undo(args: NamedTuple, context: Context) -> CommandResult:
        # TODO:
        raise NotImplementedError
