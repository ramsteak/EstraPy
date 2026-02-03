import pandas as pd
import numpy as np
from lark import Token, Tree

from dataclasses import dataclass
from numpy import typing as npt
from functools import partial
from typing import Self

from ..core.context import Command, CommandResult
from ..core.context import Context, ParseContext
from ..core.commandparser2 import CommandArgumentParser, CommandArguments, field_arg

from ..core.datastore import Domain, ColumnDescription, ColumnKind


@dataclass(slots=True)
class CommandArguments_Noise(CommandArguments):
    x: str = field_arg(
        flags=['--xaxiscol'],
        type=str,
        required=False,
        default='E'
    )

    y: str = field_arg(
        flags=['--yaxiscol'],
        type=str,
        required=False,
        default='a'
    )

@dataclass(slots=True)
class CommandResult_Noise(CommandResult):
    ...


parse_noise_command = CommandArgumentParser(CommandArguments_Noise, 'noise')


@dataclass(slots=True)
class Command_Noise(Command[CommandArguments_Noise, CommandResult_Noise]):
    @classmethod
    def parse(
        cls: type[Self], commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext
    ) -> Self:
        arguments = parse_noise_command.parse(commandtoken, tokens)

        return cls(
            line=commandtoken.line or -1,
            name=commandtoken.value,
            args=arguments,
        )

    def execute(self, context: Context) -> CommandResult_Noise:
        expr = partial(estimate_noise, xcol=self.args.x, ycol=self.args.y, name='sa')

        for _, page in context.datastore.pages.items():
            datadomain = page.domains[Domain.RECIPROCAL]
            col = ColumnDescription('sa', None, ColumnKind.DATA, [self.args.x, self.args.y], expr, 'Standard Deviation of absorbance', ['a'])
            datadomain.add_column(col.name, col)
        
        return CommandResult_Noise()


def _estimate_noise_np(xy: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    # Split even and odd indices and perform linear interpolation
    even, odd = xy[::2], xy[1::2]

    interp_even_to_odd = np.interp(odd[:, 0], even[:, 0], even[:, 1])
    interp_odd_to_even = np.interp(even[:, 0], odd[:, 0], odd[:, 1])
    interp = np.zeros(xy.shape[0])
    interp[1::2] = interp_even_to_odd
    interp[0::2] = interp_odd_to_even
    #    After statistical analysis of the even-odd method,
    #    Scale by sqrt(0.75)=0.8165 (equispaced data) to|----vvvvv
    #    account for accurate error propagation.        |
    #    Best value found was 0.65 on the current data. |
    return interp


def estimate_noise(df: pd.DataFrame, xcol: str, ycol: str, name: str) -> pd.Series:
    xy = df[[xcol, ycol]].to_numpy()
    noise = _estimate_noise_np(xy)
    return pd.Series(noise, index=df.index).rename(name)
