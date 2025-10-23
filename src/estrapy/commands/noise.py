import pandas as pd
import numpy as np
from lark import Token, Tree

from dataclasses import dataclass
from numpy import typing as npt
from functools import partial
from typing import Self, Callable

from ..core.grammarclasses import CommandArguments, CommandMetadata, Command
from ..core.context import Context, ParseContext, LocalContext
from ..grammar.commandparser import CommandArgumentParser

from ..core.datastore import Domain, Column, ColumnType


@dataclass(slots=True)
class CommandArguments_Noise(CommandArguments):
    x: str
    y: str


parse_noise_command = CommandArgumentParser(CommandArguments_Noise)
parse_noise_command.add_argument('x', '-x', type=str, default='E')
parse_noise_command.add_argument('y', '-y', type=str, default='a')

@dataclass(slots=True)
class LocalContext_Noise(LocalContext):
    expr: Callable[[pd.DataFrame], pd.Series]

@dataclass(slots=True)
class Command_Noise(Command[CommandArguments_Noise, LocalContext_Noise]):
    @classmethod
    def parse(cls: type[Self], commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext) -> Self:
        arguments = parse_noise_command.parse(commandtoken, tokens)
        metadata = CommandMetadata(initialize_context=False, finalize_context=False, execution_context=False, execute_with='sequential')
        return cls(
            line=commandtoken.line or -1,
            name=commandtoken.value,
            args=arguments,
            meta=metadata,
        )
    
    async def initialize(self, context: Context) -> None:
        expr = partial(estimate_noise, xcol=self.args.x, ycol=self.args.y, name='noise')
        self.local = LocalContext_Noise(expr)
    
    async def execute_on(self, page: str, context: Context) -> None:
        assert self.local is not None

        pagedata = context.datastore.pages[page]
        domain = pagedata.domains[Domain.RECIPROCAL]
        noise = self.local.expr(domain.data)

        col = Column('noise', None, ColumnType.DATA, self.local.expr)
        # TODO: properly add column
        domain.data['noise'] = noise
        domain.columns.append(col)

    async def finalize(self, context: Context) -> None:
        pass


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
