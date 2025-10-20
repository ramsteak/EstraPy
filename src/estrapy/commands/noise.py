import pandas as pd
import numpy as np

from dataclasses import dataclass
from numpy import typing as npt
from functools import partial

from ..core.grammarclasses import CommandArguments, CommandMetadata
from ..core.context import Context
from ..grammar.commandparser import CommandParser

from ..core.datastore import Domain, Column, ColumnType


@dataclass(slots=True)
class CommandArguments_noise(CommandArguments):
    x: str
    y: str


metadata = CommandMetadata(chainable=True, requires_global_context=False, cpu_bound=True)
parse_noise_command = CommandParser(CommandArguments_noise, metadata, x=None, y=None)
parse_noise_command.add_argument('x', '-x', type=str, default='E')
parse_noise_command.add_argument('y', '-y', type=str, default='a')


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


# Ready for chaining and parallel execution in the future
# def single_file_noise(df: DataDomain, expr: Callable[[pd.DataFrame], pd.Series]) -> None:
#     col = Column('noise', None, ColumnType.DATA, expr)
#     df.data['noise'] = expr(df.data)
#     df.columns.append(col)


def execute_noise_command(command: CommandArguments_noise, context: Context) -> None:
    expr = partial(estimate_noise, xcol=command.x, ycol=command.y, name='noise')
    for _, file in context.datastore.files.items():
        noise = expr(file.domains[Domain.RECIPROCAL].data)
        col = Column('noise', None, ColumnType.DATA, expr)
        file.domains[Domain.RECIPROCAL].data['noise'] = noise
        file.domains[Domain.RECIPROCAL].columns.append(col)
