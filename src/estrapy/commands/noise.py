from dataclasses import dataclass

from ..core.grammarclasses import Command
from ..core.context import Context
from ..grammar.commandparser import CommandParser

from ..core.datastore import Domain

import numpy as np
import polars as pl
from numpy import typing as npt
from matplotlib import pyplot as plt

@dataclass(slots=True)
class Command_noise(Command):
    x: str
    y: str

parse_noise_command = CommandParser(Command_noise, x=None, y=None)
parse_noise_command.add_argument('x', '-x', type=str, default="E")
parse_noise_command.add_argument('y', '-y', type=str, default="a")
# parse_noise_command.add_argument(...)

# def estimate_noise_np(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
#     ...

def estimate_noise_np(xy: pl.DataFrame) -> pl.Series:
    # First convert to numpy, then split even and odd
    ...
# from matplotlib import pyplot as plt
def estimate_noise_pl(xy: pl.DataFrame) -> pl.Series:
    # Split even and odd indices in polars, then convert to numpy arrays
    xy = xy.with_row_index('idx')
    even = xy.filter(pl.col('idx') % 2 == 0).drop('idx')
    odd = xy.filter(pl.col('idx') % 2 == 1).drop('idx')
    even_np = even.to_numpy()
    odd_np = odd.to_numpy()

    interp_even_to_odd = np.interp(odd_np[:,0], even_np[:,0], even_np[:,1])
    interp_odd_to_even = np.interp(even_np[:,0], odd_np[:,0], odd_np[:,1])
    interp = np.zeros(xy.height)
    interp[1::2] = interp_even_to_odd
    interp[0::2] = interp_odd_to_even
    s = pl.Series(interp)
    xy = xy.with_columns(s.alias('interp'))
    #    After statistical analysis of the even-odd method,
    #    Scale by sqrt(0.75)=0.8165 (equispaced data) to|----vvvvv
    #    account for accurate error propagation.        |
    #    Best value found was 0.65 on the current data. |    
    # xy = xy.with_columns(((pl.col('y') - pl.col('interp'))*0.650).alias('noise')).drop('interp')
    xy = xy.with_columns(((pl.col('y') - pl.col('interp'))).alias('noise')).drop('interp')

    # Use even-odd difference method to estimate noise
def execute_noise_command(command: Command_noise, context: Context) -> None:
    for name, file in context.datastore.files.items():
        xy = file.domains[Domain.RECIPROCAL].data.select(pl.col(command.x).alias('x'), pl.col(command.y).alias("y"))
        noise = estimate_noise_pl(xy)
    ...  # Implement the logic for the 'noise' command here
    # plt.show()