from dataclasses import dataclass

from ..core.grammarclasses import Command
from ..core.context import Context
from ..grammar.commandparser import CommandParser

from ..core.datastore import Domain, Column, ColumnType

import numpy as np
from numpy import typing as npt

@dataclass(slots=True)
class Command_noise(Command):
    x: str
    y: str

parse_noise_command = CommandParser(Command_noise, x=None, y=None)
parse_noise_command.add_argument('x', '-x', type=str, default="E")
parse_noise_command.add_argument('y', '-y', type=str, default="a")


def estimate_noise(xy: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    # Split even and odd indices and perform linear interpolation
    even, odd = xy[::2], xy[1::2]

    interp_even_to_odd = np.interp(odd[:,0], even[:,0], even[:,1])
    interp_odd_to_even = np.interp(even[:,0], odd[:,0], odd[:,1])
    interp = np.zeros(xy.shape[0])
    interp[1::2] = interp_even_to_odd
    interp[0::2] = interp_odd_to_even
    #    After statistical analysis of the even-odd method,
    #    Scale by sqrt(0.75)=0.8165 (equispaced data) to|----vvvvv
    #    account for accurate error propagation.        |
    #    Best value found was 0.65 on the current data. |    
    return interp

    # Use even-odd difference method to estimate noise
def execute_noise_command(command: Command_noise, context: Context) -> None:
    for name, file in context.datastore.files.items():
        xy = file.domains[Domain.RECIPROCAL].data[[command.x, command.y]].to_numpy()
        noise = estimate_noise(xy)
        # TODO: Handle automatic addition of columns
        file.domains[Domain.RECIPROCAL].data['noise'] = noise
        file.domains[Domain.RECIPROCAL].columns.append(Column('noise', None, ColumnType.DATA))
    ...  # Implement the logic for the 'noise' command here
    # plt.show()