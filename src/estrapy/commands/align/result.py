import numpy as np

from dataclasses import dataclass
from numpy import typing as npt

from ...core.context import CommandResult
from ...core.datastore import Domain, DataPage, ColumnDescription, ColumnKind
from ...core.number import Unit


@dataclass(slots=True)
class AlignResult:
    background: npt.NDArray[np.floating]

@dataclass(slots=True)
class CommandResult_Align(CommandResult):
    ...


def apply_shift_to_page(page: DataPage, edge_energy: float | None, shift_energy: float) -> None:
    if edge_energy is not None:
        page.meta['refE0'] = edge_energy

    domain = page.domains[Domain.RECIPROCAL]
    E_column = ColumnDescription('E', Unit.EV, ColumnKind.AXIS, deps=['E'], calc=lambda df, shift=shift_energy: df['E'] - shift, labl='Energy [eV]')
    domain.add_column('E', E_column)
