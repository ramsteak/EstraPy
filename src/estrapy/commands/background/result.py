import numpy as np

from dataclasses import dataclass
from numpy import typing as npt

from ...core.context import CommandResult

@dataclass(slots=True)
class BackgroundResult:
    background: npt.NDArray[np.floating]

@dataclass(slots=True)
class CommandResult_Background(CommandResult):
    ...
