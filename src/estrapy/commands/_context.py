import pandas as pd

from pathlib import Path
from typing import Any, NamedTuple
from enum import Enum


class Options(NamedTuple):
    interactive: bool
    verbose: bool
    version: tuple[int, ...]
    logcolor: bool
    updateinput: bool
    cache: bool
    other: dict[str, Any]


class Paths(NamedTuple):
    inputfile: Path | None
    currentdir: Path
    workdir: Path
    outputdir: Path
    userdir: Path
    logfile: Path
    configfile: Path


class AxisType(Enum):
    ENERGY = "eV"
    KVECTOR = "k"
    DISTANCE = "A"
    QVECTOR = "q"
    INDEX = "i"


class SignalType(Enum):
    FLUORESCENCE = "fl"
    TRANSMITTANCE = "tr"
    INTENSITY = "i"


class MetaData(NamedTuple):
    axis: AxisType
    signaltype: SignalType | None
    refsigtype: SignalType | None
    name: str
    vars: dict[str, str | int | float]
    history: list
    refE0: float | None
    E0: float | None


class Data:
    def __init__(self, dataframe: pd.DataFrame, metadata: MetaData) -> None:
        self.data = dataframe
        self.metadata = metadata


class DataStore:
    def __init__(self) -> None:
        self.data: list[Data] = []

    def add_data(self, data: Data) -> None:
        self.data.append(data)


class Directives(NamedTuple):
    clear: bool
    noplot: bool


class Context:
    def __init__(
        self,
        paths: Paths,
        config: dict[str, Any],
        options: Options,
        directives: Directives,
    ) -> None:
        self.paths = paths
        self.config = config
        self.options = options
        self.directives = directives

        self.vars: dict[str, Any] = {}
        self.data = DataStore()
