import pandas as pd

from pathlib import Path
from typing import Any, NamedTuple
from enum import Enum
from dataclasses import dataclass, field


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


@dataclass(slots=True)
class MetaData:
    axis: AxisType
    signaltype: SignalType | None
    refsigtype: SignalType | None
    name: str
    path: Path
    vars: dict[str, str | int | float]
    run: dict[str, Any] = field(default_factory=dict)
    refE0: float | None = None
    E0: float | None = None


class Data:
    def __init__(self, dataframe: pd.DataFrame, metadata: MetaData) -> None:
        # df holds the real-space data
        self.df = dataframe
        # fd holds the Fourier-space data
        self.fd = pd.DataFrame()
        # meta holds the metadata
        self.meta = metadata


class DataStore:
    def __init__(self) -> None:
        self.data: list[Data] = []

    def add_data(self, data: Data) -> None:
        self.data.append(data)

    def __iter__(self):
        return self.data.__iter__()


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
