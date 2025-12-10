import pandas as pd

from pathlib import Path
from dataclasses import dataclass, field

from scipy.interpolate import BSpline, make_interp_spline # type: ignore

@dataclass
class FeffPath:
    nleg: int
    mult: float
    reff: float
    rnrmav: float
    edge: float
    file: Path
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    interp: dict[str,BSpline] = field(default_factory=dict[str,BSpline])

def load_path(path: Path) -> FeffPath:
    with open(path, 'r') as file:
        while True:
            line = file.readline()
            if line.strip().startswith('--------'):
                break
        
        line = file.readline().split()
        nleg, mult, reff, rnrmav, edge = int(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])
        
        for _ in range(nleg+2):
            file.readline()
        header = ['k','realphc','mag','phase','red','lambda','realp']
        data = pd.read_csv(file, sep='\\s+', names=header) # type: ignore

        data["amp"] = data["mag"] * data["red"]
        data["phs"] = data["phase"] + data["realphc"]

        interp: dict[str, BSpline] = {
            col:make_interp_spline(data['k'], data[col], k=3)
            for col in data.columns[1:]
        }
        
        return FeffPath(
            nleg=nleg,
            mult=mult,
            reff=reff,
            rnrmav=rnrmav,
            edge=edge,
            data=data,
            interp=interp,
            file=path,
        )
