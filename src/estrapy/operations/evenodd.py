import numpy as np
from numpy import typing as npt

from typing import Any

def diff_even(x: npt.NDArray[Any], y: npt.NDArray[Any]) -> np.ndarray:
    x = np.asarray(x)
    y = np.asarray(y)

    out = np.empty_like(y)

    x0 = x[:-2]
    x1 = x[2:]
    t = (x[1:-1] - x0) / (x1 - x0)
    out[1:-1] = y[:-2] + t * (y[2:] - y[:-2])
    np.subtract(y, out, out=out)

    out[0] = 0
    out[-1] = 0

    return out
