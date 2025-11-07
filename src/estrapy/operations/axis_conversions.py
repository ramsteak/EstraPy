import numpy as np
from numpy import typing as npt
from typing import TypeVar

KCONST = 0.26246842396479844746511289042048746313978437879180
SCONST = 0.51231672231618445138819837616078387648071784401809  # sqrt(KCONST)
KICNST = 3.8099821109685837311852822702410407881602562961373  # 1 / KCONST

_N = TypeVar('_N', bound=npt.ArrayLike)

# Conversion functions between energy (E) and wave vector (k).
# k is allowed to be negative to represent direction; physically, k is always
# non-negative and negative values should be culled.
def E_to_k(v: _N, E0: float) -> _N:
    return SCONST * np.sqrt(np.abs(v - E0)) * np.sign(v-E0) # type: ignore
def k_to_E(v: _N, E0: float) -> _N:
    return KICNST * np.sign(v) * v**2 + E0  # type: ignore

class Nyquist:
    @staticmethod
    def span_diff(axis: npt.ArrayLike) -> tuple[float, float]:
        """Get the Nyquist span and difference for the transform axis, given the original axis."""
        span = float(np.pi / (2 * np.diff(axis).max()))
        diff = float(np.pi / (2 * np.ptp(axis)))
        return span, diff
    
    @staticmethod
    def axis(axis: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """Get the Nyquist axis for the transform axis, given the original axis."""
        span, diff = Nyquist.span_diff(axis)
        npoints = int(np.ceil(span / diff))
        return np.linspace(0, span, npoints)
    
    @staticmethod
    def information(axis: npt.ArrayLike, reciprocalaxis: npt.ArrayLike | None = None) -> int:
        """Get the Nyquist maximum information content."""
        if reciprocalaxis is None:
            return int(np.ptp(axis) / np.max(np.diff(axis)))
        else:
            return int(2 * np.ptp(axis) * np.ptp(reciprocalaxis) / np.pi)
