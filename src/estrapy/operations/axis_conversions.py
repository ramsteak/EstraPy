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

def rt(v: _N) -> _N:
    """Square root function that preserves the sign of the input."""
    # this implementation is faster for small arrays (200 < size < 40_000)
    return np.sign(v) * np.sqrt(np.abs(v)) # type: ignore
    # this implementation is faster for large arrays (size > 40_000) (and oddly enough very small arrays with size < 200)
    # out = np.abs(v)
    # np.sqrt(out, out=out)
    # np.copysign(out, v, out=out)
    # return out  # type: ignore

def sq(v: _N) -> _N:
    """Square function that preserves the sign of the input."""
    # This implementation is always faster
    out = np.abs(v)
    out *= v
    return out # type: ignore

def E_to_k(v: _N, E0: float) -> _N:
    """Convert energy E to k given energy shift E0."""
    return SCONST * rt(v - E0) # type: ignore

def k_to_E(v: _N, E0: float) -> _N:
    """Convert k to energy E given energy shift E0."""
    return KICNST * sq(v) + E0  # type: ignore

def k_to_k1(v: _N, de: float) -> _N:
    """Convert k to shifted k1 given energy shift de.
    Positive de shifts E1 to a lower value."""
    tmp = sq(v)
    tmp += de * KCONST # type: ignore
    return rt(tmp) # type: ignore


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
