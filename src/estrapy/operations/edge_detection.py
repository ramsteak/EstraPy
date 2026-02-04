import numpy as np
from numpy import typing as npt
from dataclasses import dataclass

def sliding_l2(f: npt.NDArray[np.floating], g: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    f = np.asarray(f, float)
    g = np.asarray(g, float)
    M = len(f)

    # Precompute energy of f
    f2 = np.sum(f * f)

    # Energy of each sliding window of g
    g2 = np.convolve(g * g, np.ones(M), mode='valid')

    # Cross term (correlation)
    corr = np.convolve(g, f[::-1], mode='valid')

    return g2 + f2 - 2*corr

@dataclass(slots=True, frozen=True)
class SlidingL2Result:
    x: float
    xidx: int
    poly: np.poly1d
    shifts: npt.NDArray[np.floating]
    l2_values: npt.NDArray[np.floating]
    fitting_idx: npt.NDArray[np.bool]
    message: str

def correlation_edge_detection(dat: npt.ArrayLike, ref: npt.ArrayLike, derivative:int, slide: int, dx: float=1.0) -> SlidingL2Result:
    """Perform edge detection using correlation with a reference edge."""
    from .derivative import nderivative
    _dat = np.asarray(dat)[slide:-slide]
    _ref = np.asarray(ref)

    # Calculate nth derivative
    d_data = nderivative(_dat, order=derivative)
    d_ref = nderivative(_ref, order=derivative)

    assert not isinstance(d_data, tuple), "nderivative should return a single array for 1 input"
    assert not isinstance(d_ref, tuple), "nderivative should return a single array for 1 input"

    # Normalize inputs. If d = 0, also subtract mean to center data.
    if derivative == 0:
        d_data = (d_data - np.mean(d_data)) / np.std(d_data)
        d_ref = (d_ref - np.mean(d_ref)) / np.std(d_ref)
    else:
        d_data = (d_data) / np.std(d_data)
        d_ref = (d_ref) / np.std(d_ref)

    # Compute sliding L2 norm
    l2 = sliding_l2(d_data, d_ref)
    N = len(l2)
    # Compute the corresponding shift values for the L2 array
    shift_x = np.linspace(-(N-1)/2,(N-1)/2,N)*dx

    # Find the minimum L2 value around the minimum, to get sub-sample accuracy
    min_index = np.argmin(l2)
    # If the minimum is exactly zero, return it directly
    if l2[min_index] == 0.0:
        return SlidingL2Result(
            x = float(shift_x[min_index]),
            xidx = int(min_index),
            poly = np.poly1d([]),
            shifts = shift_x,
            l2_values = l2,
            fitting_idx = np.zeros_like(shift_x, dtype=bool),
            message = 'Exact match found'
        )
    
    # If the minimum is at the edge, we cannot fit a parabola
    if min_index == 0 or min_index == N - 1:
        return SlidingL2Result(
            x = float(shift_x[min_index]),
            xidx = int(min_index),
            poly = np.poly1d([]),
            shifts = shift_x,
            l2_values = l2,
            fitting_idx = np.zeros_like(shift_x, dtype=bool),
            message = 'Minimum at edge, no sub-sample fitting possible'
        )
    
    # Use a window of max 5 samples on each side of the minimum for fitting
    interval_width = min(min_index, N - min_index - 1, 5)

    fitidx = np.zeros_like(shift_x, dtype=bool)
    fitidx[min_index - interval_width : min_index + interval_width + 1] = True
    _x, _l = shift_x[fitidx], l2[fitidx]

    # Fit a quadratic to the local minimum
    coeffs = np.polyfit(_x, _l, 2)
    # Return the opposite of the vertex of the parabola,  i.e. the positive shift (to be added to x) to minimize L2
    minimum = -(-coeffs[1] / (2 * coeffs[0]))

    return SlidingL2Result(
        x = float(minimum),
        xidx = int(min_index),
        poly = np.poly1d(coeffs),
        shifts = shift_x,
        l2_values = l2,
        fitting_idx = fitidx,
        message = 'Success'
    )