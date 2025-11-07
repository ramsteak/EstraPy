import numpy as np
from numpy import typing as npt
from typing import TypeVar

_N = TypeVar('_N', bound=npt.ArrayLike)

# Uses N*M memory, inefficient and can be slow and crash
# def sliding_l2(f: npt.NDArray[np.floating], g: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]: 
#     f = np.asarray(f)
#     g = np.asarray(g)

#     M = len(f)
#     # All M-length windows of g
#     windows = np.lib.stride_tricks.sliding_window_view(g, M)

#     # Compute L2 for each shift
#     return np.sum((windows - f)**2, axis=1)

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

def correlation_edge_detection(dat:_N, ref:_N, d:int, slide: int, dx: float=1.0) -> float:
    """Perform edge detection using correlation with a reference edge."""
    from .derivative import nderivative
    _dat = np.asarray(dat)[slide:-slide]
    _ref = np.asarray(ref)

    # Calculate nth derivative
    d_data = nderivative(_dat, d=d)
    d_ref = nderivative(_ref, d=d)

    assert not isinstance(d_data, tuple), "nderivative should return a single array for 1 input"
    assert not isinstance(d_ref, tuple), "nderivative should return a single array for 1 input"

    # Normalize inputs. If d = 0, also subtract mean to center data.
    if d == 0:
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
        return shift_x[min_index]
    # If the minimum is at the edge, we cannot fit a parabola
    if min_index == 0 or min_index == N - 1:
        return shift_x[min_index]
    interval = min(min_index, N - min_index - 1, 5) # Use a window of max 5 samples on each side


    _l = l2[min_index - interval : min_index + interval + 1]
    _x = shift_x[min_index - interval : min_index + interval + 1]

    # Fit a quadratic to the local minimum
    coeffs = np.polyfit(_x, _l, 2)
    return - (-coeffs[1] / (2 * coeffs[0]))  # Return the opposite of the vertex of the parabola
    # i.e. the positive shift (to be added to x) to minimize L2
