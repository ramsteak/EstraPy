import numpy as np
import numpy.typing as npt
from typing import Sequence
from numba import njit, prange # type: ignore missing stub

apodizer_functions = [
    'rectangular',
    'hanning',
    'hamming',
    'welch',
    'sine',
    'blackman',
    'gaussian',
    'exponential',
]

def _window(r: npt.NDArray[np.floating], type: str, p:float=3) -> npt.NDArray[np.floating]:
    r1 = np.clip(r, 0.0, 1.0)
    # The windows are centered at 0 and evaluate to 0 at r=1 (different from usual definition)
    match type:
        case 'rectangular':
            return (r1 <= 1.0).astype(np.floating)
        case 'hanning':
            return 0.5 + 0.5 * np.cos(np.pi * r1)
        case 'hamming':
            return 0.54 - 0.46 * np.cos(2 * np.pi * r1)
        case 'welch':
            return 1.0 - r1**2
        case 'sine':
            return np.cos(np.pi * r1 / 2)
        case 'blackman':
            return 0.42 - 0.5 * np.cos(2 * np.pi * r1) + 0.08 * np.cos(4 * np.pi * r1)
        case 'gaussian':
            return np.exp(-r**2 * p)
        case 'exponential':
            return np.exp(-r * p)
        case _:
            raise ValueError(f'Unknown window type "{type}".')

def flattop_window(x: npt.NDArray[np.floating], xs: Sequence[float], type: str, p:float=3) -> npt.NDArray[np.floating]:
    """
    Compute a generalized flattop window over an array of coordinates.

    This function constructs a smooth window that is flat (value 1) in the central
    region and tapers smoothly toward zero (or another shape depending on `type`)
    on both sides. The tapers are defined by the four positions `xs = [x00, x01, x10, x11]`:

                              ┌── Flat region ──┐
          x00                x01               x10                 x11
     zero  │<-- left taper -->│       one       │<-- right taper -->│  zero

    The window is defined as follows:
      - For `x` within [x01, x10]: window = 1 (flat region)
      - For `x < x01`: window tapers according to the left parameters (x00, x01)
      - For `x > x10`: window tapers according to the right parameters (x10, x11)
      - For `x < x00` or `x > x11`: window = 0 (beyond taper region)
      - Infinite or degenerate bounds are handled gracefully:
          * `x00 == -inf` or `x01 == -inf` → infinite on left (no taper)
          * `x10 == inf` or `x11 == inf` → infinite on right (no taper)
          * `x00 == x01` or `x10 == x11` → step function edge

    The taper shape is defined by the `type` argument and parameter `p`, which are
    passed to the internal `_window(r, type, p)` function. The relative position `r`
    represents the normalized distance from the edge:
      - `r = 0` → flat region
      - `r = 1` → outer edge (taper reaches zero)
      - `r > 1` → outside window (clipped to 1)
      - `r = inf` → hard step edge

    Parameters
    ----------
    x : ndarray of float
        Input coordinates at which to evaluate the window.
    xs : sequence of float
        Four boundary positions `[x00, x01, x10, x11]` defining left/right taper
        regions and the flat top. Must satisfy `x00 <= x01 <= x10 <= x11`.
        Infinite values are allowed to represent open boundaries.
    type : str
        The taper type to use. Must be one of the supported window types in `_window`
        (e.g., 'rectangular', 'hanning', 'hamming', 'welch', 'sine', 'blackman',
        'gaussian', 'exponential').
    p : float, optional
        Shape parameter used by some window types (default = 3).

    Returns
    -------
    ndarray of float
        Array of the same shape as `x`, containing the windowed values between 0 and 1.

    Raises
    ------
    ValueError
        If `xs` does not contain exactly four values or if the order of boundaries
        is invalid (`x00 > x01`, `x10 < x01`, or `x11 < x10`).

    Examples
    --------
    >>> x = np.linspace(-2, 2, 1000)
    >>> xs = [-1.5, -1.0, 1.0, 1.5]
    >>> w = flattop_window(x, xs, type='hanning')
    >>> plt.plot(x, w)  # Smooth flat-top window with Hanning tapers
    """

    x00,x01,x10,x11 = xs
    if x00 > x01 or x10 > x11 or x01 > x10:
        raise ValueError('Invalid flattop window specification. Require x00 <= x01 <= x10 <= x11.')

    r = np.zeros_like(x)
    # Left side
    lidx = x < x01
    match (x00, x01):
        case (x00, x01) if x00 == -np.inf or x01 == -np.inf: # infinite window on left -> ignore left side
            pass
        case (x00, x01) if x00 == x01: # degenerate width -> step function on left
            r[lidx] = np.inf
        case (x00, x01):
            r[lidx] = (x01 - x[lidx]) / (x01 - x00)

    # Right side
    ridx = x > x10
    match (x10, x11):
        case (x10, x11) if x10 == np.inf or x11 == np.inf:
            pass
        case (x10, x11) if x10 == x11:
            r[ridx] = np.inf
        case (x10, x11):
            r[ridx] = (x[ridx] - x10) / (x11 - x10)

    return _window(r, type, p)

@njit(parallel=True, nogil=True, cache=True) # type: ignore missing stub
def fourier(
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating | np.complexfloating],
    r: npt.NDArray[np.floating],
) -> npt.NDArray[np.complexfloating]:
    dx = np.diff(x)
    yl, yr = y[:-1] * dx, y[1:] * dx
    nr = len(r)
    result = np.zeros(nr, dtype=np.complex128)
    
    for j in prange(nr):
        real, imag = 0.0, 0.0
        for i in range(len(x) - 1):
            FTl = 2 * x[i] * r[j]
            FTr = 2 * x[i + 1] * r[j]
            real += yl[i] * np.cos(FTl) + yr[i] * np.cos(FTr)
            imag += yl[i] * np.sin(FTl) + yr[i] * np.sin(FTr)
        result[j] = real + 1j * imag
    
    return result / np.sqrt(4 * np.pi)
