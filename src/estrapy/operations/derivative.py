import numpy as np
from numpy import typing as npt
from typing import Literal, Any
from scipy.interpolate import UnivariateSpline # pyright: ignore[reportMissingTypeStubs]
from scipy.signal import savgol_filter # pyright: ignore[reportUnknownVariableType, reportMissingTypeStubs]

def derivative2(_f: npt.ArrayLike, dx: float | npt.ArrayLike = 1.0) -> npt.NDArray[Any]:
    """
    Calculate the second-order derivative of a 1D array with optional smoothing.
    
    Parameters
    ----------
    f : array_like
        Input 1D array
    dx : float or array_like, optional
        Spacing between points. Can be:
        - A scalar (uniform spacing)
        - A 1D array of positions (non-uniform grid)
        Default is 1.0
    smooth : float, optional
        Smoothing parameter (0 = no smoothing). Used only for non-uniform grids.
        Larger values = more smoothing. Typical range: 0-1.
        Default is 0.0
    
    Returns
    -------
    array_like
        Second derivative of f
    """
    f = np.asanyarray(_f)
    
    # Validate 1D input
    if f.ndim != 1:
        raise ValueError(f"Expected 1D array, got {f.ndim}D array")
    
    if f.shape[0] < 3:
        raise ValueError(
            "Array too small to calculate second derivative, "
            "at least 3 elements are required"
        )
    
    # Handle dtype conversion
    if np.issubdtype(f.dtype, np.integer):
        f = f.astype(np.float64)
        otype = np.float64
    elif np.issubdtype(f.dtype, np.inexact):
        otype = f.dtype
    else:
        otype = np.float64
    
    out = np.empty_like(f, dtype=otype)
    
    # Process spacing
    dx = np.asanyarray(dx)
    
    if dx.ndim == 0:
        # Uniform spacing - simple formula
        out[1:-1] = (f[:-2] - 2*f[1:-1] + f[2:]) / (dx**2)
    else:
        # Non-uniform spacing
        if dx.ndim != 1:
            raise ValueError("dx must be either a scalar or 1D array")
        
        if len(dx) != len(f):
            raise ValueError("When 1D, dx must match the length of f")
        
        if np.issubdtype(dx.dtype, np.integer):
            dx = dx.astype(np.float64)
        
        # Calculate spacing differences
        diffx = np.diff(dx)
        
        # Check if actually uniform (optimization)
        if (diffx == diffx[0]).all():
            out[1:-1] = (f[:-2] - 2*f[1:-1] + f[2:]) / (diffx[0]**2)
        else:
            dx1 = diffx[:-1]
            dx2 = diffx[1:]
            
            A = 1 / (dx1 * (dx1 + dx2))
            B = -1 / (dx1 * dx2)
            C = 1 / (dx2 * (dx1 + dx2))
            
            out[1:-1] = 2.0 * (A * f[:-2] + B * f[1:-1] + C * f[2:])
    
    # Handle edges with linear extrapolation
    if dx.ndim == 0:
        out[0] = out[1] + (out[1] - out[2])
        out[-1] = out[-2] + (out[-2] - out[-3])
    else:
        diffx = np.diff(dx)
        out[0] = out[1] + (out[1] - out[2]) / diffx[1] * diffx[0]
        out[-1] = out[-2] + (out[-2] - out[-3]) / diffx[-2] * diffx[-1]
    
    return out


def nderivative(
    _f: npt.ArrayLike, 
    order: int, 
    dx: float | npt.ArrayLike = 1.0,
    smooth: float = 0.0,
    method: Literal['decompose', 'spline', 'savgol'] = 'decompose'
) -> npt.NDArray[Any]:
    """
    Calculate the nth-order derivative of a 1D array with optional smoothing.
    
    Parameters
    ----------
    f : array_like
        Input 1D array
    order : int
        Order of derivative to compute (must be >= 0)
    dx : float or array_like, optional
        Spacing between points. Can be:
        - A scalar (uniform spacing)
        - A 1D array of x positions (non-uniform grid)
        Default is 1.0
    smooth : float, optional
        Smoothing parameter. Meaning depends on method:
        - 'decompose': not used (raw finite differences)
        - 'spline': smoothing factor for spline (0=interpolating, >0=smoothing)
                   Typical range: 0-10. Larger = more smoothing.
        - 'savgol': window length ratio (0-1). Window = int(smooth * len(f))
                   Must result in odd window >= order+2. Typical: 0.05-0.2
        Default is 0.0
    method : {'decompose', 'spline', 'savgol'}, optional
        Derivative computation method:
        - 'decompose': Repeated 2nd derivatives (your original method)
                      Fast, no smoothing. Best for clean data.
        - 'spline': Spline interpolation with analytical derivatives
                   Works with non-uniform grids, built-in smoothing.
                   Best for noisy non-uniform data.
        - 'savgol': Savitzky-Golay filter (local polynomial fit)
                   Only for uniform grids. Best for uniform noisy data.
        Default is 'decompose'
    
    Returns
    -------
    array_like
        nth derivative of f
    
    Notes
    -----
    Noise amplification in numerical differentiation grows exponentially with
    derivative order. For noisy data, use smoothing or keep order <= 4.
    
    Examples
    --------
    >>> x = np.linspace(0, 10, 100)
    >>> y = np.sin(x) + 0.1*np.random.randn(100)
    >>> 
    >>> # Clean data: use decompose
    >>> d2 = nderivative(y, order=2, dx=x[1]-x[0])
    >>> 
    >>> # Noisy uniform data: use savgol
    >>> d2_smooth = nderivative(y, order=2, dx=x[1]-x[0], 
    ...                         smooth=0.1, method='savgol')
    >>> 
    >>> # Non-uniform noisy data: use spline
    >>> x_nonuniform = x**1.5  # non-uniform spacing
    >>> d2_spline = nderivative(y, order=2, dx=x_nonuniform,
    ...                         smooth=1.0, method='spline')
    """
    f = np.asanyarray(_f)
    
    # Validate 1D input
    if f.ndim != 1:
        raise ValueError(f"Expected 1D array, got {f.ndim}D array")
    
    if order < 0:
        raise ValueError(f"Order must be non-negative, got {order}")
    
    if order == 0:
        return f
    
    # Method selection
    if method == 'decompose':
        return _derivative_decompose(f, order, dx)
    elif method == 'spline':
        return _derivative_spline(f, order, dx, smooth)
    elif method == 'savgol':
        return _derivative_savgol(f, order, dx, smooth)
    else:
        raise ValueError(f"Unknown method: {method}")


def _derivative_decompose(f: npt.ArrayLike, order: int, dx: float | npt.ArrayLike) -> npt.NDArray[Any]:
    """Original decomposition method: repeated 2nd derivatives + gradient"""
    if order == 1:
        return np.gradient(f, dx)
    elif order == 2:
        return derivative2(f, dx)
    
    # For higher orders: decompose into 2nd derivatives + optional 1st derivative
    result = f
    for _ in range(order // 2):
        result = derivative2(result, dx)
    
    if order % 2 != 0:
        result = np.asanyarray(np.gradient(result, dx))
    
    return result


def _derivative_spline(f: npt.ArrayLike, order: int, dx: float | npt.ArrayLike, smooth: float) -> npt.NDArray[Any]:
    """
    Spline-based derivative with smoothing.
    Works for non-uniform grids and provides built-in smoothing.
    """
    f = np.asanyarray(f)
    dx = np.asanyarray(dx)
    
    # Construct x positions
    if dx.ndim == 0:
        # Uniform spacing: create x positions
        x = np.arange(len(f)) * float(dx)
    else:
        # Non-uniform: dx is actually x positions
        x = dx
    
    # Create smoothing spline
    # k=5 allows up to 5th derivative, adjust if needed
    k = min(5, len(f) - 1)
    
    if smooth == 0:
        # Interpolating spline (no smoothing)
        spline = UnivariateSpline(x, f, k=k, s=0)
    else:
        # Smoothing spline
        # s parameter controls smoothing: s should be around std(noise)^2 * len(f)
        spline = UnivariateSpline(x, f, k=k, s=smooth * len(f))
    
    # Get derivative
    deriv_spline = spline.derivative(n=order)
    
    return deriv_spline(x)


def _derivative_savgol(
    f: npt.NDArray[Any],
    order: int,
    dx: float | npt.ArrayLike,
    smooth: float
) -> npt.NDArray[Any]:
    """
    Savitzky-Golay filter derivative.
    Only works for uniform grids but very effective for noisy data.
    """
    f = np.asanyarray(f)
    dx = np.asanyarray(dx)
    
    if dx.ndim != 0:
        raise ValueError("Savitzky-Golay filter only works with uniform spacing")
    
    if smooth <= 0:
        raise ValueError(f"For savgol method, smooth must be > 0, got {smooth}")
    
    if smooth >= 1:
        raise ValueError(f"For savgol method, smooth must be < 1 (ratio), got {smooth}")
    
    # Calculate window length from smooth ratio
    window_length = int(smooth * len(f))
    
    # Ensure window is odd
    if window_length % 2 == 0:
        window_length += 1
    
    # Minimum window size: must be >= order + 2
    min_window = order + 2
    if window_length < min_window:
        window_length = min_window
        if window_length % 2 == 0:
            window_length += 1
    
    # Window can't exceed array length
    if window_length > len(f):
        raise ValueError(
            f"Window length ({window_length}) exceeds array length ({len(f)}). "
            f"Reduce smooth parameter or use longer array."
        )
    
    # Polynomial order: should be >= derivative order, but not too high
    polyorder = max(order + 1, min(4, window_length - 2))
    
    # Apply Savitzky-Golay filter
    result = savgol_filter(f, window_length, polyorder, deriv=order, delta=float(dx))
    
    return result