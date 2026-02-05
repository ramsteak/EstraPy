import numpy as np
from numpy import typing as npt

import numpy as np
from numpy import typing as npt

def robust_polyfit(x: npt.ArrayLike, y: npt.ArrayLike, deg: int = 3, n_iter: int = 5, clip: float = 3.5
                   ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.bool_], float]:
    """
    Fit a polynomial to data robustly using iterative sigma-clipping.

    Parameters
    ----------
    x : ArrayLike
        Independent variable.
    y : ArrayLike
        Dependent variable.
    deg : int
        Degree of the polynomial.
    n_iter : int
        Number of sigma-clipping iterations.
    clip : float
        Number of robust sigma units to use for clipping.

    Returns
    -------
    coeffs : np.ndarray
        Polynomial coefficients (highest degree first).
    baseline : np.ndarray
        Fitted polynomial evaluated at x.
    inliers_mask : np.ndarray
        Boolean mask of inlier points after sigma clipping.
    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    mask = np.ones_like(y_arr, dtype=bool)
    
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1.")

    for _ in range(n_iter):
        coeffs = np.polynomial.polynomial.polyfit(x_arr[mask], y_arr[mask], deg)
        baseline = np.polynomial.polynomial.polyval(x_arr, coeffs)
        residuals = y_arr - baseline

        mad = np.median(np.abs(residuals[mask] - np.median(residuals[mask])))
        sigma = 1.4826 * mad

        mask = np.abs(residuals) < clip * sigma

    return coeffs, baseline, mask, sigma



def median_window(arr: npt.ArrayLike, window: int = 3) -> np.ndarray:
    """
    Apply a sliding median filter to a 1D array using NumPy vectorization.

    Parameters
    ----------
    arr : ArrayLike
        Input 1D array
    window : int
        Odd integer size of the median window

    Returns
    -------
    smoothed : np.ndarray
        Smoothed array (same length as input)
    """
    arr = np.asarray(arr)
    if window % 2 == 0:
        raise ValueError("window size must be odd")
    
    half = window // 2
    
    # Pad the array at both ends (reflect) to handle edges
    padded = np.pad(arr, pad_width=half, mode='reflect')
    
    # Create sliding windows using stride tricks
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=window)
    
    # Compute median along the last axis (the window)
    smoothed = np.median(windows, axis=-1)
    
    return smoothed