import warnings
import numpy as np
from numpy import typing as npt

from scipy.interpolate import BSpline
from scipy.linalg import lstsq

# Import the shared mathematical functions
from .fourier import flattop_window, fourier

class AutobkFitIllConditionedWarning(RuntimeWarning):
    """Raised when the least-squares matrix for AUTOBK is highly ill-conditioned.
    The fit may be numerically unstable. Consider reducing the number of knots 
    or increasing the R-background cutoff."""
    pass

class AutobkFitInsufficientDataWarning(RuntimeWarning):
    """Raised when the R-grid has too few points to reliably fit the specified number of knots."""
    pass

class AutobkFitter:
    def __init__(self,
                 rbkg: float,
                 kweight: float,
                 nknots: int,
                 k_range: tuple[float, float],
                 degree: int = 3):
        """Initializes the AUTOBK Fitter.
        
        Parameters
        ----------
        rbkg : float
            The R-space cutoff distance below which the EXAFS signal is minimized (in Å).
        kweight : float
            The k-weighting factor applied to the data. (Mainly stored for reference, 
            as the caller is expected to apply the weighting before passing data).
        nknots : int
            The number of equidistant knots to place in the k-range.
        k_range : tuple[float, float]
            The (min, max) bounds of the k-space data.
        degree : int, optional
            The degree of the B-spline polynomials (default is 3, cubic).
        """
        self.rbkg = float(rbkg)
        self.kweight = float(kweight)
        self.nknots = max(2, int(nknots))
        self.k_range = k_range
        self.degree = max(1, int(degree))

        # Generate uniform interior knots over the specific k-range
        self.inner_knots = np.linspace(k_range[0], k_range[1], self.nknots)

        # Pad knots for B-spline evaluation at the boundaries
        if self.nknots > 1:
            dk = self.inner_knots[1] - self.inner_knots[0]
        else:
            dk = 1.0

        knots_left = self.inner_knots[0] - np.arange(self.degree, 0, -1) * dk
        knots_right = self.inner_knots[-1] + np.arange(1, self.degree + 1) * dk
        
        self.t = np.concatenate([knots_left, self.inner_knots, knots_right])
        self.num_coeffs = len(self.t) - self.degree - 1

    def fit(self, x: npt.NDArray[np.floating], y: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Fit the AUTOBK spline to minimize low-R components with edge regularization."""
        x_in = np.asarray(x, dtype=np.float64)
        y_in = np.asarray(y, dtype=np.float64)

        if len(x_in) < self.num_coeffs:
            raise ValueError(f"Not enough data points ({len(x_in)}) to fit {self.num_coeffs} B-spline coefficients.")

        # 1. Construct B-spline design matrix X
        X = np.zeros((len(x_in), self.num_coeffs), dtype=np.float64)
        for j in range(self.num_coeffs):
            c = np.zeros(self.num_coeffs)
            c[j] = 1.0
            spl = BSpline(self.t, c, self.degree)
            X[:, j] = spl(x_in)

        # 2. Apply a Tukey-like flat-top window using the standard fourier tools
        span = x_in[-1] - x_in[0]
        if span <= 0:
            raise ValueError("Data x-range must be strictly positive.")
            
        taper_width = 0.1 * span  # Taper over the first and last 10% of the data
        xs = [
            float(x_in[0]), 
            float(x_in[0] + taper_width), 
            float(x_in[-1] - taper_width), 
            float(x_in[-1])
        ]
        
        # Generate the window and apply the safety clip to prevent invisible edges
        window_raw = flattop_window(x_in, xs, type='hanning')
        window = np.clip(window_raw, 0.01, 1.0)
        
        y_w = y_in * window
        X_w = X * window[:, None]

        # 3. Define R-grid for the Fourier transform (0 to Rbkg)
        dr = np.pi / (4 * span)
        r = np.arange(0, self.rbkg, dr)
        if len(r) == 0:
            r = np.array([self.rbkg / 2.0])

        if len(r) * 2 < self.num_coeffs:
            warnings.warn(
                f"The R-grid contains only {len(r)} points, but the model has {self.num_coeffs} "
                f"degrees of freedom. The system is underdetermined.",
                AutobkFitInsufficientDataWarning,
                stacklevel=2
            )

        # 4. Transform windowed data to R-space
        y_R_complex = fourier(x_in, y_w, r)
        y_R = np.concatenate([y_R_complex.real, y_R_complex.imag])

        # 5. Transform B-spline basis to R-space
        X_R_complex = np.zeros((len(r), self.num_coeffs), dtype=np.complex128)
        for j in range(self.num_coeffs):
            X_R_complex[:, j] = fourier(x_in, X_w[:, j], r)
            
        X_R = np.concatenate([X_R_complex.real, X_R_complex.imag], axis=0)

        # 6. Tikhonov Regularization (Smoothness Penalty)
        alpha = 1e-3 * np.linalg.norm(X_R)  # Adaptive penalty weight
        
        # Create a 2nd-derivative (difference) operator matrix L
        L = np.diag(2 * np.ones(self.num_coeffs)) - \
            np.diag(np.ones(self.num_coeffs-1), 1) - \
            np.diag(np.ones(self.num_coeffs-1), -1)
            
        # Append the regularization block to the least-squares matrices
        X_R_reg = np.vstack([X_R, alpha * L])
        y_R_reg = np.concatenate([y_R, np.zeros(self.num_coeffs)])

        # 7. Solve linear least-squares problem
        c_opt, _, _, _ = lstsq(X_R_reg, y_R_reg, cond=None)
        
        if c_opt is None:  # pyright: ignore[reportUnnecessaryComparison]
            raise RuntimeError("lstsq returned None during AUTOBK fitting.")

        # 8. Reconstruct the un-windowed background in k-space
        bkg = X @ np.asarray(c_opt, dtype=np.float64)
        
        return bkg