import numpy as np
from numpy import typing as npt

from itertools import pairwise
from typing import NamedTuple, TypeVar, Callable

class Segment(NamedTuple):
    knots: tuple[float, float]
    degree: int
    coeff_slice: slice
    domain: tuple[float, float] = ( -1.0, 1.0 )

_T = TypeVar('_T', float, npt.NDArray[np.floating], np.floating, np.ndarray)

class PiecewiseSplineFitter:
    def __init__(self,
                 knots: npt.NDArray[np.floating],
                 degrees: npt.NDArray[np.integer],
                 continuity: int | None = None,
                 fixed_points: list[tuple[float, float]] | None = None,
                 ):
        
        """Initializes the PiecewiseSplineFitter with the given knots and degrees.
        
        The instance can then be used to fit multiple datasets with the same spline configuration.
        
        Parameters
        ----------
        knots : array-like
            Knot positions defining the boundaries of spline segments.
            Must contain at least 2 knots.
        degrees : array-like
            Polynomial degrees for each segment. Length must be len(knots) - 1.
        continuity : int or None, optional
            Maximum derivative order to enforce continuity at internal knots.
            If None, uses the minimum degree at each boundary.
        fixed_points : list of (x, y) tuples, optional
            Points where the spline must pass through exactly.
        """

        self.knots = np.sort(np.unique(knots))
        self.degrees = degrees
        self._max_degree = max(degrees)

        # Validate inputs
        if len(knots) < 2:
            raise ValueError("At least two knots are required to define a piecewise spline.")
        if len(degrees) != len(knots) - 1:
            raise ValueError("Degrees array length must be one less than knots array length.")
        if np.any(degrees < 0):
            raise ValueError("Degrees must be non-negative integers.")
        
        if fixed_points is None:
            fixed_points = []
        
        for x_fix, _ in fixed_points:
            if x_fix < self.knots[0] or x_fix > self.knots[-1]:
                raise ValueError("Fixed points must lie within the range defined by the knots.")
        
        
        self.total_coeffs = int(np.sum(degrees) + len(degrees))
        
        # Setup segment table
        self.segments: list[Segment] = []
        coeff_idx = 0
        for (k1, k2), deg in zip(pairwise(self.knots), self.degrees):
            self.segments.append(
                Segment(
                    knots = (k1, k2),
                    degree = deg,
                    coeff_slice = slice(coeff_idx, coeff_idx + deg + 1),
                    domain = ( -1.0, 1.0 )
                )
            )
            coeff_idx += deg + 1
        
        # Setup boundary conditions
        self.boundaries: list[tuple[float, Segment, Segment, int]] = [
            (segment_1.knots[1], segment_1, segment_2, min(segment_1.degree-1, segment_2.degree-1, continuity or np.iinfo(np.int64).max))
            for segment_1, segment_2 in pairwise(self.segments)
        ]
        
        # Build constraints matrix
        constraints_rows: list[npt.NDArray[np.floating]] = []
        constraints_vals: list[float] = []

        # Fixed point constraints
        for x_fix, y_fix in fixed_points:
            row = np.zeros(self.total_coeffs, dtype=np.float64)
            seg_idx = np.clip(np.searchsorted(self.knots, x_fix) - 1, 0, len(self.degrees)-1)
            segment = self.segments[seg_idx]

            x_internal = self._map_domain(x_fix, segment.knots, segment.domain)

            basis_values = np.polynomial.chebyshev.chebvander([x_internal], segment.degree)[0]
            row[segment.coeff_slice] = basis_values
            constraints_rows.append(row)
            constraints_vals.append(y_fix)
        
        # Continuity constraints at boundaries
        eye_basis = np.eye(self._max_degree + 1, dtype=np.float64)
        der_basis: list[npt.NDArray[np.floating]] = [
            np.polynomial.chebyshev.chebder(eye_basis, m=d)
            for d in range(self._max_degree + 1)
        ]
        der_vals: npt.NDArray[np.floating] = np.array([
            np.polynomial.chebyshev.chebval([1.0, -1.0], der_basis[d]) # Hardcoded 1.0 for right end of left segment
            for d in range(self._max_degree + 1)
        ], dtype=np.float64)

        for _, segment_l, segment_r, max_continuity in self.boundaries:

            scale_l = 2 / (segment_l.knots[1] - segment_l.knots[0])
            scale_r = 2 / (segment_r.knots[1] - segment_r.knots[0])

            for deriv_order in range(0, max_continuity + 1):
                row = np.zeros(self.total_coeffs, dtype=np.float64)

                vals_l = der_vals[deriv_order, :segment_l.degree + 1, 0]
                vals_r = der_vals[deriv_order, :segment_r.degree + 1, 1]

                row[segment_l.coeff_slice] = vals_l * (scale_l ** deriv_order)
                row[segment_r.coeff_slice] = -vals_r * (scale_r ** deriv_order)

                constraints_rows.append(row)
                constraints_vals.append(0.0)
        
        # Finalize constraints matrix
        if constraints_rows:
            self.constraints_matrix = np.vstack(constraints_rows)
            self.constraints_values = np.array(constraints_vals, dtype=np.float64)
        else:
            self.constraints_matrix = np.zeros((0, self.total_coeffs), dtype=np.float64)
            self.constraints_values = np.zeros((0,), dtype=np.float64)
    
    def _map_domain(self, x: _T, domain1: tuple[float, float], domain2: tuple[float, float]) -> _T:
        """Maps x across two linear domains."""
        return (x - domain1[0]) / (domain1[1] - domain1[0]) * (domain2[1] - domain2[0]) + domain2[0]
        
    def fit(self, x: npt.NDArray[np.floating], y: npt.NDArray[np.floating], *, derivative: int = 0) -> Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]:
        """Fit the piecewise spline to the data.
        
        Parameters
        ----------
        x : array-like
            Input data points.
        y : array-like
            Output data points corresponding to x.
            
        Returns
        -------
        callable
            A function that evaluates the fitted spline at given points.
        """
        beta = self._fit(x, y)
        return self._make_spline_function(beta, derivative)
    
    def _fit(self, _x: npt.NDArray[np.floating], _y: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Internal method to fit the spline using least squares with constraints.
        Returns the fitted coefficients."""
        xsort = np.argsort(_x)
        x = _x[xsort]
        y = _y[xsort]

        ATA = np.zeros((self.total_coeffs, self.total_coeffs), dtype=np.float64)
        ATy = np.zeros((self.total_coeffs,), dtype=np.float64)

        split_indices = np.searchsorted(x, self.knots)
        for idx1, idx2, segment in zip(split_indices[:-1], split_indices[1:], self.segments):
            if idx1 == idx2:
                continue

            x_seg = x[idx1:idx2]
            y_seg = y[idx1:idx2]
            x_internal = self._map_domain(x_seg, segment.knots, segment.domain)
            vander = np.polynomial.chebyshev.chebvander(x_internal, segment.degree)
            ATA[segment.coeff_slice, segment.coeff_slice] += vander.T @ vander
            ATy[segment.coeff_slice] += vander.T @ y_seg

        # for i, ((k1, k2), deg, c_slice, domain) in enumerate(self.segments):
        #     if i == len(self.segments) - 1:
        #         mask = (x >= k1) & (x <= k2)
        #     else:
        #         mask = (x >= k1) & (x < k2)

        #     if not np.any(mask):
        #         continue

        #     x_seg = x[mask]
        #     x_internal = self._map_domain(x_seg, (k1, k2), domain)
        #     vander = np.polynomial.chebyshev.chebvander(x_internal, deg)
        #     ATA[c_slice, c_slice] = vander.T @ vander
        #     ATy[c_slice] = vander.T @ y[mask]


        n_c = self.constraints_matrix.shape[0]
        n_v = self.total_coeffs

        if n_c > 0:
            KKT_L: npt.NDArray[np.floating] = np.block([
                [2 * ATA, self.constraints_matrix.T],
                [self.constraints_matrix, np.zeros((n_c, n_c), dtype=np.float64)] # type: ignore
            ])
            KKT_R = np.concatenate([2 * ATy, self.constraints_values])

            solution = np.linalg.solve(KKT_L, KKT_R)
            beta = solution[:n_v]
        else:
            beta = np.linalg.solve(ATA, ATy)
        
        return beta

    def _make_spline_function(
        self, 
        beta: npt.NDArray[np.floating],
        derivative: int = 0
    ) -> Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]:
        """Create a callable function from the fitted coefficients."""
        def spline_func(x_in: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
            x_in = np.asarray(x_in, dtype=np.float64)
            y_out = np.zeros_like(x_in, dtype=np.float64)
            
            for i, seg in enumerate(self.segments):
                if i == len(self.segments) - 1:
                    mask = (x_in >= seg.knots[0]) & (x_in <= seg.knots[1])
                else:
                    mask = (x_in >= seg.knots[0]) & (x_in < seg.knots[1])
                
                if not np.any(mask):
                    continue

                coeffs = beta[seg.coeff_slice]
                if derivative > 0:
                    coeffs = np.polynomial.chebyshev.chebder(coeffs, m=derivative)
                # Map to internal domain and evaluate
                x_mapped = self._map_domain(x_in[mask], seg.knots, seg.domain)
                y_out[mask] = np.polynomial.chebyshev.chebval(x_mapped, coeffs)
            
            return y_out
        
        return spline_func
