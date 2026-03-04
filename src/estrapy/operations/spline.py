import numpy as np
from numpy import typing as npt

from itertools import pairwise
from typing import NamedTuple, TypeVar, Callable

from scipy.linalg import lstsq, svd # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

class SplineFitFewPointsWarning(RuntimeWarning):
    """Raised when a segment has fewer data points than its polynomial degree,
    making the fit underdetermined for that segment."""
    pass

class SplineFitEmptySegmentWarning(RuntimeWarning):
    """Raised when a segment has no data points at all. The segment coefficients
    will be unconstrained and defaulted to zero."""
    pass

class SplineFitIllConditionedWarning(RuntimeWarning):
    """Raised when ATA is ill-conditioned. The fit may be numerically unstable.
    Consider reducing the polynomial degree or adding more data points."""
    pass

class SplineFitRankDeficientWarning(RuntimeWarning):
    """Raised when the constraint matrix C is rank-deficient, meaning some
    constraints are redundant or contradictory. Redundant ones are ignored,
    contradictory ones are satisfied in a least-squares sense."""
    pass

class SplineFitOverconstrainedWarning(RuntimeWarning):
    """Raised when the number of constraints exceeds the number of free
    coefficients, leaving zero or negative degrees of freedom."""
    pass

class SplineFitOutOfRangeWarning(RuntimeWarning):
    """Raised when some x values lie outside the knot range and will be ignored."""
    pass

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

        # Precompute null-space decomposition for fit
        self.constraints = self.constraints_matrix.shape[0]
        if self.constraints > 0:
            U, S, Vt = (np.asarray(m, dtype=np.float64) for m in svd(self.constraints_matrix, full_matrices=True))  # pyright: ignore[reportUnknownVariableType]
            threshold = S.max() * max(self.constraints_matrix.shape) * np.finfo(np.float64).eps
            self.constraints_rank = int(np.sum(S > threshold))
            self.constraints_null_basis = Vt[self.constraints_rank:].T

            S_inv = np.zeros_like(S)
            S_inv[:self.constraints_rank] = 1.0 / S[:self.constraints_rank]
            self.constraints_particular = Vt[:self.constraints_rank].T @ (S_inv[:self.constraints_rank] * (U[:, :self.constraints_rank].T @ self.constraints_values))
        else:
            self.constraints_rank = 0
            self.constraints_null_basis = np.eye(self.total_coeffs, dtype=np.float64)
            self.constraints_particular = np.zeros(self.total_coeffs, dtype=np.float64)
        
        self._check_constraint_warnings()
    
    def _map_domain(self, x: _T, domain1: tuple[float, float], domain2: tuple[float, float]) -> _T:
        """Maps x across two linear domains."""
        return (x - domain1[0]) / (domain1[1] - domain1[0]) * (domain2[1] - domain2[0]) + domain2[0]
        
    def fit(self, x: npt.NDArray[np.floating], y: npt.NDArray[np.floating], *, derivative: int = 0, suppress_warnings: bool = False) -> Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]:
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
        #TODO: Add check to verify no x points are outside the knots range, and there are enough points per segment to fit the specified degree
        beta = self._fit(x, y, suppress_warnings=suppress_warnings)
        return self._make_spline_function(beta, derivative)

    def _construct_ATA(self, x: npt.NDArray[np.floating], y: npt.NDArray[np.floating]) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Construct the ATA matrix and ATy vector for the least squares problem."""
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
        
        return ATA, ATy

    def _fit(self, _x: npt.NDArray[np.floating], _y: npt.NDArray[np.floating], suppress_warnings: bool = False) -> npt.NDArray[np.floating]:
        xsort = np.argsort(_x)
        x = _x[xsort]
        y = _y[xsort]

        ATA, ATy = self._construct_ATA(x, y)
        if not suppress_warnings:
            self._check_fit_warnings(x, ATA, np.searchsorted(x, self.knots))

        if self.constraints == 0:
            result = lstsq(ATA, ATy, cond=None)  # pyright: ignore[reportUnknownVariableType]
            if result is None:
                raise RuntimeError("lstsq returned None")
            return np.asarray(result[0], dtype=np.float64)

        H_reduced = np.asarray(self.constraints_null_basis.T @ ATA @ self.constraints_null_basis, dtype=np.float64)
        g_reduced = np.asarray(self.constraints_null_basis.T @ (ATy - ATA @ self.constraints_particular), dtype=np.float64)

        z_result = lstsq(H_reduced, g_reduced, cond=None)  # pyright: ignore[reportUnknownVariableType]
        if z_result is None:
            raise RuntimeError("lstsq returned None")

        return np.asarray(self.constraints_particular + self.constraints_null_basis @ np.asarray(z_result[0], dtype=np.float64), dtype=np.float64)

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

    @property
    def num_segments(self) -> int:
        return len(self.segments)
    
    @property
    def num_constraints(self) -> int:
        return self.constraints_matrix.shape[0]

    @property
    def num_coefficients(self) -> int:
        return self.total_coeffs

    @property
    def num_degrees_of_freedom(self) -> int:
        return self.total_coeffs - self.constraints_matrix.shape[0]

    def _check_constraint_warnings(self) -> None:
        """Check for potential issues with the constraint matrix.
        Called once at __init__ time since constraints never change.
        """
        import warnings

        C = self.constraints_matrix

        if C.shape[0] == 0:
            return

        # Use precomputed rank from __init__
        rank = self.constraints_rank

        if rank < C.shape[0]:
            warnings.warn(
                f"Constraint matrix has rank {rank} but {C.shape[0]} rows: "
                f"{C.shape[0] - rank} constraint(s) are redundant or contradictory "
                f"and will be handled in a least-squares sense.",
                SplineFitRankDeficientWarning,
                stacklevel=3,
            )

        dof = self.total_coeffs - rank
        if dof <= 0:
            warnings.warn(
                f"The system has {self.total_coeffs} coefficient(s) and {rank} independent "
                f"constraint(s), leaving {dof} degree(s) of freedom. "
                f"The spline shape is fully or over-determined by constraints alone.",
                SplineFitOverconstrainedWarning,
                stacklevel=3,
            )

    def _check_fit_warnings(
        self,
        x: npt.NDArray[np.floating],
        ATA: npt.NDArray[np.floating],
        split_indices: npt.NDArray[np.integer],
    ) -> None:
        """Check for potential numerical issues with the data and ATA matrix.
        Called at each _fit invocation since x and ATA change with the data.
        """
        import warnings

        # Out-of-range points
        out_of_range = int(np.sum((x < self.knots[0]) | (x > self.knots[-1])))
        if out_of_range > 0:
            warnings.warn(
                f"{out_of_range} data point(s) lie outside the knot range "
                f"[{self.knots[0]}, {self.knots[-1]}] and will not contribute to the fit.",
                SplineFitOutOfRangeWarning,
                stacklevel=3,
            )

        # Per-segment point count and ATA conditioning
        for (idx1, idx2), segment in zip(pairwise(split_indices), self.segments):
            n_points = int(idx2 - idx1)

            if n_points == 0:
                warnings.warn(
                    f"Segment [{segment.knots[0]}, {segment.knots[1]}] has no data points. "
                    f"Its coefficients are unconstrained and will default to zero.",
                    SplineFitEmptySegmentWarning,
                    stacklevel=3,
                )
            elif n_points <= segment.degree:
                warnings.warn(
                    f"Segment [{segment.knots[0]}, {segment.knots[1]}] has {n_points} point(s) "
                    f"but degree {segment.degree}: the fit is underdetermined for this segment.",
                    SplineFitFewPointsWarning,
                    stacklevel=3,
                )
            else:
                block = ATA[segment.coeff_slice, segment.coeff_slice]
                cond = np.linalg.cond(block)
                if cond > 1 / np.finfo(np.float64).eps:
                    warnings.warn(
                        f"Segment [{segment.knots[0]}, {segment.knots[1]}] has an ill-conditioned "
                        f"ATA block (condition number ≈ {cond:.2e}). "
                        f"Consider reducing the polynomial degree.",
                        SplineFitIllConditionedWarning,
                        stacklevel=3,
                    )