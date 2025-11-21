import numpy as np
from numpy import typing as npt

from typing import SupportsIndex, Sequence
import numpy._core.numeric as _nx

from typing import TypeVar

_N = TypeVar('_N', bound=npt.ArrayLike)

def derivative2(
    _f: _N,
    *varargs: _N,
    axis: None | SupportsIndex | Sequence[SupportsIndex] = None,
) -> _N:
    
    # Modified from np.gradient to calculate second order derivative instead of first
    f = np.asanyarray(_f)
    N = f.ndim  # number of dimensions

    if axis is None:
        axes = tuple(range(N))
    else:
        axes:tuple[int, ...] = _nx.normalize_axis_tuple(axis, N) # type: ignore

    len_axes = len(axes)
    n = len(varargs)
    
    if n == 0:
        # no spacing argument - use 1 in all axes
        dx = [1.0] * len_axes
    elif n == 1 and np.ndim(varargs[0]) == 0:
        # single scalar for all axes
        dx = varargs * len_axes
    elif n == len_axes:
        # scalar or 1d array for each axis
        dx = list(varargs)
        for i, distances in enumerate(dx):
            distances = np.asanyarray(distances)
            if distances.ndim == 0:
                continue
            elif distances.ndim != 1:
                raise ValueError("distances must be either scalars or 1d")
            if len(distances) != f.shape[axes[i]]:
                raise ValueError("when 1d, distances must match "
                                 "the length of the corresponding dimension")
            if np.issubdtype(distances.dtype, np.integer):
                # Convert numpy integer types to float64 to avoid modular
                # arithmetic in np.diff(distances).
                distances = distances.astype(np.float64)
            diffx = np.diff(distances)
            # if distances are constant reduce to the scalar case
            # since it brings a consistent speedup
            if (diffx == diffx[0]).all():
                diffx = diffx[0]
            dx[i] = diffx
    else:
        raise TypeError("invalid number of arguments")
    
    slice1 = [slice(None)]*N
    slice2 = [slice(None)]*N
    slice3 = [slice(None)]*N
    slice4 = [slice(None)]*N

    otype = f.dtype
    if np.issubdtype(otype, np.inexact):
        pass
    else:
        # All other types convert to floating point.
        # First check if f is a numpy integer type; if so, convert f to float64
        # to avoid modular arithmetic when computing the changes in f.
        if np.issubdtype(otype, np.integer):
            f = f.astype(np.float64)
        otype = np.float64
    
    
    for axis, ax_dx in zip(axes, dx):
        if f.shape[axis] < 3:
            raise ValueError(
                "Shape of array too small to calculate a numerical gradient, "
                "at least 3 elements are required.")
        # result allocation
        out = np.empty_like(f, dtype=otype)

        outvals = []

        slice1[axis] = slice(1, -1)
        slice2[axis] = slice(None, -2)
        slice3[axis] = slice(1, -1)
        slice4[axis] = slice(2, None)

        # spacing for the current axis
        uniform_spacing = np.ndim(ax_dx) == 0
        if uniform_spacing:
            out[tuple(slice1)] = (f[tuple(slice2)] - 2*f[tuple(slice3)] + f[tuple(slice4)]) / (ax_dx**2)
        
        else:
            dx1 = ax_dx[:-1]
            dx2 = ax_dx[1:]

            A = 1 / (dx1 * (dx1 + dx2))
            B = -1 / (dx1 * dx2)
            C = 1 / (dx2 * (dx1 + dx2))
            # fix the shape for broadcasting
            shape = np.ones(N, dtype=int)
            shape[axis] = -1
            A.shape = B.shape = C.shape = shape
            
            out[tuple(slice1)] = (A * f[tuple(slice2)] + B * f[tuple(slice3)] + C * f[tuple(slice4)]) * 2.0

        ### DONE -----------------------------------------------------------------------------------------------------------------------

        # Edges
        slice1[axis] = 0
        slice2[axis] = 1
        slice3[axis] = 2
        out[tuple(slice1)] = out[tuple(slice2)] + (out[tuple(slice2)] - out[tuple(slice3)]) / ax_dx[1] * ax_dx[0]
        slice1[axis] = -1
        slice2[axis] = -2
        slice3[axis] = -3
        out[tuple(slice1)] = out[tuple(slice2)] + (out[tuple(slice2)] - out[tuple(slice3)]) / ax_dx[-2] * ax_dx[-1]

        outvals.append(out)

        # reset the slice object in this dimension to ":"
        slice1[axis] = slice(None)
        slice2[axis] = slice(None)
        slice3[axis] = slice(None)
        slice4[axis] = slice(None)

    if len_axes == 1:
        return outvals[0]
    return tuple(outvals)

def nderivative(f: _N, d:int,
    *varargs: npt.ArrayLike,
    axis: None | SupportsIndex | Sequence[SupportsIndex] = None,) -> _N | tuple[_N, ...]:

    f = np.asanyarray(f)
    N = f.ndim  # number of dimensions

    if axis is None:
        axes = tuple(range(N))
    else:
        axes: tuple[int, ...] = _nx.normalize_axis_tuple(axis, N)
    
    if d <= 0:
        return f
    elif d == 1:
        return np.gradient(f, *varargs, axis=axes)
    elif d == 2:
        return derivative2(f, *varargs, axis=axes)
    
    if len(axes) == 1:
        for _ in range(d//2):
            f = derivative2(f, *varargs, axis=axes)
        if d%2!=0:
            f = np.gradient(f, *varargs, axis=axes)
        return f
    else:
        fs = list(derivative2(f, *varargs, axis=axes))
        for i in range(d//2 -1):
            fs = [derivative2(_f, _x, axis=_axis) for _f,_x,_axis in zip(fs, varargs, axes)]
        if d%2!=0:
            fs = [np.gradient(_f, _x, axis=_axis) for _f,_x,_axis in zip(fs, varargs, axes)]
        return tuple(fs)
