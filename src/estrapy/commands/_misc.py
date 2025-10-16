import re
import numpy as np
import numpy.typing as npt

from typing import overload

from ._numberunit import parse_nu, SI_multiplier

def attempt_parse_number(s: str) -> str | int | float:
    if s.isdigit():
        return int(s)
    try:
        return float(s)
    except ValueError:
        return s
    
_ELEMENT_EDGE = r"(\w+)\.(\w+)(?:([+-]?\d+(?:\.\d+)?)(?:([kMGTPEZYRQmunpfazyrq])?(eV))?)?"
ELEMENT_EDGE = re.compile(_ELEMENT_EDGE)


def parse_edgeenergy(val: str) -> float:
    try:
        nu = parse_nu(val)
        if nu.unit not in ("eV", None):
            raise ValueError(f'Invalid energy unit: "{nu.unit}"')

        return nu.value
    except ValueError:
        m = ELEMENT_EDGE.match(val)

        if m is None: 
            raise ValueError(f'Could not parse "{val}" as edge energy.')
        
        element, edge, value, mult, _ = m.groups()

        shift = float(value) * SI_multiplier(mult) if value is not None else 0

        from xraydb import xray_edge

        energy = xray_edge(element, edge, True)
        return energy + shift  # type: ignore


# eV to k conversion

KCONST = 0.26246842396479844746511289042048746313978437879180
SCONST = 0.51231672231618445138819837616078387648071784401809  # sqrt(KCONST)
KICNST = 3.8099821109685837311852822702410407881602562961373  # 1 / KCONST


@overload
def E_to_k(v: float, E0: np.floating | float) -> float: ...
@overload
def E_to_k(v: np.floating, E0: np.floating | float) -> np.floating: ...
@overload
def E_to_k(v: npt.NDArray[np.floating], E0: np.floating | float) -> npt.NDArray[np.floating]: ...

def E_to_k(v: float | np.floating | npt.NDArray[np.floating], E0: np.floating | float):
    return SCONST * np.sqrt((v - E0))


@overload
def k_to_E(v: float, E0: np.floating | float) -> float: ...
@overload
def k_to_E(v: np.floating, E0: np.floating | float) -> np.floating: ...
@overload
def k_to_E(v: npt.NDArray, E0: np.floating | float) -> npt.NDArray: ...
def k_to_E(v, E0: np.floating | float):
    return v**2 * KICNST + E0


# sk = signed k


@overload
def E_to_sk(v: float, E0: np.floating | float) -> float: ...
@overload
def E_to_sk(v: np.floating, E0: np.floating | float) -> np.floating: ...
@overload
def E_to_sk(v: npt.NDArray, E0: np.floating | float) -> npt.NDArray: ...
def E_to_sk(v, E0: np.floating | float):
    return SCONST * np.sqrt((np.abs(v - E0))) * np.sign(v - E0)


@overload
def sk_to_E(v: float, E0: np.floating | float) -> float: ...
@overload
def sk_to_E(v: np.floating, E0: np.floating | float) -> np.floating: ...
@overload
def sk_to_E(v: npt.NDArray, E0: np.floating | float) -> npt.NDArray: ...
def sk_to_E(v, E0: np.floating | float):
    return v**2 * KICNST * np.sign(v) + E0


from typing import SupportsIndex, Sequence
import numpy._core.numeric as _nx

def derivative2(
    f: npt.ArrayLike,
    *varargs: npt.ArrayLike,
    axis: None | SupportsIndex | Sequence[SupportsIndex] = None,
) -> npt.NDArray:
    
    # Modified from np.gradient to calculate second order derivative instead of first
    
    f = np.asanyarray(f)
    N = f.ndim  # number of dimensions

    if axis is None:
        axes = tuple(range(N))
    else:
        axes = _nx.normalize_axis_tuple(axis, N)

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

def nderivative(f: npt.ArrayLike,d:int,
    *varargs: npt.ArrayLike,
    axis: None | SupportsIndex | Sequence[SupportsIndex] = None,):

    f = np.asanyarray(f)
    N = f.ndim  # number of dimensions

    if axis is None:
        axes = tuple(range(N))
    else:
        axes = _nx.normalize_axis_tuple(axis, N)
    
    if d <= 0:
        return f
    elif d == 1:
        return np.gradient(f, *varargs, axis=axes)
    elif d == 2:
        return derivative2(f, *varargs, axis=axes)
    
    if len(axes) == 1:
        for i in range(d//2):
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
