import numpy as np
import pandas as pd

import warnings
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from numpy import typing as npt
from dataclasses import dataclass
from typing import Callable

from .result import CommandResult_Align, apply_shift_to_page

from ...core._validators import validate_number_unit, validate_number_positive, listof
from ...core.context import Context
from ...core.datastore import Domain
from ...core.number import Number, parse_number, Unit
from ...core.commandparser import CommandArguments, field_arg


@dataclass(slots=True)
class SubCommandArguments_Align_Glitch(CommandArguments):
    glitches: list[Number] = field_arg(
        type = parse_number,
        nargs = '+',
        required = True,
        help = 'Energies of the glitches to align to.',
        validate = listof(validate_number_unit(Unit.EV))
    )

    interval: Number = field_arg(
        flags = ['--interval'],
        type = parse_number,
        required = False,
        default = Number(None, 10.0, Unit.EV),
        help = 'Energy range around each glitch to consider for alignment.',
        validate = [validate_number_unit(Unit.EV), validate_number_positive],
    )

@dataclass(slots=True)
class SubCommandResult_Align_Glitch(CommandResult_Align):
    interval: float
    glitch_positions: npt.NDArray[np.float64]
    shifts: pd.Series

    def plot_histogram(self) -> Callable[..., None]:
        """Factory for a callback that plots a histogram of the calculated shifts."""
        from matplotlib.figure import Figure
        from matplotlib.axes import Axes
        
        def plot_histogram_callback(ax: Axes, fig: Figure) -> None:
            ax.hist(self.shifts, bins='sqrt') # pyright: ignore[reportUnknownMemberType]
            ax.set_title('Histogram of Calculated Shifts') # pyright: ignore[reportUnknownMemberType]
            ax.set_xlabel('Shift (eV)') # pyright: ignore[reportUnknownMemberType]
            ax.set_ylabel('Frequency') # pyright: ignore[reportUnknownMemberType]
        return plot_histogram_callback

    def plot_shifts(self) -> Callable[..., None]:
        """Factory for a callback that plots the calculated shifts."""
        from matplotlib.figure import Figure
        from matplotlib.axes import Axes

        def plot_shifts_callback(ax: Axes, fig: Figure) -> None:
            ax.plot(range(len(self.shifts)), self.shifts, 'o', alpha=0.7) # pyright: ignore[reportUnknownMemberType]
            ax.set_title('Calculated Shifts per Spectrum') # pyright: ignore[reportUnknownMemberType]
            ax.set_xlabel('Spectrum Index') # pyright: ignore[reportUnknownMemberType]
            ax.set_ylabel('Shift (eV)') # pyright: ignore[reportUnknownMemberType]
        return plot_shifts_callback
    

def parabola_vertex(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
    """Given three points as x and y arrays of shape (3,), fit a parabola and return the x coordinate of its vertex."""
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != (3,) or y.shape != (3,):
        raise ValueError("x and y must be 1D arrays of shape (3,).")

    hL, hR = x[1] - x[0], x[2] - x[1]
    dy0, dy2 = y[0] - y[1], y[2] - y[1]
    return x[1] + 0.5 * (hR*hR*dy0 - hL*hL*dy2) / (hR*dy0 + hL*dy2)


def compute_shifts(
    df: pd.DataFrame,
    ref: str | None = None,
) -> pd.Series:
    """
    Estimate the energy shift of each file to align monochromator glitches.

    The model assumes every observed glitch position is:

        df[i, j]  =  s_i  +  g_j  +  ε_ij

    where ``s_i`` is an unknown scalar shift per file and ``g_j`` is the
    unknown true position of glitch ``j``.  The system is solved by least
    squares over the non-NaN entries only, with one file pinned to ``s = 0``
    as the reference (gauge fixing).

    Parameters
    ----------
    df : pd.DataFrame
        Rows   = file/scan names (the index).
        Columns = nominal glitch positions supplied by the user.
        Values  = observed sub-grid glitch positions; ``NaN`` where the scan
                  did not cover that glitch.
    ref : str | None
        Name (index label) of the reference file, which is pinned to
        ``shift = 0``.  If ``None`` (default), the file with the largest
        number of non-NaN observations is chosen automatically.

    Returns
    -------
    shifts : pd.Series
        Index = file names, values = estimated shift ``s_i``.
        Files that share no glitch with the reference (disconnected component)
        receive ``NaN`` and a ``UserWarning`` is issued listing them.

    Notes
    -----
    The least-squares matrix ``A`` has one row per non-NaN observation plus
    one anchor row.  Each observation row has exactly two non-zero entries
    (the coefficient 1 for ``s_i`` and 1 for ``g_j``), making ``A`` very
    sparse.  For typical XAS datasets (tens of files, handful of glitches)
    the dense ``np.linalg.lstsq`` solver is fast enough; for larger problems
    consider ``scipy.sparse.linalg.lsqr``.

    Connectivity is checked via the file-file adjacency graph: two files are
    adjacent if they share at least one non-NaN glitch column.  Shifts can
    only be determined within each connected component.
    """
    files    = list(df.index)
    glitches = list(df.columns)
    N = len(files)
    M = len(glitches)
    fi = {f: i for i, f in enumerate(files)}

    notna = df.notna().values  # bool (N, M)

    # ------------------------------------------------------------------
    # 1. Connectivity check
    # ------------------------------------------------------------------
    shared = notna.astype(np.int32) @ notna.astype(np.int32).T  # (N, N)
    np.fill_diagonal(shared, 0)
    _, labels = connected_components(
        csr_matrix(shared), directed=False
    )

    if ref is None:
        ref = df.notna().sum(axis=1).idxmax()

    ref_idx       = fi[ref]
    ref_component = labels[ref_idx]

    disconnected = [files[i] for i in range(N) if labels[i] != ref_component]
    if disconnected:
        warnings.warn(
            f"{len(disconnected)} file(s) are in a disconnected component and "
            f"cannot be aligned to {ref!r}: {disconnected}",
            UserWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # 2. Build the least-squares system for the reference component
    # ------------------------------------------------------------------
    in_ref   = [i for i in range(N) if labels[i] == ref_component]
    comp_map = {i: k for k, i in enumerate(in_ref)}   # file idx → compact idx
    Nc       = len(in_ref)
    n_params = Nc + M          # [s_0..s_{Nc-1}, g_0..g_{M-1}]

    # Collect non-NaN equations
    rows_eq, vals_eq = list[tuple[int, int]](), list[float]()
    for i in in_ref:
        for j in range(M):
            if notna[i, j]:
                rows_eq.append((i, j))
                vals_eq.append(df.iloc[i, j])

    n_eq = len(rows_eq)
    A    = np.zeros((n_eq + 1, n_params))
    b    = np.zeros(n_eq + 1)

    for eq, ((i, j), val) in enumerate(zip(rows_eq, vals_eq)):
        A[eq, comp_map[i]] = 1.0   # coefficient of s_i
        A[eq, Nc + j]      = 1.0   # coefficient of g_j
        b[eq]              = val

    # Anchor row: s_ref = 0
    A[n_eq, comp_map[ref_idx]] = 1.0
    b[n_eq]                    = 0.0

    params, *_ = np.linalg.lstsq(A, b, rcond=None)

    # ------------------------------------------------------------------
    # 3. Assemble output Series
    # ------------------------------------------------------------------
    shifts = pd.Series(np.nan, index=files, name="shift")
    for k, i in enumerate(in_ref):
        shifts.iloc[i] = params[k]

    return shifts


def execute_glitch(context: Context, sargs: SubCommandArguments_Align_Glitch) -> SubCommandResult_Align_Glitch:
    log = context.logger.getChild('command.align.glitch')

    _glitches: dict[str, npt.NDArray[np.float64]] = {}
    for name, page in context.datastore.pages.items():
        domain = page.domains[Domain.RECIPROCAL]
        df = domain.get_columns_data(['E', 'I0'])
        X, Y = df.E.to_numpy(), df.I0.to_numpy()

        glitch_list: list[float] = []
        for glitch in sargs.glitches:
            idx = (X >= (glitch.value - sargs.interval.value)) & (X <= (glitch.value + sargs.interval.value))
            if np.sum(idx) < 3:
                glitch_list.append(np.nan)
                continue
            xi, y = X[idx], Y[idx]

            idxmax = np.argmin(y)
            vertex = parabola_vertex(xi[idxmax-1:idxmax+2], y[idxmax-1:idxmax+2])

            glitch_list.append(vertex)

            pass

        _glitches[name] = np.array(glitch_list)

        pass

    glitches = pd.DataFrame(_glitches, index=[str(g) for g in sargs.glitches]).T
    shifts = compute_shifts(glitches)

    for name,result in shifts.items():
        apply_shift_to_page(context.datastore.pages[str(name)], None, float(result))
        log.debug(f'Applied shift of {float(result):0.4f} eV to page {name}.')
    
    log.info('Aligned all spectra using glitch method.')

    return SubCommandResult_Align_Glitch(
        interval = sargs.interval.value,
        glitch_positions = glitches.to_numpy(),
        shifts = shifts,
    )