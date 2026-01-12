from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ...core.grammar.axisindexpos import AxisIndexPosition

def _find_span(
    start: int,
    sizes: list[float],
    target: float | None,
    stop: int,
) -> tuple[int, int]:
    """
    Find an inclusive span [start, end].

    Parameters
    ----------
    start : int
        Start index in the grid.
    sizes : list[float]
        Logical grid unit sizes (typically all 1.0).
    target : float | None
        Desired span size in grid units.
        None means "unconstrained".
    stop : int
        Maximum inclusive index allowed for expansion.

    Returns
    -------
    (start, end) : tuple[int, int]
        Inclusive span indices.
    """
    # Unconstrained → expand only until the logical stop
    if target is None:
        return start, stop

    total = 0.0
    for i in range(start, stop + 1):
        total += sizes[i]
        if total >= target:
            return start, i

    # If target cannot be satisfied, span to the stop
    return start, stop

def define_gridspec(
    axes: list[AxisIndexPosition],
) -> tuple[
    GridSpec,
    dict[AxisIndexPosition, tuple[tuple[int, int], tuple[int, int]]],
]:
    # --------------------------------------------------
    # 1. Logical grid
    # --------------------------------------------------
    cols = sorted({ax.axisindex[0] for ax in axes})
    rows = sorted({ax.axisindex[1] for ax in axes})

    col_idx = {c: i for i, c in enumerate(cols)}
    row_idx = {r: i for i, r in enumerate(rows)}

    ncols = len(cols)
    nrows = len(rows)

    col_widths = [1.0] * ncols
    row_heights = [1.0] * nrows

    gs = GridSpec(
        nrows=nrows,
        ncols=ncols,
        width_ratios=col_widths,
        height_ratios=row_heights,
    )

    # --------------------------------------------------
    # 2. Pre-group axes by row and column
    # --------------------------------------------------
    axes_by_row: dict[int, list[int]] = {}
    axes_by_col: dict[int, list[int]] = {}

    for ax in axes:
        c, r = ax.axisindex
        axes_by_row.setdefault(r, []).append(c)
        axes_by_col.setdefault(c, []).append(r)

    for r in axes_by_row:
        axes_by_row[r].sort()
    for c in axes_by_col:
        axes_by_col[c].sort()

    # --------------------------------------------------
    # 3. Resolve spans
    # --------------------------------------------------
    axis_map: dict[AxisIndexPosition, tuple[tuple[int, int], tuple[int, int]]] = {}

    for ax in axes:
        c, r = ax.axisindex
        w, h = ax.axissize

        c0 = col_idx[c]
        r0 = row_idx[r]

        # ---- WIDTH STOP ----
        row_cols = axes_by_row[r]
        pos = row_cols.index(c)
        if pos + 1 < len(row_cols):
            c_stop = col_idx[row_cols[pos + 1]] - 1
        else:
            c_stop = ncols - 1

        # ---- HEIGHT STOP ----
        col_rows = axes_by_col[c]
        pos = col_rows.index(r)
        if pos + 1 < len(col_rows):
            r_stop = row_idx[col_rows[pos + 1]] - 1
        else:
            r_stop = nrows - 1

        c0, c1 = _find_span(c0, col_widths, w, c_stop)
        r0, r1 = _find_span(r0, row_heights, h, r_stop)

        axis_map[ax] = ((r0, r1), (c0, c1))

    return gs, axis_map

def check_overlap(spans: dict[AxisIndexPosition, tuple[tuple[int, int], tuple[int, int]]]) -> list[tuple[AxisIndexPosition, AxisIndexPosition]]:
    cells: dict[tuple[int, int], list[AxisIndexPosition]] = {}
    from itertools import product, combinations
    for span in spans:
        (r0, r1), (c0, c1) = spans[span]
        for pos in product(range(r0, r1 + 1), range(c0, c1 + 1)):
            cells.setdefault(pos, []).append(span)
    overlaps: list[tuple[AxisIndexPosition, AxisIndexPosition]] = []
    for _, axes in cells.items():
        if len(axes) < 2: continue
        overlaps.extend(combinations(axes, 2))
    return overlaps

def make_figure_gridspec(figspec: list[AxisIndexPosition]) -> tuple[Figure, GridSpec, dict[tuple[int, int], Axes]]:
    # Build gridspec + mapping
    gs, axis_map = define_gridspec(figspec)

    fig = plt.figure(constrained_layout=False) # pyright: ignore[reportUnknownMemberType]

    axes: dict[tuple[int, int], Axes] = {}
    for axspec, ((r0, r1), (c0, c1)) in axis_map.items():
        ax = fig.add_subplot(gs[r0:r1 + 1, c0:c1 + 1])
        axes[axspec.axisindex] = ax

    return fig, gs, axes

