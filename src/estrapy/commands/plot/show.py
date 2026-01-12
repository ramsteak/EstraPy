from ...core.context import FigureSpecification

from .gridspecsolve import make_figure_gridspec

def realize_figure(figspec: FigureSpecification):
    """Realize a figure from its specification"""
    fig, _, axes = make_figure_gridspec([a.pos for a in figspec.axes.values()])
    for (col, row), axis_spec in figspec.axes.items():
        ax = axes[(row, col)]
        for cb in axis_spec.callbacks:
            cb(ax, fig)
    return fig
