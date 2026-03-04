import numpy as np
from numpy import typing as npt
from logging import Logger

from dataclasses import dataclass
from functools import partial

from .result import BackgroundResult

from ...core.threaded import execute_threaded
from ...core.commandparser import CommandArguments, field_arg
from ...core._validators import validate_float_non_negative, validate_int_non_negative
from ...core.number import Number, parse_number, Unit
from ...core.context import Context
from ...core.datastore import Domain
from ...core.misc import fmt

from ...operations.spline import PiecewiseSplineFitter

def parse_fixed_point(s: str) -> tuple[Number, float]:
    xstr, ystr = s.split(",", maxsplit=1)
    return (parse_number(xstr), float(ystr))

@dataclass(slots=True)
class SubCommand_SplineBackgroundArguments(CommandArguments):
    nodes: list[Number] = field_arg(
        flags=['--nodes'],
        type=parse_number,
        nargs='+',
        required=True
    )

    degrees: list[int] = field_arg(
        flags=['--degrees'],
        type=int,
        nargs='+',
        required=True,
    )

    kweight: float = field_arg(
        flags=['--kweight'],
        type=float,
        required=False,
        default=2.0,
        validate=validate_float_non_negative
    )

    fixedpoints: list[tuple[Number, float]] | None = field_arg(
        flags=['--fixed-points'],
        type=parse_fixed_point,
        nargs=1,
        action='append',
        required=False,
        default=None
    )

    continuity: int | None = field_arg(
        flags=['--continuity'],
        type=int,
        required=False,
        default=1,
        validate=validate_int_non_negative
    )

    def __post_init__(self) -> None:
        if self.continuity == -1:
            self.continuity = None

    def validate(self) -> bool:
        """
        Validates the consistency between nodes and degrees.
        """
        if len(self.nodes) > 1:
            if len(self.nodes) != len(self.degrees)-1:
                raise ValueError(
                    f"Number of nodes ({len(self.nodes)}) must match number of degrees ({len(self.degrees)})."
                )
        else:
            # If there is only one node, we can interpret it as a number of knots
            # and generate equidistant knots within the range. Only one degree is allowed in this case.
            if len(self.degrees) != 1:
                raise ValueError(
                    f"If only one node is specified, there must be exactly one degree specified. "
                    f"Got {len(self.nodes)} node(s) and {len(self.degrees)} degree(s)."
                )
        return True


@dataclass(slots=True)
class SplineBackgroundResult(BackgroundResult):
    ...

def _compute_background_spline(xy: npt.NDArray[np.floating],
                               range: tuple[float, float],
                               sargs: SubCommand_SplineBackgroundArguments,
                               spline_fitter: PiecewiseSplineFitter,
                               name: str,
                               log: Logger,
                            ) -> SplineBackgroundResult:
    """The function calculates the background using the spline regression method,
    and returns a result object.
    The arguments are specified by the subcommand arguments, together with the data
    points and range.
    
    All Number instances are considered to be in the correct units."""
    
    # Canonicalize range, converting inf to the finite range
    kweight = sargs.kweight

    X = xy[:,0]
    minX, maxX = np.min(X), np.max(X)
    range_lower, range_upper = (float(max(range[0], minX)), float(min(range[1], maxX)))

    idx = (xy[:,0] >= range_lower) & (xy[:,0] <= range_upper)

    # Get x and y values within range
    x, y = xy[idx,0], xy[idx,1]

    # Calculate background
    bkg = spline_fitter.fit(x, y * x**kweight)(x) / (x ** kweight)
    
    Bkg = np.zeros_like(xy[:,1])
    Bkg[idx] = bkg

    log.debug(f'Computed spline background for page {name}.')

    return SplineBackgroundResult(
        background = Bkg
    )

def execute_background_spline(
    context: Context,
    sargs: SubCommand_SplineBackgroundArguments,
    range: tuple[Number, Number],
) -> dict[str, SplineBackgroundResult]:
    
    log = context.logger.getChild('command.background.spline')
    log.debug(f'Executing spline background subtraction in range [{range[0]}, {range[1]}], kweight={sargs.kweight}, nodes={' '.join(str(n) for n in sargs.nodes)}, degrees={' '.join(str(n) for n in sargs.degrees)}, fixedpoints={sargs.fixedpoints}')

    k_range = range[0].value, range[1].value

    log.debug('Preparing spline fitter')

    if len(sargs.nodes) == 1 and sargs.nodes[0].unit is None:
        nknots = int(sargs.nodes[0].value)
        knots = np.linspace(k_range[0], k_range[1], nknots)
        degrees = np.array(sargs.degrees * (nknots - 1))
        log.debug(f'Interpreted single node argument as number of knots: generated {nknots} equidistant knots in range: {knots}')
    else:
        knots = np.array([n.value for n in sargs.nodes])
        degrees = np.array(sargs.degrees)

    spline_fitter = PiecewiseSplineFitter(
        knots = knots,
        degrees = degrees,
        continuity = sargs.continuity,
        fixed_points = [(fp[0].value, fp[1]) for fp in sargs.fixedpoints] if sargs.fixedpoints is not None else None
    )

    page_fulldata: dict[str, npt.NDArray[np.floating]] = {
        name:page.domains[Domain.RECIPROCAL].get_columns_data(['k', 'chi']).to_numpy()
        for name,page in context.datastore.pages.items()
    }

    # Check that the nodes are within the data range of all pages, and log a warning if not.
    # It is acceptable if only one node per side is outside the data range
    # (i.e. data was cut to 12.0k, last actual point is at 11.9k and we fit to 12.0k),
    # but if more nodes are outside the data range, it is likely that the fit will be poor or that the matrix will be singular.
    # We set a threshold of 1e-1k for the distance between last point and node.

    # Also check if there are enough data points within the range to fit the spline, and log a warning if not.
    # Require at least as many data points as the number of degrees of freedom of the spline.
    for name, xy in page_fulldata.items():
        x = xy[:,0]
        minX, maxX = Number(None, float(np.min(x)), Unit.K), Number(None, float(np.max(x)), Unit.K)
        out_nodes = [
            node for node in sargs.nodes
            if (node.value < minX.value - 0.1 or node.value > maxX.value + 0.1)
        ]
        if out_nodes:
            log.warning(
                f'{fmt.plu(len(out_nodes), 'Node', 'Nodes')} '
                f'{', '.join(str(node) for node in out_nodes)} '
                f'{fmt.are(len(out_nodes))} '
                f'outside the data range [{format(minX, '0.2f')}, {format(maxX, '0.2f')}] of page {name}. '
                'This may lead to poor background fitting or singular matrices.'
        )
            
        if xy.shape[0] - spline_fitter.num_degrees_of_freedom <= 0:
            log.warning(
                f'Page {name} has only {xy.shape[0]} data points within the range, which is not enough to fit a spline with {spline_fitter.num_degrees_of_freedom} degrees of freedom. '
                'The fit may be poor or fail.'
            )
        elif xy.shape[0] - spline_fitter.num_degrees_of_freedom <= spline_fitter.num_degrees_of_freedom * 0.1:
            log.warning(
                f'Page {name} has only {xy.shape[0]} data points within the range, which is close to the number of degrees of freedom ({spline_fitter.num_degrees_of_freedom}) of the spline. '
                'The fit may be poor or unstable.'
            )


    compute = partial(_compute_background_spline, range=k_range, sargs=sargs, spline_fitter=spline_fitter, log=log)

    threaded = len(context.datastore.pages) >= 12 and context.options.debug is False
    page_background = execute_threaded(compute, page_fulldata, argkind='s', threaded=threaded, pass_key_as='name')

    log.info('Completed spline background calculation for all pages.')

    return page_background
