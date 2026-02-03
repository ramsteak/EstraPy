import numpy as np
from numpy import typing as npt
from logging import Logger

from dataclasses import dataclass
from functools import partial

from .result import BackgroundResult

from ...core.threaded import execute_threaded
from ...core.commandparser import CommandArguments, field_arg
from ...core._validators import validate_float_non_negative, validate_int_non_negative
from ...core.number import Number, parse_number
from ...core.context import Context
from ...core.datastore import Domain

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

    def validate(self) -> bool:
        """
        Validates the consistency between nodes and degrees.
        """
        if len(self.nodes) != len(self.degrees)-1:
            raise ValueError(
                f"Number of nodes ({len(self.nodes)}) must match number of degrees ({len(self.degrees)})."
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

    # TODO: if nodes is one element without unit, interpret as nknots and generate equidistant knots within range

    spline_fitter = PiecewiseSplineFitter(
        knots = np.array([n.value for n in sargs.nodes]),
        degrees = np.array(sargs.degrees),
        fixed_points = [(fp[0].value, fp[1]) for fp in sargs.fixedpoints] if sargs.fixedpoints is not None else None
    )

    page_fulldata: dict[str, npt.NDArray[np.floating]] = {
        name:page.domains[Domain.RECIPROCAL].get_columns_data(['k', 'chi']).to_numpy()
        for name,page in context.datastore.pages.items()
    }

    log.debug('Computing background for all pages.')

    compute = partial(_compute_background_spline, range=k_range, sargs=sargs, spline_fitter=spline_fitter, log=log)

    threaded = len(context.datastore.pages) >= 12 and context.options.debug is False
    page_background = execute_threaded(compute, page_fulldata, argkind='s', threaded=threaded, pass_key_as='name')

    log.info('Completed spline background calculation for all pages.')

    return page_background
