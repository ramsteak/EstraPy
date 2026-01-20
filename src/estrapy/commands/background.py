import numpy as np
import numpy.typing as npt

from lark import Token, Tree

from dataclasses import dataclass
from typing import Self
from functools import partial
from logging import Logger

from ..core.context import CommandArguments, Command, CommandResult
from ..core.threaded import execute_threaded
from ..core.context import Context, ParseContext
from ..core.commandparser import CommandArgumentParser
from ..core.number import Number, parse_range, Unit, parse_number
from ..core.datastore import Domain, ColumnDescription, ColumnKind, DataPage
from ..operations.fourier import flattop_window, fourier


@dataclass(slots=True)
class SubCommandArguments_Background_Fourier(CommandArguments):
    rmax: Number
    kweight: float

@dataclass(slots=True)
class SubCommandArguments_Background_Polynomial(CommandArguments):
    degree: int
    kweight: float

@dataclass(slots=True)
class SubCommandArguments_Background_Spline(CommandArguments):
    kweight: float
    nodes: list[Number] | None = None
    nknots: int | None = None

SubCommand = (
    SubCommandArguments_Background_Fourier
     | SubCommandArguments_Background_Polynomial
     | SubCommandArguments_Background_Spline
)

@dataclass(slots=True)
class CommandArguments_Background(CommandArguments):
    range: tuple[Number, Number]
    mode: SubCommand

sub_fourier = CommandArgumentParser(SubCommandArguments_Background_Fourier, name='fourier')
sub_fourier.add_argument('rmax', '--rmax', type=parse_number, required=False, default=Number(None, 1.0, Unit.A))
sub_fourier.add_argument('kweight', '--kweight', '-k', type=float, required=False, default=2)

sub_polynomial = CommandArgumentParser(SubCommandArguments_Background_Polynomial, name='polynomial')
sub_polynomial.add_argument('degree', '--degree', '-d', type=int, required=False, default=3)
sub_polynomial.add_argument('kweight', '--kweight', '-k', type=float, required=False, default=2)

sub_spline_nodes = CommandArgumentParser(SubCommandArguments_Background_Spline, name='splinenodes')
sub_spline_nodes.add_argument('nodes', '--nodes', type=parse_number, nargs='+')
sub_spline_nodes.add_argument('nknots', '--nknots', '-n', type=int, required=False)
sub_spline_nodes.add_argument('kweight', '--kweight', '-k', type=float, required=False, default=2)

_default_range = (Number(None, 0.0, Unit.K), Number(None, np.inf, Unit.K))
parse_background_command = CommandArgumentParser(CommandArguments_Background)
parse_background_command.add_argument('range', types=parse_range, nargs=2, required=False, default=_default_range)
parse_background_command.add_subparser('fourier', sub_fourier, 'mode')
parse_background_command.add_subparser('polynomial', sub_polynomial, 'mode')
parse_background_command.add_subparser('spline', sub_spline_nodes, 'mode')

@dataclass(slots=True)
class SubCommandResult_Background_Fourier(CommandArguments):
    fourier_axis: npt.NDArray[np.floating]
    fourier_window: npt.NDArray[np.floating]
    page_fourier: dict[str, npt.NDArray[np.complexfloating]]

def _compute_background_fourier(xy: npt.NDArray[np.floating], name: str, range: tuple[float,float], cutoff: float, kweight: float, log: Logger) -> npt.NDArray[np.floating]:
    # If range is infinite, set the window to be the width of the data
    range = (range[0] if range[0] != -np.inf else xy[0,0], range[1] if range[1] != np.inf else xy[-1,0])
    idx = (xy[:,0] >= range[0]) & (xy[:,0] <= range[1])
    x, y = xy[idx,0], xy[idx,1]
    w = flattop_window(x, (range[0]-0.1, range[0]+1, range[1]-0.1, range[1]+1), 'hanning')

    r = np.linspace(-5*cutoff,5*cutoff,2**10)
    W = flattop_window(r, (-cutoff-0.1,-cutoff+0.1,cutoff-0.1,cutoff+0.1), 'hanning')

    f = fourier(x, y * w * (x ** kweight), r)
    b = fourier(r, W * f.conj(), x).real / w / (x ** kweight)
    
    bkg = np.zeros_like(xy[:,0])
    bkg[idx] = b
    log.debug(f'Computed Fourier background for page {name} in range [{range[0]:0.2f}, {range[1]:0.2f}]')
    return bkg

def _compute_background_polynomial(xy: npt.NDArray[np.floating], name: str, range: tuple[float, float], degree: int, kweight: float, log: Logger) -> npt.NDArray[np.floating]:
    idx = (xy[:,0] >= range[0]) & (xy[:,0] <= range[1])
    x, y = xy[idx,0], xy[idx,1]

    coeffs = np.polyfit(x, y * x ** kweight, degree)
    p = np.poly1d(coeffs)
    log.debug(f'Computed Polynomial background for page {name} in range [{range[0]}k, {range[1]}k] with degree {degree}')
    return p(xy[:,0]) / (xy[:,0] ** kweight)

def _compute_background_spline(xy: npt.NDArray[np.floating], name: str, range: tuple[float, float], nknots: int | None, knots: npt.NDArray[np.floating] | None, kweight: float, log: Logger) -> npt.NDArray[np.floating]:
    from scipy.interpolate import LSQUnivariateSpline
    
    idx = (xy[:,0] >= range[0]) & (xy[:,0] <= range[1])
    x, y = xy[idx,0], xy[idx,1]
    
    # Determine knots: user-specified or automatic
    if knots is not None:
        # Use user-specified knots, filtering to interior points only
        knots_interior = knots[(knots > x[0]) & (knots < x[-1])]
        if len(knots_interior) == 0:
            log.error(f'Page {name}: No valid interior knots in range [{x[0]:.2f}, {x[-1]:.2f}]. Returning zero background.')
            return np.zeros_like(xy[:,0])
        knots_to_use = np.sort(knots_interior)
        log.debug(f'Page {name}: Using {len(knots_to_use)} user-specified knots')
    elif nknots is not None:
        # Generate evenly-spaced knots
        if len(x) < nknots + 4:  # Need enough points for cubic spline
            log.warning(f'Page {name}: Insufficient points ({len(x)}) in range for {nknots} knots, reducing to {max(1, len(x) - 4)}')
            nknots = max(1, len(x) - 4)
        knots_to_use = np.linspace(x[0], x[-1], nknots + 2)[1:-1]
    else:
        log.error(f'Page {name}: Must specify either nknots or knots. Returning zero background.')
        return np.zeros_like(xy[:,0])
    
    try:
        # Fit B-spline to k-weighted chi with least-squares (smoothing, not interpolating)
        spline = LSQUnivariateSpline(x, y * x ** kweight, knots_to_use, k=3)
        
        # Initialize background as zeros
        bkg = np.zeros_like(xy[:,0])
        
        # Only evaluate within the fitted range to avoid unreliable extrapolation
        bkg[idx] = spline(x) / (x ** kweight)
        
        log.debug(f'Computed B-spline background for page {name} in range [{range[0]:.2f}, {range[1]:.2f}] with {len(knots_to_use)} knots')
        
    except Exception as e:
        log.error(f'Page {name}: B-spline fitting failed: {e}. Returning zero background.')
        bkg = np.zeros_like(xy[:,0])
    
    return bkg

def _remove_background(page: DataPage, bkg: npt.NDArray[np.floating]) -> None:
    domain = page.domains[Domain.RECIPROCAL]
    bkgcol = ColumnDescription(name='bkg', type=ColumnKind.DATA, unit=None, labl='Background')
    domain.add_column_data('bkg', bkgcol, bkg)
    chicol = ColumnDescription(name='chi', type=ColumnKind.DATA, unit=None, deps=['bkg', 'chi'], calc=lambda df: df['chi'] - df['bkg'], labl='EXAFS signal')
    domain.add_column('chi', chicol)

@dataclass(slots=True)
class CommandResult_Background(CommandResult):
    ...

@dataclass(slots=True)
class Command_Background(Command[CommandArguments_Background, CommandResult_Background]):
    @classmethod
    def parse(
        cls: type[Self], commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext
    ) -> Self:
        arguments = parse_background_command(commandtoken, tokens, parsecontext)

        # Allow infinite bounds without an explicit unit (e.g. "0k ..") — treat bounds with unit None as valid only if they are infinite
        def _is_k_or_infinite(num: Number) -> bool:
            try:
                return (num.unit is Unit.K) or (num.unit is None and np.isinf(num.value))
            except Exception:
                return False

        if not (_is_k_or_infinite(arguments.range[0]) and _is_k_or_infinite(arguments.range[1])):
            raise ValueError('Background range must be specified in k units.')
        
        # Validate subcommand arguments
        match arguments.mode:
            case SubCommandArguments_Background_Fourier(rmax=rmax):
                if rmax.unit is not Unit.A:
                    raise ValueError('Background Fourier rmax must be specified in Angstroms.')
            case SubCommandArguments_Background_Polynomial(degree=degree):
                if degree < 0:
                    raise ValueError('Background Polynomial degree must be non-negative.')
            case SubCommandArguments_Background_Spline(nodes=nodes, nknots=nknots):
                if nodes is not None and nknots is not None:
                    raise ValueError('Background Spline: Specify either nodes or nknots, not both.')
                elif nodes is None and nknots is None:
                    raise ValueError('Background Spline: Must specify either nodes or nknots.')
                
                if nknots is not None and nknots <= 0:
                    raise ValueError('Background Spline nknots must be positive.')
                
                if nodes is not None:
                    for node in nodes:
                        if node.unit is not Unit.K:
                            raise ValueError('Background Spline nodes must be specified in k units.')
                
            case _:
                pass
        
        return cls(
            line=commandtoken.line or -1,
            name=commandtoken.value,
            args=arguments,
        )
    
    def _execute_fourier(self, args: SubCommandArguments_Background_Fourier, range: tuple[Number,Number], context: Context) -> None:
        log = context.logger.getChild('command.background.fourier')
        log.debug(f'Calculating background with Fourier method in range [{range[0]!s}, {range[1]!s}], rmax {args.rmax!s}, kweight {args.kweight}')

        k_range = (range[0].value, range[1].value)

        log.debug('Preparing data for all pages.')
        page_fulldata: dict[str, npt.NDArray[np.floating]] = {
            name:page.domains[Domain.RECIPROCAL].get_columns_data(['k', 'chi']).to_numpy()
            for name,page in context.datastore.pages.items()
        }

        log.debug('Calculating background for all pages.')
        compute = partial(_compute_background_fourier, range=k_range, cutoff=args.rmax.value, kweight=args.kweight, log=log)
        
        threaded = len(context.datastore.pages) >= 12 and context.options.debug is False
        page_background = execute_threaded(compute, page_fulldata, argkind='s', threaded=threaded, pass_key_as='name')

        for name, page in context.datastore.pages.items():
            _remove_background(page, page_background[name])
        log.info(f'Calculated background for {len(context.datastore.pages)} spectra using fourier method.')


    def _execute_polynomial(self, args: SubCommandArguments_Background_Polynomial, range: tuple[Number,Number], context: Context) -> None:
        log = context.logger.getChild('command.background.polynomial')
        log.debug(f'Calculating background with Polynomial method in range [{range[0]!s}, {range[1]!s}], degree {args.degree}, kweight {args.kweight}')

        k_range = (range[0].value, range[1].value)

        log.debug('Preparing data for all pages.')
        page_fulldata: dict[str, npt.NDArray[np.floating]] = {
            name:page.domains[Domain.RECIPROCAL].get_columns_data(['k', 'chi']).to_numpy()
            for name,page in context.datastore.pages.items()
        }

        log.debug('Calculating background for all pages.')
        compute = partial(_compute_background_polynomial, range=k_range, degree=args.degree, kweight=args.kweight, log=log)
        
        threaded = len(context.datastore.pages) >= 24 and context.options.debug is False
        page_background = execute_threaded(compute, page_fulldata, argkind='s', threaded=threaded, pass_key_as='name')

        for name, page in context.datastore.pages.items():
            _remove_background(page, page_background[name])
        log.info(f'Calculated background for {len(context.datastore.pages)} spectra using polynomial method.')
    
    def _execute_bspline(self, args: SubCommandArguments_Background_Spline, range: tuple[Number,Number], context: Context) -> None:
        log = context.logger.getChild('command.background.bspline')
        log.debug(f'Calculating background with B-spline method in range [{range[0]!s}, {range[1]!s}], knots {args.nknots}, kweight {args.kweight}')

        k_range = (range[0].value, range[1].value)

        log.debug('Preparing data for all pages.')
        page_fulldata: dict[str, npt.NDArray[np.floating]] = {
            name:page.domains[Domain.RECIPROCAL].get_columns_data(['k', 'chi']).to_numpy()
            for name,page in context.datastore.pages.items()
        }

        log.debug('Calculating background for all pages.')
        nodes = np.array([n.value for n in args.nodes]) if args.nodes is not None else None
        compute = partial(_compute_background_spline, range=k_range, nknots=args.nknots, knots=nodes, kweight=args.kweight, log=log)
        
        threaded = len(context.datastore.pages) >= 12 and context.options.debug is False
        page_background = execute_threaded(compute, page_fulldata, argkind='s', threaded=threaded, pass_key_as='name')

        for name, page in context.datastore.pages.items():
            _remove_background(page, page_background[name])
        log.info(f'Calculated background for {len(context.datastore.pages)} spectra using B-spline method.')

    def execute(self, context: Context) -> CommandResult_Background:
        match self.args.mode:
            case SubCommandArguments_Background_Fourier():
                self._execute_fourier(self.args.mode, self.args.range, context)
            case SubCommandArguments_Background_Polynomial():
                self._execute_polynomial(self.args.mode, self.args.range, context)
            case SubCommandArguments_Background_Spline():
                self._execute_bspline(self.args.mode, self.args.range, context)
            case _:
                raise ValueError('Invalid background mode')
        return CommandResult_Background()
