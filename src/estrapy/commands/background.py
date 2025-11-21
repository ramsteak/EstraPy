import numpy as np
import numpy.typing as npt

from lark import Token, Tree

from dataclasses import dataclass
from typing import Self
from functools import partial
from logging import Logger

from ..core.grammarclasses import CommandArguments, Command, CommandResult
from ..core.threaded import execute_threaded
from ..core.context import Context, ParseContext
from ..grammar.commandparser import CommandArgumentParser
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
class SubCommandArguments_Background_SplineNodes(CommandArguments):
    nodes: list[Number]
    kweight: float

SubCommand = SubCommandArguments_Background_Fourier | SubCommandArguments_Background_Polynomial | SubCommandArguments_Background_SplineNodes
@dataclass(slots=True)
class CommandArguments_Background(CommandArguments):
    range: tuple[Number, Number]
    mode: SubCommand

sub_fourier = CommandArgumentParser(SubCommandArguments_Background_Fourier, name='fourier')
sub_fourier.add_argument('rmax', '--rmax', type=parse_number, required=False, default=Number(None, 1.0, Unit.A))
sub_fourier.add_argument('kweight', '--kweight', type=float, required=False, default=2)

sub_polynomial = CommandArgumentParser(SubCommandArguments_Background_Polynomial, name='polynomial')
sub_polynomial.add_argument('degree', '--degree', '-d', type=int, required=False, default=3)
sub_polynomial.add_argument('kweight', '--kweight', type=float, required=False, default=2)

sub_spline_nodes = CommandArgumentParser(SubCommandArguments_Background_SplineNodes, name='splinenodes')
sub_spline_nodes.add_argument('nodes', '--nodes', '-n', type=parse_number, nargs='+', required=True)
sub_spline_nodes.add_argument('kweight', '--kweight', type=float, required=False, default=2)

_default_range = (Number(None, 0.0, Unit.K), Number(None, np.inf, Unit.K))
parse_background_command = CommandArgumentParser(CommandArguments_Background)
parse_background_command.add_argument('range', types=parse_range, nargs=2, required=False, default=_default_range)
parse_background_command.add_subparser('fourier', sub_fourier, 'mode')
parse_background_command.add_subparser('polynomial', sub_polynomial, 'mode')
parse_background_command.add_subparser('splinenodes', sub_spline_nodes, 'mode')

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

def _background_polynomial(xy: npt.NDArray[np.floating], name: str, range: tuple[float, float], degree: int, kweight: float, log: Logger) -> npt.NDArray[np.floating]:
    idx = (xy[:,0] >= range[0]) & (xy[:,0] <= range[1])
    x, y = xy[idx,0], xy[idx,1]

    coeffs = np.polyfit(x, y * x ** kweight, degree)
    p = np.poly1d(coeffs)
    log.debug(f'Computed Polynomial background for page {name} in range [{range[0]}k, {range[1]}k] with degree {degree}')
    return p(xy[:,0]) / (xy[:,0] ** kweight)

# def _background_spline_nodes(xy: npt.NDArray[np.floating], range: tuple[float, float]) -> npt.NDArray[np.floating]:...

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

        if arguments.range[0].unit is not Unit.K or arguments.range[1].unit is not Unit.K:
            raise ValueError('Background range must be specified in k units.')
        
        # Validate subcommand arguments
        match arguments.mode:
            case SubCommandArguments_Background_Fourier(rmax=rmax):
                if rmax.unit is not Unit.A:
                    raise ValueError('Background Fourier rmax must be specified in Angstroms.')
            case SubCommandArguments_Background_Polynomial(degree=degree):
                if degree < 0:
                    raise ValueError('Background Polynomial degree must be non-negative.')
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
        compute = partial(_background_polynomial, range=k_range, degree=args.degree, kweight=args.kweight, log=log)
        
        threaded = len(context.datastore.pages) >= 24 and context.options.debug is False
        page_background = execute_threaded(compute, page_fulldata, argkind='s', threaded=threaded, pass_key_as='name')

        for name, page in context.datastore.pages.items():
            _remove_background(page, page_background[name])
        log.info(f'Calculated background for {len(context.datastore.pages)} spectra using polynomial method.')

    def execute(self, context: Context) -> CommandResult_Background:
        match self.args.mode:
            case SubCommandArguments_Background_Fourier():
                self._execute_fourier(self.args.mode, self.args.range, context)
            case SubCommandArguments_Background_Polynomial():
                self._execute_polynomial(self.args.mode, self.args.range, context)
            case _:
                raise ValueError('Invalid background mode')
        return CommandResult_Background()
