import numpy as np
import numpy.typing as npt

from lark import Token, Tree

from dataclasses import dataclass
from typing import Self
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from ..core.grammarclasses import CommandArguments, Command
from ..core.context import Context, ParseContext
from ..grammar.commandparser import CommandArgumentParser
from ..core.number import Number, parse_range, Unit, parse_number
from ..core.datastore import Domain, ColumnDescription, ColumnKind
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

def _background_fourier(xy: npt.NDArray[np.floating], range: tuple[float,float], cutoff: float, kweight: float) -> npt.NDArray[np.floating]:
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
    return bkg

def _background_polynomial(xy: npt.NDArray[np.floating], range: tuple[float, float], degree: int, kweight: float) -> npt.NDArray[np.floating]:
    idx = (xy[:,0] >= range[0]) & (xy[:,0] <= range[1])
    x, y = xy[idx,0], xy[idx,1]

    coeffs = np.polyfit(x, y * x ** kweight, degree)
    p = np.poly1d(coeffs)
    return p(xy[:,0]) / (xy[:,0] ** kweight)

# def _background_spline_nodes(xy: npt.NDArray[np.floating], range: tuple[float, float]) -> npt.NDArray[np.floating]:...

@dataclass(slots=True)
class Command_Background(Command[CommandArguments_Background]):
    @classmethod
    def parse(
        cls: type[Self], commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext
    ) -> Self:
        arguments = parse_background_command(commandtoken, tokens, parsecontext)
        return cls(
            line=commandtoken.line or -1,
            name=commandtoken.value,
            args=arguments,
        )

    def execute(self, context: Context) -> None:
        log = context.logger.getChild('command.background')
        # TODO: transfer unit check to parsing
        if self.args.range[0].unit is not Unit.K or self.args.range[1].unit is not Unit.K:
            raise ValueError('Background range must be specified in k units.')
        range = (self.args.range[0].value, self.args.range[1].value)

        match self.args.mode:
            case SubCommandArguments_Background_Fourier(rmax=rmax, kweight=kweight):
                # check units
                if not rmax.unit == Unit.A:
                    raise ValueError(f'Background Fourier rmax must be in Angstroms, got {rmax.unit}')
                background_method = partial(_background_fourier, range=range, cutoff=rmax.value, kweight=kweight)
                method_name = 'fourier'
            case SubCommandArguments_Background_Polynomial(degree=degree, kweight=kweight):
                background_method = partial(_background_polynomial, range=range, degree=degree, kweight=kweight)
                method_name = 'polynomial'
            # case SubCommandArguments_Background_SplineNodes(nodes=nodes, kweight=kweight):
            #     background_method = _background_spline_nodes
            case _:
                raise ValueError('Invalid background mode')
            
        dat = {name:page.domains[Domain.RECIPROCAL].get_columns_data(['k', 'chi']).to_numpy() for name,page in context.datastore.pages.items()}
        
        if len(context.datastore.pages) >= 12 and context.options.debug is False:
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                bkgs = [executor.submit(background_method, d) for _,d in dat.items()]
            bkgs = {name:bkg.result() for name, bkg in zip(context.datastore.pages.keys(), bkgs)}
        else:
            bkgs = {
                name:background_method(d)
                for name,d in dat.items()
            }

        for name, page in context.datastore.pages.items():
            domain = page.domains[Domain.RECIPROCAL]
            # # TODO: Check that the columns exist and (if needed) E0 is set

            # df = domain.get_columns_data(['k', 'chi'])

            # bkg = background_method(df.to_numpy())
            domain.add_column_data(
                'bkg',
                ColumnDescription(
                    name='bkg',
                    type=ColumnKind.DATA,
                    unit=None,
                    labl='Background'
                ),
                bkgs[name]
            )
            domain.add_column_data('chi',None,dat[name][:,1] - bkgs[name])
        log.info(f'Calculated background for {len(context.datastore.pages)} spectra using {method_name} method.')
