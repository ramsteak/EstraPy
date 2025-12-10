import numpy as np

from lark import Token, Tree

from dataclasses import dataclass
from typing import Self

from ..core.grammarclasses import CommandArguments, Command
from ..core.context import Context, ParseContext
from ..core.commandparser import CommandArgumentParser
from ..core.number import Number, parse_range, parse_number, Unit
from ..operations.fourier import fourier, apodizer_functions, flattop_window
from ..operations.axis_conversions import Nyquist
from ..core.datastore import Domain, ColumnKind, ColumnDescription, DataDomain
from ..core.misc import fmt, fuzzy_match

@dataclass(slots=True)
class CommandArguments_Fourier(CommandArguments):
    range: tuple[Number, Number]
    maxR: Number | None
    spacing: Number | None
    kweight: float
    width: float | None
    apodizer: str
    apodizerp: float

parse_fourier_command = CommandArgumentParser(CommandArguments_Fourier)
parse_fourier_command.add_argument('range', nargs=2, types=parse_range, required=True)
parse_fourier_command.add_argument('maxR', type=parse_number, default=None, required=False)
parse_fourier_command.add_argument('spacing', '--dr', '--spacing', type=parse_number, required=False, default=None)
parse_fourier_command.add_argument('kweight', '--kweight', '-k', type=float, required=False, default=0.0)
parse_fourier_command.add_argument('width', '--width', '-w', type=float, required=False, default=None)
parse_fourier_command.add_argument('apodizer', '--apodizer', '-a', type=str, required=False, default='hanning')
parse_fourier_command.add_argument('apodizerp', '--parameter', '-p', type=float, required=False, default=3)


@dataclass(slots=True)
class Command_Fourier(Command[CommandArguments_Fourier, None]):
    @classmethod
    def parse(
        cls: type[Self], commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext
    ) -> Self:
        arguments = parse_fourier_command(commandtoken, tokens, parsecontext)

        # Validate parameters
        if arguments.range[0].unit != Unit.K or arguments.range[1].unit != Unit.K:
            raise ValueError('Fourier range must be specified in k units.')
        if arguments.range[0].value < 0.0:
            raise ValueError('Fourier range start must be non-negative.')
        if arguments.range[1].value <= arguments.range[0].value:
            raise ValueError('Fourier range end must be greater than start.')
        if arguments.maxR is not None and arguments.maxR.unit != Unit.A:
            raise ValueError('Fourier maxR must be specified in Angstroms.')
        if arguments.spacing is not None and arguments.spacing.unit != Unit.A:
            raise ValueError('Fourier spacing must be specified in Angstroms.')
        if arguments.kweight < 0.0:
            raise ValueError('Fourier kweight must be non-negative.')
        if arguments.width is not None and arguments.width <= 0.0:
            raise ValueError('Window taper width must be positive.')
        if arguments.apodizer not in apodizer_functions:
            apodizer = fuzzy_match(arguments.apodizer, apodizer_functions)
            if apodizer is not None:
                arguments.apodizer = apodizer
            else:
                raise ValueError(f"Unknown apodizer function '{arguments.apodizer}'. Available options are: {', '.join(apodizer_functions)}.")


        return cls(
            line=commandtoken.line or -1,
            name=commandtoken.value,
            args=arguments,
        )

    def execute(self, context: Context) -> None:
        log = context.logger.getChild('command.fourier')

        for name, page in context.datastore.pages.items():
            domain = page.domains[Domain.RECIPROCAL]

            if Domain.FOURIER not in page.domains:
                newdom = DataDomain()
                page.domains[Domain.FOURIER] = newdom
            
            fourier_domain = page.domains[Domain.FOURIER]
            
            # Get the valid range
            range = self.args.range[0].value, self.args.range[1].value
            df = domain.get_columns_data(['k', 'chi'])  # Ensure columns exist
            idx = (df['k'] >= range[0]) & (df['k'] <= range[1])

            # Define the new r axis
            k_axis = domain.get_column_data('k')[idx].to_numpy()
            match self.args.spacing, self.args.maxR:
                case None, None:
                    r_axis = Nyquist.axis(k_axis)
                case None, Number(value=maxR):
                    _, diff = Nyquist.span_diff(k_axis)
                    r_axis = np.arange(0.0, maxR + diff, diff)
                case Number(value=spacing), None:
                    span,_ = Nyquist.span_diff(k_axis)
                    r_axis = np.arange(0.0, span + spacing, spacing)
                case Number(value=spacing), Number(value=maxR):
                    r_axis = np.arange(0.0, maxR + spacing, spacing)

            ramp_width = self.args.width if self.args.width is not None else 0.5 * (range[1] - range[0])
            w = flattop_window(k_axis, (range[0], range[0]+ramp_width, range[1]-ramp_width, range[1]), self.args.apodizer)
            chi = df['chi'][idx].to_numpy() * w * k_axis**self.args.kweight
            f = fourier(k_axis, chi, r_axis)

            fourier_domain.add_column_data(
                'r',
                ColumnDescription(
                    name='r',
                    type=ColumnKind.AXIS,
                    unit=Unit.A,
                    labl='Radial distance [Å]'
                ),
                r_axis
            )
            fourier_domain.add_column_data(
                'f',
                ColumnDescription(
                    name='f',
                    type=ColumnKind.DATA,
                    unit=None,
                    labl=f'FT{{k^{self.args.kweight}*χ(k)}} [Å⁻{fmt.sup(int(self.args.kweight-1))}]'
                ),
                f
            )
            log.debug(f'Computed Fourier transform for page "{name}" with {len(r_axis)} points.')
        log.info(f'Computed Fourier transform for {len(context.datastore.pages)} pages.')
