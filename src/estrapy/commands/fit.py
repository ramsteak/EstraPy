import numpy as np
import seaborn as sns

from pathlib import Path

from lark import Token, Tree

from dataclasses import dataclass
from typing import Self
from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm

from ..core.context import CommandArguments, Command, CommandResult
from ..core.context import Context, ParseContext
from ..core.commandparser import CommandArgumentParser
from ..core.number import Number, parse_range, Unit
from ..operations.fourier import apodizer_functions
from ..core.misc import fuzzy_match
from ..core.fit.fitmodel import ExafsModel
from ..core.datastore import Domain
from ..operations.fourier import flattop_window, fourier

@dataclass(slots=True)
class CommandArguments_Fit(CommandArguments):
    modelpath: Path
    krange: tuple[Number, Number]
    rrange: tuple[Number, Number]
    kweight: float
    apodizer: str
    apodizerp: float

    # to be filled after parsing. If still None after parsing, raise error.
    model: ExafsModel = None  # type: ignore

@dataclass(slots=True)
class CommandResult_Fit(CommandResult):
    ...

parse_fit_command = CommandArgumentParser(CommandArguments_Fit)
parse_fit_command.add_argument('modelpath', required=True, type=Path)
_default_krange = (Number(None, 0.0, Unit.K), Number(None, np.inf, Unit.K))
_default_rrange = (Number(None, 0.0, Unit.A), Number(None, np.inf, Unit.A))
parse_fit_command.add_argument('krange', '--krange', nargs=2, types=parse_range, default=_default_krange)
parse_fit_command.add_argument('rrange', '--rrange', nargs=2, types=parse_range, default=_default_rrange)
parse_fit_command.add_argument('kweight', '--kweight', '-k', type=float, required=False, default=0.0)
parse_fit_command.add_argument('apodizer', '--apodizer', '-a', type=str, required=False, default='hanning')
parse_fit_command.add_argument('apodizerp', '--parameter', '-p', type=float, required=False, default=3)


@dataclass(slots=True)
class Command_Fit(Command[CommandArguments_Fit, CommandResult_Fit]):
    @classmethod
    def parse(
        cls: type[Self], commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext
    ) -> Self:
        arguments = parse_fit_command(commandtoken, tokens, parsecontext)

        # Validate parameters --------------------------------------------------------------
        if arguments.krange[0].unit != Unit.K or arguments.krange[1].unit != Unit.K:
            raise ValueError('Fit krange must be specified in k units.')
        if arguments.krange[0].value < 0.0:
            raise ValueError('Fit krange start must be non-negative.')
        if arguments.krange[1].value <= arguments.krange[0].value:
            raise ValueError('Fit krange end must be greater than start.')
        if arguments.rrange[0].unit != Unit.A or arguments.rrange[1].unit != Unit.A:
            raise ValueError('Fit rrange must be specified in Angstroms.')
        if arguments.rrange[0].value < 0.0:
            raise ValueError('Fit rrange start must be non-negative.')
        if arguments.rrange[1].value <= arguments.rrange[0].value:
            raise ValueError('Fit rrange end must be greater than start.')
        if arguments.kweight < 0.0:
            raise ValueError('Fit kweight must be non-negative.')
        if arguments.apodizer not in apodizer_functions:
            apodizer = fuzzy_match(arguments.apodizer, apodizer_functions)
            if apodizer is not None:
                arguments.apodizer = apodizer
            else:
                raise ValueError(f"Unknown apodizer function '{arguments.apodizer}'. Available options are: {', '.join(apodizer_functions)}.")

        # Resolve and read model ---------------------------------------------------------------
        if not arguments.modelpath.is_absolute():
            arguments.modelpath = (parsecontext.paths.workingdir / arguments.modelpath).resolve()
        
        model = ExafsModel.load(arguments.modelpath)
        arguments.model = model

        # Add paths to additional paths
        parsecontext.paths.additional_paths[arguments.modelpath] = Path('.') / arguments.modelpath.name
        for phase in model.phases:
            parsecontext.paths.additional_paths[phase.feffpath.file] = Path('phases') / phase.feffpath.file.name

        return cls(
            line=commandtoken.line or -1,
            name=commandtoken.value,
            args=arguments,
        )

    def execute(self, context: Context) -> CommandResult_Fit:
        log = context.logger.getChild('command.fit')
        log.info(f'Fitting model from {self.args.modelpath}')
        
        krange = (
            self.args.krange[0].value,
            self.args.krange[1].value
        )
        
        fileout = open("output.txt", "w")

        for name, page in context.datastore.pages.items():
            log.debug(f'Fitting page: {name}')
            domain = page.domains[Domain.RECIPROCAL]
            df = domain.get_columns_data(['k', 'chi'])
            fitresult = self.args.model.fit_k(df['k'].to_numpy(), df['chi'].to_numpy(), krange, self.args.kweight)
            
            # Plotting the fit results ----------------------------------------------------------------------------------
            fig1, (ax11, ax12) = plt.subplots(1,2, figsize=(12,8), sharey=True)
            fig1.tight_layout()

            ## Plot the k-space fit
            ax11.scatter(fitresult.kaxis, fitresult.knchidata, color="green", marker="+", linewidth=0.5, s=20, label=f'$k^{self.args.kweight} \\cdot \\chi$')
            ax11.plot(fitresult.kaxis, fitresult.fittedknchi, color="black", label='Fit')

            # vshift = np.ptp(fitresult.knchidata - fitresult.fittedknchi)/3
            vshift = 0.14

            for idx,(phasename,phase) in enumerate(fitresult.partialknphases.items()):
                shift = - idx * vshift - vshift*2
                ax11.plot(fitresult.kaxis, phase + shift, color="orange")
                ax11.annotate(phasename, xy=(1.0, shift), color="black", horizontalalignment='right', verticalalignment='center', xycoords=('axes fraction', 'data'))

            shift = - len(fitresult.partialknphases) * vshift - vshift*2
            ax11.annotate('Residuals', xy=(1.0, shift), color="black", horizontalalignment='right', verticalalignment='center', xycoords=('axes fraction', 'data'))
            ax11.scatter(fitresult.kaxis, fitresult.difference + shift, color="black", linewidth=0.5, s=20, marker= "o", facecolors='none')

            ## Plot the R-space fit
            windowcoords = (fitresult.kaxis[0], fitresult.kaxis[0]+0.5, fitresult.kaxis[-1]-0.5, fitresult.kaxis[-1])
            kwindow = flattop_window(fitresult.kaxis, windowcoords, 'hanning')

            r = np.linspace(0, 7, 1000)
            fdat = fourier(fitresult.kaxis, fitresult.knchidata * kwindow, r)
            fchi = fourier(fitresult.kaxis, fitresult.fittedknchi * kwindow, r)

            ax12.scatter(r, np.abs(fdat), color="green", marker="+", linewidth=0.5, s=20, label='Data')
            ax12.scatter(r, np.imag(fdat), color="green", marker="x", linewidth=0.5, s=20)

            ax12.plot(r, np.abs(fchi), color="black", label='Fit')
            ax12.plot(r, np.imag(fchi), color="black")

            for idx,(phasename,phase) in enumerate(fitresult.partialknphases.items()):
                shift = - idx * vshift - vshift*2
                part = fourier(fitresult.kaxis, phase * kwindow, r)
                ax12.plot(r, np.imag(part) + shift, label=f'Phase {idx}', color="orange")
                ax12.plot(r, np.abs(part) + shift, label=f'Phase {idx}', color="orange", linewidth=0.5)
                ax12.plot(r, -np.abs(part) + shift, label=f'Phase {idx}', color="orange", linewidth=0.5)
            
            shift = - len(fitresult.partialknphases) * vshift - vshift*2
            ax12.scatter(r, np.imag(fdat - fchi) + shift, color="black", linewidth=0.5, s=20, marker= "o", facecolors='none')

            ax11.set_ylim(-12*vshift, 3*vshift)

            fig1.suptitle(f'EXAFS Fit Results - Page: {name}', fontsize=16)

            clustermap = sns.clustermap(fitresult.correlation, dendrogram_ratio=(0.1, 0), cbar=False, vmin=-1, vmax=1, cmap="coolwarm", figsize=(10,8.5))
            clustermap.cax.set_visible(False)

            fig3, ax3 = plt.subplots(figsize=(10,8.5))
            idxcov = np.argsort(np.diag(fitresult.covariance))[::-1]
            maxextent = float(fitresult.covariance.abs().max().max())
            sns.heatmap(fitresult.covariance.iloc[idxcov, idxcov], fmt=".2e", cmap="managua", ax=ax3, norm=SymLogNorm(1e-5, vmin=-maxextent, vmax=maxextent))

            figdir = context.paths.outputdir / f"fit_result_{name}"
            figdir.mkdir(parents=True, exist_ok=True)
            fig1.savefig(figdir / 'fit_result.png', dpi=660)
            clustermap.savefig(figdir / 'fit_correlation.png', dpi=660)
            fig3.savefig(figdir / 'fit_covariance.png', dpi=660)

            plt.close("all")

            fileout.write(f"{name}\n{fitresult.fitted_vars}\n\n")
            fileout.flush()
            pass
        fileout.close()

        return CommandResult_Fit()