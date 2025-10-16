from dataclasses import dataclass
from logging import getLogger

from ..core.number import Number
from ..core.grammarclasses import Directive
from ..core.context import Context


@dataclass(slots=True)
class Directive_define(Directive):
    name: str
    value: Number | str | int

@dataclass(slots=True)
class Directive_clear(Directive):
    ...

def execute_directive(directive: Directive, context: Context) -> None:
    match directive:
        case Directive_define(name, value):
            context.vars[name] = value
        case Directive_clear():
            # Clear output directory of all files
            for file in context.paths.outputdir.glob('*'):
                try:
                    if file.is_file():
                        file.unlink(missing_ok=True)
                except PermissionError:
                    if file == context.paths.logfile:
                        # We cannot remove the log file while it's being used
                        continue
                    _file = file.relative_to(context.paths.outputdir)
                    getLogger('estrapy.directives').warning(
                        f"Could not delete file '{_file}' from output directory '{context.paths.outputdir.stem}' due to permission error."
                    )
        case _:
            raise NotImplementedError(f"Directive '{directive}' execution not implemented.")
