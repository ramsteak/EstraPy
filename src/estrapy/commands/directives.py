from dataclasses import dataclass
from logging import getLogger

from typing import Any

from ..core.number import Number
from ..core.context import Directive, Context


@dataclass(slots=True)
class Directive_define(Directive):
    name: str
    value: Number | str | int


@dataclass(slots=True)
class Directive_clear(Directive):
    ...


@dataclass(slots=True)
class Directive_title(Directive):
    title: str


@dataclass(slots=True)
class Directive_archive(Directive):
    ...


def execute_directive(directive: Directive, context: Context) -> None:
    match directive:
        case Directive_define(name, value):
            context.vars[name] = value
        case Directive_clear():
            # Clear output directory of all files
            for file in context.paths.outputdir.glob('**'):
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
            # After first pass, only empty directories remain
            for dir in context.paths.outputdir.glob('**'):
                if dir == context.paths.outputdir:
                    continue
                try:
                    if dir.is_dir():
                        dir.rmdir()
                except OSError:
                    _dir = dir.relative_to(context.paths.outputdir)
                    getLogger('estrapy.directives').warning(
                        f"Could not delete directory '{_dir}' from output directory '{context.paths.outputdir.stem}' because it is not empty."
                    )
        case Directive_archive():
            context.options.archive = True
        case Directive_title(title):
            context.projecttitle = title
        case _:
            raise NotImplementedError(f"Directive '{directive}' execution not implemented.")

# Lower number means higher priority
DIRECTIVE_PRIORITIES: dict[type[Directive], int] = {
    Directive_clear: 0,
    Directive_archive: 1,
    Directive_define: 10,
}

# Requirements are specified as: Directive A requires Directive B with arguments C
DIRECTIVE_REQUIRES: dict[type[Directive], list[tuple[type[Directive], tuple[Any, ...]]]] = {
    Directive_archive: [(Directive_clear, ())],
}

def sorted_directives(directives: list[Directive]) -> list[Directive]:
    """Sort directives by priority to ensure correct execution order, and add required directives."""
    # Add required directives
    directives = directives.copy()
    existing_types = {type(directive) for directive in directives}
    for directive in directives:
        requirements = DIRECTIVE_REQUIRES.get(type(directive), [])
        for req, args in requirements:
            if req not in existing_types:
                existing_types.add(req)
                directives.append(req(*args))
    
    # Stable sort to maintain order of same-priority directives
    directives.sort(key=lambda d: DIRECTIVE_PRIORITIES.get(type(d), 100))
    return directives
