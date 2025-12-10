from lark import Token, Tree

from dataclasses import dataclass, field
from typing import Self

from ..core.grammarclasses import CommandArguments, Command, CommandResult
from ..core.context import Context, ParseContext
from ..core.commandparser import CommandArgumentParser
from ..core.misc import fuzzy_match, Bag
from ..core.datastore import Domain

# Define which show modes require a page to be specified, as a tuple of
# (works without page, works with page)
SHOW_MODES = {
    "domains":  (True,  False),
    "pages":    (True,  False),
    "variables":(True,  False),
    "metadata": (False, True),
    "columns":  (False, True),
    "data":     (False, True),
    "summary":  (True,  True),
}

@dataclass(slots=True)
class CommandArguments_Show(CommandArguments):
    mode: str
    page: str | None

@dataclass(slots=True)
class CommandResult_Show(CommandResult):
    message: list[str] = field(default_factory=list[str])

parse_show_command = CommandArgumentParser(CommandArguments_Show)
parse_show_command.add_argument('mode', type=str, required=False)
parse_show_command.add_argument('page', '--page', type=str, required=False, default=None)

@dataclass(slots=True)
class Command_Show(Command[CommandArguments_Show, CommandResult_Show]):
    @classmethod
    def parse(
        cls: type[Self], commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext
    ) -> Self:
        arguments = parse_show_command(commandtoken, tokens, parsecontext)

        mode = fuzzy_match(arguments.mode, SHOW_MODES)
        if mode is None:
            raise ValueError(f"Invalid mode '{arguments.mode}' for show command. Valid modes are: {', '.join(SHOW_MODES)}")
        arguments.mode = mode

        if arguments.page is not None and not SHOW_MODES[mode][1]:
            raise ValueError(f"Mode '{mode}' does not accept a page argument.")
        elif arguments.page is None and not SHOW_MODES[mode][0]:
            raise ValueError(f"Mode '{mode}' requires a page argument.")

        return cls(
            line=commandtoken.line or -1,
            name=commandtoken.value,
            args=arguments,
        )

    def execute(self, context: Context) -> CommandResult_Show:
        log = context.logger.getChild("command.show")
        result = CommandResult_Show()

        match self.args.mode:
            case "domains":
                # Check that all pages have the same domains defined.
                domaingrps = Bag[frozenset[Domain], str].from_iter(
                    (frozenset(page.domains.keys()), name)
                    for name, page in context.datastore.pages.items()
                )
                if domaingrps.count_keys() > 1:
                    result.message.append("Different domain sets found across pages:")
                    for domains, page_names in domaingrps.groups():
                        domain_list = ', '.join(domain.name for domain in domains)
                        result.message.append(f'  Domains: {domain_list}')
                        result.message.extend(f'      - {pname}' for pname in page_names)
                else:
                    domain_list = ', '.join(domain.name for domain in next(iter(domaingrps)))
                    result.message.append(f"All pages have the same domains defined: {domain_list}")
            case "pages":
                result.message.append("Pages in datastore:")
                result.message.extend(f"  - {name}" for name in context.datastore.pages.keys())
            case "variables":
                # Show all non-hidden variables across all pages (hidden variables start with '.')
                vargrps = Bag[frozenset[str], str].from_iter(
                    (frozenset(vn for vn in page.meta._dict if not vn.startswith('.')), pname) # pyright: ignore[reportPrivateUsage]
                    for pname, page in context.datastore.pages.items()
                )
                if vargrps.count_keys() > 1:
                    result.message.append("Different variable sets found across pages:")
                    for varset, page_names in vargrps.groups():
                        var_list = ', '.join(varset)
                        result.message.append(f'  Variables: {var_list}')
                        result.message.extend(f'      - {pname}' for pname in page_names)
                else:
                    vars = next(iter(vargrps))
                    if vars:
                        var_list = ', '.join(sorted(vars))
                    else:
                        var_list = "(no variables)"
                    result.message.append(f"All pages have the same variables defined: {var_list}")
            case "metadata":
                if self.args.page not in context.datastore.pages:
                    raise ValueError(f"Page '{self.args.page}' not found in datastore.")
                page = context.datastore.pages[self.args.page]
                # Show all variable and values in the specified page
                result.message.append(f"Variables in page '{page.meta.name}':")
                result.message.extend(f"  - {vname}: {value}" for vname, value in page.meta._dict.items()) # pyright: ignore[reportPrivateUsage]
            case "columns":
                if self.args.page not in context.datastore.pages:
                    raise ValueError(f"Page '{self.args.page}' not found in datastore.")
                page = context.datastore.pages[self.args.page]

                for domain,datadomain in page.domains.items():
                    result.message.append(f"Columns in domain '{domain}':")
                    result.message.extend(f"   - {colname} -> {col[-1].desc.labl}" for colname, col in datadomain.columns.items())
                    result.message.append(f" Hidden columns in domain '{domain}':")
                    result.message.extend(f"   - {colname}" for colname in datadomain.data.columns)

            case "data":
                ... # TODO: to be implemented
            case "summary":
                ... # TODO: to be implemented
            case _:
                raise ValueError(f"Invalid mode '{self.args.mode}' for show command.")

        for line in result.message:
            log.info(line)

        return result