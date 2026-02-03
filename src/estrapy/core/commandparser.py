from lark import Token, Tree
from dataclasses import dataclass, field, fields, MISSING, Field
from typing import TypeVar, Callable, Any, Union, Protocol, Generic, Mapping, ItemsView, KeysView, ValuesView, Type, Self, Iterator, Sequence, TypeAlias, runtime_checkable
from enum import Enum
from itertools import chain, count

from ..core.errors import CommandError
from ..core.misc import peekable

class Callback_anystr(Protocol):
    def __call__(self, *args: str) -> Any: ...
class ReturnsBool(Protocol):
    def __call__(self, *args: Any) -> bool: ...
class Callback(Protocol):
    def __call__(self, *args: Any, **kwds: Any) -> Any:...

Callback_str = Union[
    Callable[[str], Any],
    Callable[[str, str], Any],
    Callable[[str, str, str], Any],
    Callable[[str, str, str, str], Any],
    Callable[[str, str, str, str, str], Any],
    Callable[[str, str, str, str, str, str], Any],
    Callable[[str, str, str, str, str, str, str], Any],
    Callback_anystr,
]

TokenLike: TypeAlias = Union[Token, Tree[Token], Sequence[Union[Token, Tree[Token]]]]

class CommandParseError(CommandError):
    """Exception raised for errors during parsing."""

# -----------------------------------------------------------------------------
# Structure definitions

@runtime_checkable
class Validatable(Protocol):
    def validate(self) -> None:
        ...


@dataclass(slots=True)
class CommandArguments:
    """Base class for command arguments. Specific commands should subclass this
    class and define their own arguments using dataclass fields with ArgumentSpecification metadata.
    
    If needed, POSITIONAL_ARGUMENT_ORDER can be defined to specify the order of positional arguments.
    If not defined, the order of definition in the dataclass is used.

    The _token_map field is used internally to map argument names to their corresponding
    Lark tokens or parse trees."""
    _token_map: dict[str, TokenLike] = field(
        default_factory=dict[str, TokenLike],
        init=False
    )

class ActionType(Enum):
    STORE = 'store'
    APPEND = 'append'

    STORE_TRUE = 'store_true'
    STORE_FALSE = 'store_false'

    STORE_CONST = 'store_const'
    APPEND_CONST = 'append_const'
    
    COUNT = 'count'

@dataclass(slots=True, frozen=True)
class ActionSpecification:
    action: ActionType
    const: Any = MISSING

    def __post_init__(self) -> None:
        if self.action in {ActionType.STORE_CONST, ActionType.APPEND_CONST} and self.const is MISSING:
            raise ValueError(f'Action {self.action} requires a const value.')
        if self.action in {ActionType.STORE_TRUE, ActionType.STORE_FALSE} and self.const is not MISSING:
            raise ValueError(f'Action {self.action} does not accept a const value.')
        if self.action in {ActionType.STORE, ActionType.APPEND, ActionType.COUNT} and self.const is not MISSING:
            raise ValueError(f'Action {self.action} does not accept a const value.')

@dataclass(slots=True, frozen=True)
class intrange:
    """Represents an integer range with optional start and end.
    Ranges are inclusive of both start and end."""
    start: int | None
    end: int | None

    def __contains__(self, item: int) -> bool:
        if self.start is not None and item < self.start:
            return False
        if self.end is not None and item > self.end:
            return False
        return True
    

@dataclass(slots=True, frozen=True)
class NargsSpecification:
    ranges: list[intrange] = field(default_factory=list[intrange])
    soft: bool = False
    single: bool = False

    def __post_init__(self) -> None:
        # Sort ranges by start value. None values come first.
        sorted_ranges = sorted(self.ranges, key=lambda r: (r.start is not None, r.start))
        merged_ranges: list[intrange] = []
        for r in sorted_ranges:
            if not merged_ranges:
                merged_ranges.append(r)
                continue

            last = merged_ranges[-1]
            if last.end is None or r.start is None or (last.end >= r.start - 1):
                # Merge ranges
                new_start = last.start
                new_end = None
                if last.end is None or r.end is None:
                    new_end = None
                else:
                    new_end = max(last.end, r.end)
                merged_ranges[-1] = intrange(new_start, new_end)
            else:
                merged_ranges.append(r)
        self.ranges.clear()
        self.ranges.extend(merged_ranges)
    
    def __str__(self) -> str:
        parts: list[str] = []
        for r in self.ranges:
            if r.start == r.end:
                parts.append(str(r.start))
            else:
                start_str = str(r.start) if r.start is not None else '-∞'
                end_str = str(r.end) if r.end is not None else '∞'
                parts.append(f'[{start_str},{end_str}]')
        return ' | '.join(parts)
    
    @property
    def allows_any(self) -> bool:
        """Returns True if the nargs specification allows values (i.e. maximum > 0)."""
        return self.maximum is None or self.maximum > 0
    
    @property
    def minimum(self) -> int:
        """Returns the minimum number of arguments allowed by the nargs specification."""
        min_args = min(r.start or 0 for r in self.ranges)
        return min_args
    @property
    def maximum(self) -> int | None:
        """Returns the maximum number of arguments allowed by the nargs specification.
        
        Returns None if there is no maximum (i.e., unbounded).
        """
        max_args = None
        for r in self.ranges:
            if r.end is None:
                return None
            if max_args is None or r.end > max_args:
                max_args = r.end
        return max_args
    
    def __contains__(self, item: int) -> bool:
        """Checks if the given item is allowed by the nargs specification."""
        for r in self.ranges:
            if item in r:
                return True
        return False

    @classmethod
    def parse(cls, value: Any) -> Self:
        """Parses a nargs specification from various input formats."""
        match value:
            case None:
                return cls([intrange(1, 1)], soft=False, single=True)
            case NargsSpecification():
                return cls(value.ranges, value.soft, value.single)
            case int() as n if n >= 0:
                return cls([intrange(n, n)], soft=False, single=(n == 1))
            case '*' | 'zero_or_more':
                return cls([intrange(0, None)], soft=False, single=False)
            case '+' | 'one_or_more':
                return cls([intrange(1, None)], soft=False, single=False)
            case '?' | 'optional':
                return cls([intrange(0, 1)], soft=False, single=True)
            case '*?' | 'zero_or_more_soft':
                return cls([intrange(0, None)], soft=True, single=False)
            case '+?' | 'one_or_more_soft':
                return cls([intrange(1, None)], soft=True, single=False)
            case '??' | 'optional_soft':
                return cls([intrange(0, 1)], soft=True, single=True)
            case [int(start), int(end)] if start <= end and isinstance(value, tuple):
                return cls([intrange(start, end)], soft=False, single=False)
            case list():
                ranges: list[intrange] = []
                for item in value: # pyright: ignore[reportUnknownVariableType]
                    match item:
                        case int() as n if n >= 0:
                            ranges.append(intrange(n, n))
                        case [int(start), int(end)] if start <= end and isinstance(item, tuple):
                            ranges.append(intrange(start, end))
                        case _: # pyright: ignore[reportUnknownVariableType]
                            raise ValueError(f'Invalid nargs specification in list: {item}')
                return cls(ranges, soft=False, single=False)
            case _:
                raise ValueError(f'Invalid nargs specification: {value}')
                    

@dataclass(slots=True, frozen=True)
class ArgumentSpecification(Mapping[str, Any]):
    type: Callable[[str], Any] | None
    types: Callback_str | None
    nargs: NargsSpecification
    action: ActionSpecification
    required: bool
    help: str | None
    destination: str | None
    validate: Callback | list[Callback] | None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)
    def __iter__(self) -> Any:
        for f in fields(self):
            yield f.name
    def __len__(self) -> int:
        return len(fields(self))
    def items(self) -> ItemsView[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}.items()
    def keys(self) -> KeysView[str]:
        return {f.name: None for f in fields(self)}.keys()
    def values(self) -> ValuesView[Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}.values()
    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)
    def __contains__(self, key: object) -> bool:
        return key in {f.name for f in fields(self)}
    

@dataclass(slots=True, frozen=True)
class SubparserSpecification(ArgumentSpecification):
    subparsers: dict[str, Type[CommandArguments]] | None

@dataclass(slots=True, frozen=True)
class PositionSpecification(ArgumentSpecification):
    position: int | None

@dataclass(slots=True, frozen=True)
class OptionSpecification(ArgumentSpecification):
    flags: list[str] | None
    const_flags: dict[str, Any] | None


def check_flag(flag: Any) -> bool:
    """Convenience function to check if a flag is valid. A valid flag is a string that starts with '-' or '--'.
    Options starting with a single '-' must only be one character, while options starting with '--' must be
    longer than two characters."""
    if not isinstance(flag, str): return False
    if not flag.startswith('-'):
        return False
    if flag.startswith('--'):
        return len(flag) > 3
    if flag.startswith('-'):
        return len(flag) == 2
    return False


def field_arg(*,
              default: Any = MISSING,
              default_factory: Callable[[], Any] | Any = MISSING,
              flags: list[str] | None = None,
              const_flags: dict[str, Any] | None = None,
              type: Callable[[str], Any] | None = None,
              types: Callback_str | None = None,
              nargs: NargsSpecification | int | str | tuple[int | None, int | None] | list[int | tuple[int | None, int | None]] | None = None,
              action: ActionSpecification | str | None = None,
              const: Any = MISSING,
              required: bool = False,
              help: str | None = None,
              subparsers: dict[str, type[CommandArguments]] | None = None,
              position: int | None = None,
              validate: Callback | list[Callback] | None = None
) -> Any:
    # Check that only one of default or default_factory is provided
    if default is not MISSING and default_factory is not MISSING:
        raise ValueError('Cannot specify both default and default_factory')
    
    # Check that if validate is provided, it is callable and that default/default_factory are valid.
    if validate is not None:
        if isinstance(validate, list):
            for v in validate:
                if not callable(v):
                    raise ValueError('All items in validate list must be callables.')

        elif not callable(validate):
            raise ValueError('Function validate must be a callable or a list of callables.')
        
        if default is not MISSING:
            try:
                if isinstance(validate, list):
                    for v in validate:
                        if not v(default):
                            raise ValueError(f'Default value {default} is not valid according to validate function {v}.')
                elif not validate(default):
                    raise ValueError(f'Default value {default} is not valid according to validate function.')
            except Exception as e:
                raise ValueError(f'Default value {default} is not valid according to validate function: {e}') from e
            
        if default_factory is not MISSING:
            try:
                value = default_factory()
            except Exception as e:
                raise ValueError(f'Default factory raised an exception when called: {e}') from e
            
            try:
                if isinstance(validate, list):
                    for v in validate:
                        if not v(value):
                            raise ValueError(f'Default factory produced an invalid value {value} according to validate function {v}.')
                elif not validate(value):
                    raise ValueError(f'Default factory produced an invalid value {value} according to validate function.')
            except Exception as e:
                raise ValueError(f'Default factory produced an invalid value {value} according to validate function: {e}') from e

    # Parse action
    match action, const:
        case ActionSpecification() as a, _:
            action_spec = a
        case str() as a_str, _:
            a_type = ActionType(a_str)
            action_spec = ActionSpecification(action=a_type, const=const)
        case _:
            action_spec = ActionSpecification(action=ActionType.STORE, const=MISSING)
    
    # Parse nargs
    nargs_spec = NargsSpecification.parse(nargs)

    # Validate action/nargs compatibility
    if action_spec.action in {ActionType.STORE, ActionType.APPEND}:
        if nargs_spec.allows_any is False:
            raise ValueError(f'Action {action_spec.action} requires nargs to allow at least one value.')
    elif action_spec.action in {ActionType.STORE_TRUE, ActionType.STORE_FALSE, ActionType.STORE_CONST, ActionType.APPEND_CONST}:
        if nargs_spec.allows_any is True:
            raise ValueError(f'Action {action_spec.action} requires nargs to not allow any values.')

    # Either type or types can be provided, not both.
    if type is not None and types is not None:
        raise ValueError('Cannot specify both type and types.')
    
    # If types is provided, nargs_spec must allow multiple values and single be False.
    if types is not None:
        if nargs_spec.single: # TODO: check if this is necessary
            raise ValueError('If types is specified, nargs cannot be single value.')

    # Check wether the argument is a subparser, positional or option argument.
    match position, flags, const_flags, subparsers:
        case int() | None, None, None, None:
            # Positional argument, with or without a specified order.
            # If position is none, the definition order will be used.
            # Nargs must allow at least one value.
            return field(
                default = default,
                default_factory = default_factory,
                metadata = {'arg':PositionSpecification(
                    type = type,
                    types = types,
                    nargs = nargs_spec,
                    action = action_spec,
                    required = required,
                    help = help,
                    destination = None,
                    position = position,
                    validate = validate
                )}
            )
        case [None, list(), None, None] | [None, list(), dict(), None] | [None, None, dict(), None]:
            # Check that flags is a non-empty list of strings, starting with '-' or '--'.
            # If they start with a single '-', they are short flags and should only be one character.
            if not flags and not const_flags:
                raise ValueError('Flags list cannot be empty for option arguments.')
            for flag in chain(flags or [], (const_flags or {}).keys()):
                if not check_flag(flag):
                    raise ValueError(f'Invalid flag format: {flag}')
            
            return field(
                default = default,
                default_factory = default_factory,
                metadata = {'arg': OptionSpecification(
                    type = type,
                    types = types,
                    nargs = nargs_spec,
                    action = action_spec,
                    required = required,
                    help = help,
                    destination = None,
                    flags = flags,
                    const_flags = const_flags,
                    validate = validate
                )}
            )

        case None, None, None, dict():
            # Check that the dict is of type dict[str, type[CommandArguments]]. Only valid CommandArguments subclasses
            # can be used as subparsers, the key type must be str and it cannot be empty.

            if not subparsers:
                raise ValueError('Subparsers dictionary cannot be empty for subparser arguments.')
            
            for key, parser in subparsers.items():
                if not isinstance(key, str): # pyright: ignore[reportUnnecessaryIsInstance]
                    raise ValueError(f'Subparser key must be a string, got {key.__class__}') # must use __class__ because we redefined type
                if not issubclass(parser, CommandArguments): # pyright: ignore[reportUnnecessaryIsInstance]
                    raise ValueError(f'Subparser value must be a subclass of CommandArguments, got {parser}')

            return field(
                default = default,
                default_factory = default_factory,
                metadata = {'arg':SubparserSpecification(
                    type = type,
                    types = types,
                    nargs = nargs_spec,
                    action = action_spec,
                    required = required,
                    help = help,
                    destination = None,
                    subparsers = subparsers,
                    validate = validate
                )}
            )

        case _:
            raise ValueError('Invalid combination of position, flags, and subparsers. Only one of these can be specified.')
        


    return field(default=default, default_factory=default_factory)

_CA = TypeVar('_CA', bound='CommandArguments')

class CommandArgumentParser(Generic[_CA]):
    """Base class for command argument parsers. Parses command arguments from a Lark parse tree,
    and returns a CommandArguments instance of type _CA.

    The CommandArguments dataclass must use the metadata structure to define the arguments, using the
    ArgumentSpecification class as metadata.
    """

    def __init__(self, return_type: type[_CA], name: str) -> None:
        """Initializes the command argument parser."""
        # Check that return_type is a subclass of CommandArguments
        if not issubclass(return_type, CommandArguments): # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f'return_type must be a subclass of CommandArguments, got {return_type}')

        self.return_type = return_type
        self.name = name

        # Get the fields of the return_type dataclass to cache and use later.
        # Ignore non-init fields (e.g. _token_map)
        self._fields: dict[str, Field[Any]] = {
            f.name: f
            for f in fields(self.return_type)
            if f.init  # only consider fields that are part of __init__
        }
    
        # Traverse the fields and check that the metadata is correct.
        # Separate arguments into subparsers, positional and option arguments.

        # Maps subparser names to their corresponding CommandArgumentParser
        self._subparsers: dict[str, CommandArgumentParser[CommandArguments]] = {}
        self._subparser_names: dict[str, str] = {}  # Maps subparser field names to their subparser names
        
        # Maps positional argument names to their Field
        self._positional_args: dict[str, Field[Any]] = {}
        self._positional_order: list[str] = []

        # Maps option argument names to their Field
        self._option_args: dict[str, Field[Any]] = {}
        # Maps the option flag ('--flag', '-f') to the argument name,
        # referencing _option_args.
        self._option_flags: dict[str, str] = {}

        self._init_metadata()

    def _init_metadata(self) -> None:
        """Initializes the metadata for the command arguments.

        This method processes the fields of the return_type dataclass,
        and populates the _subparsers, _positional_args, _option_args,
        and _option_flags attributes.
        """
        
        for fname, field_info in self._fields.items():
            argspec = field_info.metadata.get('arg', None)
            if argspec is None:
                raise TypeError(f'Field {fname} does not have ArgumentSpecification metadata.')

            match argspec:
                case SubparserSpecification() as subparser_meta:
                    if subparser_meta.subparsers is None:
                        raise ValueError(f'Subparser field {fname} must have subparsers defined.')
                    for sub_name, sub_type in subparser_meta.subparsers.items():
                        self._subparsers[sub_name] = CommandArgumentParser(sub_type, sub_name)
                        self._subparser_names[sub_name] = fname

                case PositionSpecification():
                    # TODO: respect and validate position order
                    self._positional_args[fname] = field_info
                    self._positional_order.append(fname)

                case OptionSpecification() as option_meta:
                    self._option_args[fname] = field_info
                    if option_meta.flags:
                        for flag in option_meta.flags:
                            if flag in self._option_flags:
                                raise ValueError(f'Flag {flag} is already used by another option argument.')
                            self._option_flags[flag] = fname

                    if option_meta.const_flags:
                        for flag in option_meta.const_flags.keys():
                            if flag in self._option_flags:
                                raise ValueError(f'Flag {flag} is already used by another option argument.')
                            self._option_flags[flag] = fname
                case _:
                    raise TypeError(f'Field {fname} has invalid ArgumentSpecification metadata.')
                
    def parse(self, command_token: Token, subtree: list[Token | Tree[Token]]) -> _CA:
        """Parses the command arguments from the given Lark parse tree subtree."""
        # This method is the outer wrapper for the parsing logic, implemented in _parse.
        # To parse subparsers, we use _parse, since the external logic is only needed in the outer layer.

        tokens = peekable(subtree)
        r = self._parse(command_token, tokens)
        
        if isinstance(r, Validatable):
            r.validate()
        return r
    
    def _parse(self, command_token: Token, tokens: peekable[Token | Tree[Token]]) -> _CA:
        """Internal method to parse the command arguments from the given Lark parse tree tokens.

        This method implements the core parsing logic, handling positional arguments,
        option arguments, and subparsers. To parse, use the parse() method instead.
        """
        # Dictionary to hold the dataclass initialization arguments.
        kwargs: dict[str, Any] = {}

        # Update the kwargs with default values from the dataclass fields.
        for fname, field_info in self._fields.items():
            if field_info.default is not MISSING:
                kwargs[fname] = field_info.default
            elif field_info.default_factory is not MISSING:
                kwargs[fname] = field_info.default_factory()
        
        # Set of all field names.
        _all_fields = set(self._fields.keys())
        # Set of defined field names (positional + option + subparser) to check for duplicates.
        _defined_fields: set[str] = set()
        # Map of field names to their corresponding tokens/trees.
        _token_map: dict[str, TokenLike] = {}

        # ----------------------------------------------------------------------
        # Main parsing loop

        # Positional arguments can only appear before any optional arguments.
        # The beginning of a subparser or option argument indicates the end of positional arguments.
        _could_be_positional = True
        _positional_names_iter = iter(self._positional_order)
        # When all fields have been defined, we can stop parsing.
        while tokens and (_all_fields - _defined_fields):
            # Attempt parsing as an option argument first.
            # Then, attempt parsing as a subparser.
            # Then, attempt parsing as a positional argument.
            # The _parse_option, _parse_subparser, and _parse_positional
            # methods will return:
            #  - (None, None) on failure to parse
            #  - (False, None) if they modify the token stream and need to reattempt parsing
            #  - (True, _) on successful parsing
            match self._parse_option(tokens, kwargs, _defined_fields, _token_map):
                case True | False:
                    _could_be_positional = False
                    continue
                case None:
                    pass
            
            match self._parse_subparser(tokens, kwargs, _defined_fields, _token_map):
                case True | False:
                    _could_be_positional = False
                    continue
                case None:
                    pass
            
            if _could_be_positional:
                match self._parse_positional(tokens, kwargs, _defined_fields, _token_map, _positional_names_iter):
                    case True | False:
                        continue
                    case None:
                        _could_be_positional = False
                        pass
            
            # If we reach here, no parsing method succeeded.
            # This indicates an unexpected token.
            token = next(tokens, None)
            if token is None:
                break

            raise CommandParseError(f'Unexpected token while parsing command arguments: {token}', token)
        
        
        # ----------------------------------------------------------------------
        # Post-parsing validation
        # Check for required arguments that were not defined.
        _required_fields = {
            fname for fname, field_info in self._fields.items() if field_info.metadata['arg'].required
        }

        missing_fields = _required_fields - _defined_fields

        if missing_fields:
            missing_field_names = ', '.join(missing_fields)
            raise CommandParseError(f'Missing required arguments: {missing_field_names}', command_token)
        
        # Check validation functions for each argument.
        for fname in _defined_fields:
            field_info = self._fields[fname]
            argspec = field_info.metadata['arg']
            if not isinstance(argspec, ArgumentSpecification):
                raise TypeError(f'Field {fname} has invalid ArgumentSpecification metadata.')
            
            if argspec.validate is not None:
                try:
                    if isinstance(argspec.validate, list):
                        for v in argspec.validate:
                            if not v(kwargs[fname]):
                                raise CommandParseError(f'Argument {fname}: {kwargs[fname]} failed validation.', _token_map.get(fname, command_token))
                    else:
                        if not argspec.validate(kwargs[fname]): # pyright: ignore[reportCallIssue]
                            raise CommandParseError(f'Argument {fname}: {kwargs[fname]} failed validation.', _token_map.get(fname, command_token))
                except CommandParseError:
                    raise
                except Exception as e:
                    raise CommandParseError(f'Argument {fname}: {kwargs[fname]} validation raised an exception: {e}', _token_map.get(fname, command_token)) from e
                

        # ----------------------------------------------------------------------
        # Create the CommandArguments instance
        
        try:
            result = self.return_type(**kwargs)
            result._token_map = _token_map # pyright: ignore[reportPrivateUsage]
        except TypeError as e:
            raise CommandParseError(f'Error initializing command arguments: {e}', command_token) from e
        
        return result


    def _parse_option(self, tokens: peekable[Token | Tree[Token]], kwargs: dict[str, Any], defined_fields: set[str], token_map: dict[str, TokenLike]) -> bool | None:
        """Attempts to parse an option argument from the token stream.

        Returns a tuple of (success: bool | None, value: Any).
        - success is True if an option argument was successfully parsed.
        - success is False if the token stream was modified and parsing should be reattempted.
        - success is None if no option argument was found.
        - value is the parsed value of the option argument, or None if not applicable.
        
        An option argument is identified by its flag (e.g., '--option' or '-o').
        As a tree, they appear as:
        Tree(Token('RULE', 'option'), [Token('OPTION', '--flag'), values ...])
        """

        token = next(tokens, None)
        match token:
            # Exhausted token stream
            case None:
                return None
            case Tree(Token('RULE', 'option'), [Token('OPTION', str(flag)) as _option_token, *arg_tokens]):
                # Check if the flag is registered and valid
                if flag not in self._option_flags:
                    # Not a valid option flag. Check if it could be a merged short flag.
                    if flag.startswith('-') and not flag.startswith('--') and len(flag) > 2:
                        # It is a multioption short flag. Split and push back the individual flags.
                        for ch in reversed(flag[1:]):
                            tokens.pushback(Tree(Token('RULE', 'option'), [Token('OPTION', f'-{ch}')]))
                        return False  # Indicate to reattempt parsing
                    else:
                        # Not a valid option flag, and we don't know how to parse it.
                        tokens.pushback(token)
                        return None
                    
                # Valid option flag
                arg_name = self._option_flags[flag]
                field_info = self._option_args[arg_name]
                arg_spec = field_info.metadata['arg']

                # Should be an OptionSpecification. If this fails, there was an issue in the initialization.
                if not isinstance(arg_spec, OptionSpecification):
                    raise TypeError(f'Field {arg_name} is not an OptionSpecification. CommandArgumentParser was badly initialized.')
                
                # Check for duplicate definition
                if arg_name in defined_fields:
                    raise CommandParseError(f'Option argument {flag} is defined multiple times.', _option_token)

                # Check if the flag is a const flag
                if arg_spec.const_flags and flag in arg_spec.const_flags:
                    # Reject all argument tokens back to the stream
                    for t in reversed(arg_tokens):
                        tokens.pushback(t)
                    
                    const_value = arg_spec.const_flags[flag]

                    # Apply the action to store the const value
                    match arg_spec.action.action:
                        case ActionType.APPEND | ActionType.APPEND_CONST:
                            if arg_name not in kwargs:
                                kwargs[arg_name] = []
                            kwargs[arg_name].append(const_value)
                            token_map[arg_name] = [_option_token]
                        case _:
                            kwargs[arg_name] = const_value
                            token_map[arg_name] = [_option_token]
                    # Mark the field as defined
                else:
                    # Field is not a const flag, parse the argument value(s)
                    # Check if the flag supports the number of arguments provided. If not, put the tokens outside of nargs on the stream back.
                    while len(arg_tokens) not in arg_spec.nargs and arg_tokens:
                        tokens.pushback(arg_tokens.pop())
                    if len(arg_tokens) not in arg_spec.nargs:
                        raise CommandParseError(f'Option argument {flag} received an invalid number of values.', _option_token)


                    # Get arg_tokens as a list of Tokens, and raise if any are Trees. This forces the type to be Token.
                    for t in arg_tokens:
                        if isinstance(t, Tree):
                            raise CommandParseError(f'Option argument {flag} received invalid value type.', t)                    
                    arg_tokens = [t for t in arg_tokens if isinstance(t, Token)]


                    # Parse the argument value(s) based on the action and type / types.
                    # Only one of type or types can be provided. If types is provided, nargs is not single.
                    match arg_spec.type, arg_spec.types, arg_spec.nargs.single:
                        case None, None, True:
                            # No type, single value, return the value
                            value = arg_tokens[0].value
                        case None, None, False:
                            # No type, multiple values, return list of values
                            value = [t.value for t in arg_tokens]
                        case arg_type, None, True if callable(arg_type):
                            # Single value, single type, return a single value
                            try:
                                value = arg_type(arg_tokens[0].value)
                            except Exception as e:
                                raise CommandParseError(f'Option argument {flag} value conversion error: {e}', arg_tokens[0]) from e
                        case arg_type, None, False if callable(arg_type):
                            # Multiple values, single type, return a list of values coerced to the type
                            values: list[Any] = []
                            for t in arg_tokens:
                                try:
                                    values.append(arg_type(t.value))
                                except Exception as e:
                                    raise CommandParseError(f'Option argument {flag} value conversion error: {e}', t) from e
                            value = values
                        case None, arg_types, False if callable(arg_types):
                            # Multiple values, types callback, return the value of types(*values)
                            try:
                                value = arg_types(*(t.value for t in arg_tokens))
                            except Exception as e:
                                raise CommandParseError(f'Option argument {flag} value conversion error: {e}', arg_tokens) from e
                        case _:
                            # All other combinations are invalid
                            raise TypeError(f'Invalid combination of type/types and nargs for option argument {flag}.')
                    
                    # Apply the action to store the value in kwargs
                    match arg_spec.action.action:
                        case ActionType.STORE:
                            if arg_tokens:
                                kwargs[arg_name] = value
                                token_map[arg_name] = [_option_token, *arg_tokens]
                            elif arg_spec.action.const is not MISSING:
                                kwargs[arg_name] = arg_spec.action.const
                                token_map[arg_name] = [_option_token]
                            else:
                                raise CommandParseError(f'Option argument {flag} requires a value to store.', _option_token)

                        case ActionType.APPEND:
                            if kwargs.get(arg_name) is None:
                                kwargs[arg_name] = []
                            
                            if arg_tokens:
                                kwargs[arg_name].append(value)
                                token_map[arg_name] = [_option_token, *arg_tokens]
                            elif arg_spec.action.const is not MISSING:
                                kwargs[arg_name].append(arg_spec.action.const)
                                token_map[arg_name] = [_option_token]
                            else:
                                raise CommandParseError(f'Option argument {flag} requires a value to append.', _option_token)
                        
                        case ActionType.STORE_TRUE:
                            kwargs[arg_name] = True
                            token_map[arg_name] = [_option_token]

                        case ActionType.STORE_FALSE:
                            kwargs[arg_name] = False
                            token_map[arg_name] = [_option_token]
                        
                        case ActionType.STORE_CONST:
                            kwargs[arg_name] = arg_spec.action.const
                            token_map[arg_name] = [_option_token]
                        
                        case ActionType.APPEND_CONST:
                            if arg_name not in kwargs:
                                kwargs[arg_name] = []
                            kwargs[arg_name].append(arg_spec.action.const)
                            token_map[arg_name] = [_option_token]

                        case ActionType.COUNT:
                            if arg_name not in kwargs:
                                kwargs[arg_name] = 0
                            kwargs[arg_name] += 1
                            token_map[arg_name] = [_option_token]
                        
                # Mark the field as defined
                defined_fields.add(arg_name)
                return True

            case _:
                tokens.pushback(token)
                return None

    def _parse_subparser(self, tokens: peekable[Token | Tree[Token]], kwargs: dict[str, Any], defined_fields: set[str], token_map: dict[str, TokenLike]) -> bool | None:
        """Attempts to parse a subparser argument from the token stream.

        Returns True if a subparser argument was successfully parsed,
        False if the token stream was modified and parsing should be reattempted,
        or None if no subparser argument was found.

        A subparser argument is identified by its name, akin to a positional argument.
        It is functionally indistinguishable from a positional, except that it is present in the
        subparsers mapping.
        """

        token = next(tokens, None)
        match token:
            # Exhausted token stream
            case None:
                return None
            case Token(_, str(sub_name)) as sub_token:
                if sub_name not in self._subparsers:
                    tokens.pushback(token)
                    return None
                
                # Valid subparser name
                subparser = self._subparsers[sub_name]
                field_name = self._subparser_names[sub_name]
                
                # Check for duplicate definition
                if sub_name in defined_fields:
                    raise CommandParseError(f'Subparser argument {sub_name} is defined multiple times.', sub_token)
                
                # Parse the subparser arguments using its own parser
                subparser_args = subparser._parse(sub_token, tokens)

                # Store the subparser arguments in kwargs
                kwargs[field_name] = subparser_args
                token_map[field_name] = [sub_token, *subparser_args._token_map.values()] # pyright: ignore[reportPrivateUsage, reportArgumentType]

                # Mark the field as defined
                defined_fields.add(field_name)
                return True
            case _:
                tokens.pushback(token)
                return None
            
    def _parse_positional(self, tokens: peekable[Token | Tree[Token]], kwargs: dict[str, Any], defined_fields: set[str], token_map: dict[str, TokenLike], positional_names_iter: Iterator[str]) -> bool | None:
        """Attempts to parse a positional argument from the token stream.

        Returns True if a positional argument was successfully parsed,
        False if the token stream was modified and parsing should be reattempted,
        or None if no positional argument was found.

        A positional argument is identified by its position in the token stream.
        """

        try:
            arg_name = next(positional_names_iter)
        except StopIteration:
            return None  # No more positional arguments to parse
        
        field_info = self._positional_args[arg_name]
        arg_spec = field_info.metadata['arg']

        # Should be a PositionSpecification. If this fails, there was an issue in the initialization.
        if not isinstance(arg_spec, PositionSpecification):
            raise TypeError(f'Field {arg_name} is not a PositionSpecification. CommandArgumentParser was badly initialized.')
        
        # Check for duplicate definition
        if arg_name in defined_fields:
            raise CommandParseError(f'Positional argument {arg_name} is defined multiple times.', None)
        
        # Parse the argument value(s) based on the nargs specification.
        # Collect tokens until we reach the required number of arguments.
        collected_tokens: list[Token] = []
        for _ in range(arg_spec.nargs.maximum) if arg_spec.nargs.maximum is not None else count():
            token = next(tokens, None)
            if token is None: # Exhausted token stream
                break
            match token:
                case Token() as t:
                    # Valid token, collect it
                    collected_tokens.append(t)
                case Tree():
                    # Encountered a tree, push it back and stop collecting tokens
                    tokens.pushback(token)
                    break
        # We now collected all tokens that could be part of this positional argument.
        # Now, we remove excess tokens.
        # We need to check nargs and types (with soft nargs) to determine how many tokens to keep.

        while True:
            if len(collected_tokens) not in arg_spec.nargs:
                tokens.pushback(collected_tokens.pop())
                if not collected_tokens:
                    # This argument failed to parse. We have pushed back all tokens, so we return None.
                    return None
        
            # Nargs is satisfied. We now check if types is satisfied (if applicable).
            match arg_spec.type, arg_spec.types, arg_spec.nargs.single:
                case None, None, True:
                    # No type or types, nargs is satisfied.
                    if len(collected_tokens) != 1:
                        raise CommandParseError(f'Positional argument {arg_name} expects a single value. There may have been an instancing error.', None)
                    value = collected_tokens[0].value
                    token_value = collected_tokens[0]
                    break
                case None, None, False:
                    # No type or types, nargs is satisfied.
                    value = [t.value for t in collected_tokens]
                    token_value = collected_tokens
                    break
                case arg_type, None, True if callable(arg_type):
                    # Single type, single value. If type does not work, we need to push back all tokens and return None.
                    if len(collected_tokens) != 1:
                        raise CommandParseError(f'Positional argument {arg_name} expects a single value. There may have been an instancing error.', None)
                    try:
                        value = arg_type(collected_tokens[0].value)
                        token_value = collected_tokens[0]
                        break
                    except Exception:
                        # Type conversion failed. Push back all tokens and return None.
                        for t in reversed(collected_tokens):
                            tokens.pushback(t)
                        return None
                case arg_type, None, False if callable(arg_type):
                    # Single type, multiple values. If any type conversion fails:
                    # if soft, push back one token and reattempt parsing
                    # if not soft, push back all tokens and return None
                    try:
                        value = [arg_type(t.value) for t in collected_tokens]
                        token_value = collected_tokens
                        break
                    except Exception:
                        if arg_spec.nargs.soft:
                            tokens.pushback(collected_tokens.pop())
                            if not collected_tokens:
                                return None
                        else:
                            for t in reversed(collected_tokens):
                                tokens.pushback(t)
                            return None
                case None, arg_types, False if callable(arg_types):
                    # Types callback, multiple values. If types conversion fails:
                    # if soft, push back one token and reattempt parsing
                    # if not soft, push back all tokens and return None
                    try:
                        value = arg_types(*(t.value for t in collected_tokens))
                        token_value = collected_tokens
                        break
                    except Exception:
                        if arg_spec.nargs.soft:
                            tokens.pushback(collected_tokens.pop())
                            if not collected_tokens:
                                return None
                        else:
                            for t in reversed(collected_tokens):
                                tokens.pushback(t)
                            return None
                case _:
                    # All other combinations are invalid
                    raise TypeError(f'Invalid combination of type/types and nargs for positional argument {arg_name}.')
        
        # Apply the action to store the value in kwargs
        match arg_spec.action.action:
            case ActionType.STORE:
                kwargs[arg_name] = value
                token_map[arg_name] = token_value
            case ActionType.APPEND:
                if kwargs.get(arg_name) is None:
                    kwargs[arg_name] = []
                kwargs[arg_name].append(value)
                token_map[arg_name] = token_value
            case _:
                raise TypeError(f'Unsupported action {arg_spec.action.action} for positional argument {arg_name}.')
        
        defined_fields.add(arg_name)
        return True
