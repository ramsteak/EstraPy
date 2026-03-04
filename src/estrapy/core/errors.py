from typing import TypeAlias, Union, Sequence, Self
from lark import Token, Tree
from lark.exceptions import VisitError


from types import TracebackType

TokenLike: TypeAlias = Union[Token, Tree[Token], Sequence[Union[Token, Tree[Token]]]]

class CommandError(Exception):
    """Base exception for command-related errors."""

    def __init__(self, message: str, token: TokenLike | None = None):
        super().__init__(message)
        self.token = token


class CommandParseError(CommandError):
    """Exception raised for errors during parsing."""


class CommandSyntaxError(CommandParseError):
    """Exception raised for syntax errors in commands."""


class ExecutionError(CommandError):
    """Exception raised for errors during command execution."""

    pass


class ArgumentError(Exception):
    """Exception raised for errors related to command arguments."""

    pass


class DuplicateArgumentError(ArgumentError):
    """Exception raised for duplicate command arguments."""

    pass

# Context manager that handles pretty-printing syntax errors with token information
class EstraCommandErrorContextManager:
    def __init__(self, input_file: str):
        self.input_file = input_file

    def __enter__(self) -> Self:
        return self
    
    def __exit__(self,
                 exc_type: type[BaseException] | None,
                 exc_value: BaseException | None,
                 traceback: TracebackType | None) -> None:

        if exc_type is CommandError and exc_value is not None and isinstance(exc_value, CommandError):
            exception = exc_value
            tokenstruct = exception.token
        elif exc_type is VisitError and exc_value is not None and isinstance(exc_value, VisitError):
            if isinstance(exc_value.orig_exc, CommandError):
                exception = exc_value.orig_exc
                tokenstruct = exception.token
            else:
                return  # Not a CommandError, do nothing and propagate
        else:
            return  # Not a CommandError, do nothing and propagate
        
        if tokenstruct is None:
            return
        # exception is a CommandError
        # tokenstruct is TokenLike

        lines = self.input_file.splitlines(keepends=True)

        # Normalize tokenstruct to a list of tokens
        tokens = self._extract_tokens(tokenstruct)
        
        if not tokens:
            return
        
        # Get the range of lines to display
        min_line = min(t.line for t in tokens if t.line is not None)
        max_line = int(max(t.end_line if hasattr(t, 'end_line') and t.end_line else t.line for t in tokens))
        
        # Build the error message with syntax highlighting
        error_lines:list[str] = []
        error_lines.append(f"Error on line {min_line}" + (f"-{max_line}" if max_line != min_line else ""))
        error_lines.append("")
        
        # Show context (a few lines before and after)
        context_before = 5
        context_after = 1
        start_line = max(1, min_line - context_before)
        end_line = min(len(lines), max_line + context_after)
        
        for line_num in range(start_line, end_line + 1):
            if line_num > len(lines):
                break
                
            line_text = lines[line_num - 1].rstrip('\n\r')
            
            # Check if this line contains any error tokens
            line_tokens = [t for t in tokens if t.line == line_num]
            
            if line_tokens:
                # Build highlighted line
                highlighted = self._highlight_line(line_text, line_tokens, line_num)
                error_lines.append(f"  {line_num:4d} | {highlighted}")
                
                # Add pointer line showing exactly where the error is
                pointer_line = self._create_pointer_line(line_tokens, line_num)
                if pointer_line:
                    error_lines.append(f"       | {pointer_line}")
            else:
                # Context line without highlighting
                error_lines.append(f"  {line_num:4d} | {line_text}")
        
        # Add the formatted error as a note to the exception
        exception.add_note("\n".join(error_lines))

        raise exception
    
    def _extract_tokens(self, tokenstruct: TokenLike) -> list[Token]:
        """Extract all tokens from a TokenLike structure."""
        tokens:list[Token] = []
        
        if isinstance(tokenstruct, Token):
            tokens.append(tokenstruct)
        elif isinstance(tokenstruct, Tree):
            # Extract all tokens from the tree
            for child in tokenstruct.iter_subtrees_topdown():
                for item in child.children:
                    if isinstance(item, Token):
                        tokens.append(item)
        elif isinstance(tokenstruct, Sequence): # pyright: ignore[reportUnnecessaryIsInstance]
            for item in tokenstruct:
                if isinstance(item, Token):
                    tokens.append(item)
                elif isinstance(item, Tree): # pyright: ignore[reportUnnecessaryIsInstance]
                    tokens.extend(self._extract_tokens(item))
        
        return tokens
    
    def _highlight_line(self, line: str, tokens: list[Token], line_num: int) -> str:
        """Highlight error tokens in a line using ANSI colors."""
        # ANSI color codes
        RED = '\033[91m'
        BOLD = '\033[1m'
        RESET = '\033[0m'
        
        # Create list of (start_col, end_col) tuples for highlighting
        highlights: list[tuple[int, int]] = []
        for token in tokens:
            if token.line == line_num:
                start_col = token.column - 1  # Lark columns are 1-indexed
                end_col = token.end_column - 1 if hasattr(token, 'end_column') else start_col + len(token)
                highlights.append((start_col, end_col))
        
        # Sort and merge overlapping highlights
        highlights.sort()
        merged: list[tuple[int, int]] = []
        for start, end in highlights:
            if merged and start <= merged[-1][1]:
                # Overlapping or adjacent, merge
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        
        # Build highlighted string
        if not merged:
            return line
        
        result: list[str] = []
        last_pos = 0
        
        for start, end in merged:
            # Add text before highlight
            result.append(line[last_pos:start])
            # Add highlighted text
            result.append(f"{RED}{BOLD}{line[start:end]}{RESET}")
            last_pos = end
        
        # Add remaining text
        result.append(line[last_pos:])
        
        return ''.join(result)
    
    def _create_pointer_line(self, tokens: list[Token], line_num: int) -> str:
        """Create a pointer line (with ^ characters) showing error location."""
        if not tokens:
            return ""
        
        # Find the extent of the error on this line
        min_col = min(t.column - 1 for t in tokens if t.line == line_num)
        max_col = max(
            (t.end_column - 1 if hasattr(t, 'end_column') else t.column - 1 + len(t))
            for t in tokens if t.line == line_num
        )
        
        # ANSI color codes
        RED = '\033[91m'
        BOLD = '\033[1m'
        RESET = '\033[0m'
        
        # Build pointer string
        pointer = ' ' * min_col + '^' * (max_col - min_col)
        return f"{RED}{BOLD}{pointer}{RESET}"