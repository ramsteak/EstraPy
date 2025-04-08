from .filein import FileIn, BatchIn
from ._handler import CommandHandler

commands: dict[str, CommandHandler] = {
    "filein": FileIn(),
    "batchin": BatchIn(),
}
