from .filein import FileIn, BatchIn
from .align import Align, Edge
from .preedge import PreEdge
from .postedge import PostEdge
from ._handler import CommandHandler

commands: dict[str, CommandHandler] = {
    "filein": FileIn(),
    "batchin": BatchIn(),
    "align": Align(),
    "edge": Edge(),
    "preedge": PreEdge(),
    "postedge": PostEdge(),
}
