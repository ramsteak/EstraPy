from .align import Align, Edge
from .filein import FileIn, BatchIn
from .fourier import Fourier
from .preedge import PreEdge
from .postedge import PostEdge
from .phase import Phase

from ._handler import CommandHandler

commands: dict[str, CommandHandler] = {
    "filein": FileIn(),
    "batchin": BatchIn(),
    "align": Align(),
    "edge": Edge(),
    "preedge": PreEdge(),
    "postedge": PostEdge(),
    "fourier": Fourier(),
    "phase": Phase(),
}
