from .edge import Align, Edge
from .filein import FileIn, BatchIn, FileTest
from .fourier import Fourier
from .preedge import PreEdge
from .postedge import PostEdge
from .phase import Phase
from .plot import Plot
from .background import Background
from .othercmds import Cut, Smooth, Rebin, Normalize
from .glitch import Deglitch, MultiEdge
from .save import Save
from .pca import PCA
from .average import Average

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
    "plot": Plot(),
    "background": Background(),
    "cut": Cut(),
    "smooth": Smooth(),
    "rebin": Rebin(),
    "deglitch": Deglitch(),
    "multiedge": MultiEdge(),
    "save": Save(),
    "test": FileTest(),
    "pca": PCA(),
    "average": Average(),
    "normalize": Normalize(),
}
