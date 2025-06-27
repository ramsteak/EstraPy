from __future__ import annotations

import numpy as np
import pandas as pd

from typing import NamedTuple
from logging import getLogger

from ._context import Context, Column, DataColType, AxisType, Domain, range_to_index
from ._format import pol
from ._handler import CommandHandler, Token, CommandResult
from ._numberunit import parse_range, NumberUnitRange
from ._numberunit import actualize_range

from ._parser import CommandParser


class Args_PCA(NamedTuple):
    ncomponents: int

class PCA(CommandHandler):
    @staticmethod
    def parse(tokens: list[Token], context: Context) -> Args_PCA:
        parser = CommandParser(
            "pca", description="Performs PCA decomposition on the data."
        )
        parser.add_argument("--numcomponents", "-n", type=int, default=10)

        args = parser.parse(tokens)

        return Args_PCA(args.numcomponents)

    @staticmethod
    def execute(args: Args_PCA, context: Context) -> CommandResult:
        log = getLogger("pca")
        from matplotlib import pyplot as plt

        _X = pd.DataFrame({i:data.get_xy("E", "mu") for i,data in enumerate(context.data)}).T
        X = _X.loc[:,_X.isna().sum() == 0]
        m = X.mean()
        Xm = X - m

        ev,eV = np.linalg.eigh(Xm.cov())
        ev = pd.Series(ev[::-1], index=[f"PC{i+1}" for i in range(Xm.shape[1])])
        eV = pd.DataFrame(eV[:,::-1], columns=ev.index, index = X.columns)

        pass
        return CommandResult(True)

    @staticmethod
    def undo(args: NamedTuple, context: Context) -> CommandResult:
        # TODO:
        raise NotImplementedError
