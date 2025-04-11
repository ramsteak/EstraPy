from __future__ import annotations

import numpy as np
import numpy.typing as npt

from enum import Enum
from logging import getLogger
from typing import NamedTuple
from matplotlib import pyplot as plt

from ._context import Context
from ._handler import CommandHandler, Token, CommandResult
from ._misc import parse_numberunit_range, parse_numberunit, NumberUnit
from ._parser import CommandParser

def extract_phase(x:npt.NDArray[np.floating], y:npt.NDArray[np.complexfloating]) -> tuple[npt.NDArray[np.complexfloating], np.poly1d]:
    """Extracts the linear behavior of the phase.
    Returns:
      - the corrected complex signal
      - the removed linear component as a np.poly1d
    """
    # Numpy unwrap the phase and linear fit to remove most of the windings
    p0 = np.unwrap(np.angle(y))
    p_p0 = np.poly1d(np.polyfit(x,p0, 1))
    linear_phase = p_p0(x)
    y1 = y * (np.cos(linear_phase) - 1.0j*np.sin(linear_phase))

    # Phase and phase derivative evaluation for better unwrapping

    # p1 = np.unwrap(np.angle(y1))
    # dp1 = np.diff(p1, prepend=[0])
    # _s = (np.cumsum(dp1 > np.pi/2) - np.cumsum(dp1 < - np.pi/2)) * np.pi
    # s = np.cos(_s) + 1.0j*np.sin(_s)
    # y2 = y1 * s.conj()

    # Rotate the phase so that the center of mass is real
    c = (_t:=y1.mean()) / np.abs(_t)
    
    y3 = y1 * c.conj()

    return y3, p_p0 - np.angle(c)

class PhaseAction(Enum):
    CORRECT = "corr"
    ALIGN = "align"

class Args_Phase(NamedTuple):
    action:PhaseAction
    

class Phase(CommandHandler):
    @staticmethod
    def parse(tokens: list[Token], context: Context) -> Args_Phase:
        parser = CommandParser(
            "phase", description="Alters the phase of the fourier transform."
        )
        subparsers = parser.add_subparsers(dest='action')

        correct = subparsers.add_parser('correct')
        align = subparsers.add_parser('align')

        args = parser.parse(tokens)

        match args.action:
            case "correct":
                action = PhaseAction.CORRECT
            case _:
                raise RuntimeError("Invalid phase transformation.")

        return Args_Phase(action)

    @staticmethod
    def execute(args: Args_Phase, context: Context) -> CommandResult:
        log = getLogger("phase")

        match args.action:
            case PhaseAction.CORRECT:
                for data in context.data:            
                    if "f" not in data.fd:
                        raise RuntimeError("Fourier transform was not calculated. Cannot correct phase.")

                    s,p = extract_phase(data.fd.R.to_numpy(),data.fd.f.to_numpy())
                    data.fd.f = s
                    data.meta.run["phase"] = p

            case PhaseAction.ALIGN:
                raise NotImplementedError()
            
        return CommandResult(True)

    @staticmethod
    def undo(args: NamedTuple, context: Context) -> CommandResult:
        # TODO:
        raise NotImplementedError
