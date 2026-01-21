"""Generate UFCx code for an expression."""

import logging

import numpy.typing as npt
from ffcx.ir.representation import ExpressionIR

logger = logging.getLogger("ffcx-backends")


def generator(ir: ExpressionIR, options: dict[str, int | float | npt.DTypeLike]) -> tuple[str]:
    """Generate UFCx code for an expression."""
    return ("","")
