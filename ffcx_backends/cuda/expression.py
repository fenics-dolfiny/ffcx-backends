"""CUDA expression generator."""

import logging

import numpy.typing as npt
from ffcx.ir.representation import ExpressionIR

logger = logging.getLogger("ffcx-backends")


def generator(ir: ExpressionIR, options: dict[str, int | float | npt.DTypeLike]) -> tuple[str, str]:
    """Generate UFCx code for an expression."""
    raise NotImplementedError
    # return ("", "")
