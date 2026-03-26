"""CUDA expression generator."""

import logging
import warnings

import numpy.typing as npt
from ffcx.ir.representation import ExpressionIR

logger = logging.getLogger("ffcx-backends")


def generator(ir: ExpressionIR, options: dict[str, int | float | npt.DTypeLike]) -> tuple[str, str]:
    """Generate UFCx code for an expression."""
    warnings.warn("Not implemented.", stacklevel=2)
    return ("", "")
