"""Generate UFCx code for an expression."""

import logging

import numpy.typing as npt

from ffcx.codegeneration.backend import FFCXBackend
from ffcx.codegeneration.common import template_keys, tensor_sizes
from ffcx.codegeneration.expression_generator import ExpressionGenerator
from ffcx.codegeneration.numba import expression_template
from ffcx.codegeneration.numba.formatter import Formatter
from ffcx.ir.representation import ExpressionIR

logger = logging.getLogger("ffcx-backends")


def generator(
    ir: ExpressionIR, options: dict[str, int | float | npt.DTypeLike]
) -> tuple[str]:
    """Generate UFCx code for an expression."""

    return ("",)
