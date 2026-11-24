"""CUDA expression generator."""

import logging
import warnings

import numpy.typing as npt
from ffcx.codegeneration.backend import FFCXBackend
from ffcx.codegeneration.C.formatter import Formatter
from ffcx.codegeneration.common import template_keys
from ffcx.codegeneration.expression_generator import ExpressionGenerator
from ffcx.codegeneration.utils import dtype_to_c_type, dtype_to_scalar_dtype
from ffcx.ir.representation import ExpressionIR

from ffcx_backends.cuda.expression_template import interpolation_kernel

logger = logging.getLogger("ffcx-backends")


def generator(ir: ExpressionIR, options: dict[str, int | float | npt.DTypeLike]) -> tuple[str, str]:
    """Generate CUDA code for an expression.

    Only the kernels of interpolations are supported, as device functions that
    the kernels holding the interpolation call.

    Args:
        ir: IR of the expression
        options: dict of kernel generation options

    Returns:
        Tuple of kernel (`.cu`) and metadata (`.c`) strings.
    """
    if not ir.is_interpolation_kernel:
        warnings.warn("Not implemented.", stacklevel=2)
        return ("", "")

    factory_name = ir.expression.name
    logger.info("Generating CUDA code for interpolation kernel:")
    logger.info(f"--- name: {factory_name}")

    backend = FFCXBackend(ir, options)
    parts = ExpressionGenerator(ir, backend).generate()
    formatter = Formatter(options["scalar_type"])  # type: ignore

    d: dict[str, str] = {}
    d["factory_name"] = factory_name
    d["tabulate_expression"] = formatter(parts)
    d["scalar_type"] = dtype_to_c_type(options["scalar_type"])  # type: ignore
    d["geom_type"] = dtype_to_c_type(dtype_to_scalar_dtype(options["scalar_type"]))  # type: ignore

    assert set(d.keys()) == template_keys(interpolation_kernel)
    return (interpolation_kernel.format_map(d), "")
