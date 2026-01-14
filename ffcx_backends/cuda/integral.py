"""Generate UFCx code for an integral."""

import logging
import sys

import basix
import numpy as np
import numpy.typing as npt

from ffcx.codegeneration.backend import FFCXBackend
from ffcx_backends.cuda import integral_template
from ffcx.codegeneration.C.formatter import Formatter
from ffcx.codegeneration.common import template_keys
from ffcx.codegeneration.integral_generator import IntegralGenerator
from ffcx.codegeneration.utils import dtype_to_c_type, dtype_to_scalar_dtype
from ffcx.ir.representation import IntegralIR

logger = logging.getLogger("ffcx-backends")

def generator(
    ir: IntegralIR, domain: basix.CellType, options: dict[str, int | float | npt.DTypeLike]
) -> tuple[str, str]:
    """Generate CUDA code for an integral.

    Args:
        ir: IR of the integral
        domain: basix cell type
        options: dict of kernel generation options

    Returns:
        Tuple of declaration (header) and implementation (source) strings.

    """

    logger.info("Generating CUDA code for integral:")
    logger.info(f"--- type: {ir.expression.integral_type}")
    logger.info(f"--- name: {ir.expression.name}")

    factory_name = f"{ir.expression.name}_{domain.name}"

    # Create FFCx backend
    backend = FFCXBackend(ir, options)

    # Configure kernel generator
    ig = IntegralGenerator(ir, backend)

    # Generate code ast for the tabulate_tensor body
    parts = ig.generate(domain)

    # Format code as string
    format = Formatter(options["scalar_type"])  # type: ignore
    body = format(parts)

    # Generate generic FFCx code snippets and add specific parts
    d: dict[str,str] = {}

    d["factory_name"] = factory_name

    # TODO properly handle enabled coefficients/facet permutations
    # also add tracking information about the domain
    d["tabulate_tensor"] = body
    d["scalar_type"] = dtype_to_c_type(options["scalar_type"]) # type: ignore
    d["geom_type"] = dtype_to_c_type(dtype_to_scalar_dtype(options["scalar_type"])) # type: ignore

    assert ir.expression.coordinate_element_hash is not None

    assert set(d.keys()) == template_keys(integral_template.factory)
    return (integral_template.factory.format_map(d),)
