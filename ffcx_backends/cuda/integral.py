"""Generate UFCx code for an integral."""

import logging

import basix
import numpy.typing as npt
from ffcx.codegeneration.backend import FFCXBackend
from ffcx.codegeneration.C.formatter import Formatter
from ffcx.codegeneration.common import template_keys
from ffcx.codegeneration.integral_generator import IntegralGenerator
from ffcx.codegeneration.utils import dtype_to_c_type, dtype_to_scalar_dtype
from ffcx.ir.representation import IntegralIR

from ffcx_backends.cuda.integral_template import factory, metadata

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
    kernel_d: dict[str,str] = {}

    kernel_d["factory_name"] = factory_name

    kernel_d["tabulate_tensor"] = body
    kernel_d["scalar_type"] = dtype_to_c_type(options["scalar_type"]) # type: ignore
    kernel_d["geom_type"] = dtype_to_c_type(dtype_to_scalar_dtype(options["scalar_type"])) # type: ignore

    metadata_d: dict[str, str] = {}
    metadata_d["factory_name"] = factory_name
    if len(ir.enabled_coefficients) > 0:
        values = ", ".join("1" if i else "0" for i in ir.enabled_coefficients)
        sizes = len(ir.enabled_coefficients)
        metadata_d["enabled_coefficients_init"] = (
            f"bool enabled_coefficients_{ir.expression.name}_{domain.name}[{sizes}] = {{{values}}};"
        )
        metadata_d["enabled_coefficients"] = (
            f"enabled_coefficients_{ir.expression.name}_{domain.name}"
        )
    else:
        metadata_d["enabled_coefficients_init"] = ""
        metadata_d["enabled_coefficients"] = "NULL"

    assert ir.expression.coordinate_element_hash is not None
    metadata_d["coordinate_element_hash"] = f"UINT64_C({ir.expression.coordinate_element_hash})"
    metadata_d["needs_facet_permutations"] = (
            "true" if ir.expression.needs_facet_permutations else "false"
    )
    metadata_d["domain"] = int(domain)

    assert set(kernel_d.keys()) == template_keys(factory)
    assert set(metadata_d.keys()) == template_keys(metadata)
    return (metadata.format_map(metadata_d), factory.format_map(kernel_d))
