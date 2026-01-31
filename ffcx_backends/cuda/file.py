"""Generate file output for CUDA."""

import logging
import pprint
import textwrap

import numpy.typing as npt
from ffcx import __version__ as ffcx_version
from ffcx.codegeneration import __version__ as ufc_version
from ffcx.codegeneration.common import template_keys

from ffcx_backends.cuda.file_template import kernel_factory, metadata_factory

logger = logging.getLogger("ffcx-backends")

# Treat the CUDA file as a declaration so FFCx doesn't try to
# compile it :)
suffixes = (".cu", "_cuda.c")


def generator(
    options: dict[str, int | float | npt.DTypeLike],
) -> tuple[tuple[str], tuple[str]]:
    """Generate UFCx code for file output.

    Args:
        options: Dict of options specified the kernel generation, these will be documented in the
        generated file.

    Returns: tuple of file start- and end sections, each for declaration and implementation.

    """
    logger.info("Generating code for CUDA file")
    print("Calling CUDA generator.")
    # Attributes
    d = {"ffcx_version": ffcx_version, "ufcx_version": ufc_version}
    d["options"] = textwrap.indent(pprint.pformat(options), "// ")
    assert set(d.keys()) == template_keys(kernel_factory)
    assert set(d.keys()) == template_keys(metadata_factory)
    return (kernel_factory.format_map(d), metadata_factory.format_map(d)), ("", "")
