"""Generate file output for CUDA."""

import logging
import pprint
import textwrap

import numpy.typing as npt

from ffcx import __version__ as FFCX_VERSION
from ffcx.codegeneration import __version__ as UFC_VERSION
from ffcx.codegeneration.common import template_keys
from ffcx_backends.cuda import file_template

logger = logging.getLogger("ffcx-backends")

suffixes = (".cu",)


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

    # Attributes
    d = {"ffcx_version": FFCX_VERSION, "ufcx_version": UFC_VERSION}
    d["options"] = textwrap.indent(pprint.pformat(options), "// ")
    assert set(d.keys()) == template_keys(file_template.factory)
    return (file_template.factory.format_map(d),), ("",)
