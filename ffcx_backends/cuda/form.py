"""Template for form output."""

import logging 

import numpy.typing as npt

from ffcx.codegeneration.common import integral_data, template_keys
from ffcx.ir.representation import FormIR

logger = logging.getLogger("ffcx-backends")

def generator(ir: FormIR, options: dict[str, int | float | npt.DTypeLike]) -> tuple[str]:
    """Generate UFCx code for a form."""

    return ("",) 
