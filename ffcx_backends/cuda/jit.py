"""CUDA-specific JIT compilation routines."""

from __future__ import annotations

import io
import json
import logging
import sys
import time
from contextlib import redirect_stdout

import cffi
import numpy as np
from ffcx.codegeneration.C.file_template import libraries as _libraries

logger = logging.getLogger("ffcx")
root_logger = logging.getLogger()


def compile_objects(
    decl,
    ufl_objects,
    object_names,
    module_name,
    options,
    cache_dir,
    cffi_extra_compile_args,
    cffi_verbose,
    cffi_debug,
    cffi_libraries,
    visualise: bool = False,
):
    """Specialized jit compilation routine for CUDA backend."""
    import ffcx.compiler

    libraries = _libraries + cffi_libraries if cffi_libraries is not None else _libraries

    # JIT uses module_name as prefix, which is needed to make names of all struct/function
    # unique across modules
    code, _ = ffcx.compiler.compile_ufl_objects(
        ufl_objects, namespace=module_name, options=options, visualise=visualise
    )
    cuda_source, code_body = code
    code_body += f"\nstatic char * cuda_source = {json.dumps(cuda_source)};\n"

    # Raise error immediately prior to compilation if no support for C99
    # _Complex. Doing this here allows FFCx to be used for complex codegen on
    # Windows.
    if sys.platform.startswith("win32"):
        if np.issubdtype(options["scalar_type"], np.complexfloating):
            raise NotImplementedError("win32 platform does not support C99 _Complex numbers")
        elif isinstance(options["scalar_type"], str) and "complex" in options["scalar_type"]:
            raise NotImplementedError("win32 platform does not support C99 _Complex numbers")

    # Compile in C17 mode
    if sys.platform.startswith("win32"):
        cffi_base_compile_args = ["-std:c17"]
    else:
        cffi_base_compile_args = ["-std=c17"]

    cffi_final_compile_args = cffi_base_compile_args + cffi_extra_compile_args

    ffibuilder = cffi.FFI()

    ffibuilder.set_source(
        module_name,
        code_body,
        include_dirs=[ffcx.codegeneration.get_include_path()],
        extra_compile_args=cffi_final_compile_args,
        libraries=libraries,
    )

    ffibuilder.cdef(decl)

    c_filename = cache_dir.joinpath(module_name + ".c")
    ready_name = c_filename.with_suffix(".c.cached")

    # Compile (ensuring that compile dir exists)
    cache_dir.mkdir(exist_ok=True, parents=True)

    logger.info(79 * "#")
    logger.info("Calling JIT C compiler")
    logger.info(79 * "#")

    t0 = time.time()
    f = io.StringIO()
    # Temporarily set root logger handlers to string buffer only
    # since CFFI logs into root logger
    old_handlers = root_logger.handlers.copy()
    root_logger.handlers = [logging.StreamHandler(f)]
    with redirect_stdout(f):
        ffibuilder.compile(tmpdir=cache_dir, verbose=True, debug=cffi_debug)
    s = f.getvalue()
    if cffi_verbose:
        print(s)

    logger.info(f"JIT C compiler finished in {time.time() - t0:.4f}")

    # Create a "status ready" file. If this fails, it is an error,
    # because it should not exist yet.
    # Copy the stdout verbose output of the build into the ready file
    fd = open(ready_name, "x")
    fd.write(s)
    fd.close()

    # Copy back the original handlers (in case someone is logging into
    # root logger and has custom handlers)
    root_logger.handlers = old_handlers

    return code_body
