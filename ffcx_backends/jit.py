"""Just-in-time compilation."""

from __future__ import annotations

import importlib
import logging
import os
import tempfile
from pathlib import Path

import ffcx
import ffcx.naming
import numpy as np
import numpy.typing as npt
import ufl
from ffcx.codegeneration.jit import (
    UFC_EXPRESSION_DECL,
    UFC_FORM_DECL,
    UFC_HEADER_DECL,
    UFC_INTEGRAL_DECL,
    _compilation_signature,
    _compile_objects,
    _compute_option_signature,
    _load_objects,
    get_cached_module,
)

logger = logging.getLogger("ffcx")
root_logger = logging.getLogger()


def compile_forms(
    forms: list[ufl.Form],
    options: dict = {},
    cache_dir: Path | None = None,
    timeout: int = 10,
    cffi_extra_compile_args: list[str] = [],
    cffi_verbose: bool = False,
    cffi_debug: bool = False,
    cffi_libraries: list[str] = [],
    visualise: bool = False,
):
    """Compile a list of UFL forms into UFCx Python objects.

    Args:
        forms: List of ufl.form to compile.
        options: Options
        cache_dir: Cache directory
        timeout: Timeout
        cffi_extra_compile_args: Extra compilation args for CFFI
        cffi_verbose: Use verbose compile
        cffi_debug: Use compiler debug mode
        cffi_libraries: libraries to use with compiler
        visualise: Toggle visualisation
    """
    p = ffcx.options.get_options(options)

    # If requested, replace bi-linear forms by their diagonal part
    if p["part"] == "diagonal":
        for i, form in enumerate(forms):
            arguments = form.arguments()
            numbers = tuple(sorted(set(a.number() for a in arguments)))
            arity = len(numbers)
            if arity == 2:
                blocked_form = ufl.extract_blocks(form, replace_argument=False)
                if isinstance(blocked_form, ufl.form.Form):
                    # If there are no sub-elements, continue
                    continue
                diagonal_form = ufl.ZeroBaseForm(())
                for j in range(len(blocked_form)):
                    if blocked_form[j][j] is not None:
                        diagonal_form += blocked_form[j][j]
                if diagonal_form == 0:
                    raise RuntimeError("Diagonal form seems to be zero.")
                forms[i] = diagonal_form  # type: ignore

    # Get a signature for these forms
    module_name = "libffcx_forms_" + ffcx.naming.compute_signature(
        forms,
        _compute_option_signature(p) + _compilation_signature(cffi_extra_compile_args, cffi_debug),
    )

    form_names = [ffcx.naming.form_name(form, i, module_name) for i, form in enumerate(forms)]

    # allow for custom backend-specific additions
    # to the CFFI interface
    language = ffcx.options.get_language(p)
    language_mod = importlib.import_module(language)
    extra_names = []
    extra_decl = ""
    if hasattr(language_mod.form, "get_cffi_decl"):
        extra_decl, extra_names = language_mod.form.get_cffi_decl(form_names)

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        obj, mod = get_cached_module(module_name, form_names + extra_names, cache_dir, timeout)
        if obj is not None:
            return obj, mod, (None, None)
    else:
        cache_dir = Path(tempfile.mkdtemp())

    try:
        decl = (
            UFC_HEADER_DECL.format(np.dtype(p["scalar_type"]).name)  # type: ignore
            + UFC_INTEGRAL_DECL
            + UFC_FORM_DECL
        )

        form_template = "extern ufcx_form {name};\n"
        for name in form_names:
            decl += form_template.format(name=name)
        decl += extra_decl

        # allow for custom compilation logic for a given backend
        if hasattr(language_mod, "compile_objects"):
            compile_objects = language_mod.compile_objects
        else:
            compile_objects = _compile_objects

        impl = compile_objects(
            decl,
            forms,
            form_names,
            module_name,
            p,
            cache_dir,
            cffi_extra_compile_args,
            cffi_verbose,
            cffi_debug,
            cffi_libraries,
            visualise=visualise,
        )
    except Exception as e:
        try:
            # remove c file so that it will not timeout next time
            c_filename = cache_dir.joinpath(module_name + ".c")
            os.replace(c_filename, c_filename.with_suffix(".c.failed"))
        except Exception:
            pass
        raise e

    obj, module = _load_objects(cache_dir, module_name, form_names + extra_names)
    return obj, module, (decl, impl)


def compile_expressions(
    expressions: list[tuple[ufl.Expr, npt.NDArray[np.floating]]],  # type: ignore
    options: dict = {},
    cache_dir: Path | None = None,
    timeout: int = 10,
    cffi_extra_compile_args: list[str] = [],
    cffi_verbose: bool = False,
    cffi_debug: bool = False,
    cffi_libraries: list[str] = [],
    visualise: bool = False,
):
    """Compile a list of UFL expressions into UFCx Python objects.

    Args:
        expressions: List of (UFL expression, evaluation points).
        options: Options
        cache_dir: Cache directory
        timeout: Timeout
        cffi_extra_compile_args: Extra compilation args for CFFI
        cffi_verbose: Use verbose compile
        cffi_debug: Use compiler debug mode
        cffi_libraries: libraries to use with compiler
        visualise: Toggle visualisation
    """
    p = ffcx.options.get_options(options)

    module_name = "libffcx_expressions_" + ffcx.naming.compute_signature(
        expressions,
        _compute_option_signature(p) + _compilation_signature(cffi_extra_compile_args, cffi_debug),
    )
    expr_names = [
        ffcx.naming.expression_name(expression, module_name) for expression in expressions
    ]

    # allow for custom backend-specific additions
    # to the CFFI interface
    language = ffcx.options.get_language(p)
    language_mod = importlib.import_module(language)
    extra_names = []
    extra_decl = ""
    if hasattr(language_mod.expression, "get_cffi_decl"):
        extra_decl, extra_names = language_mod.expression.get_cffi_decl(expr_names)

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        obj, mod = get_cached_module(module_name, expr_names + extra_names, cache_dir, timeout)
        if obj is not None:
            return obj, mod, (None, None)
    else:
        cache_dir = Path(tempfile.mkdtemp())

    try:
        decl = (
            UFC_HEADER_DECL.format(np.dtype(p["scalar_type"]).name)  # type: ignore
            + UFC_INTEGRAL_DECL
            + UFC_FORM_DECL
            + UFC_EXPRESSION_DECL
        )

        expression_template = "extern ufcx_expression {name};\n"
        for name in expr_names:
            decl += expression_template.format(name=name)

            decl += extra_decl

        # allow for custom compilation logic for a given backend
        if hasattr(language_mod, "compile_objects"):
            compile_objects = language_mod.compile_objects
        else:
            compile_objects = _compile_objects

        impl = compile_objects(
            decl,
            expressions,
            expr_names,
            module_name,
            p,
            cache_dir,
            cffi_extra_compile_args,
            cffi_verbose,
            cffi_debug,
            cffi_libraries,
            visualise=visualise,
        )
    except Exception as e:
        try:
            # remove c file so that it will not timeout next time
            c_filename = cache_dir.joinpath(module_name + ".c")
            os.replace(c_filename, c_filename.with_suffix(".c.failed"))
        except Exception:
            pass
        raise e

    obj, module = _load_objects(cache_dir, module_name, expr_names + extra_names)
    return obj, module, (decl, impl)
