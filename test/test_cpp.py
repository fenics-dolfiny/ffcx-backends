import re

import basix.ufl
import ffcx.codegeneration.lnodes as L  # noqa: N812
import pytest
import ufl
from ffcx.compiler import compile_ufl_objects
from ffcx.options import get_options

from ffcx_backends.cpp import Formatter


def test_integral() -> None:
    element = basix.ufl.element("Lagrange", "triangle", 1)
    domain = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
    space = ufl.FunctionSpace(domain, element)
    u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
    f = ufl.Coefficient(space)

    a = (ufl.inner(u, v) + f * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    opts = get_options({"language": "ffcx_backends.cpp"})

    compiled_objects = compile_ufl_objects([a], opts)

    assert len(compiled_objects) == 2


@pytest.mark.parametrize(("scalar_geometry", "expected"), [(False, "U"), (True, "T"), (None, "U")])
def test_scalar_geometry(scalar_geometry: bool | None, expected: str) -> None:
    """Geometry-derived temporaries are emitted in T only with ``scalar_geometry``.

    The kernel signature is unaffected: coordinate dofs are always read as ``U``,
    the option only changes the type they are computed in.
    """
    element = basix.ufl.element("Lagrange", "triangle", 1)
    domain = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
    space = ufl.FunctionSpace(domain, element)
    u, v = ufl.TrialFunction(space), ufl.TestFunction(space)

    a = (ufl.inner(u, v) + ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    opts = get_options({"language": "ffcx_backends.cpp"})
    if scalar_geometry is not None:
        opts["scalar_geometry"] = scalar_geometry

    code = compile_ufl_objects([a], opts)[0][0]

    # Quadrature weights, tabulated basis functions and the Jacobian are all
    # geometry-derived, hence declared in the geometry type.
    for declaration in [
        r"static const (\w+) weights_\w+\[",
        r"static const (\w+) FE\w+\[",
        # The Jacobian symbol carries a process-global counter, hence J<n>.
        r"^(\w+) J\d+_c0 = ",
    ]:
        matches = re.findall(declaration, code, flags=re.MULTILINE)
        assert matches, f"no declaration matching {declaration} in generated code"
        assert set(matches) == {expected}

    # The tabulate_tensor signature keeps the scalar/geometry split either way.
    signature = re.search(r"static void tabulate_tensor\((.*?)\)\s*\{", code, flags=re.DOTALL)
    assert signature is not None
    assert "const U* RESTRICT coordinate_dofs" in signature.group(1)
    assert "T* RESTRICT A" in signature.group(1)


@pytest.mark.parametrize(("scalar_geometry", "expected"), [(False, "U"), (True, "T")])
def test_formatter_dtype_to_cpp_type(scalar_geometry: bool, expected: str) -> None:
    """REAL follows the geometry type, the remaining datatypes do not."""
    formatter = Formatter(scalar_geometry)

    assert formatter.dtype_to_cpp_type(L.DataType.REAL) == expected
    assert formatter.dtype_to_cpp_type(L.DataType.SCALAR) == "T"
    assert formatter.dtype_to_cpp_type(L.DataType.INT) == "std::int32_t"
    assert formatter.dtype_to_cpp_type(L.DataType.BOOL) == "bool"
