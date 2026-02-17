import basix
import ufl
from ffcx.compiler import compile_ufl_objects
from ffcx.options import get_options


def test_integral():
    element = basix.ufl.element("Lagrange", "triangle", 1)
    domain = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
    space = ufl.FunctionSpace(domain, element)
    u, v = ufl.TrialFunction(space), ufl.TestFunction(space)

    a = (ufl.inner(u, v) + ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    opts = get_options({"language": "ffcx_backends.cpp"})

    compiled_objects = compile_ufl_objects([a], opts)

    assert len(compiled_objects) == 2
