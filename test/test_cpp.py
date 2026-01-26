import pathlib

import basix
import ufl

import ffcx
from ffcx.compiler import compile_ufl_objects
from ffcx.options import get_options

def test_integral():
    element = basix.ufl.element("Lagrange", "triangle", 1)
    domain = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
    space = ufl.FunctionSpace(domain, element)
    u, v = ufl.TrialFunction(space), ufl.TestFunction(space)

    a = (ufl.inner(u, v) + ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    opts = {"language": "ffcx_backends.cpp"}
    opts = get_options(opts)

    compiled_objects = compile_ufl_objects([a], opts)

    decl = compiled_objects[0][0]
