"""Forms and an expression that interpolate, for every backend.

Compile this file with FFCx: ffcx --language ffcx_backends.[lang] interpolation.py.

Each interpolation is evaluated by an expression kernel of its own, which the
kernels holding it call:

- `proxy_coefficient` interpolates `u**2`, which becomes a proxy coefficient.
- `runtime_table` interpolates `u * v`, linear in the test function, which
  becomes an element table built for each cell.
- `nested` nests an interpolation, so its interpolation kernel calls another
  one, and reads a constant.

Interpolating into P1 evaluates at the vertices, which are the first three dofs
of P2, so all have a closed form.
"""

import basix.ufl
import numpy as np
from ufl import (
    Coefficient,
    Constant,
    FunctionSpace,
    Interpolate,
    Mesh,
    TestFunction,
    dx,
)

mesh = Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
V1 = FunctionSpace(mesh, basix.ufl.element("Lagrange", "triangle", 1))
V2 = FunctionSpace(mesh, basix.ufl.element("Lagrange", "triangle", 2))

u = Coefficient(V2)
v = TestFunction(V2)
kappa = Constant(mesh)

# Forms
proxy_coefficient = Interpolate(u**2, V1) * dx
runtime_table = Interpolate(u * v, V1) * dx
forms = [proxy_coefficient, runtime_table]

# Expressions, named by the variable they are bound to
nested = Interpolate(kappa * Interpolate(u**2, V1) * u, V1)
points = np.array([[0.1, 0.1], [0.6, 0.2], [0.25, 0.5]])
expressions = [(nested, points)]
