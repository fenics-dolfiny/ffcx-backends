"""Optimizer."""

from collections import defaultdict, Counter

import ffcx.codegeneration.lnodes as L
from ffcx.ir.representationutils import QuadratureRule


def optimize(code: list[L.LNode], quadrature_rule: QuadratureRule) -> list[L.LNode]:
    """Optimize code.

    Args:
        code: List of LNodes to optimize.
        quadrature_rule: TODO.

    Returns:
        Optimized list of LNodes.
    """
    # Fuse sections with the same name and same annotations
    code = fuse_sections(code, "Coefficient")
    code = fuse_sections(code, "Jacobian")
    for i, section in enumerate(code):
        if isinstance(section, L.Section):
            if L.Annotation.fuse in section.annotations:
                section = fuse_loops(section)
            if L.Annotation.licm in section.annotations:
                section = licm(section, quadrature_rule)
            code[i] = section

    return code


def fuse_sections(code: list[L.LNode], name: str) -> list[L.LNode]:
    """Fuse sections with the same name.

    Args:
        code: List of LNodes to fuse.
        name: Common name used by the sections that should be fused

    Returns:
        Fused list of LNodes.
    """
    statements: list[L.LNode] = []
    indices: list[int] = []
    input: list[L.Symbol] = []
    output: list[L.Symbol] = []
    declarations: list[L.Declaration] = []
    annotations: list[L.Annotation] = []

    for i, section in enumerate(code):
        if isinstance(section, L.Section):
            if section.name == name:
                declarations.extend(section.declarations)
                statements.extend(section.statements)
                indices.append(i)
                input.extend(section.input)
                output.extend(section.output)
                annotations = section.annotations

    # Remove duplicated inputs
    input = list(set(input))
    # Remove duplicated outputs
    output = list(set(output))

    section = L.Section(name, statements, declarations, input, output, annotations)

    # Replace the first section with the fused section
    code = code.copy()
    if indices:
        code[indices[0]] = section
        # Remove the other sections
        code = [c for i, c in enumerate(code) if i not in indices[1:]]

    return code


def fuse_loops(code: L.Section) -> L.Section:
    """Fuse loops with the same range and same annotations.

    Args:
        code: List of LNodes to fuse.

    Returns:
        Fused list of LNodes.
    """
    loops = defaultdict(list)
    output_code = []
    for statement in code.statements:
        if isinstance(statement, L.ForRange):
            id = (statement.index, statement.begin, statement.end)
            loops[id].append(statement.body)
        else:
            output_code.append(statement)

    for range, body in loops.items():
        output_code.append(L.ForRange(*range, body))

    return L.Section(code.name, output_code, code.declarations, code.input, code.output)


def get_statements(statement: L.Statement | L.StatementList) -> list[L.LNode]:
    """Get statements from a statement list.

    Args:
        statement: Statement list.

    Returns:
        List of statements.
    """
    if isinstance(statement, L.StatementList):
        return [statement.expr for statement in statement.statements]
    else:
        return [statement.expr]


def check_dependency(statement: L.Statement, index: L.Symbol) -> bool:
    """Check if a statement depends on a given index.

    Args:
        statement: Statement to check.
        index: Index to check.

    Returns:
        True if statement depends on index, False otherwise.
    """
    if isinstance(statement, L.Sum) or isinstance(statement, L.Product):
        return any(check_dependency(arg, index) for arg in statement.args)
    if isinstance(statement, L.ArrayAccess):
        if index in statement.indices:
            return True
        else:
            for i in statement.indices:
                if isinstance(i, L.Sum) or isinstance(i, L.Product):
                    if index in i.args:
                        return True
    elif isinstance(statement, L.Symbol):
        return False
    elif isinstance(statement, L.LiteralFloat) or isinstance(statement, L.LiteralInt):
        return False
    else:
        raise NotImplementedError(f"Statement {statement} not supported.")

    return False

def extract_common_factors(expression: L.Sum):
    """Extract common factors from a sum of products

    Args:
        expression: LNode containing a sum of products

    Returns:
        Optimized LNode in which common factors are pulled out
    """

    # create a dict to keep track of which products a factor appears in
    factor_args = defaultdict(set)
    for product in expression.args:
        if not isinstance(product, L.Product):
            raise ValueError("Expected L.Product, got {product}!")
        for factor in product.args:
            factor_args[factor].add(product)
    
    # new list of products
    new_args = []
    # old products that have been used already
    used_args = set()
    # iterate until everything has been considered
    while len(used_args) < len(expression.args):
        top_factor = max(factor_args.keys(), key=lambda factor: len(factor_args[factor]))
        args = list(factor_args[top_factor])
        used_args.update(args)
        if len(args) == 0:
            break
        elif len(args) == 1:
            new_args.append(args[0])
        else:
            # factor is part of at least two products, pull it out
            summands = []
            for product in args:
                summands.append(
                    L.Product(
                        [factor for factor in product.args if repr(factor) != repr(top_factor)]
                    )
                )
            factored_product = L.Product(
                [top_factor, L.Sum(summands)]
            )
            new_args.append(factored_product)

        # remove these products from consideration for all other factors
        for factor in factor_args:
            factor_args[factor].difference_update(args)

    if len(used_args) != len(expression.args):
        raise RuntimeError("Error computing common factors!")

    return L.Sum(new_args)

def count_ops(statement: L.Statement):
    """Count operations required to compute value."""

    if isinstance(statement, L.Sum) or isinstance(statement, L.Product):
        return len(statement.args) - 1 + sum(count_ops(arg) for arg in statement.args)
    else:
        # TODO make this more accurate
        # but it works well enough for now
        return 0


def licm(section: L.Section, quadrature_rule: QuadratureRule) -> L.Section:
    """Perform loop invariant code motion.

    Args:
        section: List of LNodes to optimize.
        quadrature_rule: TODO.

    Returns:
        Optimized list of LNodes.
    """
    assert L.Annotation.licm in section.annotations

    counter = 0

    # Check depth of loops
    depth = L.depth(section.statements[0])
    if depth != 2:
        return section

    # Get statements in the inner loop
    outer_loop = section.statements[0]
    inner_loop = outer_loop.body.statements[0]

    # Collect all expressions in the inner loop by corresponding RHS
    expressions = [] 
    for body in inner_loop.body.statements:
        statements = get_statements(body)
        assert isinstance(statements, list)
        for statement in statements:
            assert isinstance(statement, L.AssignAdd)  # Expecting AssignAdd
            assert isinstance(statement.lhs, L.ArrayAccess)  # Expecting ArrayAccess
            if isinstance(statement.rhs, L.Sum):
                # see if we can pull out a common factor
                # this WILL change floating point computations
                statement.rhs = extract_common_factors(statement.rhs)
                expressions.extend(statement.rhs.args)
            elif isinstance(statement.rhs, L.Product):
                expressions.append(statement.rhs)
            else:
                raise RuntimeError("Expected either Product or Sum!")


    pre_loop: list[L.LNode] = []

    # compute hoist candidates for both the inner and outer loops
    # then potentially switch loop order

    inner_candidates = defaultdict(list)
    outer_candidates = defaultdict(list)

    for r in expressions:
        for arg in r.args:
            inner_dep = check_dependency(arg, inner_loop.index)
            outer_dep = check_dependency(arg, outer_loop.index)
            if not inner_dep:
                inner_candidates[r].append(arg)
            if not outer_dep:
                outer_candidates[r].append(arg)

    inner_size = inner_loop.end.value - inner_loop.begin.value
    outer_size = outer_loop.end.value - outer_loop.begin.value
    inner_ops = inner_size*sum(
        count_ops(arg) for args in inner_candidates.values() for arg in args
    )
    outer_ops = outer_size*sum(
        count_ops(arg) for args in outer_candidates.values() for arg in args
    )

    if outer_ops > inner_ops:
        print(
            f"Reversing loop order to optimize {outer_ops}"
            f" ops instead of {inner_ops}"
        )
        # TODO see if there's a better way to do this
        section.statements[0] = inner_loop
        outer_loop.body = inner_loop.body
        inner_loop.body = L.StatementList([outer_loop])
    else:
        print(
            f"Keeping loop order to optimize {inner_ops} ops"
            f"instead of {outer_ops}"
        )

    for r in expressions:
        hoist_candidates = []
        for arg in r.args:
            dependency = check_dependency(arg, inner_loop.index)
            if not dependency:
                hoist_candidates.append(arg)
        if len(hoist_candidates) > 1:
            # create new temp
            name = f"temp_{counter}"
            counter += 1
            temp = L.Symbol(name, L.DataType.SCALAR)
            for h in hoist_candidates:
                r.args.remove(h)
            # update expression with new temp
            r.args.append(L.ArrayAccess(temp, [outer_loop.index]))
            # create code for hoisted term
            size = outer_loop.end.value - outer_loop.begin.value
            pre_loop.append(L.ArrayDecl(temp, size, [0]))
            body = L.Assign(
                L.ArrayAccess(temp, [outer_loop.index]), L.Product(hoist_candidates)
            )
            pre_loop.append(
                L.ForRange(outer_loop.index, outer_loop.begin, outer_loop.end, [body])
            )

    section.statements = pre_loop + section.statements

    return section
