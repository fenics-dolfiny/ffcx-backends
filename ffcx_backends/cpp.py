"""Custom C++ FFCx backend."""

import functools
import logging
import pprint
import string
import textwrap

import basix
import ffcx.codegeneration.lnodes as L  # noqa
import numpy as np
from ffcx import __version__ as ffcx_version
from ffcx.codegeneration import __version__ as ufcx_version
from ffcx.codegeneration.backend import FFCXBackend
from ffcx.codegeneration.common import integral_data, template_keys
from ffcx.codegeneration.expression_generator import ExpressionGenerator
from ffcx.codegeneration.integral_generator import IntegralGenerator
from ffcx.codegeneration.utils import dtype_to_c_type, dtype_to_scalar_dtype
from ffcx.ir.representation import FormIR, IntegralIR

logger = logging.getLogger("ffcx")


def dtype_to_cpp_type(dtype: L.DataType, scalar_type: str, real_type: str) -> str:
    """Map L.DataType to C++ type."""
    if dtype == L.DataType.SCALAR:
        return scalar_type
    elif dtype == L.DataType.REAL:
        return real_type
    elif dtype == L.DataType.INT:
        return "std::int32_t"
    elif dtype == L.DataType.BOOL:
        return "bool"
    else:
        raise ValueError(f"Invalid datatype: {dtype}")


class Formatter:
    math_table = {
        "sqrt": "std::sqrt",
        "abs": "std::abs",
        "cos": "std::cos",
        "sin": "std::sin",
        "tan": "std::tan",
        "acos": "std::acos",
        "asin": "std::asin",
        "atan": "std::atan",
        "cosh": "std::cosh",
        "sinh": "std::sinh",
        "tanh": "std::tanh",
        "acosh": "std::acosh",
        "asinh": "std::asinh",
        "atanh": "std::atanh",
        "power": "std::pow",
        "exp": "std::exp",
        "ln": "std::log",
        "erf": "std::erf",
        "atan_2": "std::atan2",
        "min_value": "std::fmin",
        "max_value": "std::fmax",
        "bessel_y": "std::cyl_bessel_i",
        "bessel_j": "std::cyl_bessel_j",
        "conj": "std::conj",
        "real": "std::real",
        "imag": "std::imag",
    }

    @staticmethod
    def build_initializer_lists(values):
        """Build initializer lists."""
        arr = "{"
        if len(values.shape) == 1:
            return "{" + ", ".join(str(v) for v in values) + "}"
        elif len(values.shape) > 1:
            arr += ",\n".join(Formatter.build_initializer_lists(v) for v in values)
        arr += "}"
        return arr

    def __init__(self, scalar) -> None:
        """Initialise."""
        self.scalar_type = "T"
        self.real_type = "U"

    @functools.singledispatchmethod
    def __call__(self, obj: L.LNode) -> str:
        """Formatting function."""
        raise RuntimeError(f"Unknown statement: {obj.__class__.__name__}")

    @__call__.register
    def format_statement_list(self, slist: L.StatementList) -> str:
        """Format statement list."""
        return "".join(self(s) for s in slist.statements)

    @__call__.register
    def format_section(self, section: L.Section) -> str:
        """Format a section."""
        # add new line before section
        comments = "// ------------------------ \n"
        comments += "// Section: " + section.name + "\n"
        comments += "// Inputs: " + ", ".join(w.name for w in section.input) + "\n"
        comments += "// Outputs: " + ", ".join(w.name for w in section.output) + "\n"
        declarations = "".join(self(s) for s in section.declarations)

        body = ""
        if len(section.statements) > 0:
            declarations += "{\n  "
            body = "".join(self(s) for s in section.statements)
            body = body.replace("\n", "\n  ")
            body = body[:-2] + "}\n"

        body += "// ------------------------ \n"
        return comments + declarations + body

    @__call__.register
    def format_comment(self, c: L.Comment) -> str:
        """Format a comment."""
        return f"// {c.comment}\n"

    @__call__.register
    def format_array_decl(self, arr: L.ArrayDecl) -> str:
        """Format an array declaration."""
        dtype = arr.symbol.dtype
        assert dtype is not None

        typename = dtype_to_cpp_type(dtype, self.scalar_type, self.real_type)

        symbol = self(arr.symbol)
        dims = "".join([f"[{i}]" for i in arr.sizes])
        if arr.values is None:
            assert arr.const is False
            return f"{typename} {symbol}{dims};\n"

        vals = Formatter.build_initializer_lists(arr.values)
        cstr = "static const " if arr.const else ""
        return f"{cstr}{typename} {symbol}{dims} = {vals};\n"

    @__call__.register
    def format_array_access(self, arr: L.ArrayAccess) -> str:
        """Format array access."""
        name = self(arr.array)
        indices = f"[{']['.join(self(i) for i in arr.indices)}]"
        return f"{name}{indices}"

    @__call__.register
    def format_multi_index(self, index: L.MultiIndex) -> str:
        """Format a multi-index."""
        return self(index.global_index)

    @__call__.register
    def format_variable_decl(self, v: L.VariableDecl) -> str:
        """Format a variable declaration."""
        val = self(v.value)
        symbol = self(v.symbol)
        assert v.symbol.dtype
        typename = dtype_to_cpp_type(v.symbol.dtype, self.scalar_type, self.real_type)
        return f"{typename} {symbol} = {val};\n"

    @__call__.register
    def format_nary_op(self, oper: L.NaryOp) -> str:
        """Format an n-argument operation."""
        # Format children
        args = [self(arg) for arg in oper.args]

        # Apply parentheses
        for i in range(len(args)):
            if oper.args[i].precedence >= oper.precedence:
                args[i] = "(" + args[i] + ")"

        # Return combined string
        return f" {oper.op} ".join(args)

    @__call__.register
    def format_binary_op(self, oper: L.BinOp) -> str:
        """Format a binary operation."""
        # Format children
        lhs = self(oper.lhs)
        rhs = self(oper.rhs)

        # Apply parentheses
        if oper.lhs.precedence >= oper.precedence:
            lhs = f"({lhs})"
        if oper.rhs.precedence >= oper.precedence:
            rhs = f"({rhs})"

        # Return combined string
        return f"{lhs} {oper.op} {rhs}"

    @__call__.register
    def format_neg(self, val: L.Neg) -> str:
        """Format negation."""
        arg = self(val.arg)
        return f"-{arg}"

    @__call__.register
    def format_not(self, val: L.Not) -> str:
        """Format 'not' statement."""
        arg = self(val.arg)
        return f"{val.op}({arg})"

    @__call__.register
    def format_literal_float(self, val: L.LiteralFloat) -> str:
        """Format a literal float number."""
        return f"{val.value}"

    @__call__.register
    def format_literal_int(self, val: L.LiteralInt) -> str:
        """Format a literal int number."""
        return f"{val.value}"

    @__call__.register
    def format_for_range(self, r: L.ForRange) -> str:
        """Format a loop over a range."""
        begin = self(r.begin)
        end = self(r.end)
        index = self(r.index)
        output = f"for (int {index} = {begin}; {index} < {end}; ++{index})\n"
        output += "{\n"
        body = self(r.body)
        for line in body.split("\n"):
            if len(line) > 0:
                output += f"  {line}\n"
        output += "}\n"
        return output

    @__call__.register
    def format_statement(self, s: L.Statement) -> str:
        """Format a statement."""
        return self(s.expr)

    @__call__.register(L.Assign)
    @__call__.register(L.AssignAdd)
    def format_assign(self, expr: L.Assign | L.AssignAdd) -> str:
        """Format an assignment statement."""
        rhs = self(expr.rhs)
        lhs = self(expr.lhs)
        return f"{lhs} {expr.op} {rhs};\n"

    @__call__.register
    def format_conditional(self, s: L.Conditional) -> str:
        """Format a conditional."""
        # Format children
        c = self(s.condition)
        t = self(s.true)
        f = self(s.false)

        # Apply parentheses
        if s.condition.precedence >= s.precedence:
            c = "(" + c + ")"
        if s.true.precedence >= s.precedence:
            t = "(" + t + ")"
        if s.false.precedence >= s.precedence:
            f = "(" + f + ")"

        # Return combined string
        return c + " ? " + t + " : " + f

    @__call__.register
    def format_symbol(self, s: L.Symbol) -> str:
        """Format a symbol."""
        return f"{s.name}"

    @__call__.register
    def format_math_function(self, c: L.MathFunction) -> str:
        """Format a math function."""
        # Get a function from the table, if available, else just use bare name
        func = Formatter.math_table.get(c.function, c.function)
        args = ", ".join(self(arg) for arg in c.args)
        return f"{func}({args})"


class expression:
    declaration = """
extern ufcx_expression {factory_name};

// Helper used to create expression using name which was given to the
// expression in the UFL file.
// This helper is called in user C++ code.
//
extern ufcx_expression* {name_from_uflfile};
"""

    factory = """
// Code for expression {factory_name}

void tabulate_tensor_{factory_name}({scalar_type}* RESTRICT A,
                                    const {scalar_type}* RESTRICT w,
                                    const {scalar_type}* RESTRICT c,
                                    const {geom_type}* RESTRICT coordinate_dofs,
                                    const int* RESTRICT entity_local_index,
                                    const uint8_t* RESTRICT quadrature_permutation)
{{
{tabulate_expression}
}}

{points_init}
{value_shape_init}
{original_coefficient_positions_init}
{function_spaces_alloc}
{function_spaces_init}
{coefficient_names_init}
{constant_names_init}


ufcx_expression {factory_name} =
{{
  .tabulate_tensor_{np_scalar_type} = tabulate_tensor_{factory_name},
  .num_coefficients = {num_coefficients},
  .num_constants = {num_constants},
  .original_coefficient_positions = {original_coefficient_positions},
  .coefficient_names = {coefficient_names},
  .constant_names = {constant_names},
  .num_points = {num_points},
  .entity_dimension = {entity_dimension},
  .points = {points},
  .value_shape = {value_shape},
  .num_components = {num_components},
  .rank = {rank},
  .function_spaces = {function_spaces}
}};

// Alias name
ufcx_expression* {name_from_uflfile} = &{factory_name};

// End of code for expression {factory_name}
"""

    @staticmethod
    def generator(ir, options):
        """Generate UFC code for an expression."""
        logger.info("Generating code for expression:")
        assert len(ir.expression.integrand) == 1, "Expressions only support single quadrature rule"
        points = next(iter(ir.expression.integrand))[1].points
        logger.info(f"--- points: {points}")
        factory_name = ir.expression.name
        logger.info(f"--- name: {factory_name}")

        # Format declaration
        declaration = expression.declaration.format(
            factory_name=factory_name, name_from_uflfile=ir.name_from_uflfile
        )

        backend = FFCXBackend(ir, options)
        eg = ExpressionGenerator(ir, backend)

        d = {}
        d["name_from_uflfile"] = ir.name_from_uflfile
        d["factory_name"] = factory_name

        parts = eg.generate()

        cf = Formatter(options["scalar_type"])
        d["tabulate_expression"] = cf(parts)

        if len(ir.original_coefficient_positions) > 0:
            d["original_coefficient_positions"] = f"original_coefficient_positions_{factory_name}"
            sizes = len(ir.original_coefficient_positions)
            values = ", ".join(str(i) for i in ir.original_coefficient_positions)
            d["original_coefficient_positions_init"] = (
                f"static int original_coefficient_positions_{factory_name}[{sizes}] = {{{values}}};"
            )

        else:
            d["original_coefficient_positions"] = "NULL"
            d["original_coefficient_positions_init"] = ""

        values = ", ".join(str(p) for p in points.flatten())
        sizes = points.size
        d["points_init"] = f"static double points_{factory_name}[{sizes}] = {{{values}}};"
        d["points"] = f"points_{factory_name}"

        if len(ir.expression.shape) > 0:
            values = ", ".join(str(i) for i in ir.expression.shape)
            sizes = len(ir.expression.shape)
            d["value_shape_init"] = (
                f"static int value_shape_{factory_name}[{sizes}] = {{{values}}};"
            )
            d["value_shape"] = f"value_shape_{factory_name}"
        else:
            d["value_shape_init"] = ""
            d["value_shape"] = "NULL"

        d["num_components"] = len(ir.expression.shape)
        d["num_coefficients"] = len(ir.expression.coefficient_numbering)
        d["num_constants"] = len(ir.constant_names)
        d["num_points"] = points.shape[0]
        d["entity_dimension"] = points.shape[1]
        d["scalar_type"] = dtype_to_c_type(options["scalar_type"])
        d["geom_type"] = dtype_to_c_type(dtype_to_scalar_dtype(options["scalar_type"]))
        d["np_scalar_type"] = np.dtype(options["scalar_type"]).name

        d["rank"] = len(ir.expression.tensor_shape)

        if len(ir.coefficient_names) > 0:
            values = ", ".join(f'"{name}"' for name in ir.coefficient_names)
            sizes = len(ir.coefficient_names)
            d["coefficient_names_init"] = (
                f"static const char* coefficient_names_{factory_name}[{sizes}] = {{{values}}};"
            )

            d["coefficient_names"] = f"coefficient_names_{factory_name}"
        else:
            d["coefficient_names_init"] = ""
            d["coefficient_names"] = "NULL"

        if len(ir.constant_names) > 0:
            values = ", ".join(f'"{name}"' for name in ir.constant_names)
            sizes = len(ir.constant_names)
            d["constant_names_init"] = (
                f"static const char* constant_names_{factory_name}[{sizes}] = {{{values}}};"
            )
            d["constant_names"] = f"constant_names_{factory_name}"
        else:
            d["constant_names_init"] = ""
            d["constant_names"] = "NULL"

        # TODO: make cpp
        d["coordinate_element_hash"] = f"UINT64_C({ir.expression.coordinate_element_hash})"

        # FIXME: Should be handled differently, revise how
        # ufcx_function_space is generated (also for ufcx_form)
        # for name, (element, dofmap, cmap_family, cmap_degree) in ir.function_spaces.items():
        #     code += [f"static ufcx_function_space function_space_{name}_{ir.name_from_uflfile} ="]
        #     code += ["{"]
        #     code += [f".finite_element = &{element},"]
        #     code += [f".dofmap = &{dofmap},"]
        #     code += [f'.geometry_family = "{cmap_family}",']
        #     code += [f".geometry_degree = {cmap_degree}"]
        #     code += ["};"]

        # d["function_spaces_alloc"] = "\n".join(code)
        # d["function_spaces"] = ""

        # if len(ir.function_spaces) > 0:
        #     d["function_spaces"] = f"function_spaces_{ir.name}"
        #     fs_list = ", ".join(
        #         f"&function_space_{name}_{ir.name_from_uflfile}"
        #         for (name, _) in ir.function_spaces.items()
        #     )
        #     n = len(ir.function_spaces.items())
        #     d["function_spaces_init"] = (
        #         f"ufcx_function_space* function_spaces_{ir.name}[{n}] = {{{fs_list}}};"
        #     )
        # else:
        #     d["function_spaces"] = "NULL"
        #     d["function_spaces_init"] = ""

        # Check that no keys are redundant or have been missed
        fields = [fname for _, fname, _, _ in string.Formatter().parse(expression.factory) if fname]
        assert set(fields) == set(d.keys()), (
            "Mismatch between keys in template and in formatting dict"
        )

        # Format implementation code
        implementation = expression.factory.format_map(d)

        return declaration, implementation


class integral:
    declaration = """
class {factory_name}
{{
public:

// Constructor
{factory_name}();

// Kernel
template <typename T, typename U>
void tabulate_tensor(T* A,
                     const T* RESTRICT w,
                     const T* RESTRICT c,
                     const U* RESTRICT coordinate_dofs,
                     const std::int32_t* RESTRICT entity_local_index,
                     const std::uint8_t* RESTRICT quadrature_permutation);

// Data
std::vector<bool> enabled_coefficients; // TODO: std::vector<char>?
bool needs_facet_permutations;

}};
"""

    factory = """
// Code for integral {factory_name}

template <typename T, typename U>
void {factory_name}::tabulate_tensor(T* RESTRICT A,
                     const T* RESTRICT w,
                     const T* RESTRICT c,
                     const U* RESTRICT coordinate_dofs,
                     const std::int32_t* RESTRICT entity_local_index,
                     const std::uint8_t* RESTRICT quadrature_permutation)
{{
{tabulate_tensor}
}}

{factory_name}::{factory_name}()
{{
  enabled_coefficients = {enabled_coefficients};
  needs_facet_permutations = {needs_facet_permutations};
}}

// End of code for integral {factory_name}
"""

    @staticmethod
    def generator(ir: IntegralIR, domain: basix.CellType, options):
        """Generate C++ code for an integral."""
        logger.info("Generating code for integral:")
        logger.info(f"--- type: {ir.expression.integral_type}")
        logger.info(f"--- name: {ir.expression.name}")

        factory_name = f"{ir.expression.name}_{domain.name}"

        # Format declaration
        declaration = integral.declaration.format(factory_name=factory_name)

        # Create FFCx backend
        backend = FFCXBackend(ir, options)

        # Configure kernel generator
        ig = IntegralGenerator(ir, backend)

        # Generate code ast for the tabulate_tensor body
        parts = ig.generate(domain)

        # Format code as string
        formatter = Formatter(options["scalar_type"])  # type: ignore
        body = formatter(parts)

        # Generate generic FFCx code snippets and add specific parts
        code = {}
        code["class_type"] = ir.expression.integral_type + "_integral"
        code["name"] = ir.expression.name

        vals = ", ".join("true" if i else "false" for i in ir.enabled_coefficients)
        code["enabled_coefficients"] = f"{{{vals}}}"

        # FIXME: Get this out of code[]
        code["additional_includes_set"] = set()
        code["tabulate_tensor"] = body

        implementation = integral.factory.format(
            factory_name=factory_name,
            enabled_coefficients=code["enabled_coefficients"],
            tabulate_tensor=code["tabulate_tensor"],
            needs_facet_permutations="true" if ir.expression.needs_facet_permutations else "false",
            scalar_type=options["scalar_type"],
            geom_type=options["scalar_type"],
            np_scalar_type=options["scalar_type"],
            coordinate_element=ir.expression.coordinate_element_hash,
        )
        return (declaration + implementation,)


class form:
    declaration = """
    // extern ufcx_form {factory_name};

    // Helper used to create form using name which was given to the
    // form in the UFL file.
    // This helper is called in user c++ code.
    //
    // extern ufcx_form* {name_from_uflfile};

    """

    factory = r"""
    // Code for form {factory_name}

    struct  {factory_name} {{
    // TODO: that correct?
    {original_coefficient_position_init}
    {finite_element_hashes_init}
    {form_integral_offsets_init}
    // {form_integrals_init}
    {form_integral_ids_init}

    {coefficient_names_init}
    {constant_names_init}
    {constant_ranks_init}
    {constant_shapes_init}

    constexpr static const char* signature ={signature};
    constexpr static int rank = {rank};

    constexpr static int num_coefficients = {num_coefficients};
    constexpr static int* original_coefficient_positions = {original_coefficient_positions};
    constexpr static char** coefficient_name_map = {coefficient_names};

    constexpr static int num_constants = {num_constants};
    constexpr static const int* constant_ranks = {constant_ranks};
    constexpr static const int* constant_shapes = {constant_shapes};
    constexpr static const char* constant_name_map = {constant_names};

    constexpr static const uint64_t finite_element_hashes = {finite_element_hashes},

    {name_from_uflfile}::form_integrals = {form_integrals};
    {name_from_uflfile}::form_integral_ids = {form_integral_ids};
    {name_from_uflfile}::form_integral_offsets = form_integral_offsets_{factory_name};

    }};
    // Alias name
    using {name_from_uflfile} = {factory_name};

    // End of code for form {factory_name}
    """

    @staticmethod
    def generator(ir: FormIR, options):
        """Generate UFCx code for a form."""
        logger.info("Generating code for form:")
        logger.info(f"--- rank: {ir.rank}")
        logger.info(f"--- name: {ir.name}")

        d: dict[str, int | str] = {}
        d["factory_name"] = ir.name
        d["name_from_uflfile"] = ir.name_from_uflfile
        d["signature"] = f'"{ir.signature}"'
        d["rank"] = ir.rank
        d["num_coefficients"] = ir.num_coefficients

        if len(ir.original_coefficient_positions) > 0:
            values = ", ".join(str(i) for i in ir.original_coefficient_positions)
            sizes = len(ir.original_coefficient_positions)

            d["original_coefficient_position_init"] = (
                f"int original_coefficient_position_{ir.name}[{sizes}] = {{{values}}};"
            )
            d["original_coefficient_positions"] = f"original_coefficient_position_{ir.name}"
        else:
            d["original_coefficient_position_init"] = ""
            d["original_coefficient_positions"] = "NULL"

        if len(ir.coefficient_names) > 0:
            values = ", ".join(f'"{name}"' for name in ir.coefficient_names)
            sizes = len(ir.coefficient_names)
            d["coefficient_names_init"] = (
                f"constexpr static const char* coefficient_names_{ir.name}[{sizes}] = {{{values}}};"
            )
            d["coefficient_names"] = f"coefficient_names_{ir.name}"
        else:
            d["coefficient_names_init"] = ""
            d["coefficient_names"] = "NULL"

        d["num_constants"] = ir.num_constants
        if ir.num_constants > 0:
            d["constant_ranks_init"] = (
                f"constexpr static const int constant_ranks_{ir.name}[{ir.num_constants}] = "
                f"{{{str(ir.constant_ranks)[1:-1]}}};"
            )
            d["constant_ranks"] = f"constant_ranks_{ir.name}"

            shapes = [
                f"constexpr static const int constant_shapes_{ir.name}_{i}[{len(shape)}] = "
                f"{{{str(shape)[1:-1]}}};"
                for i, shape in enumerate(ir.constant_shapes)
                if len(shape) > 0
            ]
            names = [f"constant_shapes_{ir.name}_{i}" for i in range(ir.num_constants)]
            shapes1 = (
                f"constexpr static const int* constant_shapes_{ir.name}[{ir.num_constants}] = {{"
            )
            for rank, name in zip(ir.constant_ranks, names, strict=True):
                if rank > 0:
                    shapes1 += f"{name},\n"
                else:
                    shapes1 += "NULL,\n"
            shapes1 += "};"
            shapes.append(shapes1)

            d["constant_shapes_init"] = "\n".join(shapes)
            d["constant_shapes"] = f"constant_shapes_{ir.name}"
        else:
            d["constant_ranks_init"] = ""
            d["constant_ranks"] = "NULL"
            d["constant_shapes_init"] = ""
            d["constant_shapes"] = "NULL"

        if len(ir.constant_names) > 0:
            values = ", ".join(f'"{name}"' for name in ir.constant_names)
            sizes = len(ir.constant_names)
            d["constant_names_init"] = (
                f"constexpr static const char* constant_names_{ir.name}[{sizes}] = {{{values}}};"
            )
            d["constant_names"] = f"constant_names_{ir.name}"
        else:
            d["constant_names_init"] = ""
            d["constant_names"] = "NULL"

        if len(ir.finite_element_hashes) > 0:
            d["finite_element_hashes"] = f"finite_element_hashes_{ir.name}"
            values = ", ".join(
                f"UINT64_C({0 if el is None else el})" for el in ir.finite_element_hashes
            )
            sizes = len(ir.finite_element_hashes)
            d["finite_element_hashes_init"] = (
                f"uint64_t finite_element_hashes_{ir.name}[{sizes}] = {{{values}}};"
            )
        else:
            d["finite_element_hashes"] = "NULL"
            d["finite_element_hashes_init"] = ""

        integrals = integral_data(ir)

        if len(integrals.names) > 0:
            sizes = sum(len(domains) for domains in integrals.domains)
            values = ", ".join(
                [
                    f"&{name}_{domain.name}"
                    for name, domains in zip(integrals.names, integrals.domains, strict=True)
                    for domain in domains
                ]
            )
            d["form_integrals_init"] = (
                f"static ufcx_integral* form_integrals_{ir.name}[{sizes}] = {{{values}}};"
            )
            d["form_integrals"] = f"form_integrals_{ir.name}"
            values = ", ".join(
                f"{i}"
                for i, domains in zip(integrals.ids, integrals.domains, strict=True)
                for _ in domains
            )
            d["form_integral_ids_init"] = (
                f"int form_integral_ids_{ir.name}[{sizes}] = {{{values}}};"
            )
            d["form_integral_ids"] = f"form_integral_ids_{ir.name}"
        else:
            d["form_integrals_init"] = ""
            d["form_integrals"] = "NULL"
            d["form_integral_ids_init"] = ""
            d["form_integral_ids"] = "NULL"

        sizes = len(integrals.offsets)
        values = ", ".join(str(i) for i in integrals.offsets)
        d["form_integral_offsets_init"] = (
            f"int form_integral_offsets_{ir.name}[{sizes}] = {{{values}}};"
        )

        # Format implementation code
        assert set(d.keys()) == template_keys(form.factory)
        implementation = form.factory.format_map(d)

        # Format declaration
        declaration = form.declaration.format(
            factory_name=d["factory_name"], name_from_uflfile=d["name_from_uflfile"]
        )

        return (declaration + implementation,)


class file:
    suffixes = (".hpp",)

    declaration_pre = """
// This code conforms with the UFC specification version {ufcx_version}
// and was automatically generated by FFCx version {ffcx_version}.
//
// This code was generated with the following options:
//
{options}

#pragma once

#include <complex>
#include <cstdint>
#include <cmath>
#include <vector>

#include <ufcx.h>

#if defined(_MSC_VER)
#   define RESTRICT __restrict
#else
#   define RESTRICT __restrict__
#endif
    """

    declaration_post = """
    """

    @staticmethod
    def generator(options):
        """Generate UFC code for file output."""
        logger.info("Generating code for file")

        # Attributes
        d = {"ffcx_version": ffcx_version, "ufcx_version": ufcx_version}
        d["options"] = textwrap.indent(pprint.pformat(options), "//  ")
        extra_includes = []
        if "_Complex" in options["scalar_type"]:
            extra_includes += ["complex"]
        d["extra_includes"] = "\n".join(f"#include <{header}>" for header in extra_includes)

        code_pre = (file.declaration_pre.format_map(d),)

        # Format implementation code
        code_post = (file.declaration_post.format_map(d),)

        return code_pre, code_post
