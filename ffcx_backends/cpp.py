import logging
import pprint
import string
import textwrap

import basix
import ffcx.codegeneration.lnodes as L
import numpy as np
from ffcx import __version__ as FFCX_VERSION
from ffcx.codegeneration import __version__ as UFC_VERSION
from ffcx.codegeneration.backend import FFCXBackend
from ffcx.codegeneration.expression_generator import ExpressionGenerator
from ffcx.codegeneration.integral_generator import IntegralGenerator
from ffcx.codegeneration.utils import dtype_to_c_type, dtype_to_scalar_dtype
from ffcx.ir.representation import IntegralIR

logger = logging.getLogger("ffcx")


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
            arr += ",\n".join(Formatter.build_initializer_lists(v)
                              for v in values)
        arr += "}"
        return arr

    def __init__(self, scalar) -> None:
        """Initialise."""
        self.scalar_type = "T"
        self.real_type = "U"

    def format_statement_list(self, slist) -> str:
        """Format statement list."""
        return "".join(self.format(s) for s in slist.statements)

    def format_section(self, section) -> str:
        """Format a section."""
        # add new line before section
        comments = "// ------------------------ \n"
        comments += "// Section: " + section.name + "\n"
        comments += "// Inputs: " + \
            ", ".join(w.name for w in section.input) + "\n"
        comments += "// Outputs: " + \
            ", ".join(w.name for w in section.output) + "\n"
        declarations = "".join(self.format(s) for s in section.declarations)

        body = ""
        if len(section.statements) > 0:
            declarations += "{\n  "
            body = "".join(self.format(s) for s in section.statements)
            body = body.replace("\n", "\n  ")
            body = body[:-2] + "}\n"

        body += "// ------------------------ \n"
        return comments + declarations + body

    def format_comment(self, c) -> str:
        """Format a comment."""
        return "// " + c.comment + "\n"

    def format_array_decl(self, arr) -> str:
        """Format an array declaration."""
        dtype = arr.symbol.dtype
        assert dtype is not None

        if dtype == L.DataType.SCALAR:
            typename = self.scalar_type
        elif dtype == L.DataType.REAL:
            typename = self.real_type
        elif dtype == L.DataType.INT:
            typename = "int"
        else:
            raise ValueError("Invalid datatype")

        symbol = self.format(arr.symbol)
        dims = "".join([f"[{i}]" for i in arr.sizes])
        if arr.values is None:
            assert arr.const is False
            return f"{typename} {symbol}{dims};\n"

        vals = Formatter.build_initializer_lists(arr.values)
        cstr = "static const " if arr.const else ""
        return f"{cstr}{typename} {symbol}{dims} = {vals};\n"

    def format_array_access(self, arr) -> str:
        """Format array access."""
        name = self.format(arr.array)
        indices = f"[{']['.join(self.format(i) for i in arr.indices)}]"
        return f"{name}{indices}"

    def format_multi_index(self, index) -> str:
        """Format a multi-index."""
        return self.format(index.global_index)

    def format_variable_decl(self, v) -> str:
        """Format a variable declaration."""
        val = self.format(v.value)
        symbol = self.format(v.symbol)
        assert v.symbol.dtype
        # TODO: move to _dtype_to_name
        typename = ""  # tmp fix!!
        if v.symbol.dtype == L.DataType.SCALAR:
            typename = self.scalar_type
        elif v.symbol.dtype == L.DataType.REAL:
            typename = self.real_type
        elif v.symbol.dtype == L.DataType.INT:
            typename = "std::int32_t"
        elif v.symbol.dtype == L.DataType.BOOL:
            typename = "bool"
        return f"{typename} {symbol} = {val};\n"

    def format_nary_op(self, oper) -> str:
        """Format an n-argument operation."""
        # Format children
        args = [self.format(arg) for arg in oper.args]

        # Apply parentheses
        for i in range(len(args)):
            if oper.args[i].precedence >= oper.precedence:
                args[i] = "(" + args[i] + ")"

        # Return combined string
        return f" {oper.op} ".join(args)

    def format_binary_op(self, oper) -> str:
        """Format a binary operation."""
        # Format children
        lhs = self.format(oper.lhs)
        rhs = self.format(oper.rhs)

        # Apply parentheses
        if oper.lhs.precedence >= oper.precedence:
            lhs = f"({lhs})"
        if oper.rhs.precedence >= oper.precedence:
            rhs = f"({rhs})"

        # Return combined string
        return f"{lhs} {oper.op} {rhs}"

    def format_neg(self, val) -> str:
        """Format negation."""
        arg = self.format(val.arg)
        return f"-{arg}"

    def format_not(self, val) -> str:
        """Format 'not' statement."""
        arg = self.format(val.arg)
        return f"{val.op}({arg})"

    def format_literal_float(self, val) -> str:
        """Format a literal float number."""
        return f"{val.value}"

    def format_literal_int(self, val) -> str:
        """Format a literal int number."""
        return f"{val.value}"

    def format_for_range(self, r) -> str:
        """Format a loop over a range."""
        begin = self.format(r.begin)
        end = self.format(r.end)
        index = self.format(r.index)
        output = f"for (int {index} = {begin}; {index} < {end}; ++{index})\n"
        output += "{\n"
        body = self.format(r.body)
        for line in body.split("\n"):
            if len(line) > 0:
                output += f"  {line}\n"
        output += "}\n"
        return output

    def format_statement(self, s) -> str:
        """Format a statement."""
        return self.format(s.expr)

    def format_assign(self, expr) -> str:
        """Format an assignment statement."""
        rhs = self.format(expr.rhs)
        lhs = self.format(expr.lhs)
        return f"{lhs} {expr.op} {rhs};\n"

    def format_conditional(self, s) -> str:
        """Format a conditional."""
        # Format children
        c = self.format(s.condition)
        t = self.format(s.true)
        f = self.format(s.false)

        # Apply parentheses
        if s.condition.precedence >= s.precedence:
            c = "(" + c + ")"
        if s.true.precedence >= s.precedence:
            t = "(" + t + ")"
        if s.false.precedence >= s.precedence:
            f = "(" + f + ")"

        # Return combined string
        return c + " ? " + t + " : " + f

    def format_symbol(self, s) -> str:
        """Format a symbol."""
        return f"{s.name}"

    def format_math_function(self, c) -> str:
        """Format a math function."""
        # Get a function from the table, if available, else just use bare name
        func = Formatter.math_table.get(c.function, c.function)
        args = ", ".join(self.format(arg) for arg in c.args)
        return f"{func}({args})"

    c_impl = {
        "Section": format_section,
        "StatementList": format_statement_list,
        "Comment": format_comment,
        "ArrayDecl": format_array_decl,
        "ArrayAccess": format_array_access,
        "MultiIndex": format_multi_index,
        "VariableDecl": format_variable_decl,
        "ForRange": format_for_range,
        "Statement": format_statement,
        "Assign": format_assign,
        "AssignAdd": format_assign,
        "Product": format_nary_op,
        "Neg": format_neg,
        "Sum": format_nary_op,
        "Add": format_binary_op,
        "Sub": format_binary_op,
        "Mul": format_binary_op,
        "Div": format_binary_op,
        "Not": format_not,
        "LiteralFloat": format_literal_float,
        "LiteralInt": format_literal_int,
        "Symbol": format_symbol,
        "Conditional": format_conditional,
        "MathFunction": format_math_function,
        "And": format_binary_op,
        "Or": format_binary_op,
        "NE": format_binary_op,
        "EQ": format_binary_op,
        "GE": format_binary_op,
        "LE": format_binary_op,
        "GT": format_binary_op,
        "LT": format_binary_op,
    }

    def format(self, s) -> str:
        """Formatting function."""
        name = s.__class__.__name__
        try:
            return self.c_impl[name](self, s)
        except KeyError:
            raise RuntimeError("Unknown statement: ", name) from None


class expression:
    declaration = """
extern ufcx_expression {factory_name};

// Helper used to create expression using name which was given to the
// expression in the UFL file.
// This helper is called in user c++ code.
//
extern ufcx_expression* {name_from_uflfile};
"""

    factory = """
// Code for expression {factory_name}

void tabulate_tensor_{factory_name}({scalar_type}* restrict A,
                                    const {scalar_type}* restrict w,
                                    const {scalar_type}* restrict c,
                                    const {geom_type}* restrict coordinate_dofs,
                                    const int* restrict entity_local_index,
                                    const uint8_t* restrict quadrature_permutation)
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
        assert len(
            ir.expression.integrand) == 1, "Expressions only support single quadrature rule"
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

        CF = Formatter(options["scalar_type"])
        d["tabulate_expression"] = CF.format(parts)

        if len(ir.original_coefficient_positions) > 0:
            d["original_coefficient_positions"] = f"original_coefficient_positions_{factory_name}"
            sizes = len(ir.original_coefficient_positions)
            values = ", ".join(str(i)
                               for i in ir.original_coefficient_positions)
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
        d["geom_type"] = dtype_to_c_type(
            dtype_to_scalar_dtype(options["scalar_type"]))
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
        fields = [fname for _, fname, _, _ in string.Formatter().parse(
            expression.factory) if fname]
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

        factory_name = ir.expression.name

        # Format declaration
        declaration = integral.declaration.format(factory_name=factory_name)

        # Create FFCx C backend
        backend = FFCXBackend(ir, options)

        # Configure kernel generator
        ig = IntegralGenerator(ir, backend)

        # Generate code ast for the tabulate_tensor body
        parts = ig.generate(domain)

        # Format code as string
        CF = Formatter(options["scalar_type"])
        body = CF.format(parts)

        # Generate generic FFCx code snippets and add specific parts
        code = {}
        code["class_type"] = ir.expression.integral_type + "_integral"
        code["name"] = ir.expression.name

        vals = ", ".join(
            "true" if i else "false" for i in ir.enabled_coefficients)
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
    extern ufcx_form {factory_name};

    // Helper used to create form using name which was given to the
    // form in the UFL file.
    // This helper is called in user c++ code.
    //
    extern ufcx_form* {name_from_uflfile};

    """

    factory = """
    // Code for form {factory_name}

    // TODO: that correct?
    {original_coefficient_position_init}
    {finite_element_hashes_init}
    {form_integral_offsets_init}
    {form_integrals_init}
    {form_integral_ids_init}

    {coefficient_names_init}
    {constant_names_init}
    {constant_ranks_init}
    {constant_shapes_init}

    {name_from_uflfile}::signature ={signature};
    {name_from_uflfile}::rank = {rank};

    {name_from_uflfile}::num_coefficients = {num_coefficients};
    {name_from_uflfile}::original_coefficient_positions = {original_coefficient_positions};
    {name_from_uflfile}::coefficient_name_map = {coefficient_names};

    {name_from_uflfile}::num_constants = {num_constants};
    {name_from_uflfile}::constant_ranks = {constant_ranks};
    {name_from_uflfile}::constant_shapes = {constant_shapes};
    {name_from_uflfile}::constant_name_map = {constant_names};

    {name_from_uflfile}::finite_element_hashes = {finite_element_hashes},

    {name_from_uflfile}::form_integrals = {form_integrals};
    {name_from_uflfile}::form_integral_ids = {form_integral_ids};
    {name_from_uflfile}::form_integral_offsets = form_integral_offsets_{factory_name};

    // Alias name
    using {name_from_uflfile} = {factory_name};

    // End of code for form {factory_name}
    """

    @staticmethod
    def generator(ir, options):
        """Generate UFC code for a form."""
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
            values = ", ".join(str(i)
                               for i in ir.original_coefficient_positions)
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
                f"static const char* coefficient_names_{ir.name}[{sizes}] = {{{values}}};"
            )
            d["coefficient_names"] = f"coefficient_names_{ir.name}"
        else:
            d["coefficient_names_init"] = ""
            d["coefficient_names"] = "NULL"

        d["num_constants"] = ir.num_constants
        if ir.num_constants > 0:
            d["constant_ranks_init"] = (
                f"static const int constant_ranks_{ir.name}[{ir.num_constants}] = "
                f"{{{str(ir.constant_ranks)[1:-1]}}};"
            )
            d["constant_ranks"] = f"constant_ranks_{ir.name}"

            shapes = [
                f"static const int constant_shapes_{ir.name}_{i}[{len(shape)}] = "
                f"{{{str(shape)[1:-1]}}};"
                for i, shape in enumerate(ir.constant_shapes)
                if len(shape) > 0
            ]
            names = [f"constant_shapes_{ir.name}_{i}" for i in range(
                ir.num_constants)]
            shapes1 = f"static const int* constant_shapes_{ir.name}[{ir.num_constants}] = " + "{"
            for rank, name in zip(ir.constant_ranks, names):
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
                f"static const char* constant_names_{ir.name}[{sizes}] = {{{values}}};"
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

        integrals = []
        integral_ids = []
        integral_offsets = [0]
        integral_domains = []
        # Note: the order of this list is defined by the enum ufcx_integral_type in ufcx.h
        for itg_type in ("cell", "exterior_facet", "interior_facet", "vertex", "ridge"):
            unsorted_integrals = []
            unsorted_ids = []
            unsorted_domains = []
            for name, domains, id in zip(
                ir.integral_names[itg_type],
                ir.integral_domains[itg_type],
                ir.subdomain_ids[itg_type],
            ):
                unsorted_integrals += [f"&{name}"]
                unsorted_ids += [id]
                unsorted_domains += [domains]

            id_sort = np.argsort(unsorted_ids)
            integrals += [unsorted_integrals[i] for i in id_sort]
            integral_ids += [unsorted_ids[i] for i in id_sort]
            integral_domains += [unsorted_domains[i] for i in id_sort]

            integral_offsets.append(sum(len(d) for d in integral_domains))

        if len(integrals) > 0:
            sizes = sum(len(domains) for domains in integral_domains)
            values = ", ".join(
                [
                    f"{i}_{domain.name}"
                    for i, domains in zip(integrals, integral_domains)
                    for domain in domains
                ]
            )
            d["form_integrals_init"] = (
                f"static ufcx_integral* form_integrals_{ir.name}[{sizes}] = {{{values}}};"
            )
            d["form_integrals"] = f"form_integrals_{ir.name}"
            values = ", ".join(
                f"{i}" for i, domains in zip(integral_ids, integral_domains) for _ in domains
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

        sizes = len(integral_offsets)
        values = ", ".join(str(i) for i in integral_offsets)
        d["form_integral_offsets_init"] = (
            f"int form_integral_offsets_{ir.name}[{sizes}] = {{{values}}};"
        )

        fields = [fname for _, fname, _,
                  _ in string.Formatter().parse(form.factory) if fname]
        assert set(fields) == set(d.keys()), (
            "Mismatch between keys in template and in formatting dict"
        )

        # Format declaration
        declaration = form.declaration.format(
            factory_name=d["factory_name"], name_from_uflfile=d["name_from_uflfile"]
        )

        return (declaration,)


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
        d = {"ffcx_version": FFCX_VERSION, "ufcx_version": UFC_VERSION}
        d["options"] = textwrap.indent(pprint.pformat(options), "//  ")
        extra_includes = []
        if "_Complex" in options["scalar_type"]:
            extra_includes += ["complex"]
        d["extra_includes"] = "\n".join(
            f"#include <{header}>" for header in extra_includes)

        code_pre = (file.declaration_pre.format_map(d),)

        # Format implementation code
        code_post = (file.declaration_post.format_map(d),)

        return code_pre, code_post
