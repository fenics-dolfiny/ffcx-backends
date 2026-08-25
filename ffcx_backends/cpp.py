"""Custom C++ FFCx backend."""

import functools
import logging
import pprint
import textwrap
from typing import ClassVar

import basix
import ffcx.codegeneration.lnodes as L  # noqa
from ffcx import __version__ as ffcx_version
from ffcx.codegeneration.backend import FFCXBackend
from ffcx.codegeneration.common import integral_data, template_keys
from ffcx.codegeneration.expression_generator import ExpressionGenerator
from ffcx.codegeneration.integral_generator import IntegralGenerator
from ffcx.ir.representation import ExpressionIR, FormIR, IntegralIR
from numpy import typing as npt

logger = logging.getLogger("ffcx")


class Formatter:
    """Format FFCx nodes into C++."""

    math_table: ClassVar[dict[str, str]] = {
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
    def build_initializer_lists(values: npt.NDArray) -> str:
        """Build initializer lists."""
        arr = "{"
        if len(values.shape) == 1:
            return "{" + ", ".join(str(v) for v in values) + "}"
        elif len(values.shape) > 1:
            arr += ",\n".join(Formatter.build_initializer_lists(v) for v in values)
        arr += "}"
        return arr

    def __init__(self, scalar_geometry: bool = False) -> None:
        """Initialise.

        Args:
            scalar_geometry: Force scalar type == geometry type in the kernel body.
        """
        self._scalar_type = "T"
        self._geometry_type = "U"
        if scalar_geometry:
            self._geometry_type = self._scalar_type

    def dtype_to_cpp_type(self, dtype: L.DataType) -> str:
        """Map L.DataType to C++ type."""
        if dtype == L.DataType.SCALAR:
            return self._scalar_type
        elif dtype == L.DataType.REAL:
            return self._geometry_type
        elif dtype == L.DataType.INT:
            return "std::int32_t"
        elif dtype == L.DataType.BOOL:
            return "bool"
        else:
            raise ValueError(f"Invalid datatype: {dtype}")

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

        typename = self.dtype_to_cpp_type(dtype)

        symbol = self(arr.symbol)
        dims = "".join([f"[{i}]" for i in arr.sizes])
        if arr.values is None:
            assert arr.const is False  # type: ignore[unreachable]
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
        typename = self.dtype_to_cpp_type(v.symbol.dtype)
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


class expression:  # noqa: N801
    """Expression generator."""

    factory = """
// Code for expression {factory_name}

template<typename T, typename U>
class {factory_name}
{{
public:
  {points}
  {value_shape}
  {original_coefficient_positions}
  {coefficient_names}
  {constant_names}

  static constexpr int num_coefficients = {num_coefficients};
  static constexpr int num_constants = {num_constants};
  static constexpr int rank = {rank};
  static constexpr int num_components = {num_components};
  static constexpr int entity_dimension = {entity_dimension};
  static constexpr int num_points = {num_points};

  std::uint64_t coordinate_element_hash {{ {coordinate_element_hash}ULL }};
  static void tabulate_tensor(T* RESTRICT A,
                              const T* RESTRICT w,
                              const T* RESTRICT c,
                              const U* RESTRICT coordinate_dofs,
                              const int* RESTRICT entity_local_index,
                              const uint8_t* RESTRICT quadrature_permutation,
                              void* custom_data)
  {{
  {tabulate_expression}
  }}
}};

// Alias name
template <typename T, typename U>
using {name_from_uflfile} = {factory_name}<T, U>;

// End of code for expression {factory_name}
"""

    @staticmethod
    def generator(ir: ExpressionIR, options: dict[str, int | float | npt.DTypeLike]) -> tuple[str,]:
        """Generate UFC code for an expression."""
        logger.info("Generating code for expression:")
        assert len(ir.expression.integrand) == 1, "Expressions only support single quadrature rule"
        points = next(iter(ir.expression.integrand))[1].points
        logger.info(f"--- points: {points}")
        factory_name = ir.expression.name
        logger.info(f"--- name: {factory_name}")

        backend = FFCXBackend(ir, options)
        eg = ExpressionGenerator(ir, backend)

        d: dict[str, str | int] = {}
        d["name_from_uflfile"] = ir.name_from_uflfile
        d["factory_name"] = factory_name
        parts = eg.generate()

        scalar_geometry = True if options.get("scalar_geometry") else False
        formatter = Formatter(scalar_geometry)
        d["tabulate_expression"] = formatter(parts)

        if len(ir.original_coefficient_positions) > 0:
            values = ", ".join(str(i) for i in ir.original_coefficient_positions)
            sizes = len(ir.original_coefficient_positions)
            d["original_coefficient_positions"] = (
                f"static constexpr int original_coefficient_positions[{sizes}] = {{{values}}};"
            )
        else:
            d["original_coefficient_positions"] = ""

        values = ", ".join(str(p) for p in points.flatten())
        sizes = points.size
        d["points"] = f"static constexpr double points[{sizes}] = {{{values}}};"

        if len(ir.expression.shape) > 0:
            values = ", ".join(str(i) for i in ir.expression.shape)
            sizes = len(ir.expression.shape)
            d["value_shape"] = f"static constexpr int value_shape[{sizes}] = {{{values}}};"
        else:
            d["value_shape"] = ""
        d["num_components"] = len(ir.expression.shape)
        d["num_coefficients"] = len(ir.expression.coefficient_numbering)
        d["num_constants"] = len(ir.constant_names)
        d["num_points"] = points.shape[0]
        d["entity_dimension"] = points.shape[1]

        d["rank"] = len(ir.expression.tensor_shape)

        if len(ir.coefficient_names) > 0:
            values = ", ".join(f'"{name}"' for name in ir.coefficient_names)
            sizes = len(ir.coefficient_names)
            d["coefficient_names"] = (
                f"static constexpr const char* coefficient_names[{sizes}] = {{{values}}};"
            )
        else:
            d["coefficient_names"] = ""

        if len(ir.constant_names) > 0:
            values = ", ".join(f'"{name}"' for name in ir.constant_names)
            sizes = len(ir.constant_names)
            d["constant_names"] = (
                f"static constexpr const char* constant_names[{sizes}] = {{{values}}};"
            )
        else:
            d["constant_names"] = ""

        d["coordinate_element_hash"] = f"{ir.expression.coordinate_element_hash}"

        # Format implementation code
        assert set(d.keys()) == template_keys(expression.factory)
        implementation = expression.factory.format_map(d)

        return (implementation,)


class integral:  # noqa: N801
    """Integral generator."""

    factory = """
// Code for integral {factory_name}

template <typename T, typename U>
class {factory_name}
{{
public:
    // Kernel
    static void tabulate_tensor(T* RESTRICT A,
                                const T* RESTRICT w,
                                const T* RESTRICT c,
                                const U* RESTRICT coordinate_dofs,
                                const std::int32_t* RESTRICT entity_local_index,
                                const std::uint8_t* RESTRICT quadrature_permutation)
    {{
{tabulate_tensor}
    }}

    // Address helper
    static uintptr_t tabulate_tensor_addr()
    {{
        return reinterpret_cast<uintptr_t>(&{factory_name}::tabulate_tensor);
    }}

    // Data
    static constexpr int enabled_coefficients[] = {enabled_coefficients_init};
    static constexpr bool needs_facet_permutations = {needs_facet_permutations};
}};

// End of code for integral {factory_name}
"""

    @staticmethod
    def generator(
        ir: IntegralIR, domain: basix.CellType, options: dict[str, int | float | npt.DTypeLike]
    ) -> tuple[str,]:
        """Generate C++ code for an integral."""
        logger.info("Generating code for integral:")
        logger.info(f"--- type: {ir.expression.integral_type}")
        logger.info(f"--- name: {ir.expression.name}")

        factory_name = f"{ir.expression.name}_{domain.name}"

        # Create FFCx backend
        backend = FFCXBackend(ir, options)

        # Configure kernel generator
        ig = IntegralGenerator(ir, backend)

        # Generate code ast for the tabulate_tensor body
        parts = ig.generate(domain)

        # Format code as string
        scalar_geometry = True if options.get("scalar_geometry") else False
        formatter = Formatter(scalar_geometry)
        body = formatter(parts)

        # Generate generic FFCx code snippets and add specific parts
        code = {}
        code["class_type"] = ir.expression.integral_type + "_integral"
        code["name"] = ir.expression.name

        vals = ", ".join("1" if i else "0" for i in ir.enabled_coefficients)
        # Set to {0} to avoid empty array declaration
        code["enabled_coefficients"] = f"{{{vals}}}" if vals != "" else "{0}"
        code["needs_facet_permutations"] = (
            "true" if ir.expression.needs_facet_permutations else "false"
        )

        code["tabulate_tensor"] = body

        # Format factory with all values
        implementation = integral.factory.format(
            factory_name=factory_name,
            enabled_coefficients_init=code["enabled_coefficients"],
            tabulate_tensor=code["tabulate_tensor"],
            needs_facet_permutations=code["needs_facet_permutations"],
            scalar_type=options["scalar_type"],
            geom_type=options["scalar_type"],
            np_scalar_type=options["scalar_type"],
            coordinate_element=ir.expression.coordinate_element_hash,
        )
        return (implementation,)


class form:  # noqa: N801
    """Form generator."""

    factory = r"""
// Code for form {factory_name}

template <typename T, typename U>
class {factory_name}
{{
public:
    // Signature and rank
    static constexpr const char* signature = {signature};
    static constexpr int rank = {rank};

    // Coefficients
    static constexpr int num_coefficients = {num_coefficients};
    {original_coefficient_positions_member}
    {coefficient_names_member}

    // Constants
    static constexpr int num_constants = {num_constants};
    {constant_ranks_member}
    {constant_shapes_members}
    {constant_names_member}

    // Finite elements
    {finite_element_hashes_member}

    // Integrals
    {form_integral_ids_member}
    {form_integral_offsets_member}

    // Integral type aliases for convenient access
    {integral_type_aliases}
}};

// Alias name
template <typename T, typename U>
using {name_from_uflfile} = {factory_name}<T, U>;

// End of code for form {factory_name}
"""

    @staticmethod
    def generator(ir: FormIR, options: dict[str, int | float | npt.DTypeLike]) -> tuple[str,]:
        """Generate C++ code for a form."""
        logger.info("Generating code for form:")
        logger.info(f"--- rank: {ir.rank}")
        logger.info(f"--- name: {ir.name}")

        d: dict[str, int | str] = {}
        d["factory_name"] = ir.name
        d["name_from_uflfile"] = ir.name_from_uflfile
        d["signature"] = f'"{ir.signature}"'
        d["rank"] = ir.rank
        d["num_coefficients"] = ir.num_coefficients

        # Original coefficient positions (inline member)
        if len(ir.original_coefficient_positions) > 0:
            values = ", ".join(str(i) for i in ir.original_coefficient_positions)
            sizes = len(ir.original_coefficient_positions)
            d["original_coefficient_positions_member"] = (
                f"static constexpr int original_coefficient_positions[{sizes}] = {{{values}}};"
            )
        else:
            d["original_coefficient_positions_member"] = ""

        # Coefficient names (inline member)
        if len(ir.coefficient_names) > 0:
            values = ", ".join(f'"{name}"' for name in ir.coefficient_names)
            sizes = len(ir.coefficient_names)
            d["coefficient_names_member"] = (
                f"static constexpr const char* coefficient_name_map[{sizes}] = {{{values}}};"
            )
        else:
            d["coefficient_names_member"] = ""

        # Constants (inline members)
        d["num_constants"] = ir.num_constants
        if ir.num_constants > 0:
            # Constant ranks
            d["constant_ranks_member"] = (
                f"static constexpr int constant_ranks[{ir.num_constants}] = "
                f"{{{str(ir.constant_ranks)[1:-1]}}};"
            )

            # Constant shapes (individual arrays)
            shapes = []
            for i, shape in enumerate(ir.constant_shapes):
                if len(shape) > 0:
                    shapes.append(
                        f"static constexpr int constant_shape_{i}[{len(shape)}] = "
                        f"{{{str(shape)[1:-1]}}};"
                    )

            # Constant shapes pointer array
            names = [
                f"constant_shape_{i}" if rank > 0 else "nullptr"
                for i, rank in enumerate(ir.constant_ranks)
            ]
            shapes.append(
                f"static constexpr const int* constant_shapes[{ir.num_constants}] = "
                f"{{{', '.join(names)}}};"
            )
            d["constant_shapes_members"] = "\n    ".join(shapes)

            # Constant names
            values = ", ".join(f'"{name}"' for name in ir.constant_names)
            d["constant_names_member"] = (
                f"static constexpr const char* constant_name_map[{ir.num_constants}] = "
                f"{{{values}}};"
            )
        else:
            d["constant_ranks_member"] = ""
            d["constant_shapes_members"] = ""
            d["constant_names_member"] = ""

        # Finite element hashes (inline member)
        if len(ir.finite_element_hashes) > 0:
            values = ", ".join(f"{0 if el is None else el}ULL" for el in ir.finite_element_hashes)
            sizes = len(ir.finite_element_hashes)
            d["finite_element_hashes_member"] = (
                f"static constexpr std::uint64_t finite_element_hashes[{sizes}] = {{{values}}};"
            )
        else:
            d["finite_element_hashes_member"] = ""

        integrals = integral_data(ir)

        # Integral IDs and offsets (inline members)
        if len(integrals.names) > 0:
            # Integral IDs
            values = ", ".join(
                f"{i}"
                for i, domains in zip(integrals.ids, integrals.domains, strict=True)
                for _ in domains
            )
            sizes = sum(len(domains) for domains in integrals.domains)
            d["form_integral_ids_member"] = (
                f"static constexpr int form_integral_ids[{sizes}] = {{{values}}};"
            )

            # Generate type aliases for integral classes using domain (cell type)
            aliases = []
            for name, domains, ids in zip(
                integrals.names, integrals.domains, integrals.ids, strict=True
            ):
                for domain in domains:
                    class_name = f"{name}_{domain.name}"
                    # Create alias using domain name and subdomain ID
                    # Handle negative IDs (typically -1 for default/all subdomains)
                    if ids == -1:
                        alias_name = f"integral_{domain.name}_all"
                    elif ids > 0:
                        alias_name = f"integral_{domain.name}_id{ids}"
                    else:
                        raise ValueError(f"Invalid integral ID: {ids}")
                    alias = f"using {alias_name} = {class_name}<T, U>;"
                    aliases.append(alias)

            d["integral_type_aliases"] = "\n    ".join(aliases) if aliases else "// No integrals"
        else:
            d["form_integral_ids_member"] = ""
            d["integral_type_aliases"] = "// No integrals"

        # Integral offsets
        sizes = len(integrals.offsets)
        values = ", ".join(str(i) for i in integrals.offsets)
        d["form_integral_offsets_member"] = (
            f"static constexpr int form_integral_offsets[{sizes}] = {{{values}}};"
        )

        # Format implementation code
        assert set(d.keys()) == template_keys(form.factory)
        implementation = form.factory.format_map(d)

        return (implementation,)


class file:  # noqa: N801
    """File generator."""

    suffixes = (".hpp",)

    declaration_pre = """
// This code conforms was automatically generated by FFCx version {ffcx_version}.
//
// This code was generated with the following options:
//
{options}

#pragma once

#include <complex>
#include <cstdint>
#include <cmath>
#include <vector>

#if defined(_MSC_VER)
#   define RESTRICT __restrict
#else
#   define RESTRICT __restrict__
#endif
    """

    declaration_post = """
    """

    @staticmethod
    def generator(
        options: dict[str, str | int | float | npt.DTypeLike],
    ) -> tuple[tuple[str,], tuple[str,]]:
        """Generate UFC code for file output."""
        logger.info("Generating code for file")

        # Attributes
        d = {"ffcx_version": ffcx_version}
        d["options"] = textwrap.indent(pprint.pformat(options), "//  ")
        extra_includes = []
        scalar_type = options.get("scalar_type", "")
        assert isinstance(scalar_type, str)
        if "_Complex" in scalar_type:
            extra_includes += ["complex"]
        d["extra_includes"] = "\n".join(f"#include <{header}>" for header in extra_includes)

        code_pre = (file.declaration_pre.format_map(d),)

        # Format implementation code
        code_post = (file.declaration_post.format_map(d),)

        return code_pre, code_post
