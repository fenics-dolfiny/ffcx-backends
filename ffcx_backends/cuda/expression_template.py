"""Code generation strings for an expression."""

# Called by name, see `ffcx.codegeneration.C.formatter`, from the kernels that
# hold the interpolation, which are generated after it in the same file.
interpolation_kernel = """
// Code for interpolation kernel {factory_name}

static __device__
void tabulate_tensor_{factory_name}({scalar_type}* restrict A,
                                    const {scalar_type}* restrict w,
                                    const {scalar_type}* restrict c,
                                    const {geom_type}* restrict coordinate_dofs,
                                    const int* restrict entity_local_index,
                                    const uint8_t* restrict quadrature_permutation,
                                    void* custom_data)
{{
{tabulate_expression}
}}

// End of code for interpolation kernel {factory_name}
"""
