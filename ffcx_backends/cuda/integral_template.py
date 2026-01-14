"""Code generation strings for an integral."""

factory = """
// Code for integral {factory_name}

extern "C" __global__
void tabulate_tensor_{factory_name}({scalar_type}* restrict A,
                                    const {scalar_type}* restrict w,
                                    const {scalar_type}* restrict c,
                                    const {geom_type}* restrict coordinate_dofs,
                                    const int* restrict entity_local_index,
                                    const uint8_t* restrict quadrature_permutation,
                                    void* custom_data)
{{
{tabulate_tensor}
}}

// end code for integral {factory_name}
"""
