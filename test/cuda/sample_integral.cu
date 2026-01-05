
    #define alignas(x)
    #define restrict __restrict__
    
    typedef unsigned char uint8_t;
    typedef unsigned int uint32_t;
    typedef double ufc_scalar_t;
    
    extern "C" __global__
    void tabulate_tensor_integral_f47b8d1b79d48a8e8a81d53d75ec75321bc46ed1_tetrahedron(double* restrict A,
                                        const double* restrict w,
                                        const double* restrict c,
                                        const double* restrict coordinate_dofs,
                                        const int* restrict entity_local_index,
                                        const uint8_t* restrict quadrature_permutation
                                        )
{
// Quadrature rules
static const double weights_421[1] = {0.1666666666666667};
// Precomputed values of basis functions and precomputations
// FE* dimensions: [permutation][entities][points][dofs]
static const double FE0_C0_D100_Q421[1][1][1][4] = {{{{-1.0, 1.0, 0.0, 0.0}}}};
static const double FE1_C1_D010_Q421[1][1][1][4] = {{{{-1.0, 0.0, 1.0, 0.0}}}};
static const double FE1_C2_D001_Q421[1][1][1][4] = {{{{-1.0, 0.0, 0.0, 1.0}}}};
// ------------------------ 
// Section: Jacobian
// Inputs: FE1_C2_D001_Q421, FE1_C1_D010_Q421, coordinate_dofs, FE0_C0_D100_Q421
// Outputs: J0_c5, J0_c4, J0_c7, J0_c8, J0_c1, J0_c6, J0_c3, J0_c2, J0_c0
double J0_c4 = 0.0;
double J0_c8 = 0.0;
double J0_c5 = 0.0;
double J0_c7 = 0.0;
double J0_c0 = 0.0;
double J0_c3 = 0.0;
double J0_c6 = 0.0;
double J0_c1 = 0.0;
double J0_c2 = 0.0;
{
  for (int ic = 0; ic < 4; ++ic)
  {
    J0_c4 += coordinate_dofs[(ic) * 3 + 1] * FE1_C1_D010_Q421[0][0][0][ic];
    J0_c8 += coordinate_dofs[(ic) * 3 + 2] * FE1_C2_D001_Q421[0][0][0][ic];
    J0_c5 += coordinate_dofs[(ic) * 3 + 1] * FE1_C2_D001_Q421[0][0][0][ic];
    J0_c7 += coordinate_dofs[(ic) * 3 + 2] * FE1_C1_D010_Q421[0][0][0][ic];
    J0_c0 += coordinate_dofs[(ic) * 3] * FE0_C0_D100_Q421[0][0][0][ic];
    J0_c3 += coordinate_dofs[(ic) * 3 + 1] * FE0_C0_D100_Q421[0][0][0][ic];
    J0_c6 += coordinate_dofs[(ic) * 3 + 2] * FE0_C0_D100_Q421[0][0][0][ic];
    J0_c1 += coordinate_dofs[(ic) * 3] * FE1_C1_D010_Q421[0][0][0][ic];
    J0_c2 += coordinate_dofs[(ic) * 3] * FE1_C2_D001_Q421[0][0][0][ic];
  }
}
// ------------------------ 
double sp_421_0 = J0_c4 * J0_c8;
double sp_421_1 = J0_c5 * J0_c7;
double sp_421_2 = -sp_421_1;
double sp_421_3 = sp_421_0 + sp_421_2;
double sp_421_4 = J0_c0 * sp_421_3;
double sp_421_5 = J0_c3 * J0_c8;
double sp_421_6 = J0_c5 * J0_c6;
double sp_421_7 = -sp_421_6;
double sp_421_8 = sp_421_5 + sp_421_7;
double sp_421_9 = -J0_c1;
double sp_421_10 = sp_421_8 * sp_421_9;
double sp_421_11 = sp_421_4 + sp_421_10;
double sp_421_12 = J0_c3 * J0_c7;
double sp_421_13 = J0_c4 * J0_c6;
double sp_421_14 = -sp_421_13;
double sp_421_15 = sp_421_12 + sp_421_14;
double sp_421_16 = J0_c2 * sp_421_15;
double sp_421_17 = sp_421_11 + sp_421_16;
double sp_421_18 = sp_421_3 / sp_421_17;
double sp_421_19 = -J0_c8;
double sp_421_20 = J0_c3 * sp_421_19;
double sp_421_21 = sp_421_6 + sp_421_20;
double sp_421_22 = sp_421_21 / sp_421_17;
double sp_421_23 = sp_421_15 / sp_421_17;
double sp_421_24 = 5.0 * sp_421_18;
double sp_421_25 = 5.0 * sp_421_22;
double sp_421_26 = 5.0 * sp_421_23;
double sp_421_27 = sp_421_24 * sp_421_18;
double sp_421_28 = sp_421_25 * sp_421_18;
double sp_421_29 = sp_421_26 * sp_421_18;
double sp_421_30 = sp_421_24 * sp_421_22;
double sp_421_31 = sp_421_25 * sp_421_22;
double sp_421_32 = sp_421_26 * sp_421_22;
double sp_421_33 = sp_421_24 * sp_421_23;
double sp_421_34 = sp_421_25 * sp_421_23;
double sp_421_35 = sp_421_26 * sp_421_23;
double sp_421_36 = J0_c2 * J0_c7;
double sp_421_37 = J0_c8 * sp_421_9;
double sp_421_38 = sp_421_36 + sp_421_37;
double sp_421_39 = sp_421_38 / sp_421_17;
double sp_421_40 = J0_c0 * J0_c8;
double sp_421_41 = -J0_c2;
double sp_421_42 = J0_c6 * sp_421_41;
double sp_421_43 = sp_421_40 + sp_421_42;
double sp_421_44 = sp_421_43 / sp_421_17;
double sp_421_45 = J0_c1 * J0_c6;
double sp_421_46 = J0_c0 * J0_c7;
double sp_421_47 = -sp_421_46;
double sp_421_48 = sp_421_45 + sp_421_47;
double sp_421_49 = sp_421_48 / sp_421_17;
double sp_421_50 = 5.0 * sp_421_39;
double sp_421_51 = 5.0 * sp_421_44;
double sp_421_52 = 5.0 * sp_421_49;
double sp_421_53 = sp_421_50 * sp_421_39;
double sp_421_54 = sp_421_51 * sp_421_39;
double sp_421_55 = sp_421_52 * sp_421_39;
double sp_421_56 = sp_421_50 * sp_421_44;
double sp_421_57 = sp_421_51 * sp_421_44;
double sp_421_58 = sp_421_52 * sp_421_44;
double sp_421_59 = sp_421_50 * sp_421_49;
double sp_421_60 = sp_421_51 * sp_421_49;
double sp_421_61 = sp_421_52 * sp_421_49;
double sp_421_62 = sp_421_53 + sp_421_27;
double sp_421_63 = sp_421_54 + sp_421_28;
double sp_421_64 = sp_421_55 + sp_421_29;
double sp_421_65 = sp_421_56 + sp_421_30;
double sp_421_66 = sp_421_57 + sp_421_31;
double sp_421_67 = sp_421_58 + sp_421_32;
double sp_421_68 = sp_421_33 + sp_421_59;
double sp_421_69 = sp_421_34 + sp_421_60;
double sp_421_70 = sp_421_35 + sp_421_61;
double sp_421_71 = J0_c1 * J0_c5;
double sp_421_72 = J0_c2 * J0_c4;
double sp_421_73 = -sp_421_72;
double sp_421_74 = sp_421_71 + sp_421_73;
double sp_421_75 = sp_421_74 / sp_421_17;
double sp_421_76 = J0_c2 * J0_c3;
double sp_421_77 = J0_c0 * J0_c5;
double sp_421_78 = -sp_421_77;
double sp_421_79 = sp_421_76 + sp_421_78;
double sp_421_80 = sp_421_79 / sp_421_17;
double sp_421_81 = J0_c0 * J0_c4;
double sp_421_82 = J0_c1 * J0_c3;
double sp_421_83 = -sp_421_82;
double sp_421_84 = sp_421_81 + sp_421_83;
double sp_421_85 = sp_421_84 / sp_421_17;
double sp_421_86 = 5.0 * sp_421_75;
double sp_421_87 = 5.0 * sp_421_80;
double sp_421_88 = 5.0 * sp_421_85;
double sp_421_89 = sp_421_86 * sp_421_75;
double sp_421_90 = sp_421_87 * sp_421_75;
double sp_421_91 = sp_421_88 * sp_421_75;
double sp_421_92 = sp_421_86 * sp_421_80;
double sp_421_93 = sp_421_87 * sp_421_80;
double sp_421_94 = sp_421_88 * sp_421_80;
double sp_421_95 = sp_421_86 * sp_421_85;
double sp_421_96 = sp_421_87 * sp_421_85;
double sp_421_97 = sp_421_88 * sp_421_85;
double sp_421_98 = sp_421_62 + sp_421_89;
double sp_421_99 = sp_421_63 + sp_421_90;
double sp_421_100 = sp_421_64 + sp_421_91;
double sp_421_101 = sp_421_65 + sp_421_92;
double sp_421_102 = sp_421_66 + sp_421_93;
double sp_421_103 = sp_421_67 + sp_421_94;
double sp_421_104 = sp_421_68 + sp_421_95;
double sp_421_105 = sp_421_69 + sp_421_96;
double sp_421_106 = sp_421_70 + sp_421_97;
double sp_421_107 = 0.005 * sp_421_98;
double sp_421_108 = 0.005 * sp_421_99;
double sp_421_109 = 0.005 * sp_421_100;
double sp_421_110 = 0.005 * sp_421_101;
double sp_421_111 = 0.005 * sp_421_102;
double sp_421_112 = 0.005 * sp_421_103;
double sp_421_113 = 0.005 * sp_421_104;
double sp_421_114 = 0.005 * sp_421_105;
double sp_421_115 = 0.005 * sp_421_106;
double sp_421_116 = fabs(sp_421_17);
double sp_421_117 = sp_421_107 * sp_421_116;
double sp_421_118 = sp_421_108 * sp_421_116;
double sp_421_119 = sp_421_109 * sp_421_116;
double sp_421_120 = sp_421_110 * sp_421_116;
double sp_421_121 = sp_421_111 * sp_421_116;
double sp_421_122 = sp_421_112 * sp_421_116;
double sp_421_123 = sp_421_113 * sp_421_116;
double sp_421_124 = sp_421_114 * sp_421_116;
double sp_421_125 = sp_421_115 * sp_421_116;
for (int iq = 0; iq < 1; ++iq)
{
  // ------------------------ 
  // Section: Intermediates
  // Inputs: 
  // Outputs: fw0, fw1, fw2, fw3, fw4, fw5, fw6, fw7, fw8
  double fw0 = 0;
  double fw1 = 0;
  double fw2 = 0;
  double fw3 = 0;
  double fw4 = 0;
  double fw5 = 0;
  double fw6 = 0;
  double fw7 = 0;
  double fw8 = 0;
  {
    fw0 = sp_421_117 * weights_421[iq];
    fw1 = sp_421_118 * weights_421[iq];
    fw2 = sp_421_119 * weights_421[iq];
    fw3 = sp_421_120 * weights_421[iq];
    fw4 = sp_421_121 * weights_421[iq];
    fw5 = sp_421_122 * weights_421[iq];
    fw6 = sp_421_123 * weights_421[iq];
    fw7 = sp_421_124 * weights_421[iq];
    fw8 = sp_421_125 * weights_421[iq];
  }
  // ------------------------ 
  // ------------------------ 
  // Section: Tensor Computation
  // Inputs: fw0, fw3, fw6, FE1_C2_D001_Q421, fw2, FE1_C1_D010_Q421, fw1, fw5, fw4, fw7, fw8, FE0_C0_D100_Q421
  // Outputs: A
  {
    double temp_0[4] = {0};
    for (int j = 0; j < 4; ++j)
    {
      temp_0[j] = fw0 * FE0_C0_D100_Q421[0][0][0][j];
    }
    double temp_1[4] = {0};
    for (int j = 0; j < 4; ++j)
    {
      temp_1[j] = fw1 * FE1_C1_D010_Q421[0][0][0][j];
    }
    double temp_2[4] = {0};
    for (int j = 0; j < 4; ++j)
    {
      temp_2[j] = fw2 * FE1_C2_D001_Q421[0][0][0][j];
    }
    double temp_3[4] = {0};
    for (int j = 0; j < 4; ++j)
    {
      temp_3[j] = fw3 * FE0_C0_D100_Q421[0][0][0][j];
    }
    double temp_4[4] = {0};
    for (int j = 0; j < 4; ++j)
    {
      temp_4[j] = fw4 * FE1_C1_D010_Q421[0][0][0][j];
    }
    double temp_5[4] = {0};
    for (int j = 0; j < 4; ++j)
    {
      temp_5[j] = fw5 * FE1_C2_D001_Q421[0][0][0][j];
    }
    double temp_6[4] = {0};
    for (int j = 0; j < 4; ++j)
    {
      temp_6[j] = fw6 * FE0_C0_D100_Q421[0][0][0][j];
    }
    double temp_7[4] = {0};
    for (int j = 0; j < 4; ++j)
    {
      temp_7[j] = fw7 * FE1_C1_D010_Q421[0][0][0][j];
    }
    double temp_8[4] = {0};
    for (int j = 0; j < 4; ++j)
    {
      temp_8[j] = fw8 * FE1_C2_D001_Q421[0][0][0][j];
    }
    for (int j = 0; j < 4; ++j)
    {
      for (int i = 0; i < 4; ++i)
      {
        A[4 * (i) + (j)] += FE0_C0_D100_Q421[0][0][0][i] * temp_0[j];
        A[4 * (i) + (j)] += FE0_C0_D100_Q421[0][0][0][i] * temp_1[j];
        A[4 * (i) + (j)] += FE0_C0_D100_Q421[0][0][0][i] * temp_2[j];
        A[4 * (i) + (j)] += FE1_C1_D010_Q421[0][0][0][i] * temp_3[j];
        A[4 * (i) + (j)] += FE1_C1_D010_Q421[0][0][0][i] * temp_4[j];
        A[4 * (i) + (j)] += FE1_C1_D010_Q421[0][0][0][i] * temp_5[j];
        A[4 * (i) + (j)] += FE1_C2_D001_Q421[0][0][0][i] * temp_6[j];
        A[4 * (i) + (j)] += FE1_C2_D001_Q421[0][0][0][i] * temp_7[j];
        A[4 * (i) + (j)] += FE1_C2_D001_Q421[0][0][0][i] * temp_8[j];
      }
    }
  }
  // ------------------------ 
}

}

typedef int int32_t;
typedef long long int int64_t;

/**
 * `binary_search()` performs a binary search to find the location
 * of a given element in a sorted array of integers.
 */
extern "C" __device__ int binary_search(
  int num_elements,
  const int * __restrict__ elements,
  int key,
  int * __restrict__ out_index)
{
  if (num_elements <= 0)
    return -1;

  int p = 0;
  int q = num_elements;
  int r = (p + q) / 2;
  while (p < q) {
    if (elements[r] == key) break;
    else if (elements[r] < key) p = r + 1;
    else q = r - 1;
    r = (p + q) / 2;
  }
  if (elements[r] != key)
    return -1;
  *out_index = r;
  return 0;
}


extern "C" int printf(const char * format, ...);

extern "C" void __global__
assemble__integral_f47b8d1b79d48a8e8a81d53d75ec75321bc46ed1_tetrahedron_c_mat(
  int32_t num_active_cells,
  const int32_t* __restrict__ active_cells,
  int num_vertices_per_cell,
  const int32_t* __restrict__ vertex_indices_per_cell,
  int num_coordinates_per_vertex,
  const double* __restrict__ vertex_coordinates,
  int num_coeffs_per_cell,
  const ufc_scalar_t* __restrict__ coeffs,
  const ufc_scalar_t* __restrict__ constant_values,
  int num_dofs_per_cell0,
  int num_dofs_per_cell1,
  const int32_t* __restrict__ dofmap0,
  const int32_t* __restrict__ dofmap1,
  const char* __restrict__ bc0,
  const char* __restrict__ bc1,
  int32_t num_local_rows,
  int32_t num_local_columns,
  const int32_t* __restrict__ row_ptr,
  const int32_t* __restrict__ column_indices,
  ufc_scalar_t* __restrict__ values,
  const int32_t* __restrict__ offdiag_row_ptr,
  const int32_t* __restrict__ offdiag_column_indices,
  ufc_scalar_t* __restrict__ offdiag_values,
  int32_t num_local_offdiag_columns,
  const int32_t* __restrict__ colmap_sorted,
  const int32_t* __restrict__ colmap_sorted_indices)
{
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  assert(num_vertices_per_cell == 4);
  assert(num_coordinates_per_vertex == 3);
  double cell_vertex_coordinates[4*3];

  assert(num_dofs_per_cell0 == 4);
  assert(num_dofs_per_cell1 == 4);
  ufc_scalar_t Ae[4*4];

  for (int i = thread_idx;
    i < num_active_cells;
    i += blockDim.x * gridDim.x)
  {
    int32_t c = active_cells[i];

    // Set element matrix values to zero
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
        Ae[j*4+k] = 0.0;
      }
    }

    const ufc_scalar_t* coeff_cell = &coeffs[c*num_coeffs_per_cell];

    // Gather cell vertex coordinates
    for (int j = 0; j < 4; j++) {
      int vertex = vertex_indices_per_cell[
        c*4+j];
      for (int k = 0; k < 3; k++) {
        cell_vertex_coordinates[j*3+k] =
          vertex_coordinates[vertex*3+k];
      }
    }

    int* entity_local_index = NULL;
    uint8_t* quadrature_permutation = NULL;

    // Compute element matrix
    tabulate_tensor_integral_f47b8d1b79d48a8e8a81d53d75ec75321bc46ed1_tetrahedron(
      Ae,
      coeff_cell,
      constant_values,
      cell_vertex_coordinates,
      entity_local_index,
      quadrature_permutation);

    // Add element matrix values to the global matrix,
    // skipping entries related to degrees of freedom
    // that are subject to essential boundary conditions.
    const int32_t* dofs0 = &dofmap0[c*4];
    const int32_t* dofs1 = &dofmap1[c*4];
    for (int j = 0; j < 4; j++) {
      int32_t row = dofs0[j];
      if (bc0 && bc0[row]) continue;
      if (row < num_local_rows) {
        for (int k = 0; k < 4; k++) {
          int32_t column = dofs1[k];
          if (bc1 && bc1[column]) continue;
          if (column < num_local_columns) {
            int r;
            int err = binary_search(
              row_ptr[row+1] - row_ptr[row],
              &column_indices[row_ptr[row]],
              column, &r);
            assert(!err && "Failed to find column index in assemble_matrix_cell_global!");
            r += row_ptr[row];
            values[r] += Ae[j*4+k];
            // atomicAdd(&values[r],
            //   Ae[j*4+k]);
          } else {
            /* Search for the correct column index in the column map
             * of the off-diagonal part of the local matrix. */
            int32_t sorted_idx;
            int err = binary_search(num_local_offdiag_columns, colmap_sorted, column, &sorted_idx);
            assert(!err && "Failed to find offdiag column index in colmap_sorted!");
            int32_t colmap_idx = colmap_sorted_indices[sorted_idx];
            
            int r;
            err = binary_search(
              offdiag_row_ptr[row+1] - offdiag_row_ptr[row],
              &offdiag_column_indices[offdiag_row_ptr[row]],
              colmap_idx, &r);
            assert(!err && "Failed to find offdiag column index!");
            r += offdiag_row_ptr[row];
            offdiag_values[r] += Ae[j*4+k];
            // atomicAdd(&offdiag_values[r],
            //   Ae[j*4+k]);
          }
        }
      }
    }
  }
}

extern "C" void __global__
lift_bc__integral_f47b8d1b79d48a8e8a81d53d75ec75321bc46ed1_tetrahedron(
  int32_t num_cells,
  int num_vertices_per_cell,
  const int32_t* __restrict__ vertex_indices_per_cell,
  int num_coordinates_per_vertex,
  const double* __restrict__ vertex_coordinates,
  int num_coeffs_per_cell,
  const ufc_scalar_t* __restrict__ coeffs,
  int num_constant_values,
  const ufc_scalar_t* __restrict__ constant_values,
  int num_dofs_per_cell0,
  int num_dofs_per_cell1,
  const int32_t* __restrict__ dofmap0,
  const int32_t* __restrict__ dofmap1,
  const char* __restrict__ bc_markers1,
  const ufc_scalar_t* __restrict__ bc_values1,
  double scale,
  int32_t num_columns,
  int32_t num_rows,
  const ufc_scalar_t* x0,
  ufc_scalar_t* b)
{
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  assert(num_vertices_per_cell == 4);
  assert(num_coordinates_per_vertex == 3);
  double cell_vertex_coordinates[4*3];

  assert(num_dofs_per_cell0 == 4);
  assert(num_dofs_per_cell1 == 4);
  ufc_scalar_t Ae[4*4];
  ufc_scalar_t be[4];

  for (int c = thread_idx;
    c < num_cells;
    c += blockDim.x * gridDim.x)
  {
    // Skip cell if boundary conditions do not apply
    const int32_t* dofs1 = &dofmap1[c*4];
    bool has_bc = false;
    for (int k = 0; k < 4; k++) {
      int32_t column = dofs1[k];
      if (bc_markers1 && bc_markers1[column]) {
        has_bc = true;
        break;
      }
    }
    if (!has_bc)
      continue;

    // Set element matrix and vector values to zero
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
        Ae[j*4+k] = 0.0;
      }
      be[j] = 0.0;
    }

    const ufc_scalar_t* coeff_cell = &coeffs[c*num_coeffs_per_cell];

    // Gather cell vertex coordinates
    for (int j = 0; j < 4; j++) {
      int vertex = vertex_indices_per_cell[
        c*4+j];
      for (int k = 0; k < 3; k++) {
        cell_vertex_coordinates[j*3+k] =
          vertex_coordinates[vertex*3+k];
      }
    }

    int* entity_local_index = NULL;
    uint8_t* quadrature_permutation = NULL;

    // Compute element matrix
    tabulate_tensor_integral_f47b8d1b79d48a8e8a81d53d75ec75321bc46ed1_tetrahedron(
      Ae,
      coeff_cell,
      constant_values,
      cell_vertex_coordinates,
      entity_local_index,
      quadrature_permutation);

    // Compute modified element vector
    const int32_t* dofs0 = &dofmap0[c*4];
    for (int k = 0; k < 4; k++) {
      int32_t column = dofs1[k];
      if (bc_markers1 && bc_markers1[column]) {
        const ufc_scalar_t _x0 = (x0) ? x0[column] : 0.0;
        ufc_scalar_t bc = bc_values1[column];
        for (int j = 0; j < 4; j++) {
          be[j] -= Ae[j*4+k] * scale * (bc - _x0);
        }
      }
    }

    // Add element vector values to the global vector
    for (int j = 0; j < 4; j++) {
      int32_t row = dofs0[j];
      if (row >= num_rows) continue;
      b[row] += be[j];
      // atomicAdd(&b[row], be[j]);
    }
  }
}
