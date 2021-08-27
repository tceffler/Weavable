/**
 * Copyright (c) 2014, NVIDIA Corporation
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 * * Neither the name of the NVIDIA Corporation nor the
 *   names of its contributors may be used to endorse or promote products
 *   derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **/

#ifndef MATRIX_GPU_H
#define MATRIX_GPU_H

#include <nvToolsExt.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Operations on matrices
  void device_spmv(void *Ah, int num_rows, int num_cols, int nnz, double alpha, int *A_i, int *A_j,
                   double *data, double *x, int x_size, double beta, double *y, int y_size, int is_transpose);

  void device_l1_norm(void *Ah, int n, int m, double *u_data, double *f_data, double *l1_norms, double *d_data, double relax_weight);
  void device_set_l1_norms(void *Ah, double *l1_norms);
  int device_has_send_maps(void *Ah, int is_transpose);
  void device_create_matrix(void *Ah, int num_rows, int num_cols, int num_nnz, int *A_i, int *A_j, double *A_data, int x_size, int y_size, int is_transpose);
  void device_create_comm_buffer(void *Ah, int send_size, int *send_maps, double *v_data, double *u_data);
  void device_set_comm_map(void *Ah, int map_size, int *map, int is_transpose);
  void device_assemble_transpose_result(void *Ah, int num_rows, int num_cols, int size, double *out, double *in);
  void device_set_hyper_sparse(void *Ah, int num_rownnz, int *rownnz);
  void device_create_diagonal(void *Ah);

// get a new vector
  double *device_get_vector(int size, int idx);

  void device_cheby_loop1(void *Ah, double *ds_data, double *r_data, double *f_data);
  void device_cheby_loop2(double *r_data, double *ds_data, double *tmp_data,
                          double *orig_u, double *u_data, 
                          double coef, int num_rows);
  void device_cheby_loop3(double *tmp_data, double *ds_data, double *u_data, int num_rows);
  void device_cheby_loop4(double *ds_data, double *v_data, double *u_data,
                          double *r_data, double mult, int num_rows);
  void device_cheby_loop5(double *u_data, double *orig_u,
                          double *ds_data, int num_rows);
  void device_report_memory();
#ifdef __cplusplus
}
#endif

#endif // end #ifndef GPU_LEVEL_H
