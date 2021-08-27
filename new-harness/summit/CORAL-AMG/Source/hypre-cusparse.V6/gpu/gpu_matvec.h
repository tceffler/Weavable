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

#ifndef MATVEC_GPU_H
#define MATVEC_GPU_H

#include <cusparse.h>

struct matrix {
  // matrix data
  int *I_d, *J_d, rows, cols, nnz;
  double *val_d;

  // vector data
  int x_size, y_size;
  double *x_d, *y_d;
  // transpose vector data
  int xt_size, yt_size;
  double *xt_d, *yt_d;

  // smoother data
  int smoother_set;
  double *f_data;
  double *l1_norms;

  // send maps vector
  int *send_maps;
  double *send_data;

  // hyper-sparse data
  int is_hypersparse;
  int num_rownnz;
  int *mask;

  // diagonals
  int diag_set;
  double *diag_d;

  cusparseHybMat_t hybA;
  int *I_h, *J_h;
  double *val_h;
};

void matvec_cuda(matrix A, double *x, double *y, double alpha, double beta);
void matvecT_cuda(matrix A, double *x, double *y, double alpha, double beta);

#endif
