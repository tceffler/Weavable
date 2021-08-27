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
 
#include <assert.h>
#include <cstdio>
#include <map>
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include "gpu_matrix.h"
#include "gpu_utilities.h"
#include "gpu_matvec.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FatalError(s) {                                                 \
  std::stringstream _where;                                             \
  _where << __FILE__ << ':' << __LINE__;                                \
  printf("[E]: %s at %s\n",s.c_str(),_where.str().c_str());             \
  cudaDeviceSynchronize();                                              \
  exit(0);                                                              \
}


// device-sync and check error only if in debug mode
#if !defined(NDEBUG)
#define cudaCheckError() {                                              \
  cudaDeviceSynchronize();                                              \
  cudaError_t e=cudaGetLastError();                                     \
  if(e!=cudaSuccess) {                                                  \
    std::stringstream _error;                                           \
    _error << "Cuda failure: '" << cudaGetErrorString(e) << "'";        \
    FatalError(_error.str());                                           \
  }                                                                     \
}
#else
#define cudaCheckError() {}                                              
#endif

#define CUCHK(call) {                                    \
  cudaError_t err = call;                                                    \
  if( cudaSuccess != err) {                                                \
  fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
          __FILE__, __LINE__, cudaGetErrorString( err) );              \
  fflush(stderr); \
  exit(EXIT_FAILURE);                                                  \
  } }
      
extern cusparseHandle_t handle;
extern cusparseMatDescr_t descr;
extern cudaStream_t active_stream;
extern int GPU_SOLVE_PHASE;
extern int gpu_level;
extern int mype, npe;
size_t matrix_memory = 0, vector_memory = 0;

  void device_report_memory()
  {
    size_t free_mem, tot_mem;
    CUCHK(cudaMemGetInfo(&free_mem, &tot_mem));
    if (mype == 0) {
    printf("P(%d): matrix = %f GB, vector = %f GB, free = %f GB, tot = %f\n", 
           mype, 
           (double)matrix_memory/pow(1024.0,3), 
           (double)vector_memory/pow(1024.0,3),
           (double)free_mem/pow(1024.0,3), 
           (double)tot_mem/pow(1024.0,3));
    }
  }

// matrices map 
typedef std::map<std::pair<void *,int>, matrix> map_type;
map_type matrices;
#ifdef GPU_USE_CUSPARSE_HYB
  std::map<void*,int> is_hyb;
#endif

// pool of temp vectors 
std::map<int,double*> temp_device_vec[3];

double *device_get_vector(int size, int idx)
{
  if (temp_device_vec[idx].find(size) == temp_device_vec[idx].end()) {
    // allocate on device
    CUCHK(cudaMalloc(&temp_device_vec[idx][size], sizeof(double) * size));
    vector_memory += sizeof(double) * size;
  }
  return temp_device_vec[idx][size];
}

#ifdef GPU_STORE_EXPLICIT_TRANSPOSE
matrix *set_matrix(void *Ah, int r, int c, int nnz, int *I, int *J, double *v, int xsize, int ysize, int is_transpose)
{
  matrix A;

  // default is non-hyper-sparse
  A.is_hypersparse = 0;
  A.num_rownnz = r;
  A.mask = NULL;

  // default has no explicit diagonal stored
  A.diag_set = 0;
  A.diag_d = NULL;

  // set dimensions
  if (is_transpose) {
    A.rows = c;
    A.cols = r;
    A.nnz = nnz;
  } else {
    A.rows = r;
    A.cols = c;
    A.nnz = nnz;
  }

  if (is_transpose) {
    A.x_size = r;
    A.y_size = c;
  } else {
    A.x_size = xsize;
    A.y_size = ysize;
  }

  // allocate memory for device matrix
  if (is_transpose) { 
#ifdef GPU_USE_CUSPARSE_HYB
    if (gpu_level <= HYB_LEVEL) {
      CUCHK(cudaHostAlloc(&A.I_h, sizeof(int)*(c+1), cudaHostAllocMapped));
      CUCHK(cudaHostAlloc(&A.J_h, sizeof(int)*nnz, cudaHostAllocMapped));
      CUCHK(cudaHostAlloc(&A.val_h, sizeof(double)*nnz, cudaHostAllocMapped));
      CUCHK(cudaHostGetDevicePointer(&A.I_d, A.I_h, 0));
      if (nnz > 0) CUCHK(cudaHostGetDevicePointer(&A.J_d, A.J_h, 0));
      if (nnz > 0) CUCHK(cudaHostGetDevicePointer(&A.val_d, A.val_h, 0));        
    } else { 
      matrix_memory += sizeof(int)*(c+1) + sizeof(int)*nnz + sizeof(double)*nnz;
      CUCHK(cudaMalloc(&A.I_d, sizeof(int)*(c+1)));
      CUCHK(cudaMalloc(&A.J_d, sizeof(int)*nnz));
      CUCHK(cudaMalloc(&A.val_d, sizeof(double)*nnz));
    }      
#else
    matrix_memory += sizeof(int)*(c+1) + sizeof(int)*nnz + sizeof(double)*nnz;
    CUCHK(cudaMalloc(&A.I_d, sizeof(int)*(c+1)));
    CUCHK(cudaMalloc(&A.J_d, sizeof(int)*nnz));
    CUCHK(cudaMalloc(&A.val_d, sizeof(double)*nnz));
#endif
  } else {
#ifdef GPU_USE_CUSPARSE_HYB
    if (gpu_level <= HYB_LEVEL) {
      CUCHK(cudaHostAlloc(&A.I_h, sizeof(int)*(r+1), cudaHostAllocMapped));
      CUCHK(cudaHostAlloc(&A.J_h, sizeof(int)*nnz, cudaHostAllocMapped));
      CUCHK(cudaHostAlloc(&A.val_h, sizeof(double)*nnz, cudaHostAllocMapped));
      CUCHK(cudaHostGetDevicePointer(&A.I_d, A.I_h, 0));
      if (nnz > 0) CUCHK(cudaHostGetDevicePointer(&A.J_d, A.J_h, 0));
      if (nnz > 0) CUCHK(cudaHostGetDevicePointer(&A.val_d, A.val_h, 0));        
    } else {
      matrix_memory += sizeof(int)*(r+1) + sizeof(int)*nnz + sizeof(double)*nnz;
      CUCHK(cudaMalloc(&A.I_d, sizeof(int)*(r+1)));
      CUCHK(cudaMalloc(&A.J_d, sizeof(int)*nnz));
      CUCHK(cudaMalloc(&A.val_d, sizeof(double)*nnz));
    }
#else
    matrix_memory += sizeof(int)*(r+1) + sizeof(int)*nnz + sizeof(double)*nnz;
    CUCHK(cudaMalloc(&A.I_d, sizeof(int)*(r+1)));
    CUCHK(cudaMalloc(&A.J_d, sizeof(int)*nnz));
    CUCHK(cudaMalloc(&A.val_d, sizeof(double)*nnz));
#endif
  }

  int *J_sorted = NULL;
  double *v_sorted = NULL;

  if (is_transpose) {
    // create a temporary sorted by column indices version
    std::vector<std::pair<int, double> > V;
    J_sorted = new int[nnz];
    v_sorted = new double[nnz];

    for (int i=0; i<r; i++) {
      V.clear();

      for (int jj=I[i]; jj<I[i+1]; jj++) {
        V.push_back(std::make_pair(J[jj],v[jj]));
      }

      std::sort(V.begin(), V.end());

      for (int j=0; j<(int)V.size(); j++) {
        J_sorted[I[i]+j] = V[j].first;
        v_sorted[I[i]+j] = V[j].second;
      }
    }
  }

  if (is_transpose) {
    // if storing explicit transpose, need temp vectors
    int *csrRow, *csrCol;
    double *csrVal;

    CUCHK(cudaMalloc(&csrRow, sizeof(int)*(r+1)));
    cudaCheckError();
    CUCHK(cudaMalloc(&csrCol, sizeof(int)*nnz));
    cudaCheckError();
    CUCHK(cudaMalloc(&csrVal, sizeof(double)*nnz));
    cudaCheckError();

    // copy memory up
    cudaMemcpy(csrRow, I, sizeof(int)*(r+1),cudaMemcpyHostToDevice);
    cudaCheckError();
    cudaMemcpy(csrCol, J_sorted, sizeof(int)*nnz, cudaMemcpyHostToDevice);
    cudaCheckError();
    cudaMemcpy(csrVal, v_sorted, sizeof(double)*nnz, cudaMemcpyHostToDevice);
    cudaCheckError();

    // convert to CSC and store in A
    cusparseStatus_t status = (cusparseStatus_t)0;
    status = cusparseDcsr2csc(handle, r, c, nnz, csrVal, csrRow, csrCol, A.val_d, A.J_d, A.I_d, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      printf("[E]: csr2csc conversion failed with status: %d\n",(int)status);
      exit(0);
    }
    cudaCheckError();

    CUCHK(cudaFree(csrRow));
    CUCHK(cudaFree(csrCol));
    CUCHK(cudaFree(csrVal));
    cudaCheckError();
  } else {
    // copy memory up
    cudaMemcpy(A.I_d, I, sizeof(int)*(r+1),cudaMemcpyHostToDevice);
    cudaCheckError();
    cudaMemcpy(A.J_d, J, sizeof(int)*nnz, cudaMemcpyHostToDevice);
    cudaCheckError();
    cudaMemcpy(A.val_d, v, sizeof(double)*nnz, cudaMemcpyHostToDevice);
    cudaCheckError();
  }

  // // allocate memory for device vectors
  // CUCHK(cudaMalloc(&A.x_d, sizeof(double)*A.x_size));
  // cudaCheckError();
  // CUCHK(cudaMalloc(&A.y_d, sizeof(double)*A.y_size));
  // cudaCheckError();
  // vector_memory += sizeof(double)*(A.x_size + A.y_size);

  // additional smoother data
  // CUCHK(cudaMalloc(&A.f_data, sizeof(double)*r));
  // cudaCheckError();
  CUCHK(cudaMalloc(&A.l1_norms, sizeof(double)*r));
  cudaCheckError();
  A.smoother_set = 0;
  vector_memory += sizeof(double)*r;

  // initialize send_maps 
  A.send_maps = NULL;
  A.send_data = NULL;
  cudaCheckError();
 
#ifdef GPU_USE_CUSPARSE_HYB
  if (gpu_level <= HYB_LEVEL) {
    int *h_rows = (int*)malloc((A.rows+1)*sizeof(int));
    cudaMemcpy(h_rows, A.I_d, (A.rows+1)*sizeof(int), cudaMemcpyDeviceToHost);
    int max_col = 0;
    for (int i = 0; i < A.rows; i++) {
      int ncol = h_rows[i+1] - h_rows[i];
      if (ncol > max_col) max_col = ncol;
    }
    size_t ell_memory = (sizeof(double)+sizeof(int))*max_col*A.rows;
    matrix_memory += ell_memory;
    free(h_rows);
    cusparseStatus_t status;
    CUSPARSE_CHK(cusparseCreateHybMat(&A.hybA));
    status = cusparseDcsr2hyb(handle, A.rows, A.cols, descr,
                                  A.val_d, A.I_d, A.J_d,
                                  A.hybA, 0, CUSPARSE_HYB_PARTITION_MAX);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      printf("P(%d): csr2hyb error %d\n", mype, status);
      exit(1);
    }
    is_hyb[Ah] = 1;
  } else {
    is_hyb[Ah] = 0;
  }
#endif

  // add matrix to the map
  matrices[std::make_pair(Ah, is_transpose)] = A;

  // clean up temporary arrays
  delete [] J_sorted;
  delete [] v_sorted;
 
  return &matrices[std::make_pair(Ah, is_transpose)];
}
#else 
matrix *set_matrix(void *Ah, int r, int c, int nnz, int *I, int *J, double *v, int xsize, int ysize, int is_transpose)
{
  matrix A;

  // default is non-hyper-sparse
  A.is_hypersparse = 0;
  A.num_rownnz = r;
  A.mask = NULL;

  // default has no explicit diagonal stored
  A.diag_set = 0;
  A.diag_d = NULL;

  // set dimensions
  if (is_transpose) {
    A.rows = r;
    A.cols = c;
    A.nnz = nnz;
  } else {
    A.rows = r;
    A.cols = c;
    A.nnz = nnz;
  }

  if (is_transpose) {
    A.x_size = r;
    A.y_size = c;
  } else {
    A.x_size = xsize;
    A.y_size = ysize;
  }

  // allocate memory for device matrix
  if (is_transpose) {
    CUCHK(cudaMalloc(&A.I_d, sizeof(int)*(r+1)));
    if (nnz > 0) CUCHK(cudaMalloc(&A.J_d, sizeof(int)*nnz));
    if (nnz > 0) CUCHK(cudaMalloc(&A.val_d, sizeof(double)*nnz));    
    matrix_memory += sizeof(int)*(r+1) + sizeof(int)*nnz + sizeof(double)*nnz;
  } else {
#ifdef GPU_USE_CUSPARSE_HYB
    if (gpu_level <= HYB_LEVEL) {
      CUCHK(cudaHostAlloc(&A.I_h, sizeof(int)*(r+1), cudaHostAllocMapped));
      if (nnz > 0) CUCHK(cudaHostAlloc(&A.J_h, sizeof(int)*nnz, cudaHostAllocMapped));
      if (nnz > 0) CUCHK(cudaHostAlloc(&A.val_h, sizeof(double)*nnz, cudaHostAllocMapped));
      CUCHK(cudaHostGetDevicePointer(&A.I_d, A.I_h, 0));
      if (nnz > 0) CUCHK(cudaHostGetDevicePointer(&A.J_d, A.J_h, 0));
      if (nnz > 0) CUCHK(cudaHostGetDevicePointer(&A.val_d, A.val_h, 0));
    } else {
      CUCHK(cudaMalloc(&A.I_d, sizeof(int)*(r+1)));
      if (nnz > 0) CUCHK(cudaMalloc(&A.J_d, sizeof(int)*nnz));
      if (nnz > 0) CUCHK(cudaMalloc(&A.val_d, sizeof(double)*nnz));
      matrix_memory += sizeof(int)*(r+1) + sizeof(int)*nnz + sizeof(double)*nnz;
    }
#else
    CUCHK(cudaMalloc(&A.I_d, sizeof(int)*(r+1)));
    if (nnz > 0) CUCHK(cudaMalloc(&A.J_d, sizeof(int)*nnz));
    if (nnz > 0) CUCHK(cudaMalloc(&A.val_d, sizeof(double)*nnz));
    matrix_memory += sizeof(int)*(r+1) + sizeof(int)*nnz + sizeof(double)*nnz;
#endif
  }

  int *J_sorted = NULL;
  double *v_sorted = NULL;

  if (is_transpose) {
    // create a temporary sorted by column indices version
    std::vector<std::pair<int, double> > V;
    J_sorted = new int[nnz];
    v_sorted = new double[nnz];

    for (int i=0; i<r; i++) {
      V.clear();

      for (int jj=I[i]; jj<I[i+1]; jj++) {
        V.push_back(std::make_pair(J[jj],v[jj]));
      }

      std::sort(V.begin(), V.end());

      for (int j=0; j<(int)V.size(); j++) {
        J_sorted[I[i]+j] = V[j].first;
        v_sorted[I[i]+j] = V[j].second;
      }
    }
  }

  if (is_transpose) {
    // copy memory up
    cudaMemcpy(A.I_d, I, sizeof(int)*(r+1),cudaMemcpyHostToDevice);
    cudaCheckError();
    cudaMemcpy(A.J_d, J_sorted, sizeof(int)*nnz, cudaMemcpyHostToDevice);
    cudaCheckError();
    cudaMemcpy(A.val_d, v_sorted, sizeof(double)*nnz, cudaMemcpyHostToDevice);
    cudaCheckError();
  } else {
    // copy memory up
    cudaMemcpy(A.I_d, I, sizeof(int)*(r+1),cudaMemcpyHostToDevice);
    cudaCheckError();
    cudaMemcpy(A.J_d, J, sizeof(int)*nnz, cudaMemcpyHostToDevice);
    cudaCheckError();
    cudaMemcpy(A.val_d, v, sizeof(double)*nnz, cudaMemcpyHostToDevice);
    cudaCheckError();
#ifdef GPU_USE_CUSPARSE_HYB 
    if (nnz > 0 && gpu_level <= HYB_LEVEL) {
      int *h_rows = (int*)malloc((A.rows+1)*sizeof(int));
      cudaMemcpy(h_rows, A.I_d, (A.rows+1)*sizeof(int), cudaMemcpyDeviceToHost);
      int max_col = 0;
      for (int i = 0; i < A.rows; i++) {
        int ncol = h_rows[i+1] - h_rows[i];
        if (ncol > max_col) max_col = ncol;
      }
      size_t ell_memory = (sizeof(double)+sizeof(int))*max_col*A.rows;
      matrix_memory += ell_memory;
      free(h_rows);
      CUSPARSE_CHK(cusparseCreateHybMat(&A.hybA));
      CUSPARSE_CHK(cusparseDcsr2hyb(handle, A.rows, A.cols, descr,
                                    A.val_d, A.I_d, A.J_d,
                                    A.hybA, 0, CUSPARSE_HYB_PARTITION_MAX));
    }
#endif
  }

  // // allocate memory for device vectors
  // CUCHK(cudaMalloc(&A.x_d, sizeof(double)*A.x_size));
  // cudaCheckError();
  // CUCHK(cudaMalloc(&A.y_d, sizeof(double)*A.y_size));
  // cudaCheckError();

  // // additional smoother data
  // CUCHK(cudaMalloc(&A.f_data, sizeof(double)*r));
  // cudaCheckError();
  CUCHK(cudaMalloc(&A.l1_norms, sizeof(double)*r));
  cudaCheckError();
  A.smoother_set = 0;
  vector_memory += sizeof(double)*r;

  // initialize send_maps 
  A.send_maps = NULL;
  A.send_data = NULL;
  cudaCheckError();

  // add matrix to the map
  matrices[std::make_pair(Ah, is_transpose)] = A;

  // clean up temporary arrays
  delete [] J_sorted;
  delete [] v_sorted;

  return &matrices[std::make_pair(Ah, is_transpose)];
}
#endif

matrix *get_matrix(void *Ah, int is_transpose)
{
  cudaCheckError();
  if (handle == NULL) {
    // create cusparse handle 
    cusparseCreate(&handle);
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
    cudaCheckError();
  }

  if (descr == NULL) {
    // create cusparse matrix descriptor
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    cudaCheckError();
  

  //initiate kernel

    cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS;
    double alpha = 1.0;
    double beta = 1.0;
    double val_d = 1.0, d_x = 1.0, d_y = 1.0;
    int i_d[2];
    i_d[0] = 0;
    i_d[1] = 1;
    int j_d = 0;
    status = cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  1, 1, 1, &alpha, descr,
                                  &val_d, &i_d[0], &j_d, &d_x, &beta, &d_y);
    cudaCheckError();
    cudaDeviceSynchronize();

    beta = 0.0;
    status = cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  1, 1, 1, &alpha, descr,
                                  &val_d, &i_d[0], &j_d, &d_x, &beta, &d_y);
    cudaCheckError();
    cudaDeviceSynchronize();
  }

  map_type::iterator it;
  // find the matrix if it exists
  it = matrices.find(std::make_pair(Ah,is_transpose));

  if (it != matrices.end()) {
    // return pointer to the level found
    return &(it->second);
  } else {
    // if not found, return a null pointer
    return NULL;
  }
}

void device_spmv(void *Ah, int num_rows, int num_cols, int nnz, double alpha, int *A_i, int *A_j,
                 double *data, double *x, int x_size, double beta, double *y, int y_size, int is_transpose)
{
  // find matrix in our GPU collection
  matrix *A = get_matrix(Ah, is_transpose);
#ifdef GPU_USE_CUSPARSE_HYB
  int use_hyb = is_hyb[Ah];
#endif
  cudaCheckError()
  
  // if matrix does not exist: create and copy data
  if (A == NULL) {
    A = set_matrix(Ah, num_rows, num_cols, nnz, A_i, A_j, data, x_size, y_size, is_transpose);
    cudaCheckError()
  }

  // check that vector sizes fit in allocated memory
  if (is_transpose) {
#ifdef GPU_STORE_EXPLICIT_TRANSPOSE
    assert(x_size <= A->x_size); 
    assert(y_size <= A->y_size);
#else
    assert(x_size <= A->xt_size); 
    assert(y_size <= A->yt_size);
#endif
  } else {
    assert(x_size <= A->x_size); 
    assert(y_size <= A->y_size);
  }

  // check if x vector is on device already - then just use it w/o copy
  double *d_x = NULL; 
  cudaPointerAttributes attrib;
  cudaCheckError();
  if (x_size > 0) {
    cudaPointerGetAttributes(&attrib, x);
    // cudaGetLastError();
    cudaCheckError();
  }
  if (x_size == 0 || attrib.memoryType == cudaMemoryTypeDevice) {
    d_x = x;
  }
  else {
    cudaCheckError();
    int count;
    if (is_transpose) {
  #ifdef GPU_STORE_EXPLICIT_TRANSPOSE
      d_x = A->x_d;
      count = A->x_size;
  #else
      d_x = A->xt_d;
      count = A->xt_size;
  #endif
    } else {
      d_x = A->x_d;
      count = x_size;
    }
    cudaMemcpy(d_x, x, sizeof(double)*count, cudaMemcpyHostToDevice);
    cudaCheckError()
  }

  // check if result vector is on device already - then just use it w/o copy
  double *d_y = NULL;
  cudaCheckError();
  if (y_size > 0) {
    cudaPointerGetAttributes(&attrib, y);
    cudaGetLastError();
  }
  if (y_size == 0 || attrib.memoryType == cudaMemoryTypeDevice) {
    d_y = y;
  }
  else {
    cudaCheckError();
    int count;
    if (is_transpose) {
  #ifdef GPU_STORE_EXPLICIT_TRANSPOSE
      d_y = A->y_d;
      count = A->y_size;
  #else
      d_y = A->yt_d;
      count = A->yt_size;
  #endif
    } else {
      d_y = A->y_d;
      count = y_size;
    }
    cudaMemcpy(d_y, y, sizeof(double)*count, cudaMemcpyHostToDevice);
    cudaCheckError()
  }

  // call spmv
  cudaCheckError();
  cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS;

  // handle the case where we have a hyper-sparse matrix
#ifndef GPU_USE_CUSPARSE_HYB
  if (A->is_hypersparse) {
    status = cusparseDbsrxmv(handle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                             A->num_rownnz, A->rows, A->cols, A->nnz,
                             &alpha, descr, A->val_d, A->mask, A->I_d, A->I_d+1, A->J_d, 1, d_x, &beta, d_y);
    cudaCheckError();
  } else 
#endif
  {
    if (is_transpose) {
#ifdef GPU_STORE_EXPLICIT_TRANSPOSE
#ifdef GPU_USE_CUSPARSE_MATVEC
#ifdef GPU_USE_CUSPARSE_HYB
      if (A->nnz > 0) {
        if (use_hyb) {
          status = cusparseDhybmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &alpha, descr, A->hybA, d_x, &beta, d_y);
          if (status != CUSPARSE_STATUS_SUCCESS) {
            printf("hybmv error %d\n", status);
            exit(1);
          }
        } else {
          status = cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                  A->rows, A->cols, A->nnz, &alpha, descr, 
                                  A->val_d, A->I_d, A->J_d, d_x, &beta, d_y);
          if (status != CUSPARSE_STATUS_SUCCESS) {
            printf("csrmv error %d\n", status);
            exit(1);
          }
        }
      } 
#else
      status = cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                              A->rows, A->cols, A->nnz, &alpha, descr, 
                              A->val_d, A->I_d, A->J_d, d_x, &beta, d_y);
#endif
  #else
      status = CUSPARSE_STATUS_SUCCESS;
      // call custom cuda kernel
      matvec_cuda(*A, d_x, d_y, alpha, beta);
  #endif
#else
  #ifdef GPU_USE_CUSPARSE_MATVEC
      // call cusparse library function
      status = cusparseDcsrmv(handle, CUSPARSE_OPERATION_TRANSPOSE, A->rows, A->cols, A->nnz, &alpha, descr, A->val_d, A->I_d, A->J_d, d_x, &beta, d_y);
  #else
      status = CUSPARSE_STATUS_SUCCESS;
      // call custom cuda kernel
      matvecT_cuda(*A, d_x, d_y, alpha, beta);
  #endif
#endif
      cudaCheckError();
    } else {
#ifdef GPU_USE_CUSPARSE_MATVEC
#ifdef GPU_USE_CUSPARSE_HYB
      if (A->nnz > 0) {
        if (use_hyb) {
          status = cusparseDhybmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &alpha, descr, A->hybA, d_x, &beta, d_y);
          if (status != CUSPARSE_STATUS_SUCCESS) {
            printf("hybmv error %d\n", status);
            exit(1);
          }
        } else {
          status = cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                  A->rows, A->cols, A->nnz, &alpha, descr, 
                                  A->val_d, A->I_d, A->J_d, d_x, &beta, d_y);
          if (status != CUSPARSE_STATUS_SUCCESS) {
            printf("csrmv error %d\n", status);
            exit(1);
          }
        }
      } 
#else
      status = cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                              A->rows, A->cols, A->nnz, &alpha, descr, 
                              A->val_d, A->I_d, A->J_d, d_x, &beta, d_y);
      if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("csrmv error %d\n", status);
        exit(1);
      }
#endif
#else
      // call custom cuda kernel
      matvec_cuda(*A, d_x, d_y, alpha, beta);
#endif
      cudaCheckError();
    } 
  }
  if (status != CUSPARSE_STATUS_SUCCESS) {
    // output matrix to file 
    std::ofstream out("matrix.mtx");
    out << "%%%%MatrixMarket real general\n";
    out << num_rows << " " << num_cols << " " << nnz << "\n";
    for (int i=0; i<num_rows; i++) {
      for (int jj=A_i[i]; jj<A_i[i+1]; jj++) {
        int j = A_j[jj];
        out << i << " " << j << " " << data[jj] << "\n";
      }
    }
    out.close();

    printf("[E]: csrmv call failed (%d)\n",(int)status);
    exit(0);
  }
  cudaCheckError()

  // copy vector data back only if necessary
  if (d_y == A->y_d) {
    cudaMemcpy(y, A->y_d, sizeof(double)*y_size, cudaMemcpyDeviceToHost);
    cudaCheckError()
  }
}

__global__ void l1_norm_kernel(int rows, double *u_data, double *f_data, double *l1_norms, double *d_data, double *diag_data, double relax_weight)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows) return;

  if (diag_data[i] != 0.0)
  {
    u_data[i] += (relax_weight * f_data[i] + d_data[i]) * l1_norms[i];
  }
}
__global__ void l1_norm_kernel2(int rows, double *u_data, double *f_data, double *l1_norms, double *d_data, int *I_d, double *diag_data, double relax_weight)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows) return;

  if (diag_data[I_d[i]] != 0.0)
  {
    u_data[i] += (relax_weight * f_data[i] + d_data[i]) * l1_norms[i];
  }
}

__global__ void l1_norm_kernel_v2(int rows, double *u_data, double *f_data, double *l1_norms,
                                  double *d_data, double *diag_data, double relax_weight)
{
  # pragma unroll
  for (int i=threadIdx.x+blockIdx.x*blockDim.x; i < rows; i += gridDim.x*blockDim.x) {
#if 1
    if (diag_data[i] != 0.0) {
      u_data[i] += (relax_weight * f_data[i] + d_data[i]) * l1_norms[i];
    }
#else
    u_data[i] += (relax_weight * f_data[i] + d_data[i]) / l1_norms[i] * (diag_data[i] != 0.0);
#endif
  }
}


__global__ void reciprocal_kernel(int size, double *data)
{
  for (int i=threadIdx.x+blockIdx.x*blockDim.x; i<size; i+=gridDim.x*blockDim.x) {
    if (fabs(data[i]) < 1e-10) data[i] = 0.;
    else data[i] = 1./data[i];
  }
}

void device_l1_norm(void *Ah, int rows, int cols, double *u_data, double *f_data, double *l1_norms, double *d_data, double relax_weight)
{
  // early exit if nothing to do
  if (rows == 0) return;

  cudaCheckError()

  // find matrix in our GPU collection (must have one)
  matrix *A = get_matrix(Ah, 0);
  cudaCheckError()

  if (A == NULL) {
    printf("Internal error: cannot find matrix in our collection!\n");
    exit(1);
  }

  // lazily copy l1_norms, everything else is on device already
  if (A->smoother_set == 0) {
    cudaMemcpy(A->l1_norms, l1_norms, sizeof(double) * rows, cudaMemcpyHostToDevice);
    cudaCheckError();
    reciprocal_kernel<<<4096, 256>>>(rows, A->l1_norms);
    CUCHK(cudaGetLastError());
    cudaCheckError();
    A->smoother_set = 1;
  }

  // lazily copy diagonals
  if (!A->diag_set) {
    /*
    if (A->diag_d == NULL) {
      CUCHK(cudaMalloc(&A->diag_d, sizeof(double)*rows));
      cudaCheckError();
    }
    grab_diagonals_kernel<<<4096, 256>>>(rows, A->I_d, A->val_d, A->diag_d);
    cudaCheckError();
    A->diag_set = 1;
    */
    printf("External error - diagonal should be set\n");
    exit(1);
  }


  // launch custom kernel
// #if 1
//   int block = 256;
//   int grid = (rows + block-1)/block;
//   // l1_norm_kernel<<<grid, block, 0, active_stream>>>(rows, u_data, f_data, A->l1_norms, d_data, A->diag_d, relax_weight);
//   l1_norm_kernel2<<<grid, block, 0, active_stream>>>(rows, u_data, f_data, A->l1_norms, d_data, A->I_d, A->val_d, relax_weight);
//   cudaCheckError();
// #else
//   l1_norm_kernel_v2<<<4096, 128, 0, active_stream>>>(rows, u_data, f_data, A->l1_norms, d_data, A->diag_d, relax_weight);
//   cudaCheckError();
// #endif

#if 0
  int block = 256;
  int grid = (rows + block-1)/block;
  if (grid > 14336) grid = 14336;
  // l1_norm_kernel<<<grid, block, 0, active_stream>>>(rows, u_data, f_data, A->l1_norms, d_data, A->diag_d, relax_weight);
  l1_norm_kernel2<<<grid, block, 0, active_stream>>>(rows, u_data, f_data, A->l1_norms, d_data, A->I_d, A->val_d, relax_weight);
  cudaCheckError();
#else
  int block = 128;
  int grid = (rows + block-1)/block;
  if (grid > 14336) grid = 14336;
  l1_norm_kernel_v2<<<grid, block, 0, active_stream>>>(rows, u_data, f_data, A->l1_norms, d_data, A->diag_d, relax_weight);
  cudaCheckError();
  // cudaDeviceSynchronize();
  // exit(0);
#endif
  CUCHK(cudaGetLastError());
}

__global__ void grab_diagonals_kernel(int rows, int *offsets, double *values, double *diags)
{
  for (int i=threadIdx.x+blockIdx.x*blockDim.x; i<rows; i+=gridDim.x*blockDim.x) {
    // check if this row actually has any values
    if ((offsets[i+1]-offsets[i]) == 0) {
      diags[i] = 0.0;
    } else {
      diags[i] = values[offsets[i]]; 
    }
  }
}

void device_create_diagonal(void *Ah)
{
  matrix *A = get_matrix(Ah, 0);
  cudaCheckError()

  if (A == NULL) {
    printf("Internal error: cannot find matrix in our collection!\n");
    exit(1);
  }

  if (!A->diag_set) {
    if (A->diag_d == NULL) {
      CUCHK(cudaMalloc(&A->diag_d, sizeof(double)*A->rows));
      vector_memory += sizeof(double)*A->rows;
      cudaCheckError();
    }
    grab_diagonals_kernel<<<4096, 256>>>(A->rows, A->I_d, A->val_d, A->diag_d);
    CUCHK(cudaGetLastError());
    cudaCheckError();
    A->diag_set = 1;
  }
}

int device_has_send_maps(void *Ah, int is_transpose)
{
  // find matrix in our GPU collection
  matrix *A = get_matrix(Ah, is_transpose);
  cudaCheckError()
 
  if (A == NULL) return 0;
  if (A->send_maps == NULL) return 0;
  
  return 1;
}

__global__ void create_comm_buffer(int n, double *in_data, int *map, double *out_data)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n) return;

  out_data[i] = in_data[map[i]];
}

void device_create_matrix(void *Ah, int num_rows, int num_cols, int num_nnz, int *A_i, int *A_j, double *A_data, int x_size, int y_size, int is_transpose)
{
  // find matrix in our GPU collection
  matrix *A = get_matrix(Ah, is_transpose);
  cudaCheckError()

  // if matrix does not exist: create and copy data
  if (A == NULL) {
    A = set_matrix(Ah, num_rows, num_cols, num_nnz, A_i, A_j, A_data, x_size, y_size, is_transpose);
    cudaCheckError()
  }
}

void device_create_comm_buffer(void *Ah, int send_size, int *send_maps, double *v_data, double *u_data)
{
  // early exit
  if (send_size == 0) return;

  // find matrix in our GPU collection
  matrix *A = get_matrix(Ah, 0);
  cudaCheckError()
  
  if (A == NULL) {
    printf("Internal error: cannot find matrix in our collection!\n");
    exit(1);
  }

  // if send maps is not initialized - copy from host
  if (A->send_maps == NULL) {
    CUCHK(cudaMalloc(&A->send_maps, sizeof(int) * send_size));
    CUCHK(cudaMalloc(&A->send_data, sizeof(double) * send_size));
    vector_memory += sizeof(int) * send_size + sizeof(double) * send_size;
    cudaMemcpy(A->send_maps, send_maps, sizeof(int) * send_size, cudaMemcpyHostToDevice);
    cudaCheckError()
  }

  // launch comm buffers kernel
  int block = 256;
  int grid = (send_size + block-1)/block;
  create_comm_buffer<<<grid, block, 0, active_stream>>>(send_size, u_data, A->send_maps, A->send_data);
  CUCHK(cudaGetLastError());
  cudaCheckError()

  // copy send data to host
  cudaMemcpyAsync(v_data, A->send_data, sizeof(double) * send_size, cudaMemcpyDeviceToHost, active_stream);
  cudaCheckError()
}

void device_set_comm_map(void *Ah, int map_size, int *map, int is_transpose)
{
  if (map_size == 0) return;

  // find the matrix
  matrix *A = get_matrix(Ah, is_transpose);
  cudaCheckError();
  
  if (A == NULL) {
    printf("Internal error: cannot find matrix in our collection!\n");
    exit(1);
  }

  // if send maps is not initialized - copy from host
  if (A->send_maps == NULL) {
    CUCHK(cudaMalloc(&A->send_maps, sizeof(int) * map_size));
    CUCHK(cudaMalloc(&A->send_data, sizeof(double) * map_size));
    vector_memory += sizeof(int) * map_size + sizeof(double) * map_size;
    cudaMemcpy(A->send_maps, map, sizeof(int) * map_size, cudaMemcpyHostToDevice);
    cudaCheckError()
  }
}

//#if (__CUDACC_VER_MAJOR__ < 8)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                        __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif
//#endif

__global__
void kernel_assemble_transpose_result(int size, double *out, double *in, int *map)
{
  for (int idx=threadIdx.x+blockIdx.x*blockDim.x; idx < size; idx += gridDim.x*blockDim.x) {
    atomicAdd(&out[map[idx]], in[idx]);
  }
}

void device_assemble_transpose_result(void *Ah, int num_rows, int num_cols, int size, double *out, double *in)
{
  if (size == 0) return;

  // find the matrix
  matrix *A = get_matrix(Ah, 1);
  cudaCheckError()
  
  if (A == NULL) {
    printf("Internal error: cannot find matrix in our collection!\n");
    exit(1);
  }

  if (A->send_maps == NULL) {
    printf("Internal error: send maps not set\n");
    exit(1);
  }

  // "in" array is on the host -- copy to device (send_data)
  cudaMemcpy(A->send_data, in, sizeof(double)*size, cudaMemcpyHostToDevice);
  cudaCheckError()

  // now both arrays, and the map are on the device -- assemble
  kernel_assemble_transpose_result<<<4096, 512>>>(size, out, A->send_data, A->send_maps);
  CUCHK(cudaGetLastError());
  cudaCheckError()
}

void device_set_hyper_sparse(void *Ah, int num_rownnz, int *rownnz)
{
  if (num_rownnz == 0) return;

  matrix *A = get_matrix(Ah, 0);
  cudaCheckError();

  if (A == NULL) {
    printf("Internal error: cannot find matrix in our collection!\n");
    exit(1);
  }

  A->is_hypersparse = 1;
  A->num_rownnz = num_rownnz;
  // allocate memory
  CUCHK(cudaMalloc(&A->mask, sizeof(int)*A->num_rownnz));
  vector_memory += sizeof(int)*A->num_rownnz;
  cudaCheckError();

  // copy up row map
  cudaMemcpy(A->mask, rownnz, sizeof(int)*num_rownnz, cudaMemcpyHostToDevice);
  cudaCheckError();
}

void device_set_l1_norms(void *Ah, double *l1_norms)
{
  // find matrix in our GPU collection (must have one)
  matrix *A = get_matrix(Ah, 0);
  cudaCheckError()

  if (A == NULL) {
    printf("Internal error in %s: cannot find matrix in our collection!\n",__FUNCTION__);
    exit(1);
  }

  /*
  if (A->l1_norms == NULL) {
    printf("Internal error in %s: A->l1_norms == NULL\n",__FUNCTION__);
    exit(1);
  }
  */

  // lazily copy l1_norms, everything else is on device already
  if (A->smoother_set == 0) {
    cudaMemcpy(A->l1_norms, l1_norms, sizeof(double) * A->rows, cudaMemcpyHostToDevice);
    cudaCheckError();  
    reciprocal_kernel<<<4096, 256>>>(A->rows, A->l1_norms);
    CUCHK(cudaGetLastError());
    cudaCheckError();
    A->smoother_set = 1;
  }
}


__global__ void cheby_loop1(double *A_diag_data,   
                            double *ds_data, double *r_data, double *f_data,
                            int num_rows)
{
  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  for (int j = gid; j < num_rows; j += gridDim.x*blockDim.x) {
    double diag = A_diag_data[j];
    ds_data[j] = 1/sqrt(diag);
    
    r_data[j] = ds_data[j] * f_data[j];    
  }
}

void device_cheby_loop1(void *Ah, double *ds_data, double *r_data, double *f_data)
{
  matrix *A = get_matrix(Ah, 0);
  cudaCheckError()

  if (A == NULL) {
    printf("Internal error in %s: cannot find matrix in our collection!\n",__FUNCTION__);
    exit(1);
  }  
  if (A->rows == 0) return;
 
  // printf("diag_set = %d\n", A->diag_set);


  int block = 128;
  int grid = (A->rows + block - 1) / block;
  cheby_loop1<<<grid, block>>>(A->diag_d, ds_data, r_data, f_data, A->rows);
  CUCHK(cudaGetLastError());
}

__global__ void cheby_loop2(double *r_data, double *ds_data, double *tmp_data,
                           double *orig_u, double *u_data, 
                           double coef, int num_rows)
{
  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  for (int j = gid; j < num_rows; j += gridDim.x*blockDim.x) {
    r_data[j] += ds_data[j] * tmp_data[j];
    orig_u[j] = u_data[j];
    u_data[j] = r_data[j] * coef;
  }
}
 
void device_cheby_loop2(double *r_data, double *ds_data, double *tmp_data,
                        double *orig_u, double *u_data, 
                        double coef, int num_rows)
{ 
  if (num_rows == 0) return;
  int block = 128;
  int grid = (num_rows + block - 1) / block;
  cheby_loop2<<<grid, block>>>(r_data, ds_data, tmp_data, orig_u, u_data,
                               coef, num_rows);
  CUCHK(cudaGetLastError());
}

__global__ void cheby_loop3(double *tmp_data, double *ds_data, 
                            double *u_data, int num_rows)
{
  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  for (int j = gid; j < num_rows; j += gridDim.x*blockDim.x) {
    tmp_data[j]  =  ds_data[j] * u_data[j];    
  }
}

void device_cheby_loop3(double *tmp_data, double *ds_data, 
                        double *u_data, int num_rows)
{
  if (num_rows == 0) return;
  int block = 128;
  int grid = (num_rows + block - 1) / block;
  cheby_loop3<<<grid, block>>>(tmp_data, ds_data, u_data, num_rows);
  CUCHK(cudaGetLastError());
}

__global__ void cheby_loop4(double *ds_data, double *v_data, double *u_data,
                            double *r_data, double mult, int num_rows)
{
  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  for (int j = gid; j < num_rows; j += gridDim.x*blockDim.x) {
    double tmp_d = ds_data[j]* v_data[j];
    u_data[j] = mult * r_data[j] + tmp_d;
  }
}

void device_cheby_loop4(double *ds_data, double *v_data, double *u_data,
                        double *r_data, double mult, int num_rows)
{
  if (num_rows == 0) return;
  int block = 128;
  int grid = (num_rows + block - 1) / block;
  cheby_loop4<<<grid, block>>>(ds_data, v_data, u_data, r_data, mult, num_rows);
  CUCHK(cudaGetLastError());
}

__global__ void cheby_loop5(double *u_data, double *orig_u, 
                            double *ds_data, int num_rows)
{
  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  for (int j = gid; j < num_rows; j += gridDim.x*blockDim.x) {
    u_data[j] = orig_u[j] + ds_data[j]*u_data[j];
  }
}

void device_cheby_loop5(double *u_data, double *orig_u,
                        double *ds_data, int num_rows)
{
  if (num_rows == 0) return;
  int block = 128;
  int grid = (num_rows + block - 1) / block;
  cheby_loop5<<<grid, block>>>(u_data, orig_u, ds_data, num_rows);
  CUCHK(cudaGetLastError());
  CUCHK(cudaDeviceSynchronize());
}

#ifdef __cplusplus
} // end extern "C"
#endif
