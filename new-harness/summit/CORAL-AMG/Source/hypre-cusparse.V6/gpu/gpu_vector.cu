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

#include "gpu_vector.h"
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/transform_reduce.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <sstream>

#include <cublas_v2.h>

cublasHandle_t blas_handle = NULL;

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

__global__
void kernel_SetConstantValue(double value, double *data, int size)
{
  for (int idx=threadIdx.x+blockIdx.x*blockDim.x; idx<size; idx+=gridDim.x*blockDim.x) {
    data[idx] = value;
  }
}

void device_checkErrors()
{
  cudaCheckError();
  CUCHK(cudaGetLastError());
}

void device_createCublas()
{
  cublasStatus_t status;
  if (blas_handle == NULL) {
    status = cublasCreate(&blas_handle);
  }
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("[E]: cublasCreate in %s failed (%d)\n",__FUNCTION__,(int)status);
  }
  cudaCheckError();
}

double device_VectorSumElts(double *data, int size)
{
  double sum = thrust::reduce(data, data+size);
  cudaCheckError();
  return sum;
}

void device_SeqVectorScale(double alpha, double *data, int size)
{
  cublasStatus_t status;
  if (blas_handle == NULL) {
    status = cublasCreate(&blas_handle);
  }

  status = cublasDscal(blas_handle, size, &alpha, data, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("[E]: cublasDscal in device_SeqVectorScale failed (%d)\n",(int)status);
  }
  cudaCheckError();
}

void device_SeqVectorAxpy(double alpha, double *x, double *y, int size)
{
  cublasStatus_t status;
  if (blas_handle == NULL) {
    status = cublasCreate(&blas_handle);
  }

  status = cublasDaxpy(blas_handle, size, &alpha, x, 1, y, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("[E]: cublasDaxpy in device_SeqVectorAxpy failed (%d)\n",(int)status);
  }
  cudaCheckError();
}

double device_SeqVectorInnerProd(double *x, double *y, int size)
{
  cublasStatus_t status;
  if (blas_handle == NULL) {
    status = cublasCreate(&blas_handle);
  }

  double r = 0.;

  // r = thrust::inner_product(thrust::device, x, x+size, y, 0.0);
  status = cublasDdot(blas_handle, size, x, 1, y, 1, &r);
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("[E]: cublasDdot in device_SeqVectorInnerProd failed (%d)\n",(int)status);
  }
  cudaCheckError();

  return r;
} 

void device_SeqVectorCopy(double *x, double *y, int size)
{
  cublasStatus_t status;
  if (blas_handle == NULL) {
    status = cublasCreate(&blas_handle);
  }

  // copy x into y
  status = cublasDcopy(blas_handle, size, x, 1, y, 1);

  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("[E]: cublasDcopy in device_SeqVectorCopy failed (%d)\n",(int)status);
  }
  cudaCheckError();
}

void device_SeqVectorSetConstantValues(double value, double *data, int size)
{
  kernel_SetConstantValue<<<4096, 512>>>(value, data, size);
  CUCHK(cudaGetLastError());
  cudaCheckError();
}

#ifdef __cplusplus
}; // end extern "C"
#endif
