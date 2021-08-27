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

#ifndef UTILITIES_GPU_H
#define UTILITIES_GPU_H

#include <nvToolsExt.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { H2D, D2H, D2D } memcpy_direction;

// basic cuda API functions
void device_synchronize(); // cudaDeviceSynchronize
void device_memcpyAsync(void *dest, void *src, size_t size, memcpy_direction direction);
void device_memcpy(void *dest, void *src, size_t size, memcpy_direction direction);
void device_malloc(void **ptr, size_t size);
void device_free(void *ptr);
void device_hostRegister(void *ptr, size_t size, unsigned int flags);
void device_hostUnregister(void *ptr);
void device_memInfo(size_t *free, size_t *total);
int device_numDevices();
void device_setDevice(int dev);

// utility functions
void device_set_stream(int id);
cudaStream_t device_get_stream();
void device_sync_stream();

void nvtxRangePushColor(char *message, uint32_t color);

extern int GPU_SOLVE_PHASE;

#ifdef __cplusplus
}; // extern "C"
#endif

#define CUCHK(call) {                                    \
  cudaError_t err = call;                                                    \
  if( cudaSuccess != err) {                                                \
  fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
          __FILE__, __LINE__, cudaGetErrorString( err) );              \
  fflush(stderr); \
  exit(EXIT_FAILURE);                                                  \
  } }

#define CUSPARSE_CHK(call) {                                    \
  int err = call;                                                    \
  if( CUSPARSE_STATUS_SUCCESS != err) {                                                \
  fprintf(stderr, "cusparse error in file '%s' in line %i : %d.\n",        \
          __FILE__, __LINE__, err );              \
  fflush(stderr); \
  exit(EXIT_FAILURE);                                                  \
  } }

#endif // UTILITIES_GPU_H
