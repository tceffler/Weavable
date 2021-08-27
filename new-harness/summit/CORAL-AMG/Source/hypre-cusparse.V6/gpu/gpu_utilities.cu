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

#include "gpu_utilities.h"
#include "gpu_matrix.h"
#include "gpu_matvec.h"

class cudaDeviceSelector {
 public:
  cudaDeviceSelector() {
    char* str;
    int local_rank = 0;
    int numDev=4;

    //No MPI at this time so go by enviornment variables. 
    //This may need to be updated to match your MPI flavor
    if((str = getenv("MV2_COMM_WORLD_LOCAL_RANK")) != NULL) {
      local_rank = atoi(str);
    }
    else if((str = getenv("OMPI_COMM_WORLD_LOCAL_RANK")) != NULL) {
      local_rank = atoi(str);
    }
    else if((str = getenv("SLURM_LOCALID")) != NULL) {
      local_rank = atoi(str);
    }

    printf("MPS: local rank: %d (%d)\n", local_rank, local_rank % numDev);

    //Use MPS,  need to figure out how to set numDev, perhaps and enviornment varaible?
    char var[100];
    sprintf(var,"/tmp/nvidia-mps-slayton/mps-%d",local_rank%numDev);
    setenv("CUDA_MPS_PIPE_DIRECTORY",var,1);

  }
};

// cudaDeviceSelector __selector_;

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

// timings
double timing[20];

// cusparse handle & description
cusparseHandle_t handle = NULL;
cusparseMatDescr_t descr = NULL;

// pool of streams
std::map<int,cudaStream_t> streams;
cudaStream_t active_stream = NULL;

// global flag to note if we are in the solve phase
int GPU_SOLVE_PHASE = 0;

void nvtxRangePushColor(char *message, uint32_t color)
{
  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = color;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii = message;
  nvtxRangePushEx(&eventAttrib);
}

void device_set_stream(int id)
{
  if (id > 0) {
    if (streams.find(id) == streams.end()) {
      // create new stream
      cudaStreamCreate(&streams[id]);
    }
  
    // set current active stream
    active_stream = streams[id];
  }
  else
    // set NULL stream
    active_stream = NULL;

  // set cusparse stream
  cusparseSetStream(handle, active_stream);
}

cudaStream_t device_get_stream()
{
  return active_stream;
}

void device_sync_stream()
{
  cudaStreamSynchronize(active_stream);
}

/* cuda API wrappers */
void device_synchronize()
{
  cudaDeviceSynchronize();
}

void device_memcpyAsync(void *dest, void *src, size_t size, memcpy_direction direction)
{
  cudaStream_t current_stream = active_stream;

  switch (direction) {
    case H2D:
      cudaMemcpyAsync(dest, src, size, cudaMemcpyHostToDevice, current_stream);
      break;
    case D2H:
      cudaMemcpyAsync(dest, src, size, cudaMemcpyDeviceToHost, current_stream);
      break;
    case D2D:
      cudaMemcpyAsync(dest, src, size, cudaMemcpyDeviceToDevice, current_stream);
      break;
    default:
      printf("[E] Invalid Memcpy direction\n");
      exit(1);
  }
}

void device_memcpy(void *dest, void *src, size_t size, memcpy_direction direction)
{
  switch (direction) {
    case H2D:
      cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
      break;
    case D2H:
      cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
      break;
    case D2D:
      cudaMemcpy(dest, src, size, cudaMemcpyDeviceToDevice);
      break;
    default:
      printf("[E] Invalid Memcpy direction\n");
      exit(1);
  }
}

  extern size_t vector_memory;
void device_malloc(void **ptr, size_t size)
{
  CUCHK(cudaMalloc(ptr, size));
  vector_memory += size;
}

void device_free(void *ptr)
{
  CUCHK(cudaFree(ptr));
}

void device_hostRegister(void *ptr, size_t size, unsigned int flags)
{
  CUCHK(cudaHostRegister(ptr, size, flags));
}

void device_hostUnregister(void *ptr)
{
  CUCHK(cudaHostUnregister(ptr));
}

void device_memInfo(size_t *free, size_t *total)
{
  cudaMemGetInfo(free, total);
}

int device_numDevices()
{
  int devices;
  cudaGetDeviceCount(&devices);
  return devices;
}

void device_setDevice(int dev)
{
  cudaSetDevice(dev);
} 


#ifdef __cplusplus
}; // extern C
#endif
