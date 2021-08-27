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

#include <stdio.h>
#include "gpu_matvec.h"

#define CUCHK(call) {                                    \
  cudaError_t err = call;                                                    \
  if( cudaSuccess != err) {                                                \
  fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
          __FILE__, __LINE__, cudaGetErrorString( err) );              \
  fflush(stderr); \
  exit(EXIT_FAILURE);                                                  \
  } }

//#if (__CUDACC_VER_MAJOR__ < 8)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__
double atomicAdd(double *address, double val)
{
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif
/*
__device__ __forceinline__
double __shfl_down(double var, unsigned int laneMask, int warpWidth)
{
  int lo = __shfl_down( __double2loint(var), laneMask, warpWidth );
  int hi = __shfl_down( __double2hiint(var), laneMask, warpWidth );
  return __hiloint2double( hi, lo );
}
*/
// choose vector size depending on average number of nnz per row
// if vector size >> number of nnz in a row, implementation will be inefficient for gpu, many threads just idle
#define VECTOR_SIZE 4

__device__ __inline__ double warpReduceSum(double val) {
  if(VECTOR_SIZE>16) val+=__shfl_down(val,16,VECTOR_SIZE);
  if(VECTOR_SIZE>8) val+=__shfl_down(val,8,VECTOR_SIZE);
  if(VECTOR_SIZE>4) val+=__shfl_down(val,4,VECTOR_SIZE);
  if(VECTOR_SIZE>2) val+=__shfl_down(val,2,VECTOR_SIZE);
  if(VECTOR_SIZE>1) val+=__shfl_down(val,1,VECTOR_SIZE);
  return val;
}

template<int vec_size>
__device__ __inline__ double warpReduceSum2(double val) {
  if(vec_size>16) val+=__shfl_down(val,16,vec_size);
  if(vec_size>8) val+=__shfl_down(val,8,vec_size);
  if(vec_size>4) val+=__shfl_down(val,4,vec_size);
  if(vec_size>2) val+=__shfl_down(val,2,vec_size);
  if(vec_size>1) val+=__shfl_down(val,1,vec_size);
  return val;
}

// here is how OpenACC code for SpMV should look like in CUDA
// template<int non_zero_beta>
// __global__ void matvec_csr_vector_kernel(matrix A, double *X, double *Y, double alpha, double beta){

//   // 1 warp per row...
//   for(int row_idx=blockIdx.y*blockDim.y+threadIdx.y;row_idx<A.rows;row_idx+=blockDim.y*gridDim.y)
//   {
//     double sum=0;
//     int col_start=A.I_d[row_idx];
//     int col_end=A.I_d[row_idx+1];

//     for(int j=col_start+threadIdx.x;j<col_end;j+=VECTOR_SIZE)
//     {
//       if(j<col_end) {
//         int c=__ldg(A.J_d+j);    //Load with LDG to help with missaligned loads
//         double a=__ldg(A.val_d+j); //Load with LDG to help with missaligned loads
//         double x=__ldg(X+c);     //Load with LDG to help with scattered loads
//         sum+=a*x;
//       }
//     }
//     sum=warpReduceSum(sum);

//     if(threadIdx.x==0) {
//       if (non_zero_beta)
//        Y[row_idx] = alpha * sum + beta * Y[row_idx];
//       else
//        Y[row_idx] = alpha * sum;
//     }
//   }
// }

texture<int,1> tex_J;
texture<int2,1> tex_val;
texture<int2,1> tex_x;

#define BLOCK_DIM 128

static __inline__ __device__ double fetch_double(texture<int2, 1> t, int i)
{

  int2 v = tex1Dfetch(t,i);

  return __hiloint2double(v.y, v.x);

}


template<int non_zero_beta, int vec_size>
__global__ void matvec_csr_vector_kernel(matrix A, double *X, double *Y, double alpha, double beta){
  // __shared__ double buf[BLOCK_DIM/vec_size];
  int tid = threadIdx.y*vec_size + threadIdx.x;

  for (int row_idx=blockIdx.x*blockDim.y+threadIdx.y; row_idx<A.rows; row_idx+=blockDim.y*gridDim.x)
  {
    double sum=0;
    int col_start=A.I_d[row_idx];
    int col_end=A.I_d[row_idx+1];

    for(int j=col_start+threadIdx.x;j<col_end;j+=vec_size)
    {
      // if(j<col_end) {
        int c=__ldg(A.J_d+j);    //Load with LDG to help with missaligned loads
        double a=__ldg(A.val_d+j); //Load with LDG to help with missaligned loads
        // int c = A.J_d[j];
        // double a = A.val_d[j]; 
        double x=__ldg(X+c);     //Load with LDG to help with scattered loads
        // int c = tex1Dfetch(tex_J, j);
        // double a = fetch_double(tex_val, j);
        // double x = fetch_double(tex_x, c);
        sum+=a*x;
      // }
    }
    sum=warpReduceSum2<vec_size>(sum);
    // __syncthreads();
    // if (threadIdx.x==0) {
    //   buf[threadIdx.y] = sum;
    // }
    // __syncthreads();

    // if (tid < BLOCK_DIM/vec_size) {
    //   int row_idx_remapped = blockIdx.x*blockDim.y + tid;
    //   if (non_zero_beta) {
    //     Y[row_idx_remapped] = alpha * buf[tid] + beta*Y[row_idx_remapped];
    //   } else {
    //     Y[row_idx_remapped] = alpha * buf[tid];
    //   }
    // }
    if(threadIdx.x==0) {
      if (non_zero_beta)
       Y[row_idx] = alpha * sum + beta * Y[row_idx];
      else
       Y[row_idx] = alpha * sum;
    }
  }
}

__global__
void add_scaled_vector(int N, double *in, double scale, double *out)
{
  for (int i=threadIdx.x+blockIdx.x*blockDim.x; i<N; i+=gridDim.x*blockDim.x) {
    out[i] += scale*in[i];
  }
} 

__global__
void zero_vector(int N, double *in)
{
  for (int i=threadIdx.x+blockIdx.x*blockDim.x; i<N; i+=gridDim.x*blockDim.x) {
    in[i] = 0.;
  }
}

__global__ void matvecT_csr_vector_kernel(matrix A, double *X, double *Y, double alpha, double beta) {

  // 1 warp per row
  for (int row_idx=blockIdx.y*blockDim.y+threadIdx.y;row_idx<A.rows;row_idx+=blockDim.y*gridDim.y)
  {
    double Xi = X[row_idx];
    int col_start=A.I_d[row_idx];
    int col_end=A.I_d[row_idx+1];

    for (int jj=col_start+threadIdx.x; jj<col_end; jj += VECTOR_SIZE) {
      double val=0.;
      if (jj < col_end) {
        //int j = __ldg(A.J_d+jj);
        //double Aij = __ldg(A.val_d+jj);
        int j = A.J_d[jj];
        double Aij = A.val_d[jj];
        val = alpha*Aij*Xi;

        atomicAdd(&Y[j], val);
        // Y[j] += val;
      }
    }
  }
}

__global__
void transpose_matvec(matrix A, double *x, double *y, double alpha)
{
  for (int row=threadIdx.x+blockIdx.x*blockDim.x; row<A.rows; row+=gridDim.x*blockDim.x) {
    int col_start=A.I_d[row];
    int col_end=A.I_d[row+1];
    
    for (int jj=col_start; jj<col_end; jj++) {
      int j = A.J_d[jj];
      double Aij = A.val_d[jj];

      double val = alpha*Aij*x[row];

      atomicAdd(&y[j], val);
      // y[j] += val;
    }
  }
}

// void matvec_cuda(matrix A, double *x, double *y, double alpha, double beta){
//   dim3 BLOCK_SIZE;
//   BLOCK_SIZE.x=VECTOR_SIZE;
//   BLOCK_SIZE.y=BLOCK_DIM/VECTOR_SIZE;

//   dim3 BLOCKS;

//   BLOCKS.x=1;
//   BLOCKS.y=min((A.rows+BLOCK_SIZE.y-1)/BLOCK_SIZE.y,4096);
//   // BLOCKS.y=min((A.rows+BLOCK_SIZE.y-1)/BLOCK_SIZE.y,65535);

//   if (BLOCKS.y > 0) {
//     if (beta == 0.0)
//       matvec_csr_vector_kernel<0><<<BLOCKS,BLOCK_SIZE>>>(A, x, y, alpha, beta);
//     else
//       matvec_csr_vector_kernel<1><<<BLOCKS,BLOCK_SIZE>>>(A, x, y, alpha, beta);
//   }
// }


template <int vec_size>
void launch_matvec_cuda(matrix A, double *x, double *y, double alpha, double beta)
{
  dim3 BLOCK_SIZE;
  BLOCK_SIZE.x=vec_size;
  BLOCK_SIZE.y=BLOCK_DIM/vec_size;
  
  int BLOCKS;

  BLOCKS=(A.rows+BLOCK_SIZE.y-1)/BLOCK_SIZE.y;

  // CUCHK(cudaBindTexture(NULL, tex_J, A.J_d));
  // CUCHK(cudaBindTexture(NULL, tex_val, A.val_d));
  // CUCHK(cudaBindTexture(NULL, tex_x, x));

  if (BLOCKS > 0) {
    if (beta == 0.0)
      matvec_csr_vector_kernel<0,vec_size><<<BLOCKS,BLOCK_SIZE>>>(A, x, y, alpha, beta);
    else
      matvec_csr_vector_kernel<1,vec_size><<<BLOCKS,BLOCK_SIZE>>>(A, x, y, alpha, beta);
  }
  CUCHK(cudaGetLastError());
  // CUCHK(cudaUnbindTexture(tex_J));
  // CUCHK(cudaUnbindTexture(tex_val));
  // CUCHK(cudaUnbindTexture(tex_x));
}
void matvec_cuda(matrix A, double *x, double *y, double alpha, double beta)
{  
  //printf("A.rows=%d, A.nnz=%d\n", A.rows, A.nnz);
  int nzpr = A.nnz/A.rows;
  if (nzpr <= 4)   
    launch_matvec_cuda<1>(A, x, y, alpha, beta);
  else if (nzpr <= 16)
    launch_matvec_cuda<4>(A, x, y, alpha, beta);
  else if (nzpr <= 32) 
    launch_matvec_cuda<8>(A, x, y, alpha, beta);
  else if (nzpr <= 64)
    launch_matvec_cuda<16>(A, x, y, alpha, beta);
  else
    launch_matvec_cuda<32>(A, x, y, alpha, beta);
  // cudaDeviceSynchronize();
  // exit(0);
}

// temporary vector for non-zero beta
double *Y_initial = NULL;
int current_y_size = 0;

void matvecT_cuda(matrix A, double *x, double *y, double alpha, double beta)
{
  dim3 BLOCK_SIZE;
  BLOCK_SIZE.x = VECTOR_SIZE;
  BLOCK_SIZE.y = 256 / VECTOR_SIZE;

  dim3 BLOCKS;

  BLOCKS.x = 1;
  BLOCKS.y = min((A.rows+BLOCK_SIZE.y-1)/BLOCK_SIZE.y, 4096);

  // initialise and copy original Y if necessary
  if (beta != 0.0) {
    if (Y_initial != NULL && current_y_size < A.cols) {
      cudaFree(Y_initial);
      printf("allocating for %d cols\n",A.cols);
      CUCHK(cudaMalloc((void **)&Y_initial, A.cols*sizeof(double)));
      current_y_size = A.cols;
    } else {
      CUCHK(cudaMalloc((void **)&Y_initial, A.cols*sizeof(double)));
      current_y_size = A.cols;
    }
    cudaMemcpy(Y_initial, y, A.cols*sizeof(double), cudaMemcpyDeviceToDevice);
  }

  zero_vector<<<4096,512>>>(A.cols, y);
  CUCHK(cudaGetLastError());
  if (BLOCKS.y > 0) {
    matvecT_csr_vector_kernel<<<BLOCKS, BLOCK_SIZE>>>(A,x,y,alpha,beta);
    CUCHK(cudaGetLastError());
    // transpose_matvec<<<4096, 512>>>(A,x,y,alpha);

    // if necessary, add on beta*y
    if (beta != 0.0) {
      add_scaled_vector<<<4096, 512>>>(A.cols, y, beta, Y_initial);
      CUCHK(cudaGetLastError());
    }
  }
}


