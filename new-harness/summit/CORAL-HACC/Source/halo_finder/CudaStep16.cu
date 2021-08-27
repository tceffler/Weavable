#include <stdio.h>
#include "batch.h"
//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                          \
        cudaError_t e=cudaGetLastError();                                 \
        if(e!=cudaSuccess) {                                              \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }

#define __shfl_sync(a,b,c) __shfl(b,c)  //work around, inefficient volta compiler

static __device__ __forceinline__ float __internal_fast_rsqrtf(float a)
{
  float r;
  asm ("rsqrt.approx.ftz.f32 %0,%1;" : "=f"(r) : "f"(a));
  return r;
}

__global__ void sm_multi_copy_kernel(float *d_nx, float *d_ny, float *d_nz, float *d_nm,
                                 float *nx, float *ny, float *nz, float *nm, int count) {
  for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x) {
    d_nx[i]=nx[i];
    d_ny[i]=ny[i];
    d_nz[i]=nz[i];
    d_nm[i]=nm[i];
  }
}

extern "C" {
	void sm_multi_copy(float *d_nx, float *d_ny, float *d_nz, float *d_nm,
			const float *nx, const float *ny, const float *nz, const float *nm, int count, cudaStream_t stream) {
    float *dh_nx, *dh_ny, *dh_nz, *dh_nm;
		//TODO is this necessary?
		cudaHostGetDevicePointer(&dh_nx,const_cast<float*>(nx),0);
		cudaHostGetDevicePointer(&dh_ny,const_cast<float*>(ny),0);
		cudaHostGetDevicePointer(&dh_nz,const_cast<float*>(nz),0);
		cudaHostGetDevicePointer(&dh_nm,const_cast<float*>(nm),0);

		int threads=128;
		int lim=128;
		size_t blocks=(count+threads-1)/threads;
		if(blocks>lim) blocks=lim;
		sm_multi_copy_kernel<<<blocks,threads,0,stream>>>(d_nx, d_ny, d_nz, d_nm, dh_nx, dh_ny, dh_nz, dh_nm, count);
	}
}




#define BLOCKX (32)                     //Block size in the x dimension (should be >=32)
#define BLOCKY 2                     //Block size in the y dimension,
#define MAXX 1                       //Maximum blocks in the X dimension, smaller=more reuse but less parallelism (should be 1)
#define MAXY (2560)                      //Maximum blocks in the Y dimension, smaller = more overlap

      
__launch_bounds__(BLOCKX*BLOCKY,2048/BLOCKX/BLOCKY) //100% occupancy
__global__
void Step16_cuda_kernel(int count, int count1,
                        const float* __restrict__ xx, const float* __restrict__ yy,
                        const float* __restrict__ zz, const float* __restrict__ mass,
                        const float* __restrict__ xx1, const float* __restrict__ yy1,
                        const float* __restrict__ zz1, const float* __restrict__ mass1,
                        float* __restrict__ vx, float* __restrict__ vy,
                        float* __restrict__ vz, float fsrrmax2, float mp_rsm2, float fcoeff)
{
  __shared__ float smem[BLOCKY][3][32];
    float *sx=smem[threadIdx.y][0];
    float *sy=smem[threadIdx.y][1];
    float *sz=smem[threadIdx.y][2];
  
    const float ma0 = 0.269327, ma1 = -0.0750978, ma2 = 0.0114808, ma3 = -0.00109313, ma4 = 0.0000605491, ma5 = -0.00000147177;

    for(int i=blockIdx.y*BLOCKY+threadIdx.y;i<count;i+=BLOCKY*gridDim.y) {
      float xi = 0., yi = 0., zi = 0.;
      float xxi, yyi, zzi;
      xxi=xx[i];
      yyi=yy[i];
      zzi=zz[i];
      float xxj, yyj, zzj, massj;


      for (int j = blockIdx.x*BLOCKX+threadIdx.x;j<count1;j+=BLOCKX*gridDim.x) {   //1 IMAD 1 ISETP

        xxj = xx1[j];                                                                      //1 IMAD.WIDE, 1 LD
        yyj = yy1[j];                                                                      //1 IMAD.WIDE, 1 LD
        zzj = zz1[j];                                                                      //1 IMAD.WIDE, 1 LD
        massj = mass1[j];                                                                  //1 IMAD.WIDE, 1 LD
        __threadfence_block();                                                             //1 MEMBAR, threadfence to help with cache hits

        float dxc = xxj - xxi;                                                          //1 FADD
        float dyc = yyj - yyi;                                                          //1 FADD
        float dzc = zzj - zzi;                                                          //1 FADD

        float r2 = dxc * dxc + dyc * dyc + dzc * dzc;                                   //1 FMUL 2 FMA

        if (r2<fsrrmax2 && r2>0.0f) {                                                      //2 FSETP
          float v=r2+mp_rsm2;                                                           //1 FADD
          float v3=v*v*v;                                                               //2 FMUL
          float f = __internal_fast_rsqrtf(v3)                                          //1 MUFU,
                - ( ma0 + r2*(ma1 + r2*(ma2 + r2*(ma3 + r2*(ma4 + r2*ma5)))));            //5 FMA, 1 FADD

          f*=massj;                                                                        //1 FMUL

          xi += f * dxc;                                                                   //1 FMA
          yi += f * dyc;                                                                   //1 FMA
          zi += f * dzc;                                                                   //1 FMA
        }
      }
      if(BLOCKX>16) {
        xi+=__shfl_down_sync(0xFFFFFFFF,xi, 16);
        yi+=__shfl_down_sync(0xFFFFFFFF,yi, 16);
        zi+=__shfl_down_sync(0xFFFFFFFF,zi, 16);
      }
      if(BLOCKX>8) {
        xi+=__shfl_down_sync(0xFFFFFFFF,xi, 8);
        yi+=__shfl_down_sync(0xFFFFFFFF,yi, 8);
        zi+=__shfl_down_sync(0xFFFFFFFF,zi, 8);
      }
      if(BLOCKX>4) {
        xi+=__shfl_down_sync(0xFFFFFFFF,xi, 4);
        yi+=__shfl_down_sync(0xFFFFFFFF,yi, 4);
        zi+=__shfl_down_sync(0xFFFFFFFF,zi, 4);
      }
      if(BLOCKX>2) {
        xi+=__shfl_down_sync(0xFFFFFFFF,xi, 2);
        yi+=__shfl_down_sync(0xFFFFFFFF,yi, 2);
        zi+=__shfl_down_sync(0xFFFFFFFF,zi, 2);
      }
      if(BLOCKX>1) {
        xi+=__shfl_down_sync(0xFFFFFFFF,xi, 1);
        yi+=__shfl_down_sync(0xFFFFFFFF,yi, 1);
        zi+=__shfl_down_sync(0xFFFFFFFF,zi, 1);
      }
      //reduced across each warp here
      
      //Check if we need to reduce across warps
      if(BLOCKX>32) {
        //swizzle data to the first warp
        int widx = threadIdx.x/32;
        int woff = threadIdx.x%32; 
        if(woff==0) {
          sx[widx]=xi;
          sy[widx]=yi;
          sz[widx]=zi;
        }
        __syncthreads();  //wait for everyone to write, could be diveged if BLOCKY>1 but we don't want to run with BLOCKY>1 anyway

        if(widx==0) {
					xi = sx[woff];
					yi = sy[woff];
					zi = sz[woff];

					if(BLOCKX/32>16) {
						xi+=__shfl_down_sync(0xFFFFFFFF,xi, 16);
						yi+=__shfl_down_sync(0xFFFFFFFF,yi, 16);
						zi+=__shfl_down_sync(0xFFFFFFFF,zi, 16);
					}
					if(BLOCKX/32>8) {
						xi+=__shfl_down_sync(0xFFFFFFFF,xi, 8);
						yi+=__shfl_down_sync(0xFFFFFFFF,yi, 8);
						zi+=__shfl_down_sync(0xFFFFFFFF,zi, 8);
					}
					if(BLOCKX/32>4) {
						xi+=__shfl_down_sync(0xFFFFFFFF,xi, 4);
						yi+=__shfl_down_sync(0xFFFFFFFF,yi, 4);
						zi+=__shfl_down_sync(0xFFFFFFFF,zi, 4);
					}
					if(BLOCKX/32>2) {
						xi+=__shfl_down_sync(0xFFFFFFFF,xi, 2);
						yi+=__shfl_down_sync(0xFFFFFFFF,yi, 2);
						zi+=__shfl_down_sync(0xFFFFFFFF,zi, 2);
					}
					if(BLOCKX/32>1) {
						xi+=__shfl_down_sync(0xFFFFFFFF,xi, 1);
						yi+=__shfl_down_sync(0xFFFFFFFF,yi, 1);
						zi+=__shfl_down_sync(0xFFFFFFFF,zi, 1);
					}
        }
      }
      //reduced across blockDim.x now

			//apply update to global memory
      if(threadIdx.x==0) {
        //atomics only needed if we more more than 1 block in the x dimension
        if(MAXX==1) {
          vx[i]+=fcoeff*xi;
          vy[i]+=fcoeff*yi;
          vz[i]+=fcoeff*zi;
        } else {
          atomicAdd(vx+i,fcoeff*xi);
          atomicAdd(vy+i,fcoeff*yi);
          atomicAdd(vz+i,fcoeff*zi);
        }
      }
  }
}

//__launch_bounds__(BLOCKX*BLOCKY,24) //100% occupancy
__launch_bounds__(BLOCKX*BLOCKY,2048/BLOCKX/BLOCKY) //100% occupancy
__global__
void Step16_cuda_kernel_batched(BatchInfo batch, float fsrrmax2, float mp_rsm2, float fcoeff)
{
  
  int count=batch.count_[blockIdx.z];
  int count1=batch.count1_[blockIdx.z];

  const float ma0 = 0.269327, ma1 = -0.0750978, ma2 = 0.0114808, ma3 = -0.00109313, ma4 = 0.0000605491, ma5 = -0.00000147177;

  //for(int i=count-1-blockIdx.y*BLOCKY+threadIdx.y;i>=0;i-=BLOCKY*gridDim.y) {
  for(int i=blockIdx.y*BLOCKY+threadIdx.y;i<count;i+=BLOCKY*gridDim.y) {
  
    float *xx=batch.xx_[blockIdx.z];
    float *yy=batch.yy_[blockIdx.z];
    float *zz=batch.zz_[blockIdx.z];
    
    float xi = 0., yi = 0., zi = 0.;
    float xxi, yyi, zzi;
    xxi=xx[i];
    yyi=yy[i];
    zzi=zz[i];
      
    float *xx1=batch.xx1_[blockIdx.z];
    float *yy1=batch.yy1_[blockIdx.z];
    float *zz1=batch.zz1_[blockIdx.z];
    float *mass1=batch.mass1_[blockIdx.z];

    //for (int j = count1-1-blockIdx.x*BLOCKX+threadIdx.x;j>=0;j-=BLOCKX*gridDim.x) {   //1 IMAD 1 ISETP
    for (int j = blockIdx.x*BLOCKX+threadIdx.x;j<count1;j+=BLOCKX*gridDim.x) {   //1 IMAD 1 ISETP
      float xxj, yyj, zzj, massj;

      xxj = xx1[j];                                                                      //1 IMAD.WIDE, 1 LD
      yyj = yy1[j];                                                                      //1 IMAD.WIDE, 1 LD
      zzj = zz1[j];                                                                      //1 IMAD.WIDE, 1 LD
      massj = mass1[j];                                                                  //1 IMAD.WIDE, 1 LD
      __threadfence_block();                                                             //1 MEMBAR, threadfence to help with cache hits

      float dxc = xxj - xxi;                                                          //1 FADD
      float dyc = yyj - yyi;                                                          //1 FADD
      float dzc = zzj - zzi;                                                          //1 FADD

      float r2 = dxc * dxc + dyc * dyc + dzc * dzc;                                   //1 FMUL 2 FMA

      if (r2<fsrrmax2 && r2>0.0f) {                                                      //2 FSETP
        float v=r2+mp_rsm2;                                                           //1 FADD
        float v3=v*v*v;                                                               //2 FMUL
        float f = __internal_fast_rsqrtf(v3)                                          //1 MUFU,
              - ( ma0 + r2*(ma1 + r2*(ma2 + r2*(ma3 + r2*(ma4 + r2*ma5)))));            //5 FMA, 1 FADD

        f*=massj;                                                                        //1 FMUL

        xi += f * dxc;                                                                   //1 FMA
        yi += f * dyc;                                                                   //1 FMA
        zi += f * dzc;                                                                   //1 FMA
      }
    }
    
    if(BLOCKX>16) {
      xi+=__shfl_down_sync(0xFFFFFFFF,xi, 16);
      yi+=__shfl_down_sync(0xFFFFFFFF,yi, 16);
      zi+=__shfl_down_sync(0xFFFFFFFF,zi, 16);
    }
    if(BLOCKX>8) {
      xi+=__shfl_down_sync(0xFFFFFFFF,xi, 8);
      yi+=__shfl_down_sync(0xFFFFFFFF,yi, 8);
      zi+=__shfl_down_sync(0xFFFFFFFF,zi, 8);
    }
    if(BLOCKX>4) {
      xi+=__shfl_down_sync(0xFFFFFFFF,xi, 4);
      yi+=__shfl_down_sync(0xFFFFFFFF,yi, 4);
      zi+=__shfl_down_sync(0xFFFFFFFF,zi, 4);
    }
    if(BLOCKX>2) {
      xi+=__shfl_down_sync(0xFFFFFFFF,xi, 2);
      yi+=__shfl_down_sync(0xFFFFFFFF,yi, 2);
      zi+=__shfl_down_sync(0xFFFFFFFF,zi, 2);
    }
    if(BLOCKX>1) {
      xi+=__shfl_down_sync(0xFFFFFFFF,xi, 1);
      yi+=__shfl_down_sync(0xFFFFFFFF,yi, 1);
      zi+=__shfl_down_sync(0xFFFFFFFF,zi, 1);
    }
    //reduced across each warp here

    //Check if we need to reduce across warps
    if(BLOCKX>32) {
      __shared__ float smem[BLOCKY][3][32];
      
      float *sx=smem[threadIdx.y][0];
      float *sy=smem[threadIdx.y][1];
      float *sz=smem[threadIdx.y][2];
      //swizzle data to the first warp
      int widx = threadIdx.x/32;
      int woff = threadIdx.x%32; 
      if(woff==0) {
        sx[widx]=xi;
        sy[widx]=yi;
        sz[widx]=zi;
      }
      __syncthreads();  //wait for everyone to write, could be diveged if BLOCKY>1 but we don't want to run with BLOCKY>1 anyway

      if(widx==0) {
        xi = sx[woff];
        yi = sy[woff];
        zi = sz[woff];

        if(BLOCKX/32>16) {
          xi+=__shfl_down_sync(0xFFFFFFFF,xi, 16);
          yi+=__shfl_down_sync(0xFFFFFFFF,yi, 16);
          zi+=__shfl_down_sync(0xFFFFFFFF,zi, 16);
        }
        if(BLOCKX/32>8) {
          xi+=__shfl_down_sync(0xFFFFFFFF,xi, 8);
          yi+=__shfl_down_sync(0xFFFFFFFF,yi, 8);
          zi+=__shfl_down_sync(0xFFFFFFFF,zi, 8);
        }
        if(BLOCKX/32>4) {
          xi+=__shfl_down_sync(0xFFFFFFFF,xi, 4);
          yi+=__shfl_down_sync(0xFFFFFFFF,yi, 4);
          zi+=__shfl_down_sync(0xFFFFFFFF,zi, 4);
        }
        if(BLOCKX/32>2) {
          xi+=__shfl_down_sync(0xFFFFFFFF,xi, 2);
          yi+=__shfl_down_sync(0xFFFFFFFF,yi, 2);
          zi+=__shfl_down_sync(0xFFFFFFFF,zi, 2);
        }
        if(BLOCKX/32>1) {
          xi+=__shfl_down_sync(0xFFFFFFFF,xi, 1);
          yi+=__shfl_down_sync(0xFFFFFFFF,yi, 1);
          zi+=__shfl_down_sync(0xFFFFFFFF,zi, 1);
        }
      }
    }
    //reduced across blockDim.x now

    //apply update to global memory
    if(threadIdx.x==0) {
      float *vx=batch.vx_[blockIdx.z];
      float *vy=batch.vy_[blockIdx.z];
      float *vz=batch.vz_[blockIdx.z];
      //atomics only needed if we more more than 1 block in the x dimension
      if(MAXX==1) {
        vx[i]+=fcoeff*xi;
        vy[i]+=fcoeff*yi;
        vz[i]+=fcoeff*zi;
      } else {
        atomicAdd(vx+i,fcoeff*xi);
        atomicAdd(vy+i,fcoeff*yi);
        atomicAdd(vz+i,fcoeff*zi);
      }
    }
  }
}


extern "C" {
  void CudaStep16(int count, int count1, const float* __restrict xx, const float* __restrict yy,
      const float* __restrict zz, const float* __restrict mass,
      const float* __restrict xx1, const float* __restrict yy1,
      const float* __restrict zz1, const float* __restrict mass1,
      float* __restrict vx, float* __restrict vy, float* __restrict vz,
      float fcoeff, float fsrrmax, float rsm, cudaStream_t stream) {
    dim3 threads(BLOCKX,BLOCKY);
    int blocksX=(count1+threads.x-1)/threads.x;
    int blocksY=(count+threads.y-1)/threads.y;
    dim3 blocks( min(blocksX,MAXX), min(blocksY,MAXY));

		//call kernel
    Step16_cuda_kernel<<<blocks,threads,0,stream>>>(count,count1,xx,yy,zz,mass,xx1,yy1,zz1,mass1, vx, vy, vz, fsrrmax*fsrrmax, rsm*rsm, fcoeff);
  }
  
  void CudaStep16Batched(BatchInfo batch, float fcoeff, float fsrrmax, float rsm, cudaStream_t stream) {
    dim3 threads(BLOCKX,BLOCKY);
    //TODO TUNE THIS
    int blocksX=1;
    int blocksY=128;
    int blocksZ=batch.size;
    dim3 blocks( blocksX, blocksY, blocksZ);

		//call kernel
    Step16_cuda_kernel_batched<<<blocks,threads,0,stream>>>(batch, fsrrmax*fsrrmax, rsm*rsm, fcoeff);
  }


  void my_acc_wait_async(cudaStream_t stream1, cudaStream_t stream2) {
    static cudaEvent_t event;
    static bool first=true;
    if(first) {
      cudaEventCreateWithFlags(&event,cudaEventDisableTiming);
      first=false;
    }

    cudaEventRecord(event,stream1);
    cudaStreamWaitEvent(stream2,event,0);
    
    //cudaEventDestroy(event);
  }
}


