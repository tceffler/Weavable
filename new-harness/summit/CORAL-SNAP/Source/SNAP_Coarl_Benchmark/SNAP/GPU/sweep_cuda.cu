#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string>
#include "mpi.h"
#include "omp.h"
#include "cuda_var.h"
#include <unistd.h>
#include <stdint.h>
#include <sched.h>

extern "C" {
void destroy_ib();
int check_recv(unsigned int *recvV);
int post_control(int Q);
void post_send(unsigned int imm, int Q, size_t Soffset, size_t Roffset, int size);
void post_recv(unsigned int imm, int Q);
void query_qp();
void setup_ib_yz(double* h_Buf, size_t h_BufSize, int npey, int npez, int pey, int pez, int NA, int NG, int NC, int nTBG, MPI_Comm y_comm, MPI_Comm z_comm,int myrank);
__global__ void 
dim3_kernel(
  int *dogrp,  
  int ichunk, int jchunk, int kchunk, int achunk, int oct, int ndimen, 
  int nx, int ny, int nz, int nang, int noct,
  int NA, int NC, int NG, int cmom, int src_opt, int fixup,
  int tbY, int tbZ, int nTBG,
  int ptrNple,
  int timedep,
  const double* __restrict__ vdelt, const double* __restrict__ w, double *t_xs,
  double tolr, double hi, double *hj, double *hk, double *mu,
  const double* __restrict__ qtot, const double* __restrict__ ec, 
  double *dinv, double *qim,
  double *psi_save, double *flux, double* fluxm,
  volatile double* d_buf_y, volatile double* d_buf_z, int bSizeY,int bSizeZ,int dNple,
  double* d_ptrin, double* d_ptrout, volatile int* ptrin_rdy, volatile int* ptrin_dne,
  volatile int* seqINy, volatile int* seqINz, volatile int* d_seqOUTy, volatile int* d_seqOUTz,
  int* angrpBG, int maxAngrp, int rank, int yzFlip ,
  int bufNple, double* h_RBufY, double* h_RBufZ, double* h_SBufY, double* h_SBufZ
  );
}

#define NOTYRECV 0x4
#define NOTZRECV 0x8

class CTimer
{

protected:

  timeval start,end;
  double et;

public:

  CTimer() : et(0) {;}

  void Start() {gettimeofday(&start, NULL);}

  void End() {
    gettimeofday(&end,NULL);
    et+=(end.tv_sec+end.tv_usec*0.000001)-(start.tv_sec+start.tv_usec*0.000001);
  }

  double GetET() { return et; }
};

unsigned int clp2(unsigned int x) {
  x = x - 1; 
  x = x | (x >> 1); 
  x = x | (x >> 2); 
  x = x | (x >> 4); 
  x = x | (x >> 8); 
  x = x | (x >>16); 
  return x + 1; 
} 


extern "C"
void setgpu_(int *iproc)
{
  int myrank, nDevice;
  myrank = *iproc;
  CUDA_SAFE_CALL(cudaGetDeviceCount(&nDevice));
  int iDevice = myrank % nDevice;
  CUDA_SAFE_CALL( cudaSetDevice(iDevice) );  
}


extern "C"
void setup_cuda_(int *timedep, int *iproc, int *fort_nang, int *fort_ichunk, 
                 int *fort_nx, int *fort_ny, int *fort_nz, 
                 int *fort_ng, int *fort_noct, int *fort_cmom, int *fort_nmom, 
                 int *src_opt, int *fort_nmat,
                 int *ycomm, int *zcomm,
                 int *npey, int *npez, int *pey, int *pez)

{
  // convert to c format
  tbY=4;
  tbZ=4;
  nTBG=5;
  bufNple = 8;
  ptrNple = 12;
  ichunk = *fort_ichunk;
  jchunk = 4;
  kchunk = 4;
  achunk = WARP_SIZE;
  bchunk = achunk >> 1;
  nx = *fort_nx; ny = *fort_ny; nz = *fort_nz;
  nang = *fort_nang; noct=*fort_noct; nmat = *fort_nmat;
  NC = nx/ichunk;
  NA = 3;
  NG = *fort_ng;
  ng = NG;
  nmom = *fort_nmom;
  cmom = *fort_cmom;

  //note that maxAngrp can be different in kernel
  //is that OK?
  maxAngrp = (NA*NG)/nTBG + (((NA*NG)%nTBG)==0?0:1);
  if (*iproc == 0) printf("NA=%d NG=%d maxAgnrp=%d\n",NA,NG,maxAngrp);

  // some parameter check
  if (nang > 1024) {
    printf("Maximal supported nang for GPU: 1024\n");
    exit(1);
  }

  char proc_name[MPI_MAX_PROCESSOR_NAME];
  int proc_len;
  MPI_Get_processor_name(proc_name, &proc_len);

  int igpu, ngpu;
#ifdef ORNL
  int gpumap[4]={0,1,3,4};  // ORNL
#else
  int gpumap[4]={0,1,2,3};  // LLNL
#endif
  CUDA_SAFE_CALL( cudaGetDeviceCount(&ngpu) );
  //igpu = (*iproc) % ngpu;
  igpu = gpumap[(*iproc)%4];
  CUDA_SAFE_CALL( cudaSetDevice(igpu) );
  CUDA_SAFE_CALL( cudaGetDevice(&igpu) );
  cudaDeviceProp prop;
  CUDA_SAFE_CALL( cudaGetDeviceProperties(&prop, igpu) );
//   printf("P(%d) on %s: Using GPU %d (%s) of %d on %d\n", 
//          *iproc, proc_name, igpu, prop.name, ngpu,sched_getcpu());

  CUDA_SAFE_CALL( cudaSetDeviceFlags(cudaDeviceMapHost) );
  CUDA_SAFE_CALL( cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte) );
  CUDA_SAFE_CALL( cudaDeviceSetCacheConfig(cudaFuncCachePreferL1) );

  //
  // Compute parameters for sweep kernel
  //
  

  // gpu_batch_size is set by the user
  //gpu_batch_size = GPU_BATCH_SIZE;

  // compute #sm used per kernel
  int numsm; cudaDeviceGetAttribute(&numsm,  cudaDevAttrMultiProcessorCount,  igpu);  
  int regcnt; cudaDeviceGetAttribute(&regcnt, cudaDevAttrMaxRegistersPerBlock, igpu);
  if (*iproc == 0)
  {
    printf("My gpu has %d streaming multiprocessors\n",numsm);
    printf("My gpu has %d registers\n",regcnt);
  }

  cublasStatus_t stat = cublasCreate(&cublasHandle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("cublasCreate Failed\n");
    exit(1);
  }
  
  //device memory allocation

  double memUsed=0;
  // geom module
  CUDA_SAFE_CALL(cudaMalloc(&d_dinv, ((size_t) nang)*((size_t) nx)*((size_t) ny)*((size_t) nz)*((size_t) ng)*sizeof(double))); memUsed+= ((size_t) nang)*((size_t) nx)*((size_t) ny)*((size_t) nz)*((size_t) ng)*sizeof(double); 
  CUDA_SAFE_CALL(cudaMalloc(&d_hj, nang*sizeof(double))); memUsed+= nang*sizeof(double); 
  CUDA_SAFE_CALL(cudaMalloc(&d_hk, nang*sizeof(double))); memUsed+= nang*sizeof(double); 
  CUDA_SAFE_CALL( cudaGetLastError() );

  // sn module
  CUDA_SAFE_CALL(cudaMalloc(&d_lma, nmom*sizeof(int))); memUsed+= nmom*sizeof(int); 
  CUDA_SAFE_CALL(cudaMalloc(&d_ec, ((size_t) nang)*((size_t) cmom)*((size_t) noct)*sizeof(double))); memUsed+= ((size_t) nang)*((size_t) cmom)*((size_t) noct)*sizeof(double); 
  CUDA_SAFE_CALL(cudaMalloc(&d_mu, nang*sizeof(double))); memUsed+= nang*sizeof(double); 
  CUDA_SAFE_CALL(cudaMalloc(&d_w, nang*sizeof(double))); memUsed+= nang*sizeof(double); 
  CUDA_SAFE_CALL(cudaMalloc(&d_eta, nang*sizeof(double))); memUsed+= nang*sizeof(double); 
  CUDA_SAFE_CALL(cudaMalloc(&d_xi, nang*sizeof(double))); memUsed+= nang*sizeof(double); 
  CUDA_SAFE_CALL( cudaGetLastError() );

  // data module
  CUDA_SAFE_CALL(cudaMalloc(&d_mat, ((size_t) nx)*((size_t) ny)*((size_t) nz)*sizeof(int))); memUsed+= ((size_t) nx)*((size_t) ny)*((size_t) nz)*sizeof(int); 
  CUDA_SAFE_CALL(cudaMalloc(&d_vdelt, ng*sizeof(double))); memUsed+= ng*sizeof(double); 
  CUDA_SAFE_CALL(cudaMalloc(&d_sigt, ((size_t) nmat)*((size_t) ng)*sizeof(double))); memUsed+= ((size_t) nmat)*((size_t) ng)*sizeof(double); 
  CUDA_SAFE_CALL(cudaMalloc(&d_siga, ((size_t) nmat)*((size_t) ng)*sizeof(double))); memUsed+= ((size_t) nmat)*((size_t) ng)*sizeof(double); 
  CUDA_SAFE_CALL(cudaMalloc(&d_slgg, ((size_t) nmat)*((size_t) nmom)*((size_t) ng)*((size_t) ng)*sizeof(double))); memUsed+= ((size_t) nmat)*((size_t) nmom)*((size_t) ng)*((size_t) ng)*sizeof(double); 
  
  CUDA_SAFE_CALL( cudaGetLastError() );

  // solvar module
  //cudaMalloc(&d_psii, ((size_t) *nang)*((size_t) *ny)*((size_t) *nz)*((size_t) *ng)*sizeof(double)); memUsed+= ((size_t) *nang)*((size_t) *ny)*((size_t) *nz)*((size_t) *ng)*sizeof(double; printf("&d_psii %e\n",memUsed);
  //cudaMalloc(&d_psij, ((size_t) *nang)*((size_t) *ichunk)*((size_t) *nz)*((size_t) *ng)*sizeof(double)); memUsed+= ((size_t) *nang)*((size_t) *ichunk)*((size_t) *nz)*((size_t) *ng)*sizeof(double; printf("&d_psij %e\n",memUsed);
  //cudaMalloc(&d_psik, ((size_t) *nang)*((size_t) *ichunk)*((size_t) *ny)*((size_t) *ng)*sizeof(double)); memUsed+= ((size_t) *nang)*((size_t) *ichunk)*((size_t) *ny)*((size_t) *ng)*sizeof(double; printf("&d_psik %e\n",memUsed);

  const int NG=ng;
  const int NA=(int)(nang/(achunk>>1)) + (nang%(achunk>>1)==0?0:1); //only work for 48 angles
  if (nang%(achunk>>1)) printf("nang is not multiple of 16. No so efficient\n");
  // psi_save
  int numtb = 2 * numsm;  //nTBG * tbY * tbZ
  //double *d_psi_save;
  CUDA_SAFE_CALL(cudaMalloc(&d_psi_save, ((size_t) numtb)*((size_t) WARP_SIZE)*((size_t) nx)*((size_t) jchunk)*((size_t) kchunk)*sizeof(double) )); memUsed+= ((size_t) numtb)*((size_t) WARP_SIZE)*((size_t) nx)*((size_t) jchunk)*((size_t) kchunk)*sizeof(double) ; 

  if (*timedep == 1)
  {
    //buffer in the device
    CUDA_SAFE_CALL(cudaMalloc(&d_ptrin, ptrNple * nTBG * tbY*tbZ * jchunk*kchunk * NC*ichunk * achunk * sizeof(double))); memUsed+= ptrNple * nTBG * tbY*tbZ * jchunk*kchunk * NC*ichunk * achunk * sizeof(double); 
//    #define d_ptrin(a,t,j,k) d_ptrin[jchunk*kchunk * NC*ichunk * achunk * ( (a) % ptrNple + ptrNple *( t + nTBG*(j + tbY*(k))))]
    #define d_ptrin(a,t,j,k) d_ptrin[jchunk*kchunk * NC*ichunk * achunk * (j + tbY*(k+tbZ*( (a) % ptrNple + ptrNple * (t) )))]
    CUDA_SAFE_CALL(cudaMalloc(&d_ptrout, ptrNple * nTBG * tbY*tbZ * jchunk*kchunk * NC*ichunk * achunk * sizeof(double))); memUsed+= ptrNple * nTBG * tbY*tbZ * jchunk*kchunk * NC*ichunk * achunk * sizeof(double); 
//    #define d_ptrout(a,t,j,k) d_ptrout[jchunk*kchunk * NC*ichunk * achunk * ( (a) % ptrNple + ptrNple *( t + nTBG*(j + tbY*(k))))]
    #define d_ptrout(a,t,j,k) d_ptrout[jchunk*kchunk * NC*ichunk * achunk * (j + tbY*(k+tbZ*( (a) % ptrNple + ptrNple * (t) )))]

    //pinned mem in host
    size_t ptr_size = ((size_t)NG)*((size_t)NC)*((size_t)ichunk)*((size_t)NA)*((size_t)achunk)*((size_t)tbY)*((size_t)jchunk)*((size_t)tbZ)*((size_t)kchunk)* 4 ;
    
    CUDA_SAFE_CALL(cudaMallocHost(&h_ptrout, ptr_size *  sizeof(double) ));
    CUDA_SAFE_CALL(cudaMallocHost(&h_ptrin, ptr_size * sizeof(double) ));
    #define h_ptrin(j,k,g,e,o)   h_ptrin[(size_t)(achunk*jchunk*kchunk*NC*ichunk)*((size_t)(j) + tbY*(k + tbZ*(g + NG*(e + NA*(o)))))]
    #define h_ptrout(j,k,g,e,o) h_ptrout[(size_t)(achunk*jchunk*kchunk*NC*ichunk)*((size_t)(j) + tbY*(k + tbZ*(g + NG*(e + NA*(o)))))]
//    printf("pinned succeeded\n");
    
  }
  CUDA_SAFE_CALL( cudaGetLastError() );

  CUDA_SAFE_CALL(cudaMalloc(&d_t_xs, ((size_t) nx)*((size_t) ny)*((size_t) nz)*((size_t) ng)*sizeof(double))); memUsed+= ((size_t) nx)*((size_t) ny)*((size_t) nz)*((size_t) ng)*sizeof(double); 
  CUDA_SAFE_CALL(cudaMalloc(&d_a_xs, ((size_t) nx)*((size_t) ny)*((size_t) nz)*((size_t) ng)*sizeof(double))); memUsed+= ((size_t) nx)*((size_t) ny)*((size_t) nz)*((size_t) ng)*sizeof(double); 
  CUDA_SAFE_CALL(cudaMalloc(&d_s_xs, ((size_t) nmom)*((size_t) nx)*((size_t) ny)*((size_t) nz)*((size_t) ng)*sizeof(double))); memUsed+= ((size_t) nmom)*((size_t) nx)*((size_t) ny)*((size_t) nz)*((size_t) ng)*sizeof(double); 
  CUDA_SAFE_CALL(cudaMalloc(&d_qi, ((size_t) nx)*((size_t) ny)*((size_t) nz)*((size_t) ng)*sizeof(double))); memUsed+= ((size_t) nx)*((size_t) ny)*((size_t) nz)*((size_t) ng)*sizeof(double); 
  CUDA_SAFE_CALL(cudaMalloc(&d_q2grp, ((size_t) cmom)*((size_t) nx)*((size_t) ny)*((size_t) nz)*((size_t) ng)*sizeof(double))); memUsed+= ((size_t) cmom)*((size_t) nx)*((size_t) ny)*((size_t) nz)*((size_t) ng)*sizeof(double); 
  CUDA_SAFE_CALL(cudaMemset(d_q2grp, 0, ((size_t) cmom)*((size_t) nx)*((size_t) ny)*((size_t) nz)*((size_t) ng)*sizeof(double)));
  
  if (*src_opt == 3) {
    //cannot handle this now. qim will be large
    printf("this version may not support src_opt ==3 because of memory\n");
    CUDA_SAFE_CALL(cudaMalloc(&d_qim, ((size_t) nang)*((size_t) nx)*((size_t) ny)*((size_t) nz)*((size_t) noct)*((size_t) ng)*sizeof(double))); memUsed+= ((size_t) nang)*((size_t) nx)*((size_t) ny)*((size_t) nz)*((size_t) noct)*((size_t) ng)*sizeof(double); 
    CUDA_SAFE_CALL(cudaMemset(d_qim, 0, ((size_t) nang)*((size_t) nx)*((size_t) ny)*((size_t) nz)*((size_t) noct)*((size_t) ng)*sizeof(double)));
  }
  CUDA_SAFE_CALL(cudaMalloc(&d_qtot, ((size_t) cmom)*((size_t) nx)*((size_t) ny)*((size_t) nz)*((size_t) ng)*sizeof(double))); memUsed+= ((size_t) cmom)*((size_t) nx)*((size_t) ny)*((size_t) nz)*((size_t) ng)*sizeof(double); 
  CUDA_SAFE_CALL(cudaMemset(d_qtot, 0, ((size_t) cmom)*((size_t) nx)*((size_t) ny)*((size_t) nz)*((size_t) ng)*sizeof(double)));
  
  CUDA_SAFE_CALL(cudaMalloc(&d_flux, ((size_t) nx)*((size_t) ny)*((size_t) nz)*((size_t) ng)*sizeof(double))); memUsed+= ((size_t) nx)*((size_t) ny)*((size_t) nz)*((size_t) ng)*sizeof(double); 
  if (cmom > 1) {
    CUDA_SAFE_CALL(cudaMalloc(&d_fluxm, ((size_t) cmom - 1)*((size_t) nx)*((size_t) ny)*((size_t) nz)*((size_t) ng)*sizeof(double))); memUsed+= ((size_t) cmom - 1)*((size_t) nx)*((size_t) ny)*((size_t) nz)*((size_t) ng)*sizeof(double); 
    
  }
  CUDA_SAFE_CALL(cudaMalloc(&d_fluxpi, ((size_t) nx)*((size_t) ny)*((size_t) nz)*((size_t) ng)*sizeof(double))); memUsed+= ((size_t) nx)*((size_t) ny)*((size_t) nz)*((size_t) ng)*sizeof(double); 
  CUDA_SAFE_CALL(cudaMalloc(&d_df, ((size_t) nx)*((size_t) ny)*((size_t) nz)*((size_t) ng)*sizeof(double))); memUsed+= ((size_t) nx)*((size_t) ny)*((size_t) nz)*((size_t) ng)*sizeof(double); 
  CUDA_SAFE_CALL( cudaGetLastError() );


  // CUDA sweep data
  CUDA_SAFE_CALL(cudaMalloc(&d_dogrp, ng*sizeof(int))); memUsed+= ng*sizeof(int); 
  y_comm = MPI_Comm_f2c(*ycomm);
  z_comm = MPI_Comm_f2c(*zcomm); 

  bSizeY = achunk*ichunk*kchunk+16;
  bSizeZ = achunk*ichunk*jchunk+16;

  CUDA_SAFE_CALL(cudaMallocHost(&angrpBG, (nTBG+1) * sizeof(int))); 
  CUDA_SAFE_CALL(cudaMallocHost(&angrpL, (nTBG) * sizeof(int))); 


  //sequence Numbers 
  //volatile int *seqRVy, *seqRVz, *seqINy, *seqINz, *seqOUTy, *seqOUTz,*d_seqOUTy, *d_seqOUTz;
  CUDA_SAFE_CALL(cudaMallocHost(&seqRVy, nTBG*tbZ*sizeof(int))); 
  CUDA_SAFE_CALL(cudaMallocHost(&seqRVz, nTBG*tbY*sizeof(int))); 
  CUDA_SAFE_CALL(cudaMallocHost(&seqINy, 32*nTBG*tbZ*sizeof(int))); 
  CUDA_SAFE_CALL(cudaMallocHost(&seqINz, 32*nTBG*tbY*sizeof(int))); 
  CUDA_SAFE_CALL(cudaMallocHost(&seqOUTy, nTBG*tbZ*sizeof(int))); 
  CUDA_SAFE_CALL(cudaMallocHost(&seqOUTz, nTBG*tbY*sizeof(int))); 
  CUDA_SAFE_CALL(cudaMallocHost(&d_seqOUTy, nTBG*tbZ*128)); 
  CUDA_SAFE_CALL(cudaMallocHost(&d_seqOUTz, nTBG*tbY*128)); 
  
  //accessor
  #define seqRVz(a,b) seqRVz[b + tbY * (a)]
  #define seqRVy(a,b) seqRVy[b + tbZ * (a)]
  #define seqINz(a,b) seqINz[32*(b + tbY * (a))]
  #define seqINy(a,b) seqINy[32*(b + tbZ * (a))]
  #define seqOUTz(a,b) seqOUTz[b + tbY * (a)]
  #define seqOUTy(a,b) seqOUTy[b + tbZ * (a)]
  #define d_seqOUTz(a,b) d_seqOUTz[32*(b + tbY * (a))]
  #define d_seqOUTy(a,b) d_seqOUTy[32*(b + tbZ * (a))]

  
  for(int tt=0;tt<nTBG; tt++)
  for(int jj=0;jj<tbY;jj++)
  {
      seqRVz(tt,jj)=-1;
      seqINz(tt,jj)=-1;
      seqOUTz(tt,jj)=-1;
      d_seqOUTz(tt,jj)=-1;
  }

  for(int tt=0;tt<nTBG; tt++)
  for(int kk=0;kk<tbZ;kk++)
  {
      seqRVy(tt,kk)=-1;
      seqINy(tt,kk)=-1;
      seqOUTy(tt,kk)=-1;
      d_seqOUTy(tt,kk)=-1;
  }
  //allocate requests
  //MPI_Request *ySreq, *zSreq;
  ySreq = (MPI_Request*) malloc( sizeof(MPI_Request) * (maxAngrp+1) * NC * nTBG * tbZ );
  zSreq = (MPI_Request*) malloc( sizeof(MPI_Request) * (maxAngrp+1) * NC * nTBG * tbY );
  #define ySreq(s,t,z) ySreq[s + NC*(maxAngrp+1)*(z + tbZ *(t))]
  #define zSreq(s,t,y) zSreq[s + NC*(maxAngrp+1)*(y + tbY *(t))]

  //MPI_Request *yRreq, *zRreq;
  yRreq = (MPI_Request*) malloc( sizeof(MPI_Request) * nTBG * tbZ );
  zRreq = (MPI_Request*) malloc( sizeof(MPI_Request) * nTBG * tbY );
  #define yRreq(a,b) yRreq[b + tbZ * (a)]
  #define zRreq(a,b) zRreq[b + tbY * (a)]

  //alocate page aligned send-recv buffer
  int page_size = sysconf(_SC_PAGESIZE);
  bufNple = NC*NA*(maxAngrp+1);
  size_t h_bufSize = sizeof(double)*nTBG*((size_t)tbZ*bSizeY*(bufNple + NC*NA*(maxAngrp+1))+(size_t)tbY*bSizeZ*(bufNple + NC*NA*(maxAngrp+1)));
  posix_memalign((void**)(&h_Buf), page_size, h_bufSize); 
  CUDA_SAFE_CALL(cudaHostRegister(h_Buf,h_bufSize,cudaHostRegisterMapped));
  h_RBufY = h_Buf;
  h_RBufZ = h_RBufY + nTBG * tbZ * bSizeY * bufNple;
  h_SBufY = h_RBufZ + nTBG * tbY * bSizeZ * bufNple;
  h_SBufZ = h_SBufY + nTBG * tbZ * bSizeY * NC * (maxAngrp+1) * NA;

  CUDA_SAFE_CALL(cudaHostGetDevicePointer(&d_RBufY, h_RBufY, 0));
  CUDA_SAFE_CALL(cudaHostGetDevicePointer(&d_RBufZ, h_RBufZ, 0));
  CUDA_SAFE_CALL(cudaHostGetDevicePointer(&d_SBufY, h_SBufY, 0));
  CUDA_SAFE_CALL(cudaHostGetDevicePointer(&d_SBufZ, h_SBufZ, 0));
  

  //allocate recv buffers
  #define h_RBufY(s,t,z,k) h_RBufY[achunk*ichunk*k + bSizeY*( z + tbZ*(t + nTBG*(s)) )]
  #define h_RBufZ(s,t,y,j) h_RBufZ[achunk*ichunk*j + bSizeZ*( y + tbY*(t + nTBG*(s)) )]

  #define h_SBufY(s,t,z,k) h_SBufY[achunk*ichunk*k + bSizeY*( z + tbZ*(t + nTBG*(s)) )]
  #define h_SBufZ(s,t,y,j) h_SBufZ[achunk*ichunk*j + bSizeZ*( y + tbY*(t + nTBG*(s)) )]

  //int *h_ptrin_rdy;
  //volatile int *ptrin_rdy, *ptrin_dne;
  //communicating for ptr_in consumption
  CUDA_SAFE_CALL(cudaMallocHost(&ptrin_rdy, nTBG * tbY * tbZ * 128 ));
  CUDA_SAFE_CALL(cudaMallocHost(&ptrin_dne, nTBG * tbY * tbZ * 128 ));
  CUDA_SAFE_CALL(cudaMallocHost(&h_ptrin_rdy, nTBG * tbY * tbZ * sizeof(int) ));
  h_ptrout_dne=(int*)malloc(sizeof(int)*nTBG*tbY*tbZ);
  #define ptrin_dne(t,a,b) ptrin_dne[ 32 * ( b + tbZ * ( (a) + tbY * (t)))]
  #define ptrin_rdy(t,a,b) ptrin_rdy[ 32 * ( b + tbZ * ( (a) + tbY * (t)))]
  #define h_ptrin_rdy(t,a,b) h_ptrin_rdy[b + tbZ * ( (a) + tbY * (t))]
  #define h_ptrout_dne(t,a,b) h_ptrout_dne[b + tbZ * ( (a) + tbY * (t))]
  for(int tt=0;tt<nTBG;tt++)
  for(int jj=0;jj<tbY;jj++)
  for(int kk=0;kk<tbZ;kk++)
  {
    ptrin_rdy(tt,jj,kk) = 3;
    h_ptrin_rdy(tt,jj,kk) = -1;
    ptrin_dne(tt,jj,kk) = -1;
    h_ptrout_dne(tt,jj,kk) = -1;
  }

  //device relay buffers
  //since we do not check the consumption of the buffer, whole buffer is allocated
  //double *d_buf_y, *d_buf_z;
  dNple = NC;
  CUDA_SAFE_CALL(cudaMalloc(&d_buf_y, dNple* bSizeY * (tbY+1) * tbZ * nTBG * sizeof(double) )); memUsed+= dNple* bSizeY * (tbY+1) * tbZ * nTBG * sizeof(double) ; 
  CUDA_SAFE_CALL(cudaMalloc(&d_buf_z, dNple* bSizeZ * tbY * (tbZ+1) * nTBG * sizeof(double) )); memUsed+= dNple* bSizeZ * tbY * (tbZ+1) * nTBG * sizeof(double) ; 
  CUDA_SAFE_CALL(cudaMemset(d_buf_y, 0,dNple* bSizeY * (tbY+1) * tbZ * nTBG * sizeof(double) )); 
  CUDA_SAFE_CALL(cudaMemset(d_buf_z, 0,dNple* bSizeZ * tbY * (tbZ+1) * nTBG * sizeof(double) ));
  //#define d_buf_y(s,t,y,z) d_buf_y[bSizeY*( (s) + dNple*( (z) + tbZ*(y  + (tbY+1)*(t))))]
  //#define d_buf_z(s,t,y,z) d_buf_z[bSizeZ*( (s) + dNple*( (z) + (tbZ+1)*(y  + tbY*(t))))]

  #define d_buf_y_chunk(s,t,y) d_buf_y[tbZ*bSizeY*((s)+dNple*((y) +(tbY+1)*(t)))]
  #define d_buf_y(s,t,y,z)     d_buf_y[bSizeY*( (z) + tbZ*( (s) + dNple*( (y) + (tbY+1)*(t))))]

  #define d_buf_z_chunk(s,t,z) d_buf_z[tbY*bSizeZ*((s)+dNple*((z) +(tbZ+1)*(t)))]
  #define d_buf_z(s,t,y,z)     d_buf_z[bSizeZ*( (y) + tbY*( (s) + dNple*( (z) + (tbZ+1)*(t))))]
  //bSizeZ = achunk*ichunk*jchunk+1;

  CUDA_SAFE_CALL(cudaFuncSetAttribute(dim3_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 50));
  CUDA_SAFE_CALL( cudaGetLastError() );

  
  setup_ib_yz(h_Buf, h_bufSize, *npey, *npez, *pey, *pez, NA, NG, NC, nTBG, y_comm,z_comm,(*iproc)%4); 
}


extern "C"
void dealloc_cuda_(int *src_opt, int *timedep, int *ng, int *cmom)
{
  //
  // geom module
  //
  CUDA_SAFE_CALL(cudaFree(d_dinv));
  CUDA_SAFE_CALL(cudaFree(d_hj));
  CUDA_SAFE_CALL(cudaFree(d_hk));
  CUDA_SAFE_CALL(cudaFree(d_diag_len));
  CUDA_SAFE_CALL(cudaFree(d_diag_ic));
  CUDA_SAFE_CALL(cudaFree(d_diag_j));
  CUDA_SAFE_CALL(cudaFree(d_diag_k));
  CUDA_SAFE_CALL(cudaFree(d_diag_count));
  //
  // sn module
  //
  CUDA_SAFE_CALL(cudaFree(d_lma));
  CUDA_SAFE_CALL(cudaFree(d_ec));
  CUDA_SAFE_CALL(cudaFree(d_mu));
  CUDA_SAFE_CALL(cudaFree(d_w));
  CUDA_SAFE_CALL(cudaFree(d_eta));
  CUDA_SAFE_CALL(cudaFree(d_xi));
  //                                            
  // data module
  //
  CUDA_SAFE_CALL(cudaFree(d_mat));
  CUDA_SAFE_CALL(cudaFree(d_vdelt));
  CUDA_SAFE_CALL(cudaFree(d_sigt));
  CUDA_SAFE_CALL(cudaFree(d_siga));
  CUDA_SAFE_CALL(cudaFree(d_slgg));
  //
  // solvar module
  //
  if (*timedep == 1) {
      CUDA_SAFE_CALL(cudaFree(d_ptrin));
      CUDA_SAFE_CALL(cudaFree(d_ptrout));
      CUDA_SAFE_CALL(cudaFreeHost(h_ptrin));
      CUDA_SAFE_CALL(cudaFreeHost(h_ptrout));
  }

  CUDA_SAFE_CALL(cudaFree(d_t_xs));
  CUDA_SAFE_CALL(cudaFree(d_a_xs));
  CUDA_SAFE_CALL(cudaFree(d_s_xs));
  CUDA_SAFE_CALL(cudaFree(d_qi));
  CUDA_SAFE_CALL(cudaFree(d_q2grp));
  if (*src_opt == 3) {
    CUDA_SAFE_CALL(cudaFree(d_qim));
  }
  CUDA_SAFE_CALL(cudaFree(d_qtot));
  CUDA_SAFE_CALL(cudaFree(d_flux));
  if (*cmom > 1)
    CUDA_SAFE_CALL(cudaFree(d_fluxm));
  CUDA_SAFE_CALL(cudaFree(d_fluxpi));
  CUDA_SAFE_CALL(cudaFree(d_df));

  //
  // CUDA sweep data
  //
  CUDA_SAFE_CALL(cudaFree((void*)d_dogrp));
  CUDA_SAFE_CALL(cudaFreeHost((void*)seqRVy));
  CUDA_SAFE_CALL(cudaFreeHost((void*)seqRVz));
  CUDA_SAFE_CALL(cudaFreeHost((void*)seqINy));
  CUDA_SAFE_CALL(cudaFreeHost((void*)seqINz));
  CUDA_SAFE_CALL(cudaFreeHost((void*)seqOUTy));
  CUDA_SAFE_CALL(cudaFreeHost((void*)seqOUTz));
  CUDA_SAFE_CALL(cudaFreeHost((void*)d_seqOUTy));
  CUDA_SAFE_CALL(cudaFreeHost((void*)d_seqOUTz));

  free(yRreq);
  free(zRreq);
  free(ySreq);
  free(zSreq);


  CUDA_SAFE_CALL(cudaFreeHost((void*)ptrin_rdy));
  CUDA_SAFE_CALL(cudaFreeHost((void*)ptrin_dne));
  CUDA_SAFE_CALL(cudaFreeHost(h_ptrin_rdy));
  free(h_ptrout_dne);

  CUDA_SAFE_CALL(cudaFree(d_buf_y));
  CUDA_SAFE_CALL(cudaFree(d_buf_z));
  CUDA_SAFE_CALL( cudaGetLastError() );


  MPI_Barrier(MPI_COMM_WORLD);
  destroy_ib();
  CUDA_SAFE_CALL(cudaHostUnregister(h_Buf));
  free(h_Buf);
}

extern "C"
void zero_flux_(int *cmom, int *nx, int *ny, int *nz, int *ng)
{
  CUDA_SAFE_CALL(cudaMemset(d_flux, 0, (*nx)*(*ny)*(*nz)*(*ng)*sizeof(double)));
  if (*cmom > 1)
    CUDA_SAFE_CALL(cudaMemset(d_fluxm, 0, (*cmom-1)*(*nx)*(*ny)*(*nz)*(*ng)*sizeof(double)));
  CUDA_SAFE_CALL( cudaGetLastError() );
}

extern "C"
void swap_ptr_inout_cuda_()
{
  double *tmp;
  tmp = h_ptrout;
  h_ptrout = h_ptrin;
  h_ptrin = tmp;
}

__global__ 
void expxs_sigt_kernel(const double* __restrict__ sigt, int *mat, double *t_xs,
                       int nx, int ny, int nz, int nmat)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  int g = blockIdx.y + 1;
  if (tid < nx*ny*nz) {
    int k = tid / (nx*ny);
    int j = (tid - k*nx*ny) / nx;
    int i = tid - j*nx - k*nx*ny;
    i++, j++, k++;
    
    t_xs(i,j,k,g) = sigt(mat(i,j,k),g);
  }
}

extern "C"
void expxs_sigt_cuda_(int *nx, int *ny, int *nz, int *nmat, int *ng)
{
  const int block = 256;
  dim3 grid;
  grid.x = (int)ceil((float)(*nx)*(*ny)*(*nz) / block);
  grid.y = *ng;

  expxs_sigt_kernel<<<grid, block>>>(d_sigt, d_mat, d_t_xs, 
                                     *nx, *ny, *nz, *nmat);
}

__global__ 
void expxs_siga_kernel(const double* __restrict__ siga, int *mat, double *a_xs,
                                 int nx, int ny, int nz, int nmat)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  int g = blockIdx.y + 1;
  if (tid < nx*ny*nz) {
    int k = tid / (nx*ny);
    int j = (tid - k*nx*ny) / nx;
    int i = tid - j*nx - k*nx*ny;
    i++, j++, k++;
    
    a_xs(i,j,k,g) = siga(mat(i,j,k),g);
  }
}

extern "C"
void expxs_siga_cuda_(int *nx, int *ny, int *nz, int *nmat, int *ng)
{
  const int block = 256;
  dim3 grid;
  grid.x = (int)ceil((float)(*nx)*(*ny)*(*nz) / block);
  grid.y = *ng;

  expxs_siga_kernel<<<grid, block>>>(d_siga, d_mat, d_a_xs, 
                                     *nx, *ny, *nz, *nmat);
}

__global__ 
void expxs_slgg_kernel(int nmat, int nmom, int ng, int nx, int ny, int nz,
                       const double* __restrict__ slgg, int *mat, double *s_xs)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  int g = blockIdx.y + 1;
  if (tid < nx*ny*nz) {
    int k = tid / (nx*ny);
    int j = (tid - k*nx*ny) / nx;
    int i = tid - j*nx - k*nx*ny;
    i++, j++, k++;

    for (int l = 1; l <= nmom; l++) 
      s_xs(l,i,j,k,g) = slgg(mat(i,j,k),l,g,g);
  }
}

extern "C"
void expxs_slgg_cuda_(int *nmat, int *nmom, int *ng, int *nx, int *ny, int *nz,
                      double *s_xs)
{
  const int block = 256;
  dim3 grid;
  grid.x = (int)ceil((float)(*nx)*(*ny)*(*nz) / block);
  grid.y = *ng;
  expxs_slgg_kernel<<<grid,block>>>(*nmat, *nmom, *ng, *nx, *ny, *nz, 
                                    d_slgg, d_mat, d_s_xs);
}

__global__ 
void compute_hj_kernel(int nang, double dy, double *eta, double *hj)
{
  int i = threadIdx.x + 1;
  if (i <= nang) {
    hj(i) = (2.0/dy)*eta(i);
  }
}

__global__ 
void compute_hk_kernel(int nang, double dz, double *xi, double *hk)
{
  int i = threadIdx.x + 1;
  if (i <= nang) {
    hk(i) = (2.0/dz)*xi(i);
  }
}

__global__ 
void compute_dinv_kernel(int nang, int nx, int ny, int nz, double *t_xs, 
                         double hi, double *hj, double *hk, 
                         double *mu, double *vdelt, double *dinv)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  int wid = tid / WARP_SIZE;

  if (wid < nx*ny*nz) {
    int k = wid / (nx*ny);
    int j = (wid - k*nx*ny) / nx;
    int i = wid - j*nx - k*nx*ny;
    i++, j++, k++;
    int g = blockIdx.y + 1;
    int lid = threadIdx.x % WARP_SIZE + 1;
    
    for (int m = lid; m <= nang; m += WARP_SIZE)
      dinv(m,i,j,k,g) = 1.0 / 
        (t_xs(i,j,k,g) + vdelt(g) + mu(m)*hi + hj(m) + hk(m));
  }
}

extern "C"
void param_calc_cuda_(int *ichunk, int *nx, int *ny, int *nz,  int *nc, 
                      double *dx, double *dy, double *dz, 
                      int *nang, int *ng, int *ndimen, double *hi)
{
  *nc = *nx / *ichunk;
  *hi = 2.0/(*dx);

  if (*ndimen > 1) {
    compute_hj_kernel<<<1, *nang>>>(*nang, *dy, d_eta, d_hj);
    if (*ndimen > 2) 
      compute_hk_kernel<<<1, *nang>>>(*nang, *dz, d_xi, d_hk);
  }

  const int block = 256;
  dim3 grid;
  grid.x = (int)ceil((float)(*nx)*(*ny)*(*nz)/(block/WARP_SIZE));
  grid.y = *ng;
  compute_dinv_kernel<<<grid, block>>>(*nang, *nx, *ny, *nz, d_t_xs, 
                                       *hi, d_hj, d_hk, d_mu, d_vdelt, d_dinv);
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}

__global__ 
void otr_src_kernel(int nang, int nx, int ny, int nz, int ng, 
                    int cmom, int nmom, int nmat,
                    int *lma, double *qi, 
                    const double* __restrict__ slgg, const int* mat, 
                    const double* __restrict__ flux, 
                    const double* __restrict__ fluxm, double *q2grp)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  int g = blockIdx.y + 1;
  if (tid < nx*ny*nz) {
    int k = tid / (nx*ny);
    int j = (tid - k*nx*ny) / nx;
    int i = tid - j*nx - k*nx*ny;
    i++, j++, k++;
    
    q2grp(1,i,j,k,g) = qi(i,j,k,g);
    
    for (int gp = 1; gp <= ng; gp++) {
      if (gp == g) continue;
      q2grp(1,i,j,k,g) += slgg(mat(i,j,k),1,gp,g)*flux(i,j,k,gp);
      
      int mom = 2;
      for (int l = 2; l <= nmom; l++) {
        for (int m = 1; m <= lma(l); m++) {
          q2grp(mom,i,j,k,g) += slgg(mat(i,j,k),l,gp,g)*fluxm(mom-1,i,j,k,gp);
          mom++;
        }
      }
    }
  }  
}

extern "C"
void otr_src_cuda_(int *nang, int *nx, int *ny, int *nz, int *ng, 
                   int *cmom, int *nmom, int *nmat, double *q2grp)
{
  const int block = 256;
  dim3 grid;
  grid.x = (int)ceil((float)(*nx)*(*ny)*(*nz)/block);
  grid.y = *ng;
  otr_src_kernel<<<grid, block>>>(*nang, *nx, *ny, *nz, *ng, *cmom, *nmom, *nmat,
      d_lma, d_qi, d_slgg, d_mat, d_flux, d_fluxm, d_q2grp);
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}

__global__ 
void inr_src_scat_kernel(int nang, int cmom, int nmom, int nx, int ny, int nz,
                         int g, int *lma, double *q2grp, double *s_xs, 
                         double *flux, double *fluxm, double *qtot)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid < nx*ny*nz) {
    int k = tid / (nx*ny);
    int j = (tid - k*nx*ny) / nx;
    int i = tid - j*nx - k*nx*ny;
    i++, j++, k++;
    
    qtot(1,i,j,k,g) = q2grp(1,i,j,k,g) + s_xs(1,i,j,k,g)*flux(i,j,k,g);

    int mom = 2;
    for (int l = 2; l <= nmom; l++) {
      for (int m = 1; m <= lma(l); m++) {
        qtot(mom,i,j,k,g) = q2grp(mom,i,j,k,g) + 
          s_xs(l,i,j,k,g)*fluxm(mom-1,i,j,k,g);
        mom++;
      }
    }
//    if ( i<3 && j<3 && k<3 && g==1) printf("qtot0= %d %d %d %e %e q2grp=%e %e %d\n",i,j,k,qtot(1,i,j,k,g),flux(i,j,k,g),q2grp(1,i,j,k,g),s_xs(1,i,j,k,g),g);
//    if ( i<3 && j<3 && k<3 && g==1) printf("qtot1= %d %d %d %e %e q2grp=%e %e  %d\n",i,j,k,qtot(2,i,j,k,g),fluxm(1,i,j,k,g),q2grp(2,i,j,k,g),s_xs(2,i,j,k,g),g);
  }
}

extern "C"
void inr_src_scat_cuda_(int *nang, int *cmom, int *nmom, 
                        int *nx, int *ny, int *nz, int *g)
{
  const int block = 256;
  int grid = (int)ceil((float)(*nx)*(*ny)*(*nz)/block);
  inr_src_scat_kernel<<<grid, block>>>(*nang, *cmom, *nmom, 
                                       *nx, *ny, *nz, *g, d_lma,
                                       d_q2grp, d_s_xs, 
                                       d_flux, d_fluxm, d_qtot);
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}

__global__ 
void compute_df_kernel(int nx, int ny, int nz, int g, double tolr,
                       double *flux, double *fluxpi, double *df)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid < nx*ny*nz) {
    int k = tid / (nx*ny);
    int j = (tid - k*nx*ny) / nx;
    int i = tid - j*nx - k*nx*ny;
    i++, j++, k++;

    if (fabs(fluxpi(i,j,k,g)) > tolr) 
      df(i,j,k,g) = fabs(flux(i,j,k,g)/fluxpi(i,j,k,g) - 1.0);
    else 
      df(i,j,k,g) = fabs(flux(i,j,k,g) - fluxpi(i,j,k,g));
  }
}

extern "C" 
void compute_df_cuda_(int *nx, int *ny, int *nz, int *g, 
                      double *tolr, double *dfmxi)
{
  const int block = 256;
  int grid = (int)ceil((float)(*nx)*(*ny)*(*nz)/block);
  compute_df_kernel<<<grid, block>>>(*nx, *ny, *nz, *g, 
                                     *tolr, d_flux, d_fluxpi, d_df);
  int n = (*nx)*(*ny)*(*nz);
  int ind;
  cublasIdamax(cublasHandle, n, d_df+(*g-1)*n, 1, &ind);
  d2h<double>(dfmxi+(*g-1), d_df+(*g-1)*n+ind-1, 1);
}

__global__ 
void scale_qim_kernel(long size, int cy, double time, double *qim)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if ((long) tid < size) {
    if (cy == 1)
      qim[tid] *= time;
    else {
      double sf = (2.0*(double)cy - 1.0)/(2.0*(double)cy - 3);
      qim[tid] *= sf;
    }
  }
}

extern "C"
void scale_qim_cuda_(double *h_qim, int *nang, int *nx, int *ny, int *nz,
                     int *noct, int *ng, int *cy, double *time)
{
  long nang_l = *nang;
  long nx_l = *nx;
  long ny_l = *ny;
  long nz_l = *nz;
  long noct_l = *noct;
  long ng_l = *ng;
  long size = (nang_l)*(nx_l)*(ny_l)*(nz_l)*(noct_l)*(ng_l);
  int block = 256;
  int grid = (size + block - 1) / block;
  scale_qim_kernel<<<grid, block>>>(size, *cy, *time, d_qim);
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}

__forceinline__ __device__ 
double __shfl_bcast_double(double a, unsigned int pos)
{
  return __hiloint2double(__shfl(__double2hiint(a), pos), 
                          __shfl(__double2loint(a), pos));
}

__forceinline__ __device__ 
double __shfl_down_double(double a, unsigned int delta)
{
  return __hiloint2double(__shfl_down(__double2hiint(a), delta), 
                          __shfl_down(__double2loint(a), delta));
}

//reduce in a warp using warp shuffle instructions
__forceinline__ __device__ 
double wreduce_shfl(double a)
{
  double b;
  #pragma unroll 5
  for (int i = WARP_SIZE/2; i >= 1; i = i >> 1) {
    b = __shfl_down_double(a, i);
    a += b;
  }
  return a;
}

// reduce in a block
__forceinline__ __device__
double breduce(double a, volatile double *buf, unsigned int nwarp)
{
  int wid = threadIdx.x / WARP_SIZE;
  int laneid = threadIdx.x % WARP_SIZE;
  double b = wreduce_shfl(a);
  if (laneid == 0)
    buf[wid] = b;
  __syncthreads();
  for (int offset = nwarp/2; offset >= 1; offset = offset >> 1) {
    if (threadIdx.x < offset)
      buf[threadIdx.x] += buf[threadIdx.x + offset];
    __syncthreads();
  }
  return buf[0];
}

static double tmpirecv1 = 0, tmpirecv2 = 0, tmpisend = 0, ttot = 0, tkernel;
static int yrecvtot = 0, zrecvtot = 0, ysendtot = 0, zsendtot = 0;
extern "C"
void print_gpu_time_(int *iproc, int *nang, int *ichunk, int *ny, int *nz)
{
  tkernel = ttot - tmpirecv1 - tmpirecv2 - tmpisend;
  double recvbw = ((double)yrecvtot*(*nang)*(*ichunk)*(*nz)+(double)zrecvtot*(*nang)*(*ichunk)*(*ny))*
    sizeof(double)*1e-9/tmpirecv2;
  double sendbw = ((double)ysendtot*(*nang)*(*ichunk)*(*nz)+(double)zsendtot*(*nang)*(*ichunk)*(*ny))*
    sizeof(double)*1e-9/tmpisend;

  int nrecv = yrecvtot + zrecvtot, max_nrecv;
  MPI_Allreduce(&nrecv, &max_nrecv, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  if (*iproc == 0 || (yrecvtot + zrecvtot == max_nrecv && ysendtot + zsendtot == max_nrecv)) {
  printf("P(%d): tot = %f, recv = %f (%f, %f, %f GB/s), send = %f (%f, %f GB/s), kernel = %f, %d, %d, %d\n", 
         *iproc, ttot, tmpirecv1 + tmpirecv2, tmpirecv1, tmpirecv2, recvbw, 
         tmpisend, tmpisend*1e3/(ysendtot+zsendtot), sendbw, tkernel, yrecvtot+zrecvtot, ysendtot+zsendtot);
  }
}

#include "dim3_ker.cu"

#include "helper.cu"

extern "C"
void dim3_sweep_cuda_(
  int *h_dogrp, int *num_grth,
  int *nc, int *ndimen, 
  int *jst, int *kst, 
  int *src_opt, int *fixup,
  int *jlo, int *klo, int *jhi, int *khi, int *jd, int *kd,
  int *lasty, int *lastz, int *firsty, int *firstz, int *timedep, 
  double *tolr, double *hi, 
  int *mtag, int *yp_rcv, int *yp_snd, int *yproc, 
  int *zp_rcv, int *zp_snd, int *zproc, int *iproc)
{ 
  int ndogrp = 0;
  for (int i = 0; i < *num_grth; i++) {
    if (h_dogrp[i] == 0) break;
    ndogrp++;
  }
  if (ndogrp == 0) 
    return;

  CTimer sndY,sndZ,rcvY,rcvZ,ptrinCpy,host,sndZPoll,rcvPoll;
  CTimer preK,postK,memS;

  preK.Start();
  //need dogrp from gpu
  //seems already done
  //cudaMemcpy(d_dogrp,h_dogrp,sizeof(int)* (*num_grth),cudaMemcpyHostToDevice)

  int yzFlip = ((1-*jst)*2 + (1-*kst) ) >> 1;
  int oct = 2*(*jd-1) + 4*(*kd-1);
//  int avgAngrp = (NA * ndogrp)/nTBG;
//  int remAngrp = (NA * ndogrp)%nTBG;
//  printf("avgAngrp=%d remAngrp=%d\n",avgAngrp,remAngrp);

//  angrpBG[0]=0;
//  for(int ii=0;ii<nTBG;ii++)
//  {
//    angrpL[ii]=(avgAngrp + (remAngrp>0));
//    angrpBG[ii+1] = angrpBG[ii] + angrpL[ii]; 
//    remAngrp--;
//  }

  
  int avgGroup = (ndogrp)/nTBG;
  int remGroup = (ndogrp)%nTBG;
  angrpBG[0]=0;
  for(int ii=0;ii<nTBG;ii++)
  {
    angrpL[ii]=(avgGroup + (remGroup>ii))*3;
    angrpBG[ii+1] = angrpBG[ii] + angrpL[ii]; 
  }


  for(int tt=0;tt<nTBG; tt++)
  for(int jj=0;jj<tbY;jj++)
  {
      seqRVz(tt,jj)=-1;
      seqINz(tt,jj)=-1;
      seqOUTz(tt,jj)=-1;
      d_seqOUTz(tt,jj)=-1;
  }

  for(int tt=0;tt<nTBG; tt++)
  for(int kk=0;kk<tbZ;kk++)
  {
      seqRVy(tt,kk)=-1;
      seqINy(tt,kk)=-1;
      seqOUTy(tt,kk)=-1;
      d_seqOUTy(tt,kk)=-1;
  }

  for(int tt=0;tt<nTBG;tt++)
  for(int jj=0;jj<tbY;jj++)
  for(int kk=0;kk<tbZ;kk++)
  {
    ptrin_rdy(tt,jj,kk) = -1;
    h_ptrin_rdy(tt,jj,kk) = -1;
    ptrin_dne(tt,jj,kk) = -1;
    h_ptrout_dne(tt,jj,kk) = -1;
  }

  set_cnt<<<dim3(nTBG,tbY,tbZ),dim3(dNple,1,1)>>>(d_buf_y,d_buf_z, dNple, tbY, tbZ, bSizeY, bSizeZ, ichunk, jchunk, kchunk );

  //start======================================================================
  #define getMpiTagZ(r,s,t,j) (j+tbY*(t + nTBG*(s + NC*(maxAngrp+1)*(r))))
  #define getMpiTagY(r,s,t,k) (k+tbZ*(t + nTBG*(s + NC*(maxAngrp+1)*(r))))
  for(int tt=0;tt<nTBG; tt++)
  {
    //z-dir
    if (*zp_rcv != *zproc && z_comm != MPI_COMM_NULL) 
    {
      //determine Qid and post recv
        post_recv(angrpL[tt]*NC, tt*4+2+ *kd-1);
        yzFlip = yzFlip & (0xffff-NOTZRECV);
    }
    else
    {
      set_d_buf_z<<<dim3(nTBG,dNple,tbY),dim3(32,32,1)>>>(d_buf_z,dNple,tbY,tbZ,bSizeZ,ichunk,jchunk,kchunk,0,*zproc,(avgGroup+1)*NA);
      memS.Start();
      //memset(h_RBufZ,0,  sizeof(double) * nTBG * tbY * bSizeZ * bufNple);
      yzFlip = yzFlip | NOTZRECV;
      memS.End();

      for(int jj=0;jj<tbY;jj++)
        seqINz(tt,jj)= angrpL[tt] * NC;

    }

    //y-dir
    if (*yp_rcv != *yproc && y_comm != MPI_COMM_NULL) 
    {
        post_recv(angrpL[tt]*NC, tt*4+ *jd - 1);
      yzFlip = yzFlip & (0xffff-NOTYRECV);
    }
    else
    {
      set_d_buf_y<<<dim3(nTBG,dNple,tbZ),dim3(32,32,1)>>>(d_buf_y,dNple,tbY,tbZ,bSizeY,ichunk,jchunk,kchunk,0,*yproc,(avgGroup+1)*3);
      memS.Start();
      //memset(h_RBufY,0,  sizeof(double) * nTBG * tbZ * bSizeY * bufNple);
      yzFlip = yzFlip | NOTYRECV;
      memS.End();

      for(int kk=0;kk<tbZ;kk++)
        seqINy(tt,kk)=angrpL[tt] * NC;

    }
  }

  //dump_ptrin<<<dim3(3,1,1),dim3(3,3,3)>>>();

  __sync_synchronize();


  MPI_Barrier(MPI_COMM_WORLD);

  host.Start();
  preK.End();

  cudaStream_t kerStream,memStream,memStream0,memStream1;
  CUDA_SAFE_CALL(cudaStreamCreate ( &kerStream)) ;
  CUDA_SAFE_CALL(cudaStreamCreate ( &memStream)) ;
  CUDA_SAFE_CALL(cudaStreamCreate ( &memStream0)) ;
  CUDA_SAFE_CALL(cudaStreamCreate ( &memStream1)) ;
  //=================== device start ===================================

  dim3_kernel<<<dim3(nTBG,tbY,tbZ),dim3(WARP_SIZE,jchunk,kchunk),0,kerStream>>>(
   d_dogrp,  
   ichunk,  jchunk,  kchunk,  achunk,  oct,  *ndimen, 
   nx,  ny,  nz,  nang,  noct,
   NA,  NC,  NG,  cmom,  *src_opt,  *fixup,
   tbY,  tbZ,  nTBG,
   ptrNple,
   *timedep,
   d_vdelt, d_w,  d_t_xs,
   *tolr, *hi,  d_hj,  d_hk,  d_mu,
   d_qtot, d_ec, 
   d_dinv,  d_qim,
   d_psi_save, d_flux, d_fluxm,
   d_buf_y, d_buf_z,  bSizeY, bSizeZ, dNple,
   d_ptrin,d_ptrout, ptrin_rdy, ptrin_dne,
   seqINy, seqINz, d_seqOUTy, d_seqOUTz,
   angrpBG,  maxAngrp, *yproc, yzFlip, 
   bufNple, h_RBufY, h_RBufZ, h_SBufY, h_SBufZ ) ;
  //====================================================================
  

  uint32_t recvV[160];

  timeval w_start,w_now;
  gettimeofday(&w_start, NULL);

  omp_set_num_threads(2);

  //how to set default shared?
  #pragma omp parallel for 
  for(int threadid=0;threadid<2;threadid++)
  {

    if(threadid==1)
    {
  
  bool alldone=false;
  while(!alldone)
  {
     //=================== receive ===================================
     alldone=true;

     for(int tt=0;tt<nTBG;tt++)
     {
       int jj=0;
         if( seqINz(tt,jj)+1 < angrpL[tt]*NC ) {alldone=false; break;}

       int kk=0;
         if( seqINy(tt,kk)+1 < angrpL[tt]*NC ) {alldone=false; break;}
     }

     gettimeofday(&w_now, NULL);
     if (w_now.tv_sec - w_start.tv_sec > 180) break;

     rcvPoll.Start();
     int num_comp = check_recv(recvV);
     rcvPoll.End();
     for(int ii=0;ii<num_comp;ii++)
     {
       int yz = (recvV[ii] & 0b1000) >> 3;
       int jk = 0b111 & recvV[ii];
       int tt = (0b11110000 & recvV[ii]) >> 4;
       int seq = recvV[ii] >> 8;

       if ( yz == 0 ) //y-dir
       {
           int kk = jk;
           kk=0;
           rcvY.Start();
           seqRVy(tt,kk)++;
           
  
           if (!(seqRVy(tt,kk)<angrpL[tt]*NC)) printf("error. too many recvZ %d,%d.tt=%d kk=%d seq=%d\n",seqRVy(tt,kk),angrpL[tt]*NC,tt,kk,seq);
           __sync_synchronize();
           rcvY.End();
           seqINy(tt,kk)++;
        }
        else //z-recv
        {
           int jj = jk;
           jj = 0;
           rcvZ.Start();
           seqRVz(tt,jj)++;
           if (!(seqRVz(tt,jj)<angrpL[tt]*NC)) printf("error. too many recvZ %d,%d tt=%d jj=%d seq=%d\n",seqRVz(tt,jj),angrpL[tt]*NC,tt,jj,seq);
           __sync_synchronize();
           rcvZ.End();
         
           seqINz(tt,jj)++;

        }
     }
  
     //=================== send ===================================
     for(int tt=0;tt<nTBG; tt++)
     {
     //y-dir
       if (*yp_snd != *yproc && y_comm != MPI_COMM_NULL) 
       {
         int kk=tbZ-1;
         {
           if(seqOUTy(tt,kk)+1<angrpL[tt]*NC)
           {
             alldone=false;
             if( d_seqOUTy(tt,kk) > seqOUTy(tt,kk) )
             {
                int Q = tt*4 +  2 - *jd; // (*jd) == 1 -> minus y
                sndY.Start();
                if(post_control(Q))
                {
                  //post send
                  seqOUTy(tt,kk)++;  //it starts at -1
                  __sync_synchronize();

                  unsigned int imm = ((unsigned int)(seqOUTy(tt,kk)) << 8) + (unsigned int)(0) +  ((unsigned int)(tt)<<4);
             
                  post_send(imm,Q,  (size_t)&h_SBufY(seqOUTy(tt,kk),tt,0,0) -  (size_t)h_Buf, (size_t)&h_RBufY(seqOUTy(tt,kk),tt,0,0) - (size_t)h_Buf , sizeof(double)*tbZ*bSizeY );
             
                }
                sndY.End();
              }
           }
         }
       }
       //z-dir
       if (*zp_snd != *zproc && z_comm != MPI_COMM_NULL) 
       {
           int jj=tbY-1;
           if(seqOUTz(tt,jj)+1<angrpL[tt]*NC)
           {
             alldone=false;
             if( d_seqOUTz(tt,jj) > seqOUTz(tt,jj) )
             {
                int Q = tt*4 + 2 + 2 -  *kd;
                sndZ.Start();
                if( post_control(Q) )
                {
                  seqOUTz(tt,jj)++;  //it starts at -1
                  
                  unsigned int imm = ((unsigned int)(seqOUTz(tt,jj)) << 8) + (unsigned int)(0) + 0b1000 +  ((unsigned int)(tt)<<4);
               
                  post_send(imm, Q , (size_t)&h_SBufZ(seqOUTz(tt,jj),tt,0,0) -  (size_t)h_Buf, (size_t)&h_RBufZ(seqOUTz(tt,jj),tt,0,0) - (size_t)h_Buf ,sizeof(double)* tbY*bSizeZ );
              
                }
                sndZ.End();
             }
           }
       }
     } //tt



  } // while
  } // end of if
  else
  {
    //printf("my thread number=%d\n",omp_get_thread_num());
   
  bool alldone=false;
  while(!alldone)
  {
     alldone=true;
     #if 1
     const int slackPTR = 4;
     for(int tt=0;tt<nTBG;tt++)
     {
       int angrp, grp, ang;
       int jj=0;
       int kk=0;
       int memStream0active=0;
       int memStream1active=0;
       ptrinCpy.Start();
       {
         volatile int ptrin_dne_rd=ptrin_dne(tt,tbY-1,tbZ-1);
         if(h_ptrin_rdy(tt,jj,kk)+1<(angrpL[tt]))
         {
           alldone = false;
           if(h_ptrin_rdy(tt,jj,kk) - slackPTR < ptrin_dne_rd )
           {
             h_ptrin_rdy(tt,jj,kk)++;
             angrp = angrpBG[tt]+h_ptrin_rdy(tt,jj,kk);
             grp = angrp/NA;
             ang = angrp%NA;
             CUDA_SAFE_CALL(cudaMemcpyAsync(&d_ptrin(h_ptrin_rdy(tt,jj,kk),tt,jj,kk), &h_ptrin(jj,kk,grp,ang,oct>>1), tbY*tbZ*NC*ichunk*jchunk*kchunk*achunk*sizeof(double), cudaMemcpyHostToDevice, memStream0 ));
             memStream0active=1;
           }
         }

         
         if(h_ptrout_dne(tt,jj,kk)+1<angrpL[tt])
         {
           alldone = false;
           if(ptrin_dne_rd>h_ptrout_dne(tt,jj,kk))
           {
              h_ptrout_dne(tt,jj,kk)++;
              angrp = angrpBG[tt]+h_ptrout_dne(tt,jj,kk); grp = h_dogrp[angrp/NA]-1; ang = angrp%NA;
              CUDA_SAFE_CALL(cudaMemcpyAsync(&h_ptrout(jj,kk,grp,ang,oct>>1), &d_ptrout(h_ptrout_dne(tt,jj,kk),tt,jj,kk), tbY*tbZ*NC*ichunk*jchunk*kchunk*achunk*sizeof(double), cudaMemcpyDeviceToHost, memStream1 ));
              memStream1active=1;
           }
         }
         if(memStream0active) {CUDA_SAFE_CALL(cudaStreamSynchronize(memStream0));
                       	        ptrin_rdy(tt,jj,kk)=h_ptrin_rdy(tt,jj,kk); } // printf("tt=%d ptrin_rdy=%d\n",tt,ptrin_rdy(tt,jj,kk));
         if(memStream1active) {CUDA_SAFE_CALL(cudaStreamSynchronize(memStream1)); }

         __sync_synchronize();
       }
       ptrinCpy.End();
     }
     #endif
  } // while

  } // end of else
  } // end of prallel for
  postK.Start();
  omp_set_num_threads(1);
  host.End();


  CUDA_SAFE_CALL(cudaStreamSynchronize(kerStream)); 	
  cudaDeviceSynchronize(); 
  CUDA_SAFE_CALL(cudaStreamDestroy(kerStream));
  CUDA_SAFE_CALL(cudaStreamDestroy(memStream));
  CUDA_SAFE_CALL(cudaStreamDestroy(memStream0));
  CUDA_SAFE_CALL(cudaStreamDestroy(memStream1));

  postK.End();
}  //end of dim3_sweep_cuda_

