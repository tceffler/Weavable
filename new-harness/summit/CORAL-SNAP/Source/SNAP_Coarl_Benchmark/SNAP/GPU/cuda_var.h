#ifndef CUDA_VAR_H
#define CUDA_VAR_H

#include <stdio.h>
#include <cublas_v2.h>

#define CUDA_SAFE_CALL(call) {                                    \
  cudaError err = call;                                                    \
  if( cudaSuccess != err) {                                                \
  fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
          __FILE__, __LINE__, cudaGetErrorString( err) );              \
  fflush(stderr); \
  exit(EXIT_FAILURE);                                                  \
  } }

//
// helper routines
//
template<class T>
void h2d(T* des, T* src, size_t size)
{
  //printf("copy to devide %p,%p,%d\n",des,src,size);
  CUDA_SAFE_CALL( cudaMemcpy(des, src, size*sizeof(T), cudaMemcpyHostToDevice) );
}
template<class T>
void d2h(T* des, T* src, size_t size)
{
  CUDA_SAFE_CALL( cudaMemcpy(des, src, size*sizeof(T), cudaMemcpyDeviceToHost) );
}
template<class T>
void d2d(T* des, T*src, size_t size)
{
  CUDA_SAFE_CALL( cudaMemcpy(des, src, size*sizeof(T), cudaMemcpyDeviceToDevice) );
}
//
// geom module
//
static double *d_dinv;
  #define dinv(a,i,j,k,g) dinv[(a-1)+(i-1)*nang+(j-1)*nang*nx+(k-1)*nang*nx*ny+(g-1)*((size_t)(nang*nx*ny*nz))]
  #define dinv_aa(a,b,i,j,k,g) dinv[b+bchunk*(a)+(i)*nang+(j)*nang*nx+(k)*nang*nx*ny+(size_t)(g)*((size_t)(nang*nx*ny*nz))]

static double *d_hj;
  #define hj_aa(a,b) hj[b + bchunk*(a)]
  #define hj(a) hj[a-1]

static double *d_hk;
  #define hk_aa(a,b) hk[b + bchunk*(a)]
  #define hk(a) hk[a-1]

static int *d_diag_len;
  #define diag_len(i) diag_len[i-1]
static int *d_diag_ic;
  #define diag_ic(i) diag_ic[i-1]
static int *d_diag_j;
  #define diag_j(i) diag_j[i-1]
static int *d_diag_k;
  #define diag_k(i) diag_k[i-1]
static int *d_diag_count;
  #define diag_count(i) diag_count[i-1]
static int ndiag;
extern "C"
void h2d_diag_(
  int *diag_len, int *diag_ic, int *diag_j, int *diag_k, int *diag_count,
  int *h_ndiag, int *tot_size)
{
  //
  // allocation has to be here instead of setup_cuda because 
  // ndiag is only known after calling geom_setup
  //
  cudaMalloc(&d_diag_len, *h_ndiag*sizeof(int));
  cudaMalloc(&d_diag_ic, *tot_size*sizeof(int));
  cudaMalloc(&d_diag_j, *tot_size*sizeof(int));
  cudaMalloc(&d_diag_k, *tot_size*sizeof(int));
  cudaMalloc(&d_diag_count, *h_ndiag*sizeof(int));

  cudaMemcpy(d_diag_len, diag_len, *h_ndiag*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_diag_ic, diag_ic, *tot_size*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_diag_j, diag_j, *tot_size*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_diag_k, diag_k, *tot_size*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_diag_count, diag_count, *h_ndiag*sizeof(int), cudaMemcpyHostToDevice);

  ndiag = *h_ndiag;

}
//
// sn module
//
static int *d_lma;
  #define lma(c) lma[c-1]
  extern "C" void h2d_lma_(int *lma, int *nmom) { h2d<int>(d_lma, lma, *nmom); }
static double *d_eta;
  #define eta(a) eta[a-1]
  extern "C" void h2d_eta_(double *eta, int *nang) { h2d<double>(d_eta, eta, *nang); }
static double *d_xi;
  #define xi(a) xi[a-1]
  extern "C" void h2d_xi_(double *xi, int *nang) { h2d<double>(d_xi, xi, *nang); }
static double *d_mu;
  #define mu_aa(a,b) mu[b + bchunk*(a)]
  #define mu(a) mu[a-1]

  extern "C" void h2d_mu_(double *mu, int *nang) { h2d<double>(d_mu, mu, *nang); }
static double *d_w;
  #define w(a) w[a-1]
  #define w_aa(a,b) w[b + bchunk*(a)]
  extern "C" void h2d_w_(double *w, int *nang) { h2d<double>(d_w, w, *nang); }
static double *d_ec;
  #define ec_aa(a,b,c,o) ec[b + bchunk*(a + NA*(c + cmom*(o)))]
  #define ec(a,c,o) ec[(a-1)+(c-1)*nang+(o-1)*nang*cmom]

  extern "C" void h2d_ec_(double *ec, int *nang, int *cmom, int *noct) 
    { h2d<double>(d_ec, ec, (*nang)*(*cmom)*(*noct)); }
//
// data module
//
static int *d_mat;
  #define mat(i,j,k) mat[(i-1)+(j-1)*nx+(k-1)*nx*ny]
  extern "C" void h2d_mat_(int *mat, int *nx, int *ny, int *nz) 
    { h2d<int>(d_mat, mat, (*nx)*(*ny)*(*nz)); }
static double *d_vdelt;
  #define vdelt(g) vdelt[g-1]
  extern "C" void h2d_vdelt_(double *vdelt, int *ng) { h2d<double>(d_vdelt, vdelt, *ng); }
static double *d_sigt;
  #define sigt(i,j) sigt[(i-1)+(j-1)*nmat]
  extern "C" void h2d_sigt_(double *sigt, int *nmat, int *ng) { h2d<double>(d_sigt, sigt, (*nmat)*(*ng)); }
static double *d_siga;
  #define siga(i,j) siga[(i-1)+(j-1)*nmat]
  extern "C" void h2d_siga_(double *siga, int *nmat, int *ng) { h2d<double>(d_siga, siga, (*nmat)*(*ng)); }
static double *d_slgg;
  #define slgg(n,c,i,j) slgg[(n-1)+(c-1)*nmat+(i-1)*nmat*nmom+(j-1)*nmat*nmom*ng]
  extern "C" void h2d_slgg_(double *slgg, int *nmat, int *nmom, int *ng)
    { h2d<double>(d_slgg, slgg, (*nmat)*(*nmom)*(*ng)*(*ng)); }
//
// solvar module
//
//static double *d_psii;
//  #define psii(i,j,k,g) psii[(i-1)+(j-1)*nang+(k-1)*nang*ny+(g-1)*nang*ny*nz]
//static double *d_psij;
//  #define psij(i,j,k,g) psij[(i-1)+(j-1)*nang+(k-1)*nang*ichunk+(g-1)*nang*ichunk*nz]
//static double *d_psik;
//  #define psik(i,j,k,g) psik[(i-1)+(j-1)*nang+(k-1)*nang*ichunk+(g-1)*nang*ichunk*ny]

//static double *d_ptr_in;
//  #define ptr_in(a,i,j,k,o,g) \
    ptr_in[(long)(a-1)+(long)(i-1)* (long)nang+(long)(j-1)* (long)nang* (long)nx+(long)(k-1)* (long)nang* (long)nx* (long)ny+(long)(o-1)* (long)nang* (long)nx* (long)ny* (long)nz+(long)(g-1)* (long)nang* (long)nx* (long)ny* (long)nz* (long)noct]

//    ptr_in[(a-1)+(i-1)*d1+(j-1)*d1*d2+(k-1)*d1*d2*d3+(o-1)*d1*d2*d3*d4+(g-1)*d1*d2*d3*d4*noct]


//static double *d_ptr_out;
//  #define ptr_out(a,i,j,k,o,g) \
    ptr_out[(long)(a-1)+(long)(i-1)* (long)nang+(long)(j-1)* (long)nang* (long)nx+(long)(k-1)* (long)nang* (long)nx* (long)ny+(long)(o-1)* (long)nang* (long)nx* (long)ny* (long)nz+(long)(g-1)* (long)nang* (long)nx* (long)ny* (long)nz* (long)noct]
//ptr_out[(a-1)+(i-1)*d1+(j-1)*d1*d2+(k-1)*d1*d2*d3+(o-1)*d1*d2*d3*d4+(g-1)*d1*d2*d3*d4*noct]
static double *h_ptr_in, *h_ptr_out;
static double *d_t_xs;
  #define t_xs(i,j,k,g) t_xs[(i-1)+(j-1)*nx+(k-1)*nx*ny+(g-1)*nx*ny*nz]
  #define t_xs_aa(i,j,k,g) t_xs[i+nx*(j+ny*(k+nz*(g)))]
static double *d_a_xs;
  #define a_xs(i,j,k,g) a_xs[(i-1)+(j-1)*nx+(k-1)*nx*ny+(g-1)*nx*ny*nz]
static double *d_s_xs;
  #define s_xs(n,i,j,k,g) s_xs[(n-1)+(i-1)*nmom+(j-1)*nx*nmom+(k-1)*nx*ny*nmom+(g-1)*nx*ny*nz*nmom]
static double *d_qi;
  #define qi(i,j,k,g) qi[(i-1)+(j-1)*nx+(k-1)*nx*ny+(g-1)*nx*ny*nz]
  extern "C" void h2d_qi_(double *qi, int *nx, int *ny, int *nz, int *ng)
    { h2d<double>(d_qi, qi, (*nx)*(*ny)*(*nz)*(*ng)); }
static double *d_qim;
  #define qim(a,i,j,k,o,g) \
    qim[(a)+(i)*nang+(j)*nang*nx+(k)*nang*nx*ny+(o)*nang*nx*ny*nz+(g)*nang*nx*ny*nz*noct]
    //qim[(a-1)+(i-1)*nang+(j-1)*nang*nx+(k-1)*nang*nx*ny+(o-1)*nang*nx*ny*nz+(g-1)*nang*nx*ny*nz*noct]

  extern "C" void h2d_qim_(double *qim, int *nang, int *nx, int *ny, int *nz, int *noct, int *ng)
    { h2d<double>(d_qim, qim, (*nang)*(*nx)*(*ny)*(*nz)*(*noct)*(*ng)); }

static double *d_q2grp;
  #define q2grp(c,i,j,k,g) q2grp[(c-1)+(i-1)*cmom+(j-1)*cmom*nx+(k-1)*cmom*nx*ny+(g-1)*cmom*nx*ny*nz]

static double *d_qtot;
  #define qtot_aa(c,i,j,k,g) qtot[(c)+(i)*cmom+(j)*cmom*nx+(k)*cmom*nx*ny+(g)*cmom*nx*ny*nz]
  #define qtot(c,i,j,k,g) qtot[(c-1)+(i-1)*cmom+(j-1)*cmom*nx+(k-1)*cmom*nx*ny+(g-1)*cmom*nx*ny*nz]

  extern "C" void h2d_qtot_(double *qtot, int *cmom, int *nx, int *ny, int *nz, int *ng)
    { h2d<double>(d_qtot, qtot, (*cmom)*(*nx)*(*ny)*(*nz)*(*ng)); }
static double *d_flux;
  #define flux(i,j,k,g) flux[(i-1)+(j-1)*nx+(k-1)*nx*ny+(g-1)*nx*ny*nz]
  #define flux_aa(i,j,k,g) flux[i + nx*(j + ny*(k + nz*(g)))]
static double *d_fluxm;
  #define fluxm(c,i,j,k,g) fluxm[(i-1)+nx*((j-1)+ny*((k-1)+nz*(c-1+(cmom-1)*(g-1))))]
  #define fluxm_aa(c,i,j,k,g) fluxm[i+nx*(j+ny*(k+nz*(c+(cmom-1)*(g))))]
static double *d_fluxpi;
  #define fluxpi(i,j,k,g) fluxpi[(i-1)+(j-1)*nx+(k-1)*nx*ny+(g-1)*nx*ny*nz]
static double *d_df;
  #define df(i,j,k,g) df[(i-1)+(j-1)*nx+(k-1)*nx*ny+(g-1)*nx*ny*nz]
  extern "C" void d2h_df_(double *df, int *nx, int *ny, int *nz, int *g) 
    { int n = (*nx)*(*ny)*(*nz); d2h<double>(df+(*g-1)*n, d_df+(*g-1)*n, n); }
static double *h_jb_in, *d_jb_in;
  #define jb_in(i,j,k,g) jb_in[(i-1)+(j-1)*nang+(k-1)*nang*ichunk+(g-1)*nang*ichunk*nz]
static double *h_jb_out, *d_jb_out;
  #define jb_out(i,j,k,g) jb_out[(i-1)+(j-1)*nang+(k-1)*nang*ichunk+(g-1)*nang*ichunk*nz]
static double *h_kb_in, *d_kb_in;
  #define kb_in(i,j,k,g) kb_in[(i-1)+(j-1)*nang+(k-1)*nang*ichunk+(g-1)*nang*ichunk*ny]
static double *h_kb_out, *d_kb_out;
  #define kb_out(i,j,k,g) kb_out[(i-1)+(j-1)*nang+(k-1)*nang*ichunk+(g-1)*nang*ichunk*ny]
extern "C"
void h2d_flux_(double *flux, double *fluxm,
               int *cmom, int *nx, int *ny, int *nz, int *ng)
{
  CUDA_SAFE_CALL( cudaMemcpy(d_flux, flux, (*nx)*(*ny)*(*nz)*(*ng)*sizeof(double),
                             cudaMemcpyHostToDevice) );
  if (*cmom > 1) {
    CUDA_SAFE_CALL( cudaMemcpy(d_fluxm, fluxm, (*cmom-1)*(*nx)*(*ny)*(*nz)*(*ng)*sizeof(double),
                               cudaMemcpyHostToDevice) );
  }
}
extern "C"
void d2h_flux_(double *flux, double *fluxm,
               int *cmom, int *nx, int *ny, int *nz, int *ng)
{
  CUDA_SAFE_CALL( cudaMemcpy(flux, d_flux, (*nx)*(*ny)*(*nz)*(*ng)*sizeof(double),
                             cudaMemcpyDeviceToHost) );
  if (*cmom > 1) {
    CUDA_SAFE_CALL( cudaMemcpy(fluxm, d_fluxm, (*cmom-1)*(*nx)*(*ny)*(*nz)*(*ng)*sizeof(double),
                               cudaMemcpyDeviceToHost) );
  }
}
extern "C"
void copy_fluxpi_(int *nx, int *ny, int *nz, int *g)
{
  int n = (*nx)*(*ny)*(*nz);
  d2d<double>(d_fluxpi+(*g-1)*n, d_flux+(*g-1)*n, n);
}

__global__ void ker_dump_flux(double* flux, int nx, int ny, int nz, int g, int t,int ot)
{
  printf("flux: t=%d ot=%d xx=%d yy=%d zz=%d g=%d %.16e\n",t,ot,blockIdx.x,blockIdx.y,blockIdx.z,g,flux_aa(blockIdx.x,blockIdx.y,blockIdx.z,g));
}

extern "C"
void dump_flux_(int *nx, int *ny, int *nz, int *t, int *ot)
{
  ker_dump_flux<<<dim3(4,4,4),1>>>(d_flux,*nx,*ny,*nz,0,*t,*ot);
}

__global__ void ker_dump_fluxm(double* fluxm, int nx, int ny, int nz, int cmom, int g, int t,int ot)
{ 
  printf("fluxm: t=%d ot=%d cc=%d xx=%d yy=%d zz=%d g=%d %.16e\n",t,ot,blockIdx.x,blockIdx.y,blockIdx.z,threadIdx.x,g,fluxm_aa(blockIdx.x,blockIdx.y,blockIdx.z,threadIdx.x,g));
}
extern "C"
void dump_fluxm_(int *nx, int *ny, int *nz, int *cmom, int *t, int *ot)
{ 
  ker_dump_fluxm<<<dim3(3,3,3),3>>>(d_fluxm,*nx,*ny,*nz,*cmom,0,*t,*ot);
}

//
// CUDA sweep data
//
static int *d_dogrp;
  #define dogrp(i) dogrp[i-1]
  extern "C" void h2d_dogrp_(int *dogrp, int *num_grth) { h2d<int>(d_dogrp, dogrp, *num_grth); }
//static cudaStream_t *stream;
static MPI_Comm y_comm, z_comm;
//static MPI_Request *yrequest, *zrequest;
//static MPI_Status *ystat, *zstat;
//static int *recvdone, *kerdone, *yrecvflag, *zrecvflag;
//static int gpu_batch_size;
//static int grid_size;
cublasHandle_t cublasHandle;
//__device__ volatile int *mutexin, *mutexout;
//int *d_mutexin, *d_mutexout;

int tbY;
int tbZ;
int nTBG;
int bSizeY;
int bSizeZ;
int *angrpBG,*angrpL;

int bufNple;
double *h_Buf;
double *h_RBufY,*h_RBufZ;
double *h_SBufY,*h_SBufZ;
double *d_RBufY,*d_RBufZ;
double *d_SBufY,*d_SBufZ;
int *h_ptrin_rdy, *h_ptrout_dne;
volatile int *ptrin_rdy, *ptrin_dne;
double *d_buf_y, *d_buf_z;
double *d_psi_save;
int dNple;
volatile int *seqRVy, *seqRVz, *seqINy, *seqINz, *seqOUTy, *seqOUTz,*d_seqOUTy, *d_seqOUTz;
MPI_Request *yRreq, *zRreq;
MPI_Request *ySreq, *zSreq;
MPI_Status status;
int ptrNple;
#define WARP_SIZE 32
int ichunk,jchunk,kchunk,achunk,bchunk;
int nang, ng, noct,nmom,cmom,nmat;
int maxAngrp;
int nx,ny,nz;
double *d_ptrin, *d_ptrout;
double *h_ptrin, *h_ptrout;
int NC,NA,NG;
#endif
