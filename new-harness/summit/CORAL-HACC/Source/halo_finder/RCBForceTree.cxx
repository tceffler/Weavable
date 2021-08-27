/*=========================================================================
                                                                                
Copyright (c) 2007, Los Alamos National Security, LLC

All rights reserved.

Copyright 2007. Los Alamos National Security, LLC. 
This software was produced under U.S. Government contract DE-AC52-06NA25396 
for Los Alamos National Laboratory (LANL), which is operated by 
Los Alamos National Security, LLC for the U.S. Department of Energy. 
The U.S. Government has rights to use, reproduce, and distribute this software. 
NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY,
EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  
If software is modified to produce derivative works, such modified software 
should be clearly marked, so as not to confuse it with the version available 
from LANL.
 
Additionally, redistribution and use in source and binary forms, with or 
without modification, are permitted provided that the following conditions 
are met:
-   Redistributions of source code must retain the above copyright notice, 
    this list of conditions and the following disclaimer. 
-   Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution. 
-   Neither the name of Los Alamos National Security, LLC, Los Alamos National
    Laboratory, LANL, the U.S. Government, nor the names of its contributors
    may be used to endorse or promote products derived from this software 
    without specific prior written permission. 

THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
ARE DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR 
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; 
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
                                                                                
=========================================================================*/

/*=========================================================================

Copyright (c) 2011-2012 Argonne National Laboratory
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

=========================================================================*/


/*
BG/Q tuned version of HACC: 69.2% of peak performance on 96 racks of Sequoia
Argonne Leadership Computing Facility, Argonne, IL 60439
Vitali Morozov (morozov@anl.gov)
Hal Finkel (hfinkel@anl.gov)
*/

#include "Timings.h"
#include "RCBForceTree.h"
#include "Partition.h"

#include <cstring>
#include <cstdio>
#include <ctime>
#include <stdexcept>
#include <assert.h>
#ifdef HACC_CUDA
#else
#include <openacc.h>
#endif

extern "C" void Timer_Beg(const char *);
extern "C" void Timer_End(const char *);

extern "C" void pgi_acc_map_host_data(void* a, size_t bytes);
extern "C" void pgi_acc_unmap_host_data(void* a);
extern "C" void my_acc_wait_async(void* stream1, void* stream2);
extern "C" void sm_multi_copy(POSVEL_T *d_nx, POSVEL_T *d_ny, POSVEL_T *d_nz, POSVEL_T *d_nm, const POSVEL_T *nx, const POSVEL_T *ny, const POSVEL_T *nz, const POSVEL_T *nm, int count, void* stream);
extern "C" void CudaStep16(int count, int count1, const float* __restrict xx, const float* __restrict yy,
      const float* __restrict zz, const float* __restrict mass,
      const float* __restrict xx1, const float* __restrict yy1,
      const float* __restrict zz1, const float* __restrict mass1,
      float* __restrict vx, float* __restrict vy, float* __restrict vz,
      float fcoeff, float fsrrmax, float rsm, void* stream);
#ifdef HACC_CUDA
extern "C" void CudaStep16Batched(BatchInfo batch, float fcoeff, float fsrrmax, float rsm, void* stream);
static inline void nbody1_batched(BatchInfo batch, float fcoeff, float fsrrmax, float rsm);
#endif

void copy_down_shim(const float *h1, const float *h2, const float *h3, const float *h4, int count, void* stream) {

  float *d1=const_cast<float*>(h1);
  float *d2=const_cast<float*>(h2);
  float *d3=const_cast<float*>(h3);
  float *d4=const_cast<float*>(h4);

#pragma acc host_data use_device(d1,d2,d3,d4)
  {
    //printf("count: %d, d1: %p, d2: %p, d3: %p, d4: %p, h1: %p, h2: %p, h3: %p, h4: %p, count",count,d1,d2,d3,d4,h1,h2,h3,h4);
    sm_multi_copy(d1,d2,d3,d4,h1,h2,h3,h4,count,stream);
  }
}

using namespace std;

// References:
// Emanuel Gafton and Stephan Rosswog. A fast recursive coordinate bisection tree for
// neighbour search and gravity. Mon. Not. R. Astron. Soc. to appear, 2011.
// http://arxiv.org/abs/1108.0028v1
//
// Atsushi Kawai, Junichiro Makino and Toshikazu Ebisuzaki.
// Performance Analysis of High-Accuracy Tree Code Based on the Pseudoparticle
// Multipole Method. The Astrophysical Journal Supplement Series, 151:13-33, 2004.
// Related: http://arxiv.org/abs/astro-ph/0012041v1
//
// R. H. Hardin and N. J. Sloane
// New Spherical 4-Designs. Discrete Math, 106/107 255-264, 1992.
//
// The library of spherical designs:
// http://www2.research.att.com/~njas/sphdesigns/
namespace {
template <int TDPTS>
struct sphdesign {};

#define DECLARE_SPHDESIGN(TDPTS) \
template <> \
struct sphdesign<TDPTS> \
{ \
  static const POSVEL_T x[TDPTS]; \
  static const POSVEL_T y[TDPTS]; \
  static const POSVEL_T z[TDPTS]; \
}; \
/**/

DECLARE_SPHDESIGN(1)
DECLARE_SPHDESIGN(2)
DECLARE_SPHDESIGN(3)
DECLARE_SPHDESIGN(4)
DECLARE_SPHDESIGN(6)
DECLARE_SPHDESIGN(12)
DECLARE_SPHDESIGN(14)

#undef DECLARE_SPHDESIGN

#define VMAX (16384*2)

/* this is not a t-design, but puts the monopole moment
   at the center of mass. */
const POSVEL_T sphdesign<1>::x[] = {
  0
};

const POSVEL_T sphdesign<1>::y[] = {
  0
};

const POSVEL_T sphdesign<1>::z[] = {
  0
};

const POSVEL_T sphdesign<2>::x[] = {
  1.0,
  -1.0
};

const POSVEL_T sphdesign<2>::y[] = {
  0,
  0
};

const POSVEL_T sphdesign<2>::z[] = {
  0,
  0
};

const POSVEL_T sphdesign<3>::x[] = {
  1.0,
  -.5,
  -.5
};

const POSVEL_T sphdesign<3>::y[] = {
  0,
  .86602540378443864675,
  -.86602540378443864675
};

const POSVEL_T sphdesign<3>::z[] = {
  0,
  0,
  0
};

const POSVEL_T sphdesign<4>::x[] = {
  .577350269189625763,
  .577350269189625763,
  -.577350269189625763,
  -.577350269189625763
};

const POSVEL_T sphdesign<4>::y[] = {
  .577350269189625763,
  -.577350269189625763,
  .577350269189625763,
  -.577350269189625763
};

const POSVEL_T sphdesign<4>::z[] = {
  .577350269189625763,
  -.577350269189625763,
  -.577350269189625763,
  .577350269189625763
};

const POSVEL_T sphdesign<6>::x[] = {
  1.0,
  -1.0,
  0,
  0,
  0,
  0
};

const POSVEL_T sphdesign<6>::y[] = {
  0,
  0,
  1.0,
  -1.0,
  0,
  0
};

const POSVEL_T sphdesign<6>::z[] = {
  0,
  0,
  0,
  0,
  1.0,
  -1.0
};

// This is a 3-D 12-point spherical 4-design
// (the verticies of a icosahedron) from Hardin and Sloane.
const POSVEL_T sphdesign<12>::x[] = {
  0,
  0,
  0.525731112119134,
  -0.525731112119134,
  0.85065080835204,
  -0.85065080835204,
  0,
  0,
  -0.525731112119134,
  0.525731112119134,
  -0.85065080835204,
  0.85065080835204
};

const POSVEL_T sphdesign<12>::y[] = {
  0.85065080835204,
  0.85065080835204,
  0,
  0,
  0.525731112119134,
  0.525731112119134,
  -0.85065080835204,
  -0.85065080835204,
  0,
  0,
  -0.525731112119134,
  -0.525731112119134
};

const POSVEL_T sphdesign<12>::z[] = {
  0.525731112119134,
  -0.525731112119134,
  0.85065080835204,
  0.85065080835204,
  0,
  0,
  -0.525731112119134,
  0.525731112119134,
  -0.85065080835204,
  -0.85065080835204,
  0,
  0
};

// This is a 3-D 14-point spherical 4-design by
// R. H. Hardin and N. J. A. Sloane.
const POSVEL_T sphdesign<14>::x[] = {
  1.0e0,
  5.947189772040725e-1,
  5.947189772040725e-1,
  5.947189772040725e-1,
  -5.947189772040725e-1,
  -5.947189772040725e-1,
  -5.947189772040725e-1,
  3.012536847870683e-1,
  3.012536847870683e-1,
  3.012536847870683e-1,
  -3.012536847870683e-1,
  -3.012536847870683e-1,
  -3.012536847870683e-1,
  -1.0e0
};

const POSVEL_T sphdesign<14>::y[] = {
  0.0e0,
  1.776539926025823e-1,
  -7.678419429698292e-1,
  5.90187950367247e-1,
  1.776539926025823e-1,
  5.90187950367247e-1,
  -7.678419429698292e-1,
  8.79474443923065e-1,
  -7.588425179318781e-1,
  -1.206319259911869e-1,
  8.79474443923065e-1,
  -1.206319259911869e-1,
  -7.588425179318781e-1,
  0.0e0
};

const POSVEL_T sphdesign<14>::z[] = {
  0.0e0,
  7.840589244857197e-1,
  -2.381765915652909e-1,
  -5.458823329204288e-1,
  -7.840589244857197e-1,
  5.458823329204288e-1,
  2.381765915652909e-1,
  3.684710570566285e-1,
  5.774116818882528e-1,
  -9.458827389448813e-1,
  -3.684710570566285e-1,
  9.458827389448813e-1,
  -5.774116818882528e-1,
  0.0e0
};
} // anonymous namespace

// Note: In Gafton and Rosswog the far-field force contribution is calculated
// per-cell (at the center of mass), and then a Taylor expansion about the center
// of mass is used to calculate the force on the individual particles. For this to
// work, the functional form of the force must be known (because the Jacobian
// and Hessian are required). Here, however, the functional form is not known,
// and so the pseudo-particle method of Makino is used instead.

template <int TDPTS>
RCBForceTree<TDPTS>::RCBForceTree(
			 POSVEL_T* minLoc,
			 POSVEL_T* maxLoc,
			 POSVEL_T* minForceLoc,
			 POSVEL_T* maxForceLoc,
			 ID_T count,
			 POSVEL_T* xLoc,
			 POSVEL_T* yLoc,
			 POSVEL_T* zLoc,
			 POSVEL_T* xVel,
			 POSVEL_T* yVel,
			 POSVEL_T* zVel,
			 POSVEL_T* ms,
                         POSVEL_T* phiLoc,
                         ID_T *idLoc,
                         MASK_T *maskLoc,
			 POSVEL_T avgMass,
                         POSVEL_T fsm,
                         POSVEL_T r,
                         POSVEL_T oa,
                         ID_T nd,
                         ID_T ds,
                         ID_T tmin,
			 ForceLaw *fl,
			 float fcoeff,
                         POSVEL_T ppc)
{
  vx = xVel;
  vy = yVel;
  vz = zVel;
  
  nx_=(POSVEL_T*)malloc(BATCH_SIZE*VMAX*sizeof(POSVEL_T));
  ny_=(POSVEL_T*)malloc(BATCH_SIZE*VMAX*sizeof(POSVEL_T));
  nz_=(POSVEL_T*)malloc(BATCH_SIZE*VMAX*sizeof(POSVEL_T));
  nm_=(POSVEL_T*)malloc(BATCH_SIZE*VMAX*sizeof(POSVEL_T));

#ifdef HACC_CUDA
  cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
  cudaMalloc((void**)&vx_d,count*sizeof(POSVEL_T));
  cudaMalloc((void**)&vy_d,count*sizeof(POSVEL_T));
  cudaMalloc((void**)&vz_d,count*sizeof(POSVEL_T));
  cudaMalloc((void**)&nx_d,BATCH_SIZE*VMAX*sizeof(POSVEL_T));
  cudaMalloc((void**)&ny_d,BATCH_SIZE*VMAX*sizeof(POSVEL_T));
  cudaMalloc((void**)&nz_d,BATCH_SIZE*VMAX*sizeof(POSVEL_T));
  cudaMalloc((void**)&nm_d,BATCH_SIZE*VMAX*sizeof(POSVEL_T));
  
  cudaHostRegister(nx_,BATCH_SIZE*VMAX*sizeof(POSVEL_T),0);
  cudaHostRegister(ny_,BATCH_SIZE*VMAX*sizeof(POSVEL_T),0);
  cudaHostRegister(nz_,BATCH_SIZE*VMAX*sizeof(POSVEL_T),0);
  cudaHostRegister(nm_,BATCH_SIZE*VMAX*sizeof(POSVEL_T),0);
 
  cudaHostRegister(vx,count*sizeof(POSVEL_T),0);
  cudaHostRegister(vy,count*sizeof(POSVEL_T),0);
  cudaHostRegister(vz,count*sizeof(POSVEL_T),0);
  cudaCheckError();
#else
  #pragma acc enter data create(nx_[:BATCH_SIZE*VMAX],ny_[:BATCH_SIZE*VMAX],nz_[:BATCH_SIZE*VMAX],nm_[:BATCH_SIZE*VMAX])
  //page lock memory, openacc should have a way to do this in the future
  pgi_acc_map_host_data(nx_,BATCH_SIZE*VMAX*sizeof(POSVEL_T));
  pgi_acc_map_host_data(ny_,BATCH_SIZE*VMAX*sizeof(POSVEL_T));
  pgi_acc_map_host_data(nz_,BATCH_SIZE*VMAX*sizeof(POSVEL_T));
  pgi_acc_map_host_data(nm_,BATCH_SIZE*VMAX*sizeof(POSVEL_T));
  
  pgi_acc_map_host_data(vx,count*sizeof(POSVEL_T));
  pgi_acc_map_host_data(vy,count*sizeof(POSVEL_T));
  pgi_acc_map_host_data(vz,count*sizeof(POSVEL_T));
#endif

  batch.clear();

  // Extract the contiguous data block from a vector pointer
  particleCount = count;

  xx = xLoc;
  yy = yLoc;
  zz = zLoc;
  mass = ms;
  phi = phiLoc;
  id = idLoc;
  mask = maskLoc;

  particleMass = avgMass;
  fsrrmax = fsm;
  rsm = r;
  sinOpeningAngle = sinf(oa);
  tanOpeningAngle = tanf(oa);
  nDirect = nd;
  depthSafety = ds;
  taskPartMin = tmin;
  ppContract = ppc;

  // Find the grid size of this chaining mesh
  for (int dim = 0; dim < DIMENSION; dim++) {
    minRange[dim] = minLoc[dim];
    maxRange[dim] = maxLoc[dim];
    minForceRange[dim] = minForceLoc[dim];
    maxForceRange[dim] = maxForceLoc[dim];
  }

  if (fl) {
    m_own_fl = false;
    m_fl = fl;
    m_fcoeff = fcoeff;
  } else {
    //maybe change this to Newton's law or something
    m_own_fl = true;
    m_fl = new ForceLawNewton();
    m_fcoeff = 1.0;
  }

  // Because the tree may be built in parallel, and no efficient way of locking
  // the tree seems to be available in OpenMP (no reader/writer locks, etc.),
  // we just estimate the number of tree nodes that will be needed. Hopefully,
  // this will be an over estimate. If we need more than this, then tree nodes
  // that really should be subdivided will not be.
  //
  // If the tree were perfectly balanced, then it would have a depth of
  // log_2(particleCount/nDirect). The tree needs to have (2^depth)+1 entries.
  // To that, a safety factor is added to the depth.
  ID_T nds = (((ID_T)(particleCount/(POSVEL_T)nDirect)) << depthSafety) + 1;
  tree.reserve(nds);

#ifdef _OPENMP
  int nest = omp_get_nested();
  omp_set_nested(1);
#endif

  int nthreads = 1;
#ifdef _OPENMP
  nthreads = omp_get_max_threads();
#endif

  timespec b_start, b_end;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &b_start);
  // Create the recursive RCB tree from the particle locations
  Timer_Beg("createTree");
  createRCBForceTree();
  Timer_End("createTree");
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &b_end);
  double b_time = (b_end.tv_sec - b_start.tv_sec);
  b_time += 1e-9*(b_end.tv_nsec - b_start.tv_nsec);

  // Interaction lists.
  inx.resize(nthreads);
  iny.resize(nthreads);
  inz.resize(nthreads);
  inm.resize(nthreads);
  iq.resize(nthreads);
  
  timespec f_start, f_end;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &f_start);
  Timer_Beg("calcForces");
  //copy velocities down to device
#ifdef HACC_CUDA
  cudaMemcpyAsync(vx_d,vx,sizeof(POSVEL_T)*count,cudaMemcpyHostToDevice);
  cudaMemcpyAsync(vy_d,vy,sizeof(POSVEL_T)*count,cudaMemcpyHostToDevice);
  cudaMemcpyAsync(vz_d,vz,sizeof(POSVEL_T)*count,cudaMemcpyHostToDevice);
  cudaCheckError();
#else
  #pragma acc enter data copyin(vx[:count], vy[:count], vz[:count]) 
  #pragma acc wait
#endif

  calcInternodeForces();


#ifdef HACC_CUDA
  //process remaning batches
  if(batch.size>0) {
#ifndef HACC_NO_PCIE_TRANSFER
    cudaMemcpyAsync(nx_d,nx_,sizeof(POSVEL_T)*batch.offset,cudaMemcpyHostToDevice);
    cudaMemcpyAsync(ny_d,ny_,sizeof(POSVEL_T)*batch.offset,cudaMemcpyHostToDevice);
    cudaMemcpyAsync(nz_d,nz_,sizeof(POSVEL_T)*batch.offset,cudaMemcpyHostToDevice);
    cudaMemcpyAsync(nm_d,nm_,sizeof(POSVEL_T)*batch.offset,cudaMemcpyHostToDevice);
    cudaCheckError();
#endif
    ::nbody1_batched(batch,m_fcoeff,fsrrmax,rsm);
    batch.clear();
  }
#endif

//copy results back to host
#ifdef HACC_CUDA
  cudaMemcpyAsync(vx,vx_d,sizeof(POSVEL_T)*count,cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(vy,vy_d,sizeof(POSVEL_T)*count,cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(vz,vz_d,sizeof(POSVEL_T)*count,cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  cudaCheckError();
#else
#pragma acc wait(2)
#pragma acc wait
#pragma acc exit data copyout(vx[:count], vy[:count], vz[:count]) 
#endif

  Timer_End("calcForces");
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &f_end);
  double f_time = (f_end.tv_sec - f_start.tv_sec);
  f_time += 1e-9*(f_end.tv_nsec - f_start.tv_nsec);

  printStats(b_time,f_time);

#ifdef _OPENMP
  omp_set_nested(nest);
#endif
}

template <int TDPTS>
RCBForceTree<TDPTS>::~RCBForceTree()
{
#ifdef HACC_CUDA
  cudaHostUnregister(nx_);
  cudaHostUnregister(ny_);
  cudaHostUnregister(nz_);
  cudaHostUnregister(nm_);
  cudaHostUnregister(vx);
  cudaHostUnregister(vy);
  cudaHostUnregister(vz);
  cudaFree(nx_d);
  cudaFree(ny_d);
  cudaFree(nz_d);
  cudaFree(nm_d);
  cudaFree(vx_d);
  cudaFree(vy_d);
  cudaFree(vz_d);
  cudaEventDestroy(event);
#else
  pgi_acc_unmap_host_data(vx);
  pgi_acc_unmap_host_data(vy);
  pgi_acc_unmap_host_data(vz);
  
  pgi_acc_unmap_host_data(nx_);
  pgi_acc_unmap_host_data(ny_);
  pgi_acc_unmap_host_data(nz_);
  pgi_acc_unmap_host_data(nm_);
  
  #pragma acc exit data delete(nx_,ny_,nz_,nm_)
#endif

  free(nx_);
  free(ny_);
  free(nz_);
  free(nm_);
  if (m_own_fl) {
    delete m_fl;
  }
}

template <int TDPTS>
void RCBForceTree<TDPTS>::printStats(double buildTime, double forceTime)
{
  size_t zeroLeafNodes = 0;
  size_t nonzeroLeafNodes = 0;
  size_t maxPPN = 0;
  size_t leafParts = 0;

  for (ID_T tl = 1; tl < (ID_T) tree.size(); ++tl) {
    if (tree[tl].cl == 0 && tree[tl].cr == 0) {
      if (tree[tl].count > 0) {
        ++nonzeroLeafNodes;

        leafParts += tree[tl].count;
        maxPPN = std::max((size_t) tree[tl].count, maxPPN);
      } else {
        ++zeroLeafNodes;
      }
    }
  }

  double localParticleCount = particleCount;
  double localTreeSize = tree.size();
  double localTreeCapacity = tree.capacity();
  double localLeaves = zeroLeafNodes+nonzeroLeafNodes;
  double localEmptyLeaves = zeroLeafNodes;
  double localMeanPPN = leafParts/((double) nonzeroLeafNodes);
  unsigned long localMaxPPN = maxPPN;
  double localBuildTime = buildTime;

  /*
  double globalParticleCount;
  double globalTreeSize;
  double globalTreeCapacity;
  double globalLeaves;
  double globalEmptyLeaves;
  double globalMeanPPN;
  unsigned long globalMaxPPN;
  double globalBuildTime;

  bool printHere = true;
  */

  if ( Partition::getMyProc() == 0 ) {
    printf("\ttree post-build statistics (local for rank 0):\n");
    printf("\t\tparticles: %.2f\n", localParticleCount);
    printf("\t\tnodes: %.2f (allocated:  %.2f)\n", localTreeSize, localTreeCapacity);
    printf("\t\tleaves: %.2f (empty: %.2f)\n", localLeaves, localEmptyLeaves);
    printf("\t\tmean ppn: %.2f (max ppn: %lu)\n", localMeanPPN, localMaxPPN);
    printf("\t\tbuild time: %g s\n", localBuildTime);
    printf("\t\tforce time: %g s\n", forceTime);
  }
}


extern "C" void cm(ID_T count, const POSVEL_T* __restrict xx, const POSVEL_T* __restrict yy,
                      const POSVEL_T* __restrict zz, const POSVEL_T* __restrict mass,
                      POSVEL_T* __restrict xmin, POSVEL_T* __restrict xmax, POSVEL_T* __restrict xc);


static inline POSVEL_T pptdr(const POSVEL_T* __restrict xmin, const POSVEL_T* __restrict xmax, const POSVEL_T* __restrict xc)
{
  return std::min(xmax[0] - xc[0], std::min(xmax[1] - xc[1], std::min(xmax[2] - xc[2], std::min(xc[0] - xmin[0],
                 std::min(xc[1] - xmin[1], xc[2] - xmin[2])))));
}

template <int TDPTS>
static inline void pppts(POSVEL_T tdr, const POSVEL_T* __restrict xc,
                         POSVEL_T* __restrict ppx, POSVEL_T* __restrict ppy, POSVEL_T* __restrict ppz)
{
  for (int i = 0; i < TDPTS; ++i) {
    ppx[i] = tdr*sphdesign<TDPTS>::x[i] + xc[0];
    ppy[i] = tdr*sphdesign<TDPTS>::y[i] + xc[1];
    ppz[i] = tdr*sphdesign<TDPTS>::z[i] + xc[2];
  }
}

template <int TDPTS>
static inline void pp(ID_T count, const POSVEL_T* __restrict xx, const POSVEL_T* __restrict yy,
                      const POSVEL_T* __restrict zz, const POSVEL_T* __restrict mass, const POSVEL_T* __restrict xc,
                      const POSVEL_T* __restrict ppx, const POSVEL_T* __restrict ppy, const POSVEL_T* __restrict ppz,
                      POSVEL_T* __restrict ppm, POSVEL_T tdr)
{
  POSVEL_T K = TDPTS;
  POSVEL_T odr0 = 1/K;

  for (int i = 0; i < count; ++i) {
    POSVEL_T xi = xx[i] - xc[0];
    POSVEL_T yi = yy[i] - xc[1];
    POSVEL_T zi = zz[i] - xc[2];
    POSVEL_T ri = sqrtf(xi*xi + yi*yi + zi*zi);

    for (int j = 0; j < TDPTS; ++j) {
      POSVEL_T xj = ppx[j] - xc[0];
      POSVEL_T yj = ppy[j] - xc[1];
      POSVEL_T zj = ppz[j] - xc[2];
      POSVEL_T rj2 = xj*xj + yj*yj + zj*zj;

      POSVEL_T odr1 = 0, odr2 = 0;
      if (rj2 != 0) {
        POSVEL_T rj  = sqrtf(rj2);
        POSVEL_T aij = (xi*xj + yi*yj + zi*zj)/(ri*rj);

        odr1 = (3/K)*(ri/tdr)*aij;
        odr2 = (5/K)*(ri/tdr)*(ri/tdr)*0.5*(3*aij*aij - 1);
      }

      ppm[j] += mass[i]*(odr0 + odr1 + odr2);
    }
  }
}


#ifdef __bgq__
extern "C" void Step16_int( int count1, float xxi, float yyi, float zzi, float fsrrmax2, float mp_rsm2, const float *xx1, const float *yy1, const float *zz1,const  float *mass1, float *ax, float *ay, float *az );
#endif
#define __bgq__  //use bgq path here to get correct force

static inline void nbody1(ID_T count, ID_T count1, POSVEL_T* __restrict xx, const POSVEL_T* __restrict yy,
                         const POSVEL_T* __restrict zz, const POSVEL_T* __restrict mass,
                         const POSVEL_T* __restrict xx1, const POSVEL_T* __restrict yy1,
                         const POSVEL_T* __restrict zz1, const POSVEL_T* __restrict mass1,
                         POSVEL_T* __restrict vx, POSVEL_T* __restrict vy, POSVEL_T* __restrict vz,
                         float fcoeff, float fsrrmax, float rsm)
{
  POSVEL_T fsrrmax2 = fsrrmax*fsrrmax;
  POSVEL_T rsm2 = rsm*rsm;
  const float ma0 = 0.269327, ma1 = -0.0750978, ma2 = 0.0114808, ma3 = -0.00109313, ma4 = 0.0000605491, ma5 = -0.00000147177;
 
#ifndef HACC_CUDA
#if 0 //WAR #pragma acc wait async is broken,  acc_wait_async(2,1) is inefficient
  #pragma acc wait(2) async(1)
#else
  my_acc_wait_async(acc_get_cuda_stream(2),acc_get_cuda_stream(1));
#endif

#if 0   //WAR:  Use SM to perform small copies as the copy engines are inefficient at these sizes
  #pragma acc update device(xx1[:count1],yy1[:count1],zz1[:count1],mass1[:count1]) async(1) 
#else
  copy_down_shim(xx1,yy1,zz1,mass1,count1,acc_get_cuda_stream(1));
#endif

#if 0  //WAR #pragma acc wait async is broken,  acc_wait_async(1,2) is inefficient
  //#pragma acc wait(1) async(2)
  acc_wait_async(1,2);
#else
  my_acc_wait_async(acc_get_cuda_stream(1),acc_get_cuda_stream(2));
#endif

#if 1  //WAR use hand tuned kernel that openacc should be able to generate.
 #pragma acc host_data use_device(xx,yy,zz,mass,xx1,yy1,zz1,mass1,vx,vy,vz)
 CudaStep16(count,count1,xx,yy,zz,mass,xx1,yy1,zz1,mass1,vx,vy,vz,fcoeff,fsrrmax,rsm,acc_get_cuda_stream(2));
#else
#pragma acc data  present(xx,yy,zz,vx,mass,vy,vz,xx1,yy1,zz1,mass1)
#pragma acc parallel vector_length(64) loop gang async(2) 
  for (int i = 0; i < count; ++i)
  {
    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    float xxi = xx[i];
    float yyi = yy[i];
    float zzi = zz[i];

#pragma acc loop vector reduction(+:ax) reduction(+:ay) reduction(+:az)
    for (int k = 0; k < count1; k++ )
    {
      float dxc, dyc, dzc, m, r2, f, s0, s1, s2;
      dxc = xx1[k] - xxi;
      dyc = yy1[k] - yyi;
      dzc = zz1[k] - zzi;

      r2 = dxc * dxc + dyc * dyc + dzc * dzc;

      if(r2>0.0f && r2<fsrrmax2) {

        s0 = r2 + rsm2;
        s1 = s0 * s0 * s0;
        s2 = 1.0f / sqrtf( s1 ) - ( ma0 + r2*(ma1 + r2*(ma2 + r2*(ma3 + r2*(ma4 + r2*ma5)))));

        f = s2 * mass1[k];

        ax = ax + f * dxc;
        ay = ay + f * dyc;
        az = az + f * dzc;
      }
    }
    vx[i] = vx[i] + ax * fcoeff;
    vy[i] = vy[i] + ay * fcoeff;
    vz[i] = vz[i] + az * fcoeff;
  }
#endif

//wait for copy to finish so that the host can safely overwrite buffers
#pragma acc wait(1)
#endif
}

static inline void nbody1_batched(BatchInfo batch, float fcoeff, float fsrrmax, float rsm)
{
  POSVEL_T fsrrmax2 = fsrrmax*fsrrmax;
  POSVEL_T rsm2 = rsm*rsm;
  const float ma0 = 0.269327, ma1 = -0.0750978, ma2 = 0.0114808, ma3 = -0.00109313, ma4 = 0.0000605491, ma5 = -0.00000147177;

#ifdef HACC_CUDA
  CudaStep16Batched(batch,fcoeff,fsrrmax,rsm,0);
  cudaCheckError();
#endif
}

static inline ID_T partition(ID_T n,
                             POSVEL_T* __restrict xx, POSVEL_T* __restrict yy, POSVEL_T* __restrict zz,
                             POSVEL_T* __restrict vx, POSVEL_T* __restrict vy, POSVEL_T* __restrict vz,
                             POSVEL_T* __restrict mass, POSVEL_T* __restrict phi,
                             ID_T* __restrict id, MASK_T* __restrict mask, POSVEL_T pv
                            )
{
  float t0, t1, t2, t3, t4, t5, t6, t7;
  int32_t is, i, j;
  long i0;
  uint16_t i1;
  int idx[n];

  is = 0;
  for ( i = 0; i < n; i = i + 1 ) 
  {
    if (xx[i] < pv) 
    {
      idx[is] = i;
      is = is + 1;
    }
  }

#pragma unroll (4)
  for ( j = 0; j < is; j++ ) 
  {
    i = idx[j];
    t6 = mass[i]; mass[i] = mass[j]; mass[j] = t6;
    t7 = phi [i]; phi [i] = phi [j]; phi [j] = t7;
  }

#pragma unroll (4)
  for ( j = 0; j < is; j++ ) 
  {
    i = idx[j];
    i1 = mask[i]; mask[i] = mask[j]; mask[j] = i1;
    i0 = id  [i]; id  [i] = id  [j]; id  [j] = i0;
  }

#pragma unroll (4)
  for ( j = 0; j < is; j++ ) 
  {
    i = idx[j];
    t0 = xx[i]; xx[i] = xx[j]; xx[j] = t0;
    t1 = yy[i]; yy[i] = yy[j]; yy[j] = t1;
  }

#pragma unroll (4)
  for ( j = 0; j < is; j++ )
  {
    i = idx[j];
    t2 = zz[i]; zz[i] = zz[j]; zz[j] = t2;
    t3 = vx[i]; vx[i] = vx[j]; vx[j] = t3;
  }

#pragma unroll (4)
  for ( j = 0; j < is; j++ )
  {
    i = idx[j];
    t4 = vy[i]; vy[i] = vy[j]; vy[j] = t4;
    t5 = vz[i]; vz[i] = vz[j]; vz[j] = t5;
  }
  return is;
}


#ifndef RCB_UNTHREADED_BUILD
#ifndef PREFER_OMP_TASKS_TO_SECTIONS
#define DONT_USE_TASKS_IN_BUILD
#endif

#ifdef _OPENMP
#if _OPENMP < 200805 || defined(DONT_USE_TASKS_IN_BUILD) || (defined(__bgq__) && !defined(USE_TASKS_ON_BGQ))
#define BUILD_USES_SECTIONS
#else
#define BUILD_USES_TASKS
#endif
#endif
#endif // RCB_UNTHREADED_BUILD

template <int TDPTS>
void RCBForceTree<TDPTS>::createRCBForceSubtree(int d, ID_T tl, ID_T tlcl, ID_T tlcr)
{

  POSVEL_T *x1, *x2, *x3;
  switch (d) {
  case 0:
    x1 = xx;
    x2 = yy;
    x3 = zz;
  break;
  case 1:
    x1 = yy;
    x2 = zz;
    x3 = xx;
  break;
  /*case 2*/ default:
    x1 = zz;
    x2 = xx;
    x3 = yy;
  break;
  }

#ifdef __bgq__
  int tid = 0;
#ifdef _OPENMP
  tid = omp_get_thread_num();
#endif

#endif

  const bool geoSplit = false;
  POSVEL_T split = geoSplit ? (tree[tl].xmax[d]+tree[tl].xmin[d])/2 : tree[tl].xc[d];
  ID_T is = ::partition(tree[tl].count, x1 + tree[tl].offset, x2 + tree[tl].offset, x3 + tree[tl].offset,
                        vx + tree[tl].offset, vy + tree[tl].offset, vz + tree[tl].offset,
                        mass + tree[tl].offset, phi + tree[tl].offset,
                        id + tree[tl].offset, mask + tree[tl].offset, split
                       );

  if (is == 0 || is == tree[tl].count) {
    return;
  }

  tree[tlcl].count = is;
  tree[tlcr].count = tree[tl].count - tree[tlcl].count;

#ifdef BUILD_USES_SECTIONS
#pragma omp parallel num_threads(2)
#pragma omp sections
  {
#endif

#ifdef BUILD_USES_SECTIONS
#pragma omp section
#endif

  if (tree[tlcl].count > 0) {
    tree[tl].cl = tlcl;
    tree[tlcl].offset = tree[tl].offset;
    tree[tlcl].xmax[d] = split;

#ifdef BUILD_USES_TASKS
#pragma omp task if(tree[tlcl].count > taskPartMin)
#endif

    createRCBForceTreeInParallel(tlcl);
  }

#ifdef BUILD_USES_SECTIONS
#pragma omp section
#endif

  if (tree[tlcr].count > 0) {
    tree[tl].cr = tlcr;
    tree[tlcr].offset = tree[tl].offset + tree[tlcl].count;
    tree[tlcr].xmin[d] = split;

#ifdef BUILD_USES_TASKS
#pragma omp task if(tree[tlcr].count > taskPartMin)
#endif

    createRCBForceTreeInParallel(tlcr);
  }

#ifdef BUILD_USES_SECTIONS
  } /* end sections */
#endif
}

// This is basically the algorithm from (Gafton and Rosswog, 2011).
template <int TDPTS>
void RCBForceTree<TDPTS>::createRCBForceTreeInParallel(ID_T tl)
{
  ID_T cnt = tree[tl].count;
  ID_T off = tree[tl].offset;

  // Compute the center-of-mass coordinates (and recompute the min/max)
  ::cm(cnt, xx + off, yy + off, zz + off, mass + off,
       tree[tl].xmin, tree[tl].xmax, tree[tl].xc);

  if (cnt <= nDirect) {
    // The pseudoparticles
    tree[tl].tdr = ppContract*::pptdr(tree[tl].xmin, tree[tl].xmax, tree[tl].xc);
    memset(tree[tl].ppm, 0, sizeof(POSVEL_T)*TDPTS);
    if (cnt > TDPTS) { // Otherwise, the pseudoparticles are never used
      POSVEL_T ppx[TDPTS], ppy[TDPTS], ppz[TDPTS];
      pppts<TDPTS>(tree[tl].tdr, tree[tl].xc, ppx, ppy, ppz);
      pp<TDPTS>(cnt, xx + off, yy + off, zz + off, mass + off, tree[tl].xc,
                ppx, ppy, ppz, tree[tl].ppm, tree[tl].tdr);
    }

    return;
  }

  // Index of the right and left child levels
  ID_T tlcl, tlcr;
#ifdef _OPENMP
#pragma omp critical(RCBTreeBuild)
#endif
  {
    tlcl = tree.size();
    tlcr = tlcl+1;
    size_t newSize = tlcr+1;
#ifdef _OPENMP
    if (newSize > tree.capacity()) {
      // The tree is about to reallocate: make sure we are the only
      // one here, or else bail...
      if (omp_get_num_threads() > 1) {
        fprintf(stderr, "The tree cannot reallocate during a parallel build!\n");
        fprintf(stderr, "%ld tree nodes have already been added, please reserve a larger number!\n", tlcl);
        // throw std::runtime_error("Invalid parallel tree-build allocation");
        exit(1);
      }
    }
#endif
    tree.resize(newSize);
  }
  memset(&tree[tlcl], 0, sizeof(TreeNode)*2);

  // Both children have similar bounding boxes to the current node (the
  // parent), so copy the bounding box here, and then overwrite the changed
  // coordinate later.
  for (int i = 0; i < DIMENSION; ++i) {
          tree[tlcl].xmin[i] = tree[tl].xmin[i];
          tree[tlcr].xmin[i] = tree[tl].xmin[i];
          tree[tlcl].xmax[i] = tree[tl].xmax[i];
          tree[tlcr].xmax[i] = tree[tl].xmax[i];
  }

  // Split the longest edge at the center of mass.
  POSVEL_T xlen[DIMENSION];
  for (int i = 0; i < DIMENSION; ++i) {
    xlen[i] = tree[tl].xmax[i] - tree[tl].xmin[i];
  }

  int d;
  if (xlen[0] > xlen[1] && xlen[0] > xlen[2]) {
        d = 0; // Split in the x direction
  }
  else if (xlen[1] > xlen[2]) {
        d = 1; // Split in the y direction
  }
  else {
        d = 2; // Split in the z direction
  }

  createRCBForceSubtree(d, tl, tlcl, tlcr);

  // Compute the pseudoparticles based on those of the children
  POSVEL_T ppx[TDPTS], ppy[TDPTS], ppz[TDPTS];
  tree[tl].tdr = ppContract*::pptdr(tree[tl].xmin, tree[tl].xmax, tree[tl].xc);
  pppts<TDPTS>(tree[tl].tdr, tree[tl].xc, ppx, ppy, ppz);
  memset(tree[tl].ppm, 0, sizeof(POSVEL_T)*TDPTS);

#ifdef BUILD_USES_TASKS
#pragma omp taskwait
#endif

  if (tree[tlcl].count > 0) {
    if (tree[tlcl].count <= TDPTS) {
      ID_T offc = tree[tlcl].offset;
      pp<TDPTS>(tree[tlcl].count, xx + offc, yy + offc, zz + offc, mass + offc,
                tree[tl].xc, ppx, ppy, ppz, tree[tl].ppm, tree[tl].tdr);
    } else {
      POSVEL_T ppxc[TDPTS], ppyc[TDPTS], ppzc[TDPTS];
      pppts<TDPTS>(tree[tlcl].tdr, tree[tlcl].xc, ppxc, ppyc, ppzc);
      pp<TDPTS>(TDPTS, ppxc, ppyc, ppzc, tree[tlcl].ppm, tree[tl].xc,
                ppx, ppy, ppz, tree[tl].ppm, tree[tl].tdr);
    }
  }
  if (tree[tlcr].count > 0) {
    if (tree[tlcr].count <= TDPTS) {
      ID_T offc = tree[tlcr].offset;
      pp<TDPTS>(tree[tlcr].count, xx + offc, yy + offc, zz + offc, mass + offc,
                tree[tl].xc, ppx, ppy, ppz, tree[tl].ppm, tree[tl].tdr);
    } else {
      POSVEL_T ppxc[TDPTS], ppyc[TDPTS], ppzc[TDPTS];
      pppts<TDPTS>(tree[tlcr].tdr, tree[tlcr].xc, ppxc, ppyc, ppzc);
      pp<TDPTS>(TDPTS, ppxc, ppyc, ppzc, tree[tlcr].ppm, tree[tl].xc,
                ppx, ppy, ppz, tree[tl].ppm, tree[tl].tdr);
    }
  }
}

template <int TDPTS>
void RCBForceTree<TDPTS>::createRCBForceTree()
{
#ifdef BUILD_USES_TASKS
#pragma omp parallel
#pragma omp single
  {
#endif
  // The top tree is the entire box
  tree.resize(1);
  memset(&tree[0], 0, sizeof(TreeNode));

  tree[0].count = particleCount;
  tree[0].offset = 0;

  for (int i = 0; i < DIMENSION; ++i) {
    tree[0].xmin[i] = minRange[i];
    tree[0].xmax[i] = maxRange[i];
  }

  createRCBForceTreeInParallel();

#ifdef BUILD_USES_TASKS
#pragma omp taskwait
  }
#endif
}

// static size for the interaction list
  


template <int TDPTS>
void RCBForceTree<TDPTS>::calcInternodeForce(ID_T tl,
                                            const std::vector<ID_T> &parents) {
  POSVEL_T fsrrmax2 = fsrrmax*fsrrmax;
  const TreeNode* tree_ = &tree[0];

  int tid = 0;
#ifdef _OPENMP
  tid = omp_get_thread_num();
#endif

  POSVEL_T * __restrict__ nx=&nx_[batch.offset];
  POSVEL_T * __restrict__ ny=&ny_[batch.offset];
  POSVEL_T * __restrict__ nz=&nz_[batch.offset];
  POSVEL_T * __restrict__ nm=&nm_[batch.offset];

  std::vector<ID_T> &q = iq[tid];
  q.clear();
  q.push_back(0);

  // The interaction list.
  int SIZE = 0; // current size of these arrays
  while (!q.empty()) {
    ID_T tln = q.back();
    q.pop_back();

    // We should not interact with our own parents.
    if (tln < tl) {
      bool isParent = std::binary_search(parents.begin(), parents.end(), tln);
      if (isParent) {
        ID_T tlncr = tree_[tln].cr;
        ID_T tlncl = tree_[tln].cl;

        if (tlncl != tl && tlncl > 0 && tree_[tlncl].count > 0) {
          q.push_back(tlncl);
        }
        if (tlncr != tl && tlncr > 0 && tree_[tlncr].count > 0) {
          q.push_back(tlncr);
        }

        continue;
      }
    }

    // Is this node have a small enough opening angle to interact with?
    POSVEL_T dx = tree_[tln].xc[0] - tree_[tl].xc[0];
    POSVEL_T dy = tree_[tln].xc[1] - tree_[tl].xc[1];
    POSVEL_T dz = tree_[tln].xc[2] - tree_[tl].xc[2];
    POSVEL_T dist2 = dx*dx + dy*dy + dz*dz;

    POSVEL_T sx = tree_[tln].xmax[0]-tree_[tln].xmin[0];
    POSVEL_T sy = tree_[tln].xmax[1]-tree_[tln].xmin[1];
    POSVEL_T sz = tree_[tln].xmax[2]-tree_[tln].xmin[2];
    POSVEL_T l2 = std::min(sx*sx, std::min(sy*sy, sz*sz)); // under-estimate

    POSVEL_T dtt2 = dist2*tanOpeningAngle*tanOpeningAngle; 
    bool looksBig;
    // l2/dist2 is really tan^2 theta, for small theta, tan(theta) ~ theta
    if (l2 > dtt2) {
      // the under-estimate is too big, so this is definitely too big
      looksBig = true;
    } else {
      // There are 8 corner points of the remote node, and the maximum angular
      // size will be from one of those points to its opposite points. So there
      // are 8 vector dot products to compute to determine the maximum angular
      // size at any given reference point. (do we need to do this for each point
      // in leaf node, or will the c.m. point be sufficient?).
      looksBig = false;
      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
          POSVEL_T x1 = (i == 0 ? tree_[tln].xmin : tree_[tln].xmax)[0] - tree_[tl].xc[0];
          POSVEL_T y1 = (j == 0 ? tree_[tln].xmin : tree_[tln].xmax)[1] - tree_[tl].xc[1];
          POSVEL_T z1 = tree_[tln].xmin[2] - tree_[tl].xc[2];

          POSVEL_T x2 = (i == 0 ? tree_[tln].xmax : tree_[tln].xmin)[0] - tree_[tl].xc[0];
          POSVEL_T y2 = (j == 0 ? tree_[tln].xmax : tree_[tln].xmin)[1] - tree_[tl].xc[1];
          POSVEL_T z2 = tree_[tln].xmax[2] - tree_[tl].xc[2];

          const bool useRealOA = false;
          if (useRealOA) {
            // |a x b| = a*b*sin(theta)
            POSVEL_T cx = y1*z2 - z1*y2;
            POSVEL_T cy = z1*x2 - x1*z2;
            POSVEL_T cz = x1*y2 - y1*x2;
            if ((cx*cx + cy*cy + cz*cz) > sinOpeningAngle*sinOpeningAngle*
                (x1*x1 + y1*y1 + z1*z1)*(x2*x2 + y2*y2 + z2*z2)
               ) {
              looksBig = true;
              break;
            }
          } else {
            // Instead of using the real opening angle, use the tan approximation; this is
            // better than the opening-angle b/c it incorporates depth information.
            POSVEL_T ddx = x1 - x2, ddy = y1 - y2, ddz = z1 - z2;
            POSVEL_T dh2 = ddx*ddx + ddy*ddy + ddz*ddz;
            if (dh2 > dtt2) {
              looksBig = true;
              break;
            }
          }
        }
    }

    if (!looksBig) {
      if (dist2 > fsrrmax2) {
        // We could interact with this node, but it is too far away to make
        // any difference, so it will be skipped, along with all of its
        // children.
        continue;
      }

      // This node has fewer particles than pseudo particles, so just use the
      // particles that are actually there.
      if (tree_[tln].count <= TDPTS) {
        ID_T offn = tree_[tln].offset;
        ID_T cntn = tree_[tln].count;

        int start = SIZE;
        SIZE = SIZE + cntn;
        assert( SIZE < VMAX );

#pragma vector
        for ( int i = 0; i < cntn; ++i) {
          nx[start + i] = xx[offn + i];
          ny[start + i] = yy[offn + i];
          nz[start + i] = zz[offn + i];
          nm[start + i] = mass[offn + i];
        }

        continue;
      }

      // Interact the particles in this node with the pseudoparticles of the
      // other node.
      int start = SIZE;
      SIZE = SIZE + TDPTS;
      assert( SIZE < VMAX );

      pppts<TDPTS>(tree_[tln].tdr, tree_[tln].xc, &nx[start], &ny[start], &nz[start]);
      for ( int i = 0; i < TDPTS; ++i) {
        nm[start + i] = tree_[tln].ppm[i];
      }

      continue;
    } else if (tree_[tln].cr == 0 && tree_[tln].cl == 0) {
      // This is a leaf node with which we must interact.
      ID_T offn = tree_[tln].offset;
      ID_T cntn = tree_[tln].count;

      int start = SIZE;
      SIZE = SIZE + cntn;
      assert( SIZE < VMAX );

#pragma ivdep 
      for ( int i = 0; i < cntn; ++i) {
        nx[start + i] = xx[offn + i];
        ny[start + i] = yy[offn + i];
        nz[start + i] = zz[offn + i];
        nm[start + i] = mass[offn + i];
      }

      continue;
    }

    // This other node is not a leaf, but has too large an opening angle
    // for an approx. interaction: queue its children.

    ID_T tlncr = tree_[tln].cr;
    ID_T tlncl = tree_[tln].cl;

    if (tlncl > 0 && tree_[tlncl].count > 0) {
      bool close = true;
      for (int i = 0; i < DIMENSION; ++i) {
        POSVEL_T dist = 0;
        if (tree_[tl].xmax[i] < tree_[tlncl].xmin[i]) {
          dist = tree_[tlncl].xmin[i] - tree_[tl].xmax[i];
        } else if (tree_[tl].xmin[i] > tree_[tlncl].xmax[i]) {
          dist = tree_[tl].xmin[i] - tree_[tlncl].xmax[i];
        }

        if (dist > fsrrmax) {
          close = false;
          break;
        }
      }

      if (close) q.push_back(tlncl);
    }
    if (tlncr > 0 && tree_[tlncr].count > 0) {
      bool close = true;
      for (int i = 0; i < DIMENSION; ++i) {
        POSVEL_T dist = 0;
        if (tree_[tl].xmax[i] < tree_[tlncr].xmin[i]) {
          dist = tree_[tlncr].xmin[i] - tree_[tl].xmax[i];
        } else if (tree_[tl].xmin[i] > tree_[tlncr].xmax[i]) {
          dist = tree_[tl].xmin[i] - tree_[tlncr].xmax[i];
        }

        if (dist > fsrrmax) {
          close = false;
          break;
        }
      }

      if (close) q.push_back(tlncr);
    }
  }

  ID_T off = tree_[tl].offset;
  ID_T cnt = tree_[tl].count;

  // Add self interactions...
  int start = SIZE;
  SIZE = SIZE + cnt;
  assert( SIZE < VMAX );

#pragma ivdep 
  for ( int i = 0; i < cnt; ++i) {
    nx[start + i] = xx[off + i];
    ny[start + i] = yy[off + i];
    nz[start + i] = zz[off + i];
    nm[start + i] = mass[off + i];
  }

#ifdef HACC_CUDA
  //compute batch device pointers
  POSVEL_T *bx = nx_d + batch.offset;
  POSVEL_T *by = ny_d + batch.offset;
  POSVEL_T *bz = nz_d + batch.offset;
  POSVEL_T *bm = nm_d + batch.offset;
  batch.add(cnt,SIZE, bx + start, by + start, bz + start, bx, by, bz, bm, vx_d + off, vy_d + off, vz_d + off);
  
  if(batch.size==BATCH_SIZE) {
#ifndef HACC_NO_PCIE_TRANSFER
    cudaMemcpyAsync(nx_d,nx_,sizeof(POSVEL_T)*batch.offset,cudaMemcpyHostToDevice);
    cudaMemcpyAsync(ny_d,ny_,sizeof(POSVEL_T)*batch.offset,cudaMemcpyHostToDevice);
    cudaMemcpyAsync(nz_d,nz_,sizeof(POSVEL_T)*batch.offset,cudaMemcpyHostToDevice);
    cudaMemcpyAsync(nm_d,nm_,sizeof(POSVEL_T)*batch.offset,cudaMemcpyHostToDevice);
#endif
    cudaEventRecord(event); //record event to mark end of transfer
    ::nbody1_batched(batch,m_fcoeff,fsrrmax,rsm);
    batch.clear();
    cudaEventSynchronize(event); //wait for transfer to finish before exiting (i.e. safe to overwrite host buffers)
  }
#else
  ::nbody1(cnt, SIZE, nx + start, ny + start, nz + start, nm + start, &nx[0], &ny[0], &nz[0], &nm[0], vx + off, vy + off, vz + off, m_fcoeff, fsrrmax, rsm);
#endif
}

// Iterate through the tree nodes, for each leaf node, start a task.
// That task iterates through the tree nodes, skipping any node (and all
// of its children) if all corners are too far away. Then it compares the
// opening angle.
template <int TDPTS>
void RCBForceTree<TDPTS>::calcInternodeForces()
{
#if defined(_OPENMP) && !defined(__bgq__)
#pragma omp parallel
#pragma omp single
  {
#endif
  std::vector<ID_T> q(1, 0);
  std::vector<ID_T> parents;
  while (!q.empty()) {
    ID_T tl = q.back();
    if (tree[tl].cr == 0 && tree[tl].cl == 0) {
      // This is a leaf node.
      q.pop_back();

      bool inside = true;
      for (int i = 0; i < DIMENSION; ++i) {
        inside &= (tree[tl].xmax[i] < maxForceRange[i] && tree[tl].xmax[i] > minForceRange[i]) ||
                  (tree[tl].xmin[i] < maxForceRange[i] && tree[tl].xmin[i] > minForceRange[i]);
      }

      if (inside) {
#if defined(_OPENMP) && _OPENMP >= 200805 && !defined(__bg__)
#pragma omp task
#endif

        calcInternodeForce(tl, parents);
      }
    } else if (parents.size() > 0 && parents.back() == tl) {
      // This is second time here; we've done with all children.
      parents.pop_back();
      q.pop_back();
    } else {
      // This is the first time at this parent node, queue the children.
      if (tree[tl].cl > 0) q.push_back(tree[tl].cl);
      if (tree[tl].cr > 0) q.push_back(tree[tl].cr);
      parents.push_back(tl);
    }
  }

#if defined(_OPENMP) && _OPENMP >= 200805 && !defined(__bg__)
#pragma omp taskwait
#endif
#if defined(_OPENMP) && !defined(__bgq__)
  }
#endif
}

// Explicit template instantiation...
template class RCBForceTree<QUADRUPOLE_TDPTS>;
template class RCBForceTree<MONOPOLE_TDPTS>;

