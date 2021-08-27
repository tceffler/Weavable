#ifndef LSMS_ATOMDATA_H
#define LSMS_ATOMDATA_H

#include <vector>
#include <algorithm>

#include "Real.hpp"
#include "Complex.hpp"
#include "Matrix.hpp"
#include "Array3d.hpp"
#include "VORPOL/VORPOL.hpp"

extern "C"
{
void spin_trafo_(Real *evec, Complex * u, Complex *ud);
}


#ifdef BUILDKKRMATRIX_GPU
void * allocateDConst(void);
void freeDConst(void *);
#endif


class AtomData {
public:

  AtomData() {reset();}

  ~AtomData() {}

  void reset(void)
  {
    b_con[0] = b_con[1] = b_con[2] = 0.0;
    for(int i=0; i<9; i++) b_basis[i] = 0.0;
    b_basis[0] = b_basis[4] = b_basis[8] = 1.0;
  }


  void resizePotential(int npts)
  {
    int n = npts;

    vr.resize(n,2);
    vr = 0.0;
    vSpinShift=0.0;

    rhotot.resize(n,2);
    rhotot = 0.0;

    corden.resize(n,2);
    corden=0.0;

    semcor.resize(n,2);
    r_mesh.resize(n);
    x_mesh.resize(n);

    vrNew.resize(n,2);
    vrNew = 0.0;

    rhoNew.resize(n,2);
    rhoNew=0.0;

    dos_real.resize(40,4);
    greenint.resize(n,4);
    greenlast.resize(n,4);

    // Check if they should be put here...
    exchangeCorrelationPotential.resize(n,2);
    exchangeCorrelationPotential = 0.0;
    exchangeCorrelationEnergy.resize(n,2);
    exchangeCorrelationEnergy = 0.0;

  }


  void resizeCore(int ncs)
  {
    ec.resize(ncs,2);
    nc.resize(ncs,2);
    lc.resize(ncs,2);
    kc.resize(ncs,2);
  }


  AtomData &operator=(const AtomData &a)
  {
    jmt = a.jmt;
    jws = a.jws;
    xstart = a.xstart;
    rmt = a.rmt;
    h = a.h;
    r_mesh = a.r_mesh;
    x_mesh = a.x_mesh;

    alat = a.alat;
    efermi = a.efermi;
    vdif = a.vdif;
    ztotss = a.ztotss;
    zcorss = a.zcorss;
    zsemss = a.zsemss;
    zvalss = a.zvalss;

    nspin = a.nspin;
    numc = a.numc;
    afm = a.afm;

    evec[0] = a.evec[0];
    evec[1] = a.evec[1];
    evec[2] = a.evec[2];
    xvalws[0] = a.xvalws[0];
    xvalws[1] = a.xvalws[1];
    xvalmt[0] = a.xvalmt[0];
    xvalmt[1] = a.xvalmt[1];
    qtotws = a.qtotws;
    mtotws = a.mtotws;
    qtotmt = a.qtotmt;
    mtotmt = a.mtotmt;
    qvalws = a.qvalws;
    mvalws = a.mvalws;
    qvalmt = a.qvalmt;
    mvalmt = a.mvalmt;

    qInt = a.qInt;
    mInt = a.mInt;
    rhoInt = a.rhoInt;
    mIntComponent[0] = a.mIntComponent[0];
    mIntComponent[1] = a.mIntComponent[1];
    mIntComponent[2] = a.mIntComponent[2];

    for(int i=0; i<80; i++) header[i] = a.header[i];

    vr = a.vr;
    vSpinShift=a.vSpinShift;
    rhotot = a.rhotot;
    
    exchangeCorrelationPotential = a.exchangeCorrelationPotential;
    exchangeCorrelationEnergy = a.exchangeCorrelationEnergy;
    exchangeCorrelationV[0] = a.exchangeCorrelationV[0];
    exchangeCorrelationV[1] = a.exchangeCorrelationV[1];
    exchangeCorrelationE = a.exchangeCorrelationE;

    ec = a.ec;
    nc = a.nc;
    lc = a.lc;
    kc = a.kc;

    ecorv[0] = a.ecorv[0];
    ecorv[1] = a.ecorv[1];
    esemv[0] = a.esemv[0];
    esemv[1] = a.esemv[1];

    corden = a.corden;
    semcor = a.semcor;
    qcpsc_mt = a.qcpsc_mt;
    qcpsc_ws = a.qcpsc_ws;
    mcpsc_mt = a.mcpsc_mt;
    mcpsc_ws = a.mcpsc_ws;

    b_con[0] = a.b_con[0];
    b_con[1] = a.b_con[1];
    b_con[2] = a.b_con[2];

    for(int i=0; i<9; i++) b_basis[i] = a.b_basis[i];

    return *this;
  }


  void generateRadialMesh(void)
  {
    int N = std::max((int)r_mesh.size(), jws);
    N = std::max(N, jmt);
    if (N != r_mesh.size()) r_mesh.resize(N);
    Real xmt = std::log(rmt);
    h = (xmt-xstart) / (jmt-1);
    for(int j=0; j<N; j++)
    {
      x_mesh[j] = xstart + (Real)j*h;
      r_mesh[j] = std::exp(x_mesh[j]);
    }
    generateNewMesh = false;
  }


  void setEvec(Real x, Real y, Real z)
  {
    evec[0] = x;
    evec[1] = y;
    evec[2] = z;
    spin_trafo_(evec, ubr, ubrd);
  }


// Local Interaction Zone
  int numLIZ;
  std::vector<int> LIZGlobalIdx, LIZStoreIdx, LIZlmax;
  int nrmat;                          // sum (LIZlmax+1)^2
  std::vector<Real> LIZDist;
  Matrix<Real> LIZPos;

// Mesh Data:
  int jmt,jws;
  Real xstart,rmt,h;
  Real rInscribed; // LSMS_1.9: rins
  std::vector<Real> r_mesh, x_mesh;
  bool generateNewMesh;

// General Data
  char header[80];
  int lmax, kkrsz;
  Real alat, efermi;
  Real ztotss, zcorss, zsemss, zvalss;
  Real vdif, vdifNew;
  Real evec[3], evecNew[3];
  Complex ubr[4], ubrd[4];            // Spin transformation matrices
  Complex wx[4], wy[4], wz[4];
  Real xvalws[2], xvalwsNew[2];
  Real xvalmt[2];
  Real qtotws, mtotws;
  Real qtotmt, mtotmt;
  Real qvalws, mvalws;
  Real qvalmt, mvalmt;
  Real qInt, mInt, rhoInt;            // Interstitial charge, moment and charge density
  Real mIntComponent[3];              // Interstitial moment components
  int nspin, numc;
  int afm;                            // Flag for antiferromagnetic condition

// Volumes:
  Real omegaMT;                       // Muffin-Tin volume
  Real omegaWS;                       // Wigner-Seitz volume
  Real rws;                           // Wigner-Seitz radius

// omegaInt - interstitial volume is in voronoi.omegaInt

// Madelung matrix
  std::vector<Real> madelungMatrix;

// Potential and charge density
  Real vSpinShift; // relativ shift of the spin up and spin down potentials
                   // for use in WL-LSMS. Applied using the PotentialShifter class
  Matrix<Real> vr, rhotot;

// Storage for newly calculated potential and chage densities before mixing
  Matrix<Real> vrNew,rhoNew;

// Exchange-correlation parameters
  Matrix<Real> exchangeCorrelationPotential;     // Exchange-correlation potential
  Matrix<Real> exchangeCorrelationEnergy;        // Exchange-correlation energy
  Real exchangeCorrelationE;                     // Exchange-correlation energy
  Real exchangeCorrelationV[2];                  // Exchange-correlation potential for spin up/down

// Core state info
  Matrix<Real> ec;
  Matrix<int> nc, lc, kc;
  Real ecorv[2], esemv[2];
  Matrix<Real> corden, semcor;
  Real qcpsc_mt, qcpsc_ws, mcpsc_mt, mcpsc_ws;

// Constraint data
  Real b_con[3];
  Real b_basis[9];

// vector for the energiy points in eGroup
  std::vector<Matrix<Complex> > pmat_m;

  VoronoiPolyhedra voronoi;

// local densities
  Matrix<Real> dos_real;
  Real doslast[4];
  Real doscklast[4];
  Real evalsum[4];
  Real dosint[4];
  Real dosckint[4];
  Matrix<Real> greenint;
  Matrix<Real> greenlast;
  Real dip[6];

  void resetLocalDensities(void)
  {
    dos_real=0.0;
    greenint=0.0;
    greenlast=0.0;
    doslast[0]=doslast[1]=doslast[2]=doslast[3]=0.0;
    doscklast[0]=doscklast[1]=doscklast[2]=doscklast[3]=0.0;
    evalsum[0]=evalsum[1]=evalsum[2]=evalsum[3]=0.0;
    dosint[0]=dosint[1]=dosint[2]=dosint[3]=0.0;
    dosckint[0]=dosckint[1]=dosckint[2]=dosckint[3]=0.0;
    dip[0]=dip[1]=dip[2]=dip[3]=dip[4]=dip[5]=0.0;
  }
};

#endif
