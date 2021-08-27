// this replaces zplanint from LSMS_1.9

#include <vector>
#include <mpi.h>
#include <complex>
#include "Complex.hpp"

#include "Communication/LSMSCommunication.hpp"
#include "SingleSite/SingleSiteScattering.hpp"
#include "MultipleScattering/MultipleScattering.hpp"
#include "EnergyContourIntegration.hpp"
#include "Misc/Coeficients.hpp"
#include "calculateDensities.hpp"
// #include <omp.h>

#ifdef BUILDKKRMATRIX_GPU
void copyTmatStoreToDevice(LocalTypeInfo &local);
#include "Accelerator/buildKKRMatrix_gpu.hpp"
extern std::vector<void *> deviceConstants;
extern void * deviceStorage;
#endif
#if defined(ACCELERATOR_CULA) || defined(ACCELERATOR_LIBSCI) || defined(ACCELERATOR_CUDA_C)
#include "Accelerator/DeviceStorage.hpp"
#endif
void solveSingleScatterers(LSMSSystemParameters &lsms, LocalTypeInfo &local,
                           std::vector<Matrix<Real> > &vr, Complex energy,
                           std::vector<NonRelativisticSingleScattererSolution> &solution,int iie);

extern "C"
{
//     cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
void constraint_(int *jmt,Real *rmt,int *n_spin_pola,
                 Real*vr,Real *r_mesh,Real *pi4,
                 Real *evec,Real *evec_r,Real *b_con,Real *b_basis,
                 int *i_vdif,Real *h_app_para_mag,Real *h_app_perp_mag,
                 int *iprpts,
                 int *iprint,char *istop,int len_sitop);
//     cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
void u_sigma_u_(Complex *ubr,Complex *ubrd,
               Complex *wx,Complex *wy,Complex *wz);
//     cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
void green_function_(int *mtasa,int *n_spin_pola,int *n_spin_cant,
                     int *lmax,int *kkrsz,Complex *wx, Complex *wy, Complex *wz,
                     Real *rins,Real *r_sph,Real *r_mesh,int *jmt,int *jws,
                     Complex *pnrel,Complex *tau00_l,Complex *matom,Complex *zlr,Complex *jlr,
                     int *nprpts, int* nplmax,
                     int *ngaussr,
                     Real *cgnt, int *lmax_cg,
                     Complex *dos,Complex *dosck,Complex *green,Complex *dipole,
                     int *ncrit,Real *grwylm,Real *gwwylm,Complex *wylm,
                     int *iprint,char *istop,int len_sitop);
//     ================================================================
}

void buildEnergyContour(int igrid,Real ebot,Real etop,Real eibot, Real eitop,
                        std::vector<Complex> &egrd, std::vector<Complex> &dele1, int npts, int &nume,
                        int iprint, char *istop)
{
  Real pi=2.0*std::asin(1.0);
  int ipepts;
  if(iprint>=0) printf("Energy Contour Parameters: grid=%d npts=%d, ebot=%lf etop=%lf eibot=%lf eitop=%lf\n",
                      igrid,npts,ebot,etop,eibot,eitop);
  switch (igrid)
  {
  case 0: // single energy point
    egrd.resize(1); dele1.resize(1);
    egrd[0]=std::complex<Real>(ebot,eibot); dele1[0]=0.0; nume=1;
    break;
  case 2: // Gaussian Contour
    egrd.resize(npts+2); dele1.resize(npts+2);
    ipepts=egrd.size();
    congauss_(&ebot,&etop,&eibot,&egrd[0],&dele1[0],&npts,&nume,&pi,&ipepts,&iprint,istop,32);
    break;
  default:
    fprintf(stderr,"Unknown energy grid type %d in 'buildEnergyContour'\n",igrid);
    exit(1);
  }
}

void energyContourIntegration(LSMSCommunication &comm,LSMSSystemParameters &lsms, LocalTypeInfo &local)
{
  double timeEnergyContourIntegration_1=MPI_Wtime();
  double timeCalcDensities;

// energy grid info
  std::vector<Complex> egrd,dele1;
  int nume;
// constrained potentials:
  std::vector<Matrix<Real > > vr_con;
  Matrix<Real> evec_r;

  int i_vdif=0;

  Real pi4=4.0*2.0*std::asin(1.0);

  vr_con.resize(local.num_local);
  evec_r.resize(3,local.num_local);
#pragma omp parallel for default(none) shared(local,lsms,vr_con,evec_r)
  for(int i=0; i<local.num_local; i++)
  {
    Real pi4=4.0*2.0*std::asin(1.0);
    int i_vdif=0;
//     ================================================================
//     set up the spin space stransformation matrix....................
//     ================================================================
// check evec
    Real evec_norm=std::sqrt(local.atom[i].evec[0]*local.atom[i].evec[0]
                             +local.atom[i].evec[1]*local.atom[i].evec[1]
                             +local.atom[i].evec[2]*local.atom[i].evec[2]);
    if(std::abs(evec_norm-1.0)>1.0e-5) printf("|atom[%d].evec|=%lf\n",i,evec_norm);
//
    spin_trafo_(&local.atom[i].evec[0],&local.atom[i].ubr[0],&local.atom[i].ubrd[0]);
    u_sigma_u_(&local.atom[i].ubr[0],&local.atom[i].ubrd[0],
               &local.atom[i].wx[0],&local.atom[i].wy[0],&local.atom[i].wz[0]);
//     ================================================================
//     set up Constraint ..............................................
//     copy vr into vr_con which contains the B-field constraint.......
//     calls to gettau_c etc. require vr_con...........................
//     ================================================================
    vr_con[i]=local.atom[i].vr;
    if(lsms.n_spin_cant==2)
    {
      Real h_app_para_mag=0.0;
      Real h_app_perp_mag=0.0;
      int iprpts=local.atom[i].r_mesh.size();
// here I leave out the i_vdif<0 case!
      constraint_(&local.atom[i].jmt,&local.atom[i].rmt,&lsms.n_spin_pola,
                  &(vr_con[i])(0,0),&local.atom[i].r_mesh[0],&pi4,
                  &local.atom[i].evec[0],&evec_r(0,i),local.atom[i].b_con,
                  local.atom[i].b_basis,&i_vdif,&h_app_para_mag,&h_app_perp_mag,
                  &iprpts,
                  &lsms.global.iprint,lsms.global.istop,32);
      if(lsms.nrel_rel==0)
      {
        spin_trafo_(&evec_r(0,i),&local.atom[i].ubr[0],&local.atom[i].ubrd[0]);
      } else { //. relativistic
//        r_global(1) = 0.0;
//        r_global(2) = 0.0;
//        r_global(3) = 1.0;
//           call matrot1(r_global,evec_r,lmax,dmat,dmatp)
      }
    } else {
// call zcopy(4,u,1,ubr,1)
// call zcopy(4,ud,1,ubrd,1)
    }
  }
  // Real e_top;
  // e_top=lsms.energyContour.etop;
  // if(lsms.energyContour.etop==0.0) etop=lsms.chempot;
  buildEnergyContour(lsms.energyContour.grid, lsms.energyContour.ebot, lsms.chempot,
                     lsms.energyContour.eibot, lsms.energyContour.eitop, egrd, dele1,
                     lsms.energyContour.npts, nume, lsms.global.iprint, lsms.global.istop);

  for(int i=0; i<local.num_local; i++)
  {
    if(local.atom[i].dos_real.l_dim()<nume) 
      local.atom[i].dos_real.resize(nume,4);
  }

  std::vector<std::vector<NonRelativisticSingleScattererSolution> >solution;
  solution.resize(lsms.energyContour.groupSize());
  for(int ie=0; ie<lsms.energyContour.groupSize(); ie++) solution[ie].resize(local.num_local);
  int maxkkrsz=(lsms.maxlmax+1)*(lsms.maxlmax+1);
  int maxkkrsz_ns=lsms.n_spin_cant*maxkkrsz;
  Matrix<Complex> tau00_l(maxkkrsz_ns*maxkkrsz_ns,local.num_local);
  Matrix<Complex> dos(4,local.num_local);
  // dos=0.0;
  Matrix<Complex> dosck(4,local.num_local);
  // dosck=0.0;
  Array3d<Complex> dipole(6,4,local.num_local);
  // dipole=0.0;
  Array3d<Complex> green(local.atom[0].jws,4,local.num_local);
  // green=0.0;

// setup Device constant on GPU
  int maxNumLIZ=0;
#ifdef BUILDKKRMATRIX_GPU
  #pragma omp parallel for default(none) shared(lsms,local,deviceConstants)
  for(int i=0; i<local.num_local; i++)
  {
    setupForBuildKKRMatrix_gpu_opaque(lsms,local.atom[i],deviceConstants[i]);
  }
#endif
  for(int i=0; i<local.num_local; i++)
  {
    if(local.atom[i].numLIZ>maxNumLIZ) maxNumLIZ=local.atom[i].numLIZ;
  }
#if defined(ACCELERATOR_CULA) || defined(ACCELERATOR_LIBSCI) || defined(ACCELERATOR_CUDA_C)
  initDStore(deviceStorage,maxkkrsz,lsms.n_spin_cant,maxNumLIZ,lsms.global.GPUThreads);
#endif

// inside an omp for to ensure first touch
#pragma omp parallel for default(none) shared(local,dos,dosck,dipole,green)
  for(int i=0; i<local.num_local; i++)
  {
    for(int j=0; j<4; j++)
    {
      dos(j,i)=dosck(j,i)=0.0;
      for(int k=0; k<6; k++) dipole(k,j,i)=0.0;
      for(int k=0; k<local.atom[i].jws; k++) green(k,j,i)=0.0;
    }
    local.atom[i].resetLocalDensities();
  }

  timeEnergyContourIntegration_1=MPI_Wtime()-timeEnergyContourIntegration_1;

  double timeEnergyContourIntegration_2=MPI_Wtime();
  double timeCalculateAllTauMatrices=0.0;

// energy groups:
  int eGroupRemainder=nume%lsms.energyContour.groupSize();
  int numEGroups=nume/lsms.energyContour.groupSize()+std::min(1,eGroupRemainder);
  std::vector<int> eGroupIdx(numEGroups+1);
  for(int ig=0; ig<numEGroups; ig++) eGroupIdx[ig]=ig*lsms.energyContour.groupSize();
  eGroupIdx[numEGroups]=nume;

  for(int ig=0; ig<numEGroups; ig++)
  {
// solve single site problem
    double timeSingleScatterers=MPI_Wtime();
    expectTmatCommunication(comm,local);
    local.tmatStore=0.0;

if(lsms.global.iprint>=0) printf("calculate single scatterer solutions.\n");

#pragma omp parallel for default(none) shared(local,lsms,eGroupIdx,ig,egrd,solution,vr_con)
    for(int ie=eGroupIdx[ig]; ie<eGroupIdx[ig+1]; ie++)
    {
      int iie=ie-eGroupIdx[ig];
      Complex energy=egrd[ie];
      Complex pnrel=std::sqrt(energy);

      solveSingleScatterers(lsms,local,vr_con,energy,solution[iie],iie);
    }

    if(lsms.global.iprint>=1) printf("About to send t matrices\n");
    sendTmats(comm,local);
    if(lsms.global.iprint>=1) printf("About to finalize t matrices communication\n");
    finalizeTmatCommunication(comm);
    if(lsms.global.iprint>=1) printf("Recieved all t matricies\n");
    timeSingleScatterers=MPI_Wtime()-timeSingleScatterers;
    if(lsms.global.iprint>=0) printf("timeSingleScatteres = %lf sec\n",timeSingleScatterers);

#ifdef BUILDKKRMATRIX_GPU
  copyTmatStoreToDevice(local);
#endif

  for(int ie=eGroupIdx[ig]; ie<eGroupIdx[ig+1]; ie++)
  {
    int iie=ie-eGroupIdx[ig];
    Complex energy=egrd[ie];
    Complex pnrel=std::sqrt(energy);
    if(lsms.global.iprint>=0) printf("Energy #%d (%lf,%lf)\n",ie,real(energy),imag(energy));

    double timeCATM=MPI_Wtime();
    calculateAllTauMatrices(comm, lsms, local, vr_con, energy, iie, tau00_l);

    timeCalculateAllTauMatrices+=MPI_Wtime()-timeCATM;
    // if(!lsms.global.checkIstop("buildKKRMatrix"))
    {
       // IBM : include this output only for iprint > 0
       if (lsms.global.iprint>0) 
       {
          timeCalcDensities=MPI_Wtime();
       }
    if(lsms.nrel_rel==0)
    {
// openMP here
#pragma omp parallel for default(none) \
        shared(local,lsms,dos,dosck,green,dipole,solution,gauntCoeficients,dele1,tau00_l) \
        firstprivate(ie,iie,pnrel,energy,nume)
      for(int i=0; i<local.num_local; i++)
      {
        //Real r_sph=local.atom[i].r_mesh[local.atom[i].jws];
        //if (lsms.mtasa==0) r_sph=local.atom[i].r_mesh[local.atom[i].jmt];
        Real r_sph=local.atom[i].rInscribed;
        if(lsms.mtasa>0) r_sph=local.atom[i].rws;
        Real rins=local.atom[i].rmt;
//        int nprpts=solution[iie][i].zlr.l_dim1();
        int nprpts=local.atom[i].r_mesh.size();
//        int nplmax=solution[iie][i].zlr.l_dim2()-1;
        int nplmax=local.atom[i].lmax;
        green_function_(&lsms.mtasa,&lsms.n_spin_pola,&lsms.n_spin_cant,
                        &local.atom[i].lmax, &local.atom[i].kkrsz,
                        &local.atom[i].wx[0],&local.atom[i].wy[0],&local.atom[i].wz[0],
                        &rins,&r_sph,&local.atom[i].r_mesh[0],&local.atom[i].jmt,&local.atom[i].jws,
                        &pnrel,&tau00_l(0,i),&solution[iie][i].matom(0,0),
                        &solution[iie][i].zlr(0,0,0),&solution[iie][i].jlr(0,0,0),
                        &nprpts,&nplmax,
                        &lsms.ngaussr, &gauntCoeficients.cgnt(0,0,0), &gauntCoeficients.lmax,
                        &dos(0,i),&dosck(0,i),&green(0,0,i),&dipole(0,0,i),
                        &local.atom[i].voronoi.ncrit,&local.atom[i].voronoi.grwylm(0,0),
                        &local.atom[i].voronoi.gwwylm(0,0),&local.atom[i].voronoi.wylm(0,0,0),
                        &lsms.global.iprint,lsms.global.istop,32);
        Complex tr_pxtau[3];
        calculateDensities(lsms, i, 0, ie, nume, energy, dele1[ie],
                           dos,dosck,green,
                           dipole,
                           local.atom[i]);

      }
    } else {
        printf("Relativistic version not implemented yet\n");
        exit(1);
    }

    if (lsms.global.iprint>0) 
    {
       timeCalcDensities=MPI_Wtime()-timeCalcDensities;
       printf("timeCalculateDensities = %lf sec\n",timeCalcDensities);
    }
  }
  }
  }
  timeEnergyContourIntegration_2=MPI_Wtime()-timeEnergyContourIntegration_2;
  if(lsms.global.iprint>=0)
  {
    printf("time in energyContourIntegration = %lf sec\n",timeEnergyContourIntegration_1+timeEnergyContourIntegration_2);
    printf("  before energy loop             = %lf sec\n",timeEnergyContourIntegration_1);
    printf("  in energy loop                 = %lf sec\n",timeEnergyContourIntegration_2);
    printf("    in calculateAllTauMatrices   = %lf sec\n",timeCalculateAllTauMatrices);
  }
}
