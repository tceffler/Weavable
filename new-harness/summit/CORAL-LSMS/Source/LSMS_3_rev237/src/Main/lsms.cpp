#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// #include <fenv.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// #define USE_PAPI 1
#ifdef USE_PAPI
#include <papi.h>
#endif

#ifdef USE_GPTL
#include "gptl.h"
#endif

#include <hdf5.h>

#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"

#include "SystemParameters.hpp"
#include "PotentialIO.hpp"
#include "Communication/distributeAtoms.hpp"
#include "Communication/LSMSCommunication.hpp"
#include "Core/CoreStates.hpp"
#include "Misc/Indices.hpp"
#include "Misc/Coeficients.hpp"
#include "Madelung/Madelung.hpp"
#include "VORPOL/VORPOL.hpp"
#include "EnergyContourIntegration.hpp"
#include "Accelerator/Accelerator.hpp"
#include "calculateChemPot.hpp"
#include "calculateDensities.hpp"
#include "mixing.hpp"
#include "calculateEvec.hpp"
#include "Potential/calculateChargesPotential.hpp"
#include "Potential/interpolatePotential.hpp"
#include "TotalEnergy/calculateTotalEnergy.hpp"

#include "Misc/readLastLine.hpp"

SphericalHarmonicsCoeficients sphericalHarmonicsCoeficients;
GauntCoeficients gauntCoeficients;
IFactors iFactors;

#ifdef BUILDKKRMATRIX_GPU
#include "Accelerator/DeviceStorage.hpp"
void *allocateDConst(void);
void freeDConst(void *);

std::vector<void *> deviceConstants;
// std::vector<void *> deviceStorage;
void * deviceStorage;
#endif

void initLSMSLuaInterface(lua_State *L);
int readInput(lua_State *L, LSMSSystemParameters &lsms, CrystalParameters &crystal, MixingParameters &mix);
void buildLIZandCommLists(LSMSCommunication &comm, LSMSSystemParameters &lsms,
                          CrystalParameters &crystal, LocalTypeInfo &local);
void setupVorpol(LSMSSystemParameters &lsms, CrystalParameters &crystal, LocalTypeInfo &local,
                 SphericalHarmonicsCoeficients &shc);

void calculateVolumes(LSMSCommunication &comm, LSMSSystemParameters &lsms, CrystalParameters &crystal, LocalTypeInfo &local);

/*
static int
feenableexcept (unsigned int excepts)
{
  static fenv_t fenv;
  unsigned int new_excepts = excepts & FE_ALL_EXCEPT,
               old_excepts;  // previous masks

  if ( fegetenv (&fenv) ) return -1;
  old_excepts = fenv.__control & FE_ALL_EXCEPT;

  // unmask
  fenv.__control &= ~new_excepts;
  fenv.__mxcsr   &= ~(new_excepts << 7);

  return ( fesetenv (&fenv) ? -1 : old_excepts );
}
*/

int main(int argc, char *argv[])
{
  LSMSSystemParameters lsms;
  LSMSCommunication comm;
  CrystalParameters crystal;
  LocalTypeInfo local;
  MixingParameters mix;

  char inputFileName[128];

  Real eband;

  lua_State *L=lua_open();
  luaL_openlibs(L);
  initLSMSLuaInterface(L);

  // feenableexcept(FE_INVALID);

#ifdef USE_GPTL
  GPTLinitialize();
#endif
  initializeCommunication(comm);
  H5open();

  // set input file name (default 'i_lsms')
  strncpy(inputFileName,"i_lsms",10);
  if(argc>1) strncpy(inputFileName,argv[1],120);

  lsms.global.iprpts=1051;
  lsms.global.ipcore=15;
  lsms.global.setIstop("main");
  lsms.global.iprint=0;
  lsms.global.default_iprint=-1;
  lsms.global.print_node=0;
  lsms.ngaussr=10;
  lsms.ngaussq=40;
  lsms.vSpinShiftFlag=0;
#ifdef _OPENMP
  lsms.global.GPUThreads=std::min(12,omp_get_max_threads());
#else
  lsms.global.GPUThreads=1;
#endif
  if(comm.rank==0) lsms.global.iprint=0;

  if(comm.rank==0)
  {
    printf("LSMS_3: Program started\n");
    printf("Using %d MPI processes\n",comm.size);
#ifdef _OPENMP
    printf("Using %d OpenMP threads\n",omp_get_max_threads());
#endif
    acceleratorPrint();
#ifdef BUILDKKRMATRIX_GPU
    printf("Using GPU to build KKR matrix.\n");
#endif
    printf("Reading input file '%s'\n",inputFileName);

    if(luaL_loadfile(L, inputFileName) || lua_pcall(L,0,0,0))
    {
      fprintf(stderr,"!! Cannot run input file!!\n");
      exit(1);
    }

    if(readInput(L,lsms,crystal,mix))
    {
      fprintf(stderr,"!! Something wrong in input file!!\n");
      exit(1);
    }
  }

 
  communicateParameters(comm,lsms,crystal,mix);
  if(comm.rank!=lsms.global.print_node) lsms.global.iprint=lsms.global.default_iprint;
  // printf("maxlmax=%d\n",lsms.maxlmax);

  local.setNumLocal(distributeTypes(crystal, comm));
  local.setGlobalId(comm.rank,crystal);

  lsms.angularMomentumIndices.init(2*crystal.maxlmax);
  sphericalHarmonicsCoeficients.init(2*crystal.maxlmax);

  gauntCoeficients.init(lsms,lsms.angularMomentumIndices,sphericalHarmonicsCoeficients);
  iFactors.init(lsms,crystal.maxlmax);

  printf("before buildLIZandCommLists\n");
  buildLIZandCommLists(comm, lsms, crystal, local);
  printf("after buildLIZandCommLists: num_local=%d\n",local.num_local);


// initialize the potential accelerators (GPU)
// we need to know the max. size of the kkr matrix to invert: lsms.n_spin_cant*local.maxNrmat()
// which is only available after building the LIZ

  acceleratorInitialize(lsms.n_spin_cant*local.maxNrmat(),lsms.global.GPUThreads);
  local.tmatStore.pinMemory();

#ifdef BUILDKKRMATRIX_GPU
  deviceConstants.resize(local.num_local);
  for(int i=0; i<local.num_local; i++) deviceConstants[i]=allocateDConst();
  // deviceStorage.resize(lsms.global.GPUThreads);
  // for(int i=0; i<lsms.global.GPUThreads; i++) deviceStorage[i]=allocateDStore();
  deviceStorage=allocateDStore();
#endif

  for(int i=0; i<local.num_local; i++)
    local.atom[i].pmat_m.resize(lsms.energyContour.groupSize());  

// set maximal number of radial grid points and core states if reading from bigcell file
  local.setMaxPts(lsms.global.iprpts);
  local.setMaxCore(lsms.global.ipcore);

  if(lsms.global.iprint>=0) printLSMSGlobals(stdout,lsms);

  if(lsms.global.iprint>=2)
  {
    printLSMSSystemParameters(stdout,lsms);
    printCrystalParameters(stdout,crystal);
  }
  if(lsms.global.iprint>=0)
  {
    fprintf(stdout,"LIZ for atom 0 on this node\n");
    printLIZInfo(stdout,local.atom[0]);
  }
  if(lsms.global.iprint>=1)
  {
    printCommunicationInfo(stdout, comm);
  }

//  initialAtomSetup(comm,lsms,crystal,local);

// the next line is a hack for initialization of potentials from scratch to work.
//  if(lsms.pot_in_type<0) setupVorpol(lsms,crystal,local,sphericalHarmonicsCoeficients);

  loadPotentials(comm,lsms,crystal,local);

// for testing purposes:
//  std::vector<Matrix<Real> > vrs;
//  vrs.resize(local.num_local);
//  for(int i=0; i<local.num_local; i++) vrs[i]=local.atom[i].vr;
// -------------------------------------

  setupVorpol(lsms,crystal,local,sphericalHarmonicsCoeficients);

// Generate new grids after new rmt is defined
  for (int i=0; i<local.num_local; i++)
  {
    if(local.atom[i].generateNewMesh)
      interpolatePotential(lsms, local.atom[i]);
  }

  calculateVolumes(comm,lsms,crystal,local);

//  loadPotentials(comm,lsms,crystal,local);

// initialize Mixing
  Mixing *mixing;
  setupMixing(mix, mixing);

// need to calculate madelung matrices
  calculateMadelungMatrices(lsms,crystal,local);

  if(lsms.global.iprint>=1)
  {
    printLocalTypeInfo(stdout,local);
  }

  calculateCoreStates(comm,lsms,local);
  if(lsms.global.iprint>=0)
    printf("Finished calculateCoreStates(...)\n");

// check that vrs have not changed ...
//  bool vr_check=false;
//  for(int i=0; i<local.num_local; i++)
//  {
//    vr_check=true;
//    for(int j=0; j<vrs[i].n_row();j++)
//      for(int k=0; k<vrs[i].n_col(); k++)
//        vr_check=vr_check && (vrs[i](j,k)==local.atom[i].vr(j,k));
//    if(!vr_check)
//      printf("Potential %d has changed!!\n",i);
//  }
//  printf("Potentials have been checked\n");
// --------------------------------------------

/*
// only for test
  expectTmatCommunication(comm,local);
  sendTmats(comm,local);
  finalizeTmatCommunication(comm);
//
*/

#ifdef USE_PAPI
  #define NUM_PAPI_EVENTS 2
  int hw_counters = PAPI_num_counters();
  if(hw_counters>NUM_PAPI_EVENTS) hw_counters=NUM_PAPI_EVENTS;
  int papi_events[NUM_PAPI_EVENTS]; 
  char *papi_event_name[] = {"PAPI_TOT_INS","PAPI_FP_OPS"};
  // get events from names:
  for(int i=0; i<NUM_PAPI_EVENTS; i++)
  {
    if(PAPI_event_name_to_code(papi_event_name[i],&papi_events[i]) != PAPI_OK)
    {
      if(hw_counters>i) hw_counters=i;
    }
  }
  long long papi_values[NUM_PAPI_EVENTS+4];
  if(hw_counters>NUM_PAPI_EVENTS) hw_counters=NUM_PAPI_EVENTS;
  long long papi_real_cyc_0 = PAPI_get_real_cyc();
  long long papi_real_usec_0 = PAPI_get_real_usec();
  long long papi_virt_cyc_0 = PAPI_get_virt_cyc();
  long long papi_virt_usec_0 = PAPI_get_virt_usec();
  PAPI_start_counters(papi_events,hw_counters);
#endif

// -----------------------------------------------------------------------------
//                                 MAIN SCF LOOP
// -----------------------------------------------------------------------------

  if(lsms.global.iprint >= 0) printf("Total number of iterations:%d\n",lsms.nscf);
  double timeScfLoop=MPI_Wtime();
  double timeCalcChemPot = 0.0;

  int iterationStart=0;
  int potentialWriteCounter=0;
  FILE *kFile=NULL;
  if(comm.rank==0)
  {
    iterationStart=readNextIterationNumber("k.out");
    kFile=fopen("k.out","a");
  }

  for(int i=0; i<lsms.nscf; i++)
  {
    if(lsms.global.iprint>=0)
      printf("Iteration %d:\n",i);

    energyContourIntegration(comm,lsms,local);
    double dTimeCCP = MPI_Wtime();
    // if(!lsms.global.checkIstop("buildKKRMatrix"))
    calculateChemPot(comm,lsms,local,eband);
    dTimeCCP=MPI_Wtime() - dTimeCCP;
    timeCalcChemPot += dTimeCCP;
    calculateEvec(lsms,local);
    calculateAllLocalChargeDensities(lsms,local);
    calculateChargesPotential(comm,lsms,local,crystal,0);
    calculateTotalEnergy(comm,lsms,local,crystal);

    mixing -> updateChargeDensity(lsms,local.atom);

    // If charge is mixed, recalculate the potential  (need a flag for this from input)
    calculateChargesPotential(comm,lsms,local,crystal,1);
    mixing -> updatePotential(lsms,local.atom);

    if(comm.rank==0)
    {
      printf("Band Energy = %lf Ry %10s", eband, "");
      printf("Fermi Energy = %lf Ry\n", lsms.chempot);
      printf("Total Energy = %lf Ry\n", lsms.totalEnergy);
    }

    if(kFile!=NULL)
    {
      Real rms=0.5*(local.qrms[0]+local.qrms[1]);
      fprintf(kFile,"%4d %20.12lf %12.6lf %12.6lf  %12.6lf\n",
              iterationStart+i,lsms.totalEnergy,lsms.chempot,local.atom[0].mtotws,rms);
      fflush(kFile);
    }

    // calculate core states for new potential if we are performing scf calculations
    calculateCoreStates(comm,lsms,local);

  // periodically write the new potential for scf calculations 
    potentialWriteCounter++;
    if(lsms.pot_out_type>=0 && potentialWriteCounter>=lsms.writeSteps)
    {
      std::cout<<"Writing new potentials.\n";
      writePotentials(comm,lsms,crystal,local);
      potentialWriteCounter=0;
    }
  }

  if(kFile!=NULL)
    fclose(kFile);

  timeScfLoop=MPI_Wtime()-timeScfLoop;

// -----------------------------------------------------------------------------

#ifdef USE_PAPI
  PAPI_stop_counters(papi_values,hw_counters);
  papi_values[hw_counters  ] = PAPI_get_real_cyc()-papi_real_cyc_0;
  papi_values[hw_counters+1] = PAPI_get_real_usec()-papi_real_usec_0;
  papi_values[hw_counters+2] = PAPI_get_virt_cyc()-papi_virt_cyc_0;
  papi_values[hw_counters+3] = PAPI_get_virt_usec()-papi_virt_usec_0;
  long long accumulated_counters[NUM_PAPI_EVENTS+4];
  MPI_Reduce(papi_values,accumulated_counters,hw_counters+4,
             MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
  if(comm.rank==0)
  {
    for(int i=0; i<hw_counters; i++)
    {
      std::cout<<"Accumulated: "<<(papi_event_name[i])<<" = "<<(accumulated_counters[i])<<"\n";
    }
    std::cout<<"PAPI accumulated real cycles : "<<(accumulated_counters[hw_counters])<<"\n";
    std::cout<<"PAPI accumulated user cycles : "<<(accumulated_counters[hw_counters+2])<<"\n";
    double gflops_papi = ((double)accumulated_counters[1])/
      (1000.0*(double)papi_values[hw_counters+1]);
    double gflops_hw_double = ((double)accumulated_counters[2])/
      (1000.0*(double)papi_values[hw_counters+1]);
    double gflops_hw_single = ((double)accumulated_counters[3])/
      (1000.0*(double)papi_values[hw_counters+1]);
    double gips = ((double)accumulated_counters[0])/(1000.0*(double)papi_values[hw_counters+1]);
    std::cout<<"PAPI_FP_OPS real GFLOP/s : "<<(gflops_papi)<<"\n";
    std::cout<<"PAPI hw double real GFLOP/s : "<<(gflops_hw_double)<<"\n";
    std::cout<<"PAPI hw single real GFLOP/s : "<<(gflops_hw_single)<<"\n";
    std::cout<<"PAPI real GINST/s : "<<(gips)<<"\n";
    std::cout<<"Time (s) : " << (double)papi_values[hw_counters+1] << "\n";
  }
#endif

  if(lsms.pot_out_type>=0)
  {
    std::cout<<"Writing new potentials.\n";
    writePotentials(comm,lsms,crystal,local);
  }

  if(comm.rank==0)
  {
    printf("Band Energy = %.15lf Ry\n",eband);
    printf("Fermi Energy = %.15lf Ry\n", lsms.chempot);
    printf("Total Energy = %.15lf Ry\n", lsms.totalEnergy);
    printf("timeScfLoop[rank==0] = %lf sec\n",timeScfLoop);
    printf(".../lsms.nscf = %lf sec\n",timeScfLoop/(double)lsms.nscf);
    printf("timeCalcChemPot[rank==0]/lsms.nscf = %lf sec\n",timeCalcChemPot/(double)lsms.nscf);
  }

  local.tmatStore.unpinMemory();
#ifdef BUILDKKRMATRIX_GPU
  for(int i=0; i<local.num_local; i++) freeDConst(deviceConstants[i]);
  freeDStore(deviceStorage);
#endif
  acceleratorFinalize();
  delete mixing;
#ifdef USE_GPTL
  GPTLpr(comm.rank);
#endif
  H5close();
  finalizeCommunication();
  lua_close(L);
  return 0;
}
