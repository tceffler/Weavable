#include "mc3.h"
#include "common.h"
#include "bigchunk.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __bgq__
#include <spi/include/kernel/memory.h>
#endif
#include <malloc.h>

#if defined(__bgq__) && !defined(NO_HPM)
extern "C" void HPM_Start(char *);
extern "C" void HPM_Stop(char *);
extern "C" void HPM_Print(int, int);
#endif

extern "C" void Timer_Print(void);
extern "C" void Timer_Reset(void);
extern "C" void Timer_Beg(const char *); 
extern "C" void Timer_End(const char *);

#ifdef HACC_CUDA
#include <cuda_runtime_api.h>
#endif

#include <sys/time.h>

void print_mem_stats()
{
#ifdef __bgq__
  uint64_t shared, persist,
    heapavail, stackavail,
    stack, heap, guard, mmap;

  Kernel_GetMemorySize(KERNEL_MEMSIZE_SHARED, &shared);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_PERSIST, &persist);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAPAVAIL, &heapavail);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_STACKAVAIL, &stackavail);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_STACK, &stack);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAP, &heap);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_GUARD, &guard);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_MMAP, &mmap);

  printf("Allocated heap: %.2f MB, avail. heap: %.2f MB\n",
    double(heap)/(1024*1024), double(heapavail)/(1024*1024));
  printf("Allocated stack: %.2f MB, avail. stack: %.2f MB\n",
    double(stack)/(1024*1024), double(stackavail)/(1024*1024));
  printf("Memory: shared: %.2f MB, persist: %.2f MB, guard: %.2f MB, mmap: %.2f MB\n",
    double(shared)/(1024*1024), double(persist)/(1024*1024),
    double(guard)/(1024*1024), double(mmap)/(1024*1024));
#else
  struct mallinfo mi = mallinfo();
  printf("Allocated heap: %.2f MB (large) + %.2f MB (small) used, %.2f MB unused\n",
    double(mi.arena + mi.hblkhd)/(1024*1024),
    double(mi.usmblks + mi.uordblks)/(1024*1024),
    double(mi.fsmblks + mi.fordblks)/(1024*1024));
#endif
}

//MAIN
int main(int argc, char* argv[])
{
  int step0 = 0;
  double avgtimes[10], mintimes[10], maxtimes[10];

  if(argc < 6) {
    fprintf(stderr,"USAGE: mc3 <indat> <inBase|tfName> <outBase> <INIT|RECORD|BLOCK|COSMO|RESTART> <ROUND_ROBIN|ALL_TO_ALL|ONE_TO_ONE|restart_step>\n");
    fprintf(stderr,"-a <aliveDumpName>   : alive particle dumps\n");
    fprintf(stderr,"-r <restartDumpName> : restart particle dumps\n");
    fprintf(stderr,"-f <refreshStepName> : steps for particle refresh\n");
    fprintf(stderr,"-o <analysisdat>     : config file for analysis\n");
    fprintf(stderr,"-s <staticDumpName>  : static time analysis dumps\n");
    fprintf(stderr,"-l <LCUpdateName>    : lightcone time updates\n");
    fprintf(stderr,"-h                   : halo outputs\n");
    fprintf(stderr,"-z                   : skewerz\n");
    fprintf(stderr,"-g                   : final grid output\n");
    fprintf(stderr,"-m                   : initialize MPI_Alltoall\n");
    fprintf(stderr,"-p <pkDumpName>      : P(k) dumps (+ initial, final, restarts)\n");
    fprintf(stderr,"-i <nInterp>         : interpolated f_sr with <nInterp> (max = %d) points\n",MAX_N_INTERP);
    fprintf(stderr,"-P                   : polynomial force law\n");
    fprintf(stderr,"-b                   : reserve cell blade\n");
    //fprintf(stderr,"-c                   : use chaining mesh\n");
    fprintf(stderr,"-e                   : building chaining mesh every subcycle\n");
    fprintf(stderr,"-1                   : skip stream\n");
    fprintf(stderr,"-2                   : skip long range kick\n");
    fprintf(stderr,"-3                   : skip short range kick\n");
    fprintf(stderr,"-M                   : don't drop memory\n");
    fprintf(stderr,"-B                   : don't use bigchunk\n");
    fprintf(stderr,"-F                   : use \"fast\" tree force eval\n");
    fprintf(stderr,"-R                   : use RCB monopole tree\n");
    fprintf(stderr,"-S                   : use RCB quadrupole tree\n");
    fprintf(stderr,"-N <nLeaf>           : max RCB tree particles per leaf\n");
    fprintf(stderr,"-L <nLevAlloc>       : number of extra RCB tree levels to preallocate\n");
    fprintf(stderr,"-T <nTaskMin>        : min number of particles per task during build\n");
    fprintf(stderr,"-O                   : use RCO tree instead of RCB tree\n");
    fprintf(stderr,"-w                   : use white noise initializer\n");
    fprintf(stderr,"-I                   : use MPI IO for restart files and pseudo-outputs\n");
    fprintf(stderr,"-t <NXxNYxNZ>        : use 3D topology NX by NY by NZ\n");
    exit(-1);
  }

  //sort command line options
  MC3Options options(argc, argv);


  int argvi = optind;
  string indatName = argv[argvi++];
  string inBase = argv[argvi++];
  string outBase = argv[argvi++];
  string dataType = argv[argvi++];
  string distributeType = argv[argvi++];

  //starting from restart file
  int restartQ = 0;
  if( dataType == "RESTART") {
    restartQ = 1;
    step0 = atoi(distributeType.c_str());
  }

  //need to convert Mpc to Mpc/h for true cosmo format
  int cosmoFormatQ=0;
  if( dataType == "COSMO") {
    cosmoFormatQ=1;
    dataType = "RECORD";
  }

  //INITIALIZE MPI
  //MPI_Init(&argc, &argv);
  int provided;
  MPI_Init_thread( &argc, &argv, MPI_THREAD_FUNNELED, &provided );

  MPI_Barrier(MPI_COMM_WORLD); //wait for all threads to arrive here before starting CUDA.
#ifdef HACC_CUDA
  cudaFree(0);  //hacky way to initialize the context now
#else
  #pragma acc init device_type(nvidia)
#endif
  //MPI_Finalize();
  //return 0;
  if ( provided < MPI_THREAD_FUNNELED ) MPI_Abort( MPI_COMM_WORLD, 1 );

  // SimpleTimings::TimerRef t_total = SimpleTimings::getTimer("total");
  // SimpleTimings::TimerRef t_init = SimpleTimings::getTimer("init");
  // SimpleTimings::TimerRef t_stepr = SimpleTimings::getTimer("stepr");
  SimpleTimings::TimerRef t_step = SimpleTimings::getTimer("step");
  // SimpleTimings::TimerRef t_xtra = SimpleTimings::getTimer("xtra");
  // SimpleTimings::TimerRef t_sort = SimpleTimings::getTimer("sort");
  //// SimpleTimings::TimerRef t_map1 = SimpleTimings::getTimer("map1");
  // SimpleTimings::TimerRef t_sub = SimpleTimings::getTimer("sub");
  // SimpleTimings::TimerRef t_cm = SimpleTimings::getTimer("cm");
  // SimpleTimings::TimerRef t_map1 = SimpleTimings::getTimer("map1");
  // SimpleTimings::TimerRef t_map2 = SimpleTimings::getTimer("map2");

  //INITIALIZE TOPOLOGY
  int tmpDims[DIMENSION];
  if(options.topologyQ()) {
    char *tmpDimsStr = (char *)options.topologyString().c_str();
    char *tmpDimsTok;
    tmpDimsTok = strtok(tmpDimsStr,"x");
    tmpDims[0] = atoi(tmpDimsTok);
    tmpDimsTok = strtok(NULL,"x");
    tmpDims[1] = atoi(tmpDimsTok);
    tmpDimsTok = strtok(NULL,"x");
    tmpDims[2] = atoi(tmpDimsTok);
    int nnodes;
    MPI_Comm_size(MPI_COMM_WORLD, &nnodes);
    MY_Dims_init_3D(nnodes, DIMENSION, tmpDims);
  }
  Partition::initialize();
  int numranks = Partition::getNumProc();
  int rank = Partition::getMyProc();

#ifdef HACC_CUDA
  int numdevices;
  cudaGetDeviceCount(&numdevices);
  int mydevice = rank % numdevices;
  cudaSetDevice(mydevice);
#endif

#if FFTW3_THREADS
  if (!fftw_init_threads()) {
    fprintf(stderr, "fftw_init_threads failed!\n");
    exit(1);
  }
#endif

#ifdef _OPENMP
  int omt = omp_get_max_threads();
#if FFTW3_THREADS
  fftw_plan_with_nthreads(omt);
#endif
  if (rank == 0) printf("%s: %d\n", "Threads per process", omt);
#endif

  if (rank == 0) print_mem_stats();

  //READ INDAT FILE
  Basedata indat( indatName.c_str() );

  mallopt(M_MMAP_THRESHOLD, sysconf(_SC_PAGESIZE));
  mallopt(M_TRIM_THRESHOLD, 0); 
  mallopt(M_TOP_PAD, 0);

  //INITIALIZE GEOMETRY
  Domain::initialize(indat);


  //
  MC3Extras *extras = new MC3Extras(options, indat);


  //start some timers
  // SimpleTimings::startTimer(t_total);
  // SimpleTimings::startTimer(t_init);


  //(OPTIONALLY) INITIALIZE MPI_ALLTOALL
  if(options.initialAlltoallQ()) {
    if(rank==0) {
      printf("\nStarting MPI_Alltoall initialization\n");
      fflush(stdout);
    }
    initial_all_to_all(options.initialAlltoallQ());
    ////MPI_Barrier(MPI_COMM_WORLD);
  }


  //INSTANTIATE PARTICLES
  Particles particles(indat, options);


  /*
  if(options.interpQ()) {
    particles.forceInterp(options.nInterp());
  }
  */
   
 
  if(!restartQ) {
    //START FROM THE BEGINNING
    loadParticles(indat, particles, 
		  inBase, dataType, distributeType, 
		  cosmoFormatQ, options);
  } else {
    //RESTARTING
    if(options.mpiio()) {
      particles.readRestart( create_outName( inBase + "." + MPI_RESTART_SUFFIX, step0).c_str() );
    } else {
      particles.readRestart( create_outName( create_outName( inBase + "." + RESTART_SUFFIX, step0), rank).c_str() );
    }
    step0++;
  }
  ////MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) print_mem_stats();

  //MOVE PARTICLES TO CELL BEFORE ALLOCATING FFT MEMORY
  particles.shoveParticles();
  ////MPI_Barrier(MPI_COMM_WORLD);


  //LOCAL GRID
  GRID_T *rho_arr = particles.field();
  GRID_T *grad_phi_arr = rho_arr;
  GRID_T *phi_arr = rho_arr;


  //LOCAL COPIES OF GEOMETRIC INFO
  int ngla[DIMENSION], nglt[DIMENSION];
  Domain::ng_local_alive(ngla);
  Domain::ng_local_total(nglt);
  int Ngla = Domain::Ng_local_alive();
  int Nglt = Domain::Ng_local_total();
  int ngo = Domain::ng_overload();
  int ng = Domain::ng();


  //INITIALIZE GRID EXCHANGE
  GridExchange gexchange(nglt, ngo, ngo+1);


  //ALLOC POISSON SOLVER AND BUFFERS
  SolverQuiet *solver = new SolverQuiet(MPI_COMM_WORLD, ng);
  COMPLEX_T *fft_rho_arr, *fft_grad_phi_arr, *fft_phi_arr;
  poisson_alloc(&fft_rho_arr, &fft_grad_phi_arr, &fft_phi_arr);
  //MPI_Barrier(MPI_COMM_WORLD);


  //TIMESTEPPER VARIABLES
  TimeStepper ts(indat.alpha(), indat.ain(), indat.afin(),
		 indat.nsteps(), indat.omegatot() );
  int64_t Np_local_alive, Np_global_alive;
  double rho_local_alive, rho_global_alive;


  //MPI_Barrier(MPI_COMM_WORLD);


  //P(k) INITIAL
  if(!restartQ) {
    if(rank==0) {
      printf("P(k) initial\n");
      fflush(stdout);
    }
    //MPI_Barrier(MPI_COMM_WORLD);

    map2_poisson_forward(particles, solver, rho_arr, fft_rho_arr);
    //MPI_Barrier(MPI_COMM_WORLD);

    writePk(solver, outBase + "." + PK_SUFFIX + ".ini");
    //MPI_Barrier(MPI_COMM_WORLD);

    if(rank==0) printf("\n");
    //MPI_Barrier(MPI_COMM_WORLD);
  }


  //DONE WITH INITIALIZATION
  // SimpleTimings::stopTimerStats(t_init);
  if(rank==0) printf("\n");


  vector<int> Nplav;
  Nplav.reserve(ts.nsteps()+1);
  Nplav.push_back(particles.Np_local_alive());


  //TIMESTEPPER
  // SimpleTimings::startTimer(t_stepr);

  //if restart, get timestepper variables up to speed
  for(int step = 0; step < step0; step++) {
    ts.advanceFullStep();
    extras->setStep(step, ts.aa());
  }
  MPI_Barrier(MPI_COMM_WORLD); //barrier to help with accurate timing

  int doHalfKick = 1;

  MPI_Pcontrol(1); // start MPI profiling
//MPI_Pcontrol(101); // start MPI tracing   

  Timer_Reset();
  Timer_Beg("step");

  //actual timestepping
  for(int step = step0; step < ts.nsteps(); step++) {

#if defined(__bgq__) && !defined(NO_HPM)
    HPM_Start("step");
#endif

    SimpleTimings::startTimer(t_step);

    if(rank==0) {
      print_mem_stats();
      printf("STEP %d, pp = %f, a = %f, z = %f\n",
          step, ts.pp(), ts.aa(), ts.zz());
      fflush(stdout);
    }
    //MPI_Barrier(MPI_COMM_WORLD);


    //(SOMETIMES) HALF KICK
    if(doHalfKick) {
      //POISSON FORWARD
      map2_poisson_forward(particles, solver, rho_arr, fft_rho_arr);
      //MPI_Barrier(MPI_COMM_WORLD);

      //POISSON BACKWARD GRADIENT
      if(!options.skipKickLRQ())
        map2_poisson_backward_gradient(particles, solver,
            grad_phi_arr, fft_grad_phi_arr,
            gexchange, ts, 0.5);
      //MPI_Barrier(MPI_COMM_WORLD);
    }


    //UPDATE TIME
    ts.advanceHalfStep();
    //MPI_Barrier(MPI_COMM_WORLD);


    /*
    //STREAM
    // SimpleTimings::startTimer(t_map1);
    particles.map1(ts.pp(), ts.tau(), ts.adot());
    // SimpleTimings::stopTimerStats(t_map1);
    //MPI_Barrier(MPI_COMM_WORLD);
    */

    //DROP FFT MEMORY
    size_t bcm1 = 0;
    if (!options.dontDropMemory()) {
      delete solver;
      poisson_free(&fft_rho_arr, &fft_grad_phi_arr, &fft_phi_arr);
      gexchange.dropBuffers();
      //MPI_Barrier(MPI_COMM_WORLD);

      bcm1 = bigchunk_get_total();
      bigchunk_reset();
    }

    //SUBCYCLE
    // SimpleTimings::startTimer(t_sub);
    if(options.cmQ())
      particles.subCycleCM(&ts);
    else
      particles.subCycle(&ts);
    // SimpleTimings::stopTimer(t_sub);
    Timer_Beg("barrier");
    MPI_Barrier(MPI_COMM_WORLD);
    Timer_End("barrier");

    Timer_Beg("postcycle");
    if (!options.dontDropMemory()) {
      size_t bcm2 = bigchunk_get_total();
      bigchunk_reset();
      if (bigchunk_get_size() == 0 && !options.dontUseBigchunk()) {
        bigchunk_init(std::max(bcm1, bcm2));
      }

      //RE-ALLOC FFT MEMORY
      solver = new SolverQuiet(MPI_COMM_WORLD, ng);
      poisson_alloc(&fft_rho_arr, &fft_grad_phi_arr, &fft_phi_arr);
      gexchange.resurrectBuffers();
      //MPI_Barrier(MPI_COMM_WORLD);
    }
    Timer_End("postcycle");

    //CHECKSUM PARTICLES
    Timer_Beg("particle_sum");
    Np_local_alive = particles.Np_local_alive();
    Nplav.push_back(Np_local_alive);
    MPI_Allreduce(&Np_local_alive, &Np_global_alive, 1, 
        MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    Timer_End("particle_sum");

    //UPDATE TIME
    ts.advanceHalfStep();
    //MPI_Barrier(MPI_COMM_WORLD);

    extras->setStep(step, ts.aa());

    //FORWARD FFT SOLVE
    map2_poisson_forward(particles, solver, rho_arr, fft_rho_arr);
    //MPI_Barrier(MPI_COMM_WORLD);

    //CHECKSUM DENSITY GRID
    Timer_Beg("density_sum");
    rho_local_alive = sum_rho_alive(rho_arr);
    MPI_Allreduce(&rho_local_alive, &rho_global_alive, 1, 
        MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    Timer_End("density_sum");
    //MPI_Barrier(MPI_COMM_WORLD);


    //P(k) DUMP
    if(extras->pkStep()) {
      writePk(solver, create_outName(outBase + "." + PK_SUFFIX, step) );
      //MPI_Barrier(MPI_COMM_WORLD);
    }


    //POISSON BACKWARD
    //FIGURE OUT FULL OR HALF KICK
    TS_FLOAT stepFraction=1.0;
    if( extras->extrasStep() || step == ts.nsteps()-1 ) {
      doHalfKick = 1;
      stepFraction = 0.5;
    } else {
      doHalfKick = 0;
      stepFraction = 1.0;
    }
    if(!options.skipKickLRQ())
      map2_poisson_backward_gradient(particles, solver,
          grad_phi_arr, fft_grad_phi_arr,
          gexchange, ts, stepFraction);
    //MPI_Barrier(MPI_COMM_WORLD);

    if(rank==0) {
      printf( "total alive density   = %f\n",rho_global_alive);
      cout << "total alive particles = " << Np_global_alive << endl;
      cout.flush();
    }
    //MPI_Barrier(MPI_COMM_WORLD);


    //OPTIONAL STUFF
    // SimpleTimings::startTimer(t_xtra);
    if(extras->extrasStep()) {
      if(rank==0) {
        printf("EXTRAS: ");
        if(extras->staticStep())printf("(static output) ");
        if(extras->lcStep())printf("(light cone update) ");
        if(extras->aliveStep())printf("(alive particle output) ");
        if(extras->restartStep())printf("(restart dump) ");
        if(extras->pkStep())printf("(power spectrum) ");
        if(extras->refreshStep())printf("(overload particle refresh)");
        printf("\n");
        fflush(stdout);
      }
    }
    //MPI_Barrier(MPI_COMM_WORLD);


    //BACKWARD POTENTIAL CALCULATION
    if(extras->fftbpotStep())
      map2_poisson_backward_potential(particles, solver, 
          phi_arr, fft_phi_arr,
          gexchange);
    //MPI_Barrier(MPI_COMM_WORLD);


    //DROP FFT MEMORY
    int fftMemDropped = 0;
    if(extras->particleStep() && !options.dontDropMemory()) {
      fftMemDropped = 1;
      delete solver;
      poisson_free(&fft_rho_arr, &fft_grad_phi_arr, &fft_phi_arr);
      gexchange.dropBuffers();
      bigchunk_reset();
      //MPI_Barrier(MPI_COMM_WORLD);
    }


    if(extras->particleStep())
      extras->particleExtras(particles, indat, outBase, &Nplav);


    //RE-ALLOC FFT MEMORY
    if(fftMemDropped && !options.dontDropMemory()) {
      particles.shoveParticles();
      solver = new SolverQuiet(MPI_COMM_WORLD, ng);
      poisson_alloc(&fft_rho_arr, &fft_grad_phi_arr, &fft_phi_arr);
      gexchange.resurrectBuffers();
      //MPI_Barrier(MPI_COMM_WORLD);
    }

    // SimpleTimings::stopTimerStats(t_xtra);
    SimpleTimings::stopTimerStats(t_step);
    if(rank==0) {
      printf("\n");
      fflush(stdout);
    }

#if defined(__bgq__) && !defined(NO_HPM)
    HPM_Stop("step");
    HPM_Stop("mpiAll");
    HPM_Print(step, 0);
    HPM_Start("mpiAll");
#endif
  } // end timestepper 

  MPI_Pcontrol(0); // stop MPI profiling
//MPI_Pcontrol(100); // stop MPI tracing 
  Timer_End("step");
  Timer_Print();

  // SimpleTimings::stopTimerStats(t_stepr);


  //OUTPUT REST OF LIGHT CONE SKEWERS ACCUMULATED
  if(extras->lightconeQ() && extras->skewerQ()) {
    string outName = create_outName(create_outName(outBase+"."+LC_SKEWER_SUFFIX, ts.nsteps()), Partition::getMyProc());
    (extras->lcskewers())->WriteLocalSkewers(outName.c_str());
    (extras->lcskewers())->ClearSkewers();
  }


  //P(k) FINAL
  if(rank==0) {
    printf("P(k) final\n");
    fflush(stdout);
  }
  //MPI_Barrier(MPI_COMM_WORLD);
  map2_poisson_forward(particles, solver, rho_arr, fft_rho_arr);
  //MPI_Barrier(MPI_COMM_WORLD);
  writePk(solver, outBase + "." + PK_SUFFIX + ".fin");  
  //MPI_Barrier(MPI_COMM_WORLD);
  if(rank==0)
    printf("\n");
  //MPI_Barrier(MPI_COMM_WORLD);

  
  //GRID OUTPUT
  if(extras->gridQ()) {
    //CIC ALREADY DONE FOR P(k)
    output_array_alive(rho_arr,create_outName(create_outName(outBase+"."+GRID_SUFFIX,ts.nsteps()),rank).c_str());
  }
  //MPI_Barrier(MPI_COMM_WORLD);


  delete extras;
  delete solver;
  poisson_free(&fft_rho_arr, &fft_grad_phi_arr, &fft_phi_arr);
  //MPI_Barrier(MPI_COMM_WORLD);

  // SimpleTimings::stopTimer(t_total);
  SimpleTimings::accumStats();

  // Shut down MPI
  Partition::finalize();
  MPI_Finalize();

  return 0;
}
