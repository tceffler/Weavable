Making utilities ...
make[1]: Entering directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-AMG/Source/hypre-cusparse.V6/utilities'
mpicc  -o amg_linklist.o -c -I.. -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE -DHYPRE_TIMING amg_linklist.c
mpicc  -o binsearch.o -c -I.. -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE -DHYPRE_TIMING binsearch.c
mpicc  -o exchange_data.o -c -I.. -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE -DHYPRE_TIMING exchange_data.c
mpicc  -o hypre_memory.o -c -I.. -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE -DHYPRE_TIMING hypre_memory.c
mpicc  -o hypre_qsort.o -c -I.. -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE -DHYPRE_TIMING hypre_qsort.c
mpicc  -o memory_dmalloc.o -c -I.. -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE -DHYPRE_TIMING memory_dmalloc.c
mpicc  -o mpistubs.o -c -I.. -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE -DHYPRE_TIMING mpistubs.c
mpicc  -o qsplit.o -c -I.. -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE -DHYPRE_TIMING qsplit.c
mpicc  -o random.o -c -I.. -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE -DHYPRE_TIMING random.c
mpicc  -o threading.o -c -I.. -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE -DHYPRE_TIMING threading.c
mpicc  -o thread_mpistubs.o -c -I.. -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE -DHYPRE_TIMING thread_mpistubs.c
mpicc  -D_POSIX_SOURCE -o timer.o -c -I.. -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE -DHYPRE_TIMING timer.c
mpicc  -o timing.o -c -I.. -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE -DHYPRE_TIMING timing.c
mpicc  -o umalloc_local.o -c -I.. -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE -DHYPRE_TIMING umalloc_local.c
mpicc  -o hypre_error.o -c -I.. -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE -DHYPRE_TIMING hypre_error.c
Building libHYPRE_utilities.a ... 
ar -rcu libHYPRE_utilities.a amg_linklist.o binsearch.o exchange_data.o hypre_memory.o hypre_qsort.o memory_dmalloc.o mpistubs.o qsplit.o random.o threading.o thread_mpistubs.o timer.o timing.o umalloc_local.o hypre_error.o
ranlib libHYPRE_utilities.a
make[1]: Leaving directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-AMG/Source/hypre-cusparse.V6/utilities'

Making krylov ...
make[1]: Entering directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-AMG/Source/hypre-cusparse.V6/krylov'
mpicc  -o gmres.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE gmres.c
mpicc  -o HYPRE_gmres.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE HYPRE_gmres.c
mpicc  -o HYPRE_pcg.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE HYPRE_pcg.c
mpicc  -o pcg.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE pcg.c
pcg.c:275:4: warning: implicit declaration of function 'nvtxRangePushA' is invalid in C99 [-Wimplicit-function-declaration]
   nvtxRangePushA("PGCSolve_pre");
   ^
pcg.c:464:4: warning: implicit declaration of function 'nvtxRangePop' is invalid in C99 [-Wimplicit-function-declaration]
   nvtxRangePop();
   ^
2 warnings generated.
Building libkrylov.a ... 
ar -rcu libkrylov.a gmres.o HYPRE_gmres.o HYPRE_pcg.o pcg.o
ranlib libkrylov.a
make[1]: Leaving directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-AMG/Source/hypre-cusparse.V6/krylov'

Making IJ_mv ...
make[1]: Entering directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-AMG/Source/hypre-cusparse.V6/IJ_mv'
mpicc  -o aux_parcsr_matrix.o -c -I.. -I../utilities -I../struct_mv -I../seq_mv -I../parcsr_mv -I../parcsr_ls -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE aux_parcsr_matrix.c
mpicc  -o aux_par_vector.o -c -I.. -I../utilities -I../struct_mv -I../seq_mv -I../parcsr_mv -I../parcsr_ls -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE aux_par_vector.c
mpicc  -o HYPRE_IJMatrix.o -c -I.. -I../utilities -I../struct_mv -I../seq_mv -I../parcsr_mv -I../parcsr_ls -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE HYPRE_IJMatrix.c
mpicc  -o HYPRE_IJVector.o -c -I.. -I../utilities -I../struct_mv -I../seq_mv -I../parcsr_mv -I../parcsr_ls -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE HYPRE_IJVector.c
mpicc  -o IJMatrix.o -c -I.. -I../utilities -I../struct_mv -I../seq_mv -I../parcsr_mv -I../parcsr_ls -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE IJMatrix.c
mpicc  -o IJMatrix_parcsr.o -c -I.. -I../utilities -I../struct_mv -I../seq_mv -I../parcsr_mv -I../parcsr_ls -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE IJMatrix_parcsr.c
mpicc  -o IJVector_parcsr.o -c -I.. -I../utilities -I../struct_mv -I../seq_mv -I../parcsr_mv -I../parcsr_ls -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE IJVector_parcsr.c
     716  1500-010: (W) WARNING in hypre_IJVectorGetValuesPar: Infinite loop.  Program may not stop.
Building libIJ_mv.a ... 
ar -rcu libIJ_mv.a aux_parcsr_matrix.o aux_par_vector.o HYPRE_IJMatrix.o HYPRE_IJVector.o IJMatrix.o IJMatrix_parcsr.o IJVector_parcsr.o
ranlib libIJ_mv.a
make[1]: Leaving directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-AMG/Source/hypre-cusparse.V6/IJ_mv'

Making parcsr_ls ...
make[1]: Entering directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-AMG/Source/hypre-cusparse.V6/parcsr_ls'
mpicc  -o aux_interp.o -c -I.. -I../utilities -I../krylov -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE aux_interp.c
mpicc  -o gen_redcs_mat.o -c -I.. -I../utilities -I../krylov -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE gen_redcs_mat.c
mpicc  -o HYPRE_parcsr_amg.o -c -I.. -I../utilities -I../krylov -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE HYPRE_parcsr_amg.c
mpicc  -o HYPRE_parcsr_gmres.o -c -I.. -I../utilities -I../krylov -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE HYPRE_parcsr_gmres.c
mpicc  -o HYPRE_parcsr_pcg.o -c -I.. -I../utilities -I../krylov -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE HYPRE_parcsr_pcg.c
mpicc  -o par_amg.o -c -I.. -I../utilities -I../krylov -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE par_amg.c
mpicc  -o par_amg_setup.o -c -I.. -I../utilities -I../krylov -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE par_amg_setup.c
    1500-030: (I) INFORMATION: hypre_BoomerAMGSetup: Additional optimization may be attained by recompiling and specifying MAXMEM option with a value greater than 8192.
mpicc  -o par_amg_solve.o -c -I.. -I../utilities -I../krylov -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE par_amg_solve.c
mpicc  -o par_cg_relax_wt.o -c -I.. -I../utilities -I../krylov -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE par_cg_relax_wt.c
mpicc  -o par_coarsen.o -c -I.. -I../utilities -I../krylov -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE par_coarsen.c
    1500-030: (I) INFORMATION: hypre_BoomerAMGCoarsenRuge: Additional optimization may be attained by recompiling and specifying MAXMEM option with a value greater than 8192.
mpicc  -o par_coarse_parms.o -c -I.. -I../utilities -I../krylov -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE par_coarse_parms.c
mpicc  -o par_cycle.o -c -I.. -I../utilities -I../krylov -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE par_cycle.c
par_cycle.c:515:13: warning: implicit declaration of function 'hypre_VectorSyncHost' is invalid in C99 [-Wimplicit-function-declaration]
            hypre_VectorSyncHost(U_array[coarse_grid]->local_vector);
            ^
par_cycle.c:544:13: warning: implicit declaration of function 'hypre_VectorSyncDevice' is invalid in C99 [-Wimplicit-function-declaration]
            hypre_VectorSyncDevice(U_array[coarse_grid]->local_vector);
            ^
2 warnings generated.
mpicc  -o par_indepset.o -c -I.. -I../utilities -I../krylov -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE par_indepset.c
mpicc  -o par_interp.o -c -I.. -I../utilities -I../krylov -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE par_interp.c
mpicc  -o par_jacobi_interp.o -c -I.. -I../utilities -I../krylov -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE par_jacobi_interp.c
mpicc  -o par_multi_interp.o -c -I.. -I../utilities -I../krylov -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE par_multi_interp.c
    1500-030: (I) INFORMATION: hypre_BoomerAMGBuildMultipass: Additional optimization may be attained by recompiling and specifying MAXMEM option with a value greater than 8192.
mpicc  -o par_laplace_27pt.o -c -I.. -I../utilities -I../krylov -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE par_laplace_27pt.c
mpicc  -o par_laplace.o -c -I.. -I../utilities -I../krylov -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE par_laplace.c
mpicc  -o par_lr_interp.o -c -I.. -I../utilities -I../krylov -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE par_lr_interp.c
    1500-030: (I) INFORMATION: hypre_BoomerAMGBuildStdInterp: Additional optimization may be attained by recompiling and specifying MAXMEM option with a value greater than 8192.
mpicc  -o par_nodal_systems.o -c -I.. -I../utilities -I../krylov -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE par_nodal_systems.c
par_nodal_systems.c:134:11: warning: using floating point absolute value function 'fabs' when argument is of integer type [-Wabsolute-value]
   mode = fabs(option);
          ^
par_nodal_systems.c:134:11: note: use function 'abs' instead
   mode = fabs(option);
          ^~~~
          abs
1 warning generated.
mpicc  -o par_rap.o -c -I.. -I../utilities -I../krylov -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE par_rap.c
mpicc  -o par_rap_communication.o -c -I.. -I../utilities -I../krylov -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE par_rap_communication.c
mpicc  -o par_vardifconv.o -c -I.. -I../utilities -I../krylov -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE par_vardifconv.c
mpicc  -o par_relax.o -c -I.. -I../utilities -I../krylov -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE par_relax.c
    1500-030: (I) INFORMATION: hypre_BoomerAMGRelax: Additional optimization may be attained by recompiling and specifying MAXMEM option with a value greater than 8192.
mpicc  -o par_relax_interface.o -c -I.. -I../utilities -I../krylov -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE par_relax_interface.c
mpicc  -o par_scaled_matnorm.o -c -I.. -I../utilities -I../krylov -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE par_scaled_matnorm.c
mpicc  -o par_stats.o -c -I.. -I../utilities -I../krylov -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE par_stats.c
mpicc  -o par_strength.o -c -I.. -I../utilities -I../krylov -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE par_strength.c
    1500-030: (I) INFORMATION: hypre_BoomerAMGCreate2ndS: Additional optimization may be attained by recompiling and specifying MAXMEM option with a value greater than 8192.
mpicc  -o partial.o -c -I.. -I../utilities -I../krylov -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE partial.c
    1500-030: (I) INFORMATION: hypre_BoomerAMGBuildPartialStdInterp: Additional optimization may be attained by recompiling and specifying MAXMEM option with a value greater than 8192.
mpicc  -o pcg_par.o -c -I.. -I../utilities -I../krylov -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE pcg_par.c
mpicc  -o par_relax_more.o -c -I.. -I../utilities -I../krylov -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE par_relax_more.c
par_relax_more.c:488:23: warning: passing 'const char [24]' to parameter of type 'char *' discards qualifiers [-Wincompatible-pointer-types-discards-qualifiers]
   nvtxRangePushColor(__FUNCTION__, 0xFF00FFFF);
                      ^~~~~~~~~~~~
./../gpu/gpu_utilities.h:58:31: note: passing argument to parameter 'message' here
void nvtxRangePushColor(char *message, uint32_t color);
                              ^
par_relax_more.c:529:33: warning: implicit declaration of function 'hypre_VectorSyncDevice' is invalid in C99 [-Wimplicit-function-declaration]
     if (!u_local->initialized) hypre_VectorSyncDevice(u_local);
                                ^
par_relax_more.c:2475:24: warning: passing 'const char [28]' to parameter of type 'char *' discards qualifiers [-Wincompatible-pointer-types-discards-qualifiers]
    nvtxRangePushColor(__FUNCTION__, 0xFF00FFFF);
                       ^~~~~~~~~~~~
./../gpu/gpu_utilities.h:58:31: note: passing argument to parameter 'message' here
void nvtxRangePushColor(char *message, uint32_t color);
                              ^
3 warnings generated.
Building libparcsr_ls.a ... 
ar -rcu libparcsr_ls.a aux_interp.o gen_redcs_mat.o HYPRE_parcsr_amg.o HYPRE_parcsr_gmres.o HYPRE_parcsr_pcg.o par_amg.o par_amg_setup.o par_amg_solve.o par_cg_relax_wt.o par_coarsen.o par_coarse_parms.o par_cycle.o par_indepset.o par_interp.o par_jacobi_interp.o par_multi_interp.o par_laplace_27pt.o par_laplace.o par_lr_interp.o par_nodal_systems.o par_rap.o par_rap_communication.o par_vardifconv.o par_relax.o par_relax_interface.o par_scaled_matnorm.o par_stats.o par_strength.o partial.o pcg_par.o par_relax_more.o
ranlib libparcsr_ls.a
make[1]: Leaving directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-AMG/Source/hypre-cusparse.V6/parcsr_ls'

Making parcsr_mv ...
make[1]: Entering directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-AMG/Source/hypre-cusparse.V6/parcsr_mv'
mpicc  -o HYPRE_parcsr_matrix.o -c -I.. -I../utilities -I../seq_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE HYPRE_parcsr_matrix.c
mpicc  -o HYPRE_parcsr_vector.o -c -I.. -I../utilities -I../seq_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE HYPRE_parcsr_vector.c
mpicc  -o new_commpkg.o -c -I.. -I../utilities -I../seq_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE new_commpkg.c
mpicc  -o par_csr_assumed_part.o -c -I.. -I../utilities -I../seq_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE par_csr_assumed_part.c
mpicc  -o par_csr_communication.o -c -I.. -I../utilities -I../seq_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE par_csr_communication.c
mpicc  -o par_csr_matop.o -c -I.. -I../utilities -I../seq_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE par_csr_matop.c
mpicc  -o par_csr_matrix.o -c -I.. -I../utilities -I../seq_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE par_csr_matrix.c
mpicc  -o par_csr_matop_marked.o -c -I.. -I../utilities -I../seq_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE par_csr_matop_marked.c
mpicc  -o par_csr_matvec.o -c -I.. -I../utilities -I../seq_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE par_csr_matvec.c
par_csr_matvec.c:40:23: warning: passing 'const char [25]' to parameter of type 'char *' discards qualifiers [-Wincompatible-pointer-types-discards-qualifiers]
   nvtxRangePushColor(__FUNCTION__, 0xFFFFFF00);
                      ^~~~~~~~~~~~
./../gpu/gpu_utilities.h:58:31: note: passing argument to parameter 'message' here
void nvtxRangePushColor(char *message, uint32_t color);
                              ^
par_csr_matvec.c:139:33: warning: implicit declaration of function 'hypre_VectorSyncDevice' is invalid in C99 [-Wimplicit-function-declaration]
     if (!x_local->initialized) hypre_VectorSyncDevice(x_local);
                                ^
par_csr_matvec.c:148:11: warning: implicit declaration of function 'device_has_send_maps' is invalid in C99 [-Wimplicit-function-declaration]
     if (!device_has_send_maps(diag, 0)) {
          ^
par_csr_matvec.c:166:6: warning: implicit declaration of function 'device_create_comm_buffer' is invalid in C99 [-Wimplicit-function-declaration]
     device_create_comm_buffer(diag, send_size, send_maps, x_buf_data[0], x_local->d_data);
     ^
par_csr_matvec.c:285:23: warning: passing 'const char [26]' to parameter of type 'char *' discards qualifiers [-Wincompatible-pointer-types-discards-qualifiers]
   nvtxRangePushColor(__FUNCTION__, 0xFFFFA500);
                      ^~~~~~~~~~~~
./../gpu/gpu_utilities.h:58:31: note: passing argument to parameter 'message' here
void nvtxRangePushColor(char *message, uint32_t color);
                              ^
par_csr_matvec.c:464:8: warning: implicit declaration of function 'device_set_comm_map' is invalid in C99 [-Wimplicit-function-declaration]
       device_set_comm_map(diag,recv_size,recv_maps, 1);
       ^
par_csr_matvec.c:470:8: warning: implicit declaration of function 'device_assemble_transpose_result' is invalid in C99 [-Wimplicit-function-declaration]
       device_assemble_transpose_result(diag, hypre_CSRMatrixNumRows(diag), hypre_CSRMatrixNumCols(diag), recv_size, y_local->d_data, y_buf_data[0]);
       ^
7 warnings generated.
mpicc  -o par_vector.o -c -I.. -I../utilities -I../seq_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE par_vector.c
par_vector.c:454:4: warning: implicit declaration of function 'nvtxRangePushColor' is invalid in C99 [-Wimplicit-function-declaration]
   nvtxRangePushColor("InnerProd_MPI_Allreduce", 0xFFFF0000); 
   ^
par_vector.c:458:4: warning: implicit declaration of function 'nvtxRangePop' is invalid in C99 [-Wimplicit-function-declaration]
   nvtxRangePop();
   ^
2 warnings generated.
Building libparcsr_mv.a ... 
ar -rcu libparcsr_mv.a HYPRE_parcsr_matrix.o HYPRE_parcsr_vector.o new_commpkg.o par_csr_assumed_part.o par_csr_communication.o par_csr_matop.o par_csr_matrix.o par_csr_matop_marked.o par_csr_matvec.o par_vector.o
ranlib libparcsr_mv.a
make[1]: Leaving directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-AMG/Source/hypre-cusparse.V6/parcsr_mv'

Making seq_mv ...
make[1]: Entering directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-AMG/Source/hypre-cusparse.V6/seq_mv'
mpicc  -o big_csr_matrix.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE big_csr_matrix.c
mpicc  -o csr_matop.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE csr_matop.c
mpicc  -o csr_matrix.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE csr_matrix.c
mpicc  -o csr_matvec.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE csr_matvec.c
csr_matvec.c:61:26: warning: implicit declaration of function 'hypre_VectorSyncDevice' is invalid in C99 [-Wimplicit-function-declaration]
    if (!x->initialized) hypre_VectorSyncDevice(x);
                         ^
1 warning generated.
mpicc  -o genpart.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE genpart.c
mpicc  -o HYPRE_csr_matrix.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE HYPRE_csr_matrix.c
mpicc  -o HYPRE_vector.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE HYPRE_vector.c
mpicc  -o vector.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE vector.c
vector.c:393:8: warning: implicit declaration of function 'device_SeqVectorSetConstantValues' is invalid in C99 [-Wimplicit-function-declaration]
       device_SeqVectorSetConstantValues(value, v->d_data, size);
       ^
vector.c:475:8: warning: implicit declaration of function 'device_SeqVectorCopy' is invalid in C99 [-Wimplicit-function-declaration]
       device_SeqVectorCopy(x->d_data, y->d_data, size);
       ^
2 warnings generated.
Building libseq_mv.a ... 
ar -rcu libseq_mv.a big_csr_matrix.o csr_matop.o csr_matrix.o csr_matvec.o genpart.o HYPRE_csr_matrix.o HYPRE_vector.o vector.o
ranlib libseq_mv.a
make[1]: Leaving directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-AMG/Source/hypre-cusparse.V6/seq_mv'

Making struct_mv ...
make[1]: Entering directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-AMG/Source/hypre-cusparse.V6/struct_mv'
mpicc  -o assumed_part.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE assumed_part.c
    1500-030: (I) INFORMATION: hypre_StructAssumedPartitionCreate: Additional optimization may be attained by recompiling and specifying MAXMEM option with a value greater than 8192.
mpicc  -o box_algebra.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE box_algebra.c
mpicc  -o box_alloc.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE box_alloc.c
mpicc  -o box_boundary.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE box_boundary.c
mpicc  -o box.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE box.c
mpicc  -o box_manager.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE box_manager.c
    1500-030: (I) INFORMATION: hypre_BoxManAssemble: Additional optimization may be attained by recompiling and specifying MAXMEM option with a value greater than 8192.
mpicc  -o box_neighbors.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE box_neighbors.c
mpicc  -o communication_info.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE communication_info.c
mpicc  -o computation.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE computation.c
mpicc  -o grow.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE grow.c
mpicc  -o HYPRE_struct_grid.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE HYPRE_struct_grid.c
mpicc  -o HYPRE_struct_matrix.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE HYPRE_struct_matrix.c
mpicc  -o HYPRE_struct_stencil.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE HYPRE_struct_stencil.c
mpicc  -o HYPRE_struct_vector.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE HYPRE_struct_vector.c
mpicc  -o new_assemble.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE new_assemble.c
    1500-030: (I) INFORMATION: hypre_StructGridAssembleWithAP: Additional optimization may be attained by recompiling and specifying MAXMEM option with a value greater than 8192.
mpicc  -o new_box_neighbors.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE new_box_neighbors.c
mpicc  -o project.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE project.c
mpicc  -o struct_axpy.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE struct_axpy.c
mpicc  -o struct_communication.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE struct_communication.c
mpicc  -o struct_copy.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE struct_copy.c
mpicc  -o struct_grid.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE struct_grid.c
mpicc  -o struct_innerprod.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE struct_innerprod.c
mpicc  -o struct_io.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE struct_io.c
mpicc  -o struct_matrix.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE struct_matrix.c
mpicc  -o struct_matrix_mask.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE struct_matrix_mask.c
mpicc  -o struct_matvec.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE struct_matvec.c
mpicc  -o struct_overlap_innerprod.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE struct_overlap_innerprod.c
mpicc  -o struct_scale.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE struct_scale.c
mpicc  -o struct_stencil.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE struct_stencil.c
mpicc  -o struct_vector.o -c -I.. -I../utilities -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE struct_vector.c
Building libHYPRE_struct_mv.a ... 
ar -rcu libHYPRE_struct_mv.a assumed_part.o box_algebra.o box_alloc.o box_boundary.o box.o box_manager.o box_neighbors.o communication_info.o computation.o grow.o HYPRE_struct_grid.o HYPRE_struct_matrix.o HYPRE_struct_stencil.o HYPRE_struct_vector.o new_assemble.o new_box_neighbors.o project.o struct_axpy.o struct_communication.o struct_copy.o struct_grid.o struct_innerprod.o struct_io.o struct_matrix.o struct_matrix_mask.o struct_matvec.o struct_overlap_innerprod.o struct_scale.o struct_stencil.o struct_vector.o
ranlib libHYPRE_struct_mv.a
make[1]: Leaving directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-AMG/Source/hypre-cusparse.V6/struct_mv'

Making sstruct_mv ...
make[1]: Entering directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-AMG/Source/hypre-cusparse.V6/sstruct_mv'
mpicc  -o box_map.o -c -I.. -I../utilities -I../struct_mv -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE box_map.c
mpicc  -o HYPRE_sstruct_graph.o -c -I.. -I../utilities -I../struct_mv -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE HYPRE_sstruct_graph.c
mpicc  -o HYPRE_sstruct_grid.o -c -I.. -I../utilities -I../struct_mv -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE HYPRE_sstruct_grid.c
mpicc  -o HYPRE_sstruct_matrix.o -c -I.. -I../utilities -I../struct_mv -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE HYPRE_sstruct_matrix.c
mpicc  -o HYPRE_sstruct_stencil.o -c -I.. -I../utilities -I../struct_mv -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE HYPRE_sstruct_stencil.c
mpicc  -o HYPRE_sstruct_vector.o -c -I.. -I../utilities -I../struct_mv -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE HYPRE_sstruct_vector.c
mpicc  -o sstruct_axpy.o -c -I.. -I../utilities -I../struct_mv -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE sstruct_axpy.c
mpicc  -o sstruct_copy.o -c -I.. -I../utilities -I../struct_mv -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE sstruct_copy.c
mpicc  -o sstruct_graph.o -c -I.. -I../utilities -I../struct_mv -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE sstruct_graph.c
mpicc  -o sstruct_grid.o -c -I.. -I../utilities -I../struct_mv -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE sstruct_grid.c
mpicc  -o sstruct_innerprod.o -c -I.. -I../utilities -I../struct_mv -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE sstruct_innerprod.c
mpicc  -o sstruct_matrix.o -c -I.. -I../utilities -I../struct_mv -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE sstruct_matrix.c
mpicc  -o sstruct_matvec.o -c -I.. -I../utilities -I../struct_mv -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE sstruct_matvec.c
mpicc  -o sstruct_overlap_innerprod.o -c -I.. -I../utilities -I../struct_mv -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE sstruct_overlap_innerprod.c
mpicc  -o sstruct_scale.o -c -I.. -I../utilities -I../struct_mv -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE sstruct_scale.c
mpicc  -o sstruct_stencil.o -c -I.. -I../utilities -I../struct_mv -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE sstruct_stencil.c
mpicc  -o sstruct_vector.o -c -I.. -I../utilities -I../struct_mv -I../seq_mv -I../parcsr_mv -I../IJ_mv -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE sstruct_vector.c
Building libsstruct_mv.a ... 
ar -rcu libsstruct_mv.a box_map.o HYPRE_sstruct_graph.o HYPRE_sstruct_grid.o HYPRE_sstruct_matrix.o HYPRE_sstruct_stencil.o HYPRE_sstruct_vector.o sstruct_axpy.o sstruct_copy.o sstruct_graph.o sstruct_grid.o sstruct_innerprod.o sstruct_matrix.o sstruct_matvec.o sstruct_overlap_innerprod.o sstruct_scale.o sstruct_stencil.o sstruct_vector.o
ranlib libsstruct_mv.a
make[1]: Leaving directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-AMG/Source/hypre-cusparse.V6/sstruct_mv'

Making gpu ...
make[1]: Entering directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-AMG/Source/hypre-cusparse.V6/gpu'
nvcc  -Xptxas -dlcm=cg -m64 -use_fast_math -O3 -gencode=arch=compute_70,code=sm_70 -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE -DNDEBUG -o gpu_matrix.o -c gpu_matrix.cu
gpu_matrix.cu(497): warning: variable "status" was set but never used

gpu_matrix.cu: In function 'void device_spmv(void*, int, int, int, double, int*, int*, double*, double*, int, double, double*, int, int)':
gpu_matrix.cu:571:31: warning: 'cudaPointerAttributes::memoryType' is deprecated (declared at /sw/summit/cuda/10.1.243/bin/../targets/ppc64le-linux/include/driver_types.h:1302) [-Wdeprecated-declarations]
   if (x_size == 0 || attrib.memoryType == cudaMemoryTypeDevice) {
                               ^
gpu_matrix.cu:571:31: warning: 'cudaPointerAttributes::memoryType' is deprecated (declared at /sw/summit/cuda/10.1.243/bin/../targets/ppc64le-linux/include/driver_types.h:1302) [-Wdeprecated-declarations]
gpu_matrix.cu:600:31: warning: 'cudaPointerAttributes::memoryType' is deprecated (declared at /sw/summit/cuda/10.1.243/bin/../targets/ppc64le-linux/include/driver_types.h:1302) [-Wdeprecated-declarations]
   if (y_size == 0 || attrib.memoryType == cudaMemoryTypeDevice) {
                               ^
gpu_matrix.cu:600:31: warning: 'cudaPointerAttributes::memoryType' is deprecated (declared at /sw/summit/cuda/10.1.243/bin/../targets/ppc64le-linux/include/driver_types.h:1302) [-Wdeprecated-declarations]
nvcc  -Xptxas -dlcm=cg -m64 -use_fast_math -O3 -gencode=arch=compute_70,code=sm_70 -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE -DNDEBUG -o gpu_matvec.o -c gpu_matvec.cu
gpu_matvec.cu(71): error: identifier "__shfl_down" is undefined

gpu_matvec.cu(72): error: identifier "__shfl_down" is undefined

gpu_matvec.cu(73): error: identifier "__shfl_down" is undefined

gpu_matvec.cu(74): error: identifier "__shfl_down" is undefined

gpu_matvec.cu(75): error: identifier "__shfl_down" is undefined

gpu_matvec.cu(139): warning: variable "tid" was declared but never referenced
          detected during instantiation of "void launch_matvec_cuda<vec_size>(matrix, double *, double *, double, double) [with vec_size=1]" 
(296): here

gpu_matvec.cu(81): error: identifier "__shfl_down" is undefined
          detected during:
            instantiation of "double warpReduceSum2<vec_size>(double) [with vec_size=1]" 
(161): here
            instantiation of "void matvec_csr_vector_kernel<non_zero_beta,vec_size>(matrix, double *, double *, double, double) [with non_zero_beta=0, vec_size=1]" 
(282): here
            instantiation of "void launch_matvec_cuda<vec_size>(matrix, double *, double *, double, double) [with vec_size=1]" 
(296): here

gpu_matvec.cu(82): error: identifier "__shfl_down" is undefined
          detected during:
            instantiation of "double warpReduceSum2<vec_size>(double) [with vec_size=1]" 
(161): here
            instantiation of "void matvec_csr_vector_kernel<non_zero_beta,vec_size>(matrix, double *, double *, double, double) [with non_zero_beta=0, vec_size=1]" 
(282): here
            instantiation of "void launch_matvec_cuda<vec_size>(matrix, double *, double *, double, double) [with vec_size=1]" 
(296): here

gpu_matvec.cu(83): error: identifier "__shfl_down" is undefined
          detected during:
            instantiation of "double warpReduceSum2<vec_size>(double) [with vec_size=1]" 
(161): here
            instantiation of "void matvec_csr_vector_kernel<non_zero_beta,vec_size>(matrix, double *, double *, double, double) [with non_zero_beta=0, vec_size=1]" 
(282): here
            instantiation of "void launch_matvec_cuda<vec_size>(matrix, double *, double *, double, double) [with vec_size=1]" 
(296): here

gpu_matvec.cu(84): error: identifier "__shfl_down" is undefined
          detected during:
            instantiation of "double warpReduceSum2<vec_size>(double) [with vec_size=1]" 
(161): here
            instantiation of "void matvec_csr_vector_kernel<non_zero_beta,vec_size>(matrix, double *, double *, double, double) [with non_zero_beta=0, vec_size=1]" 
(282): here
            instantiation of "void launch_matvec_cuda<vec_size>(matrix, double *, double *, double, double) [with vec_size=1]" 
(296): here

gpu_matvec.cu(85): error: identifier "__shfl_down" is undefined
          detected during:
            instantiation of "double warpReduceSum2<vec_size>(double) [with vec_size=1]" 
(161): here
            instantiation of "void matvec_csr_vector_kernel<non_zero_beta,vec_size>(matrix, double *, double *, double, double) [with non_zero_beta=0, vec_size=1]" 
(282): here
            instantiation of "void launch_matvec_cuda<vec_size>(matrix, double *, double *, double, double) [with vec_size=1]" 
(296): here

gpu_matvec.cu(81): error: identifier "__shfl_down" is undefined
          detected during:
            instantiation of "double warpReduceSum2<vec_size>(double) [with vec_size=4]" 
(161): here
            instantiation of "void matvec_csr_vector_kernel<non_zero_beta,vec_size>(matrix, double *, double *, double, double) [with non_zero_beta=0, vec_size=4]" 
(282): here
            instantiation of "void launch_matvec_cuda<vec_size>(matrix, double *, double *, double, double) [with vec_size=4]" 
(298): here

gpu_matvec.cu(82): error: identifier "__shfl_down" is undefined
          detected during:
            instantiation of "double warpReduceSum2<vec_size>(double) [with vec_size=4]" 
(161): here
            instantiation of "void matvec_csr_vector_kernel<non_zero_beta,vec_size>(matrix, double *, double *, double, double) [with non_zero_beta=0, vec_size=4]" 
(282): here
            instantiation of "void launch_matvec_cuda<vec_size>(matrix, double *, double *, double, double) [with vec_size=4]" 
(298): here

gpu_matvec.cu(83): error: identifier "__shfl_down" is undefined
          detected during:
            instantiation of "double warpReduceSum2<vec_size>(double) [with vec_size=4]" 
(161): here
            instantiation of "void matvec_csr_vector_kernel<non_zero_beta,vec_size>(matrix, double *, double *, double, double) [with non_zero_beta=0, vec_size=4]" 
(282): here
            instantiation of "void launch_matvec_cuda<vec_size>(matrix, double *, double *, double, double) [with vec_size=4]" 
(298): here

gpu_matvec.cu(84): error: identifier "__shfl_down" is undefined
          detected during:
            instantiation of "double warpReduceSum2<vec_size>(double) [with vec_size=4]" 
(161): here
            instantiation of "void matvec_csr_vector_kernel<non_zero_beta,vec_size>(matrix, double *, double *, double, double) [with non_zero_beta=0, vec_size=4]" 
(282): here
            instantiation of "void launch_matvec_cuda<vec_size>(matrix, double *, double *, double, double) [with vec_size=4]" 
(298): here

gpu_matvec.cu(85): error: identifier "__shfl_down" is undefined
          detected during:
            instantiation of "double warpReduceSum2<vec_size>(double) [with vec_size=4]" 
(161): here
            instantiation of "void matvec_csr_vector_kernel<non_zero_beta,vec_size>(matrix, double *, double *, double, double) [with non_zero_beta=0, vec_size=4]" 
(282): here
            instantiation of "void launch_matvec_cuda<vec_size>(matrix, double *, double *, double, double) [with vec_size=4]" 
(298): here

gpu_matvec.cu(81): error: identifier "__shfl_down" is undefined
          detected during:
            instantiation of "double warpReduceSum2<vec_size>(double) [with vec_size=8]" 
(161): here
            instantiation of "void matvec_csr_vector_kernel<non_zero_beta,vec_size>(matrix, double *, double *, double, double) [with non_zero_beta=0, vec_size=8]" 
(282): here
            instantiation of "void launch_matvec_cuda<vec_size>(matrix, double *, double *, double, double) [with vec_size=8]" 
(300): here

gpu_matvec.cu(82): error: identifier "__shfl_down" is undefined
          detected during:
            instantiation of "double warpReduceSum2<vec_size>(double) [with vec_size=8]" 
(161): here
            instantiation of "void matvec_csr_vector_kernel<non_zero_beta,vec_size>(matrix, double *, double *, double, double) [with non_zero_beta=0, vec_size=8]" 
(282): here
            instantiation of "void launch_matvec_cuda<vec_size>(matrix, double *, double *, double, double) [with vec_size=8]" 
(300): here

gpu_matvec.cu(83): error: identifier "__shfl_down" is undefined
          detected during:
            instantiation of "double warpReduceSum2<vec_size>(double) [with vec_size=8]" 
(161): here
            instantiation of "void matvec_csr_vector_kernel<non_zero_beta,vec_size>(matrix, double *, double *, double, double) [with non_zero_beta=0, vec_size=8]" 
(282): here
            instantiation of "void launch_matvec_cuda<vec_size>(matrix, double *, double *, double, double) [with vec_size=8]" 
(300): here

gpu_matvec.cu(84): error: identifier "__shfl_down" is undefined
          detected during:
            instantiation of "double warpReduceSum2<vec_size>(double) [with vec_size=8]" 
(161): here
            instantiation of "void matvec_csr_vector_kernel<non_zero_beta,vec_size>(matrix, double *, double *, double, double) [with non_zero_beta=0, vec_size=8]" 
(282): here
            instantiation of "void launch_matvec_cuda<vec_size>(matrix, double *, double *, double, double) [with vec_size=8]" 
(300): here

gpu_matvec.cu(85): error: identifier "__shfl_down" is undefined
          detected during:
            instantiation of "double warpReduceSum2<vec_size>(double) [with vec_size=8]" 
(161): here
            instantiation of "void matvec_csr_vector_kernel<non_zero_beta,vec_size>(matrix, double *, double *, double, double) [with non_zero_beta=0, vec_size=8]" 
(282): here
            instantiation of "void launch_matvec_cuda<vec_size>(matrix, double *, double *, double, double) [with vec_size=8]" 
(300): here

gpu_matvec.cu(81): error: identifier "__shfl_down" is undefined
          detected during:
            instantiation of "double warpReduceSum2<vec_size>(double) [with vec_size=16]" 
(161): here
            instantiation of "void matvec_csr_vector_kernel<non_zero_beta,vec_size>(matrix, double *, double *, double, double) [with non_zero_beta=0, vec_size=16]" 
(282): here
            instantiation of "void launch_matvec_cuda<vec_size>(matrix, double *, double *, double, double) [with vec_size=16]" 
(302): here

gpu_matvec.cu(82): error: identifier "__shfl_down" is undefined
          detected during:
            instantiation of "double warpReduceSum2<vec_size>(double) [with vec_size=16]" 
(161): here
            instantiation of "void matvec_csr_vector_kernel<non_zero_beta,vec_size>(matrix, double *, double *, double, double) [with non_zero_beta=0, vec_size=16]" 
(282): here
            instantiation of "void launch_matvec_cuda<vec_size>(matrix, double *, double *, double, double) [with vec_size=16]" 
(302): here

gpu_matvec.cu(83): error: identifier "__shfl_down" is undefined
          detected during:
            instantiation of "double warpReduceSum2<vec_size>(double) [with vec_size=16]" 
(161): here
            instantiation of "void matvec_csr_vector_kernel<non_zero_beta,vec_size>(matrix, double *, double *, double, double) [with non_zero_beta=0, vec_size=16]" 
(282): here
            instantiation of "void launch_matvec_cuda<vec_size>(matrix, double *, double *, double, double) [with vec_size=16]" 
(302): here

gpu_matvec.cu(84): error: identifier "__shfl_down" is undefined
          detected during:
            instantiation of "double warpReduceSum2<vec_size>(double) [with vec_size=16]" 
(161): here
            instantiation of "void matvec_csr_vector_kernel<non_zero_beta,vec_size>(matrix, double *, double *, double, double) [with non_zero_beta=0, vec_size=16]" 
(282): here
            instantiation of "void launch_matvec_cuda<vec_size>(matrix, double *, double *, double, double) [with vec_size=16]" 
(302): here

gpu_matvec.cu(85): error: identifier "__shfl_down" is undefined
          detected during:
            instantiation of "double warpReduceSum2<vec_size>(double) [with vec_size=16]" 
(161): here
            instantiation of "void matvec_csr_vector_kernel<non_zero_beta,vec_size>(matrix, double *, double *, double, double) [with non_zero_beta=0, vec_size=16]" 
(282): here
            instantiation of "void launch_matvec_cuda<vec_size>(matrix, double *, double *, double, double) [with vec_size=16]" 
(302): here

gpu_matvec.cu(81): error: identifier "__shfl_down" is undefined
          detected during:
            instantiation of "double warpReduceSum2<vec_size>(double) [with vec_size=32]" 
(161): here
            instantiation of "void matvec_csr_vector_kernel<non_zero_beta,vec_size>(matrix, double *, double *, double, double) [with non_zero_beta=0, vec_size=32]" 
(282): here
            instantiation of "void launch_matvec_cuda<vec_size>(matrix, double *, double *, double, double) [with vec_size=32]" 
(304): here

gpu_matvec.cu(82): error: identifier "__shfl_down" is undefined
          detected during:
            instantiation of "double warpReduceSum2<vec_size>(double) [with vec_size=32]" 
(161): here
            instantiation of "void matvec_csr_vector_kernel<non_zero_beta,vec_size>(matrix, double *, double *, double, double) [with non_zero_beta=0, vec_size=32]" 
(282): here
            instantiation of "void launch_matvec_cuda<vec_size>(matrix, double *, double *, double, double) [with vec_size=32]" 
(304): here

gpu_matvec.cu(83): error: identifier "__shfl_down" is undefined
          detected during:
            instantiation of "double warpReduceSum2<vec_size>(double) [with vec_size=32]" 
(161): here
            instantiation of "void matvec_csr_vector_kernel<non_zero_beta,vec_size>(matrix, double *, double *, double, double) [with non_zero_beta=0, vec_size=32]" 
(282): here
            instantiation of "void launch_matvec_cuda<vec_size>(matrix, double *, double *, double, double) [with vec_size=32]" 
(304): here

gpu_matvec.cu(84): error: identifier "__shfl_down" is undefined
          detected during:
            instantiation of "double warpReduceSum2<vec_size>(double) [with vec_size=32]" 
(161): here
            instantiation of "void matvec_csr_vector_kernel<non_zero_beta,vec_size>(matrix, double *, double *, double, double) [with non_zero_beta=0, vec_size=32]" 
(282): here
            instantiation of "void launch_matvec_cuda<vec_size>(matrix, double *, double *, double, double) [with vec_size=32]" 
(304): here

gpu_matvec.cu(85): error: identifier "__shfl_down" is undefined
          detected during:
            instantiation of "double warpReduceSum2<vec_size>(double) [with vec_size=32]" 
(161): here
            instantiation of "void matvec_csr_vector_kernel<non_zero_beta,vec_size>(matrix, double *, double *, double, double) [with non_zero_beta=0, vec_size=32]" 
(282): here
            instantiation of "void launch_matvec_cuda<vec_size>(matrix, double *, double *, double, double) [with vec_size=32]" 
(304): here

gpu_matvec.cu(126): warning: function "fetch_double" was declared but never referenced

30 errors detected in the compilation of "/tmp/tmpxft_0001162d_00000000-6_gpu_matvec.cpp1.ii".
make[1]: *** [gpu_matvec.o] Error 1
make[1]: Leaving directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-AMG/Source/hypre-cusparse.V6/gpu'

Making test ...
make[1]: Entering directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-AMG/Source/hypre-cusparse.V6/test'
mpicc  -o amg2013.o -c -I.. -I../utilities -I../struct_mv -I../sstruct_mv -I../IJ_mv -I../seq_mv -I../parcsr_mv -I../parcsr_ls -I../krylov -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DMPIPCONTROL -I/sw/summit/cuda/10.1.243/bin/..//include -DHYPRE_USING_GPU -DGPU_STORE_EXPLICIT_TRANSPOSE -DGPU_USE_CUSPARSE_MATVEC  -DSWITCH_HOST_DEVICE -DHYPRE_TIMING amg2013.c
amg2013.c:105:1: warning: control reaches end of non-void function [-Wreturn-type]
}
^
amg2013.c:118:3: warning: implicit declaration of function 'hypre_VectorSyncDevice' is invalid in C99 [-Wimplicit-function-declaration]
  hypre_VectorSyncDevice(r->local_vector);
  ^
amg2013.c:205:7: warning: implicit declaration of function 'device_create_matrix' is invalid in C99 [-Wimplicit-function-declaration]
      device_create_matrix(diag, rows, cols, nnz, I, J, data, cols, rows, 0); 
      ^
amg2013.c:206:7: warning: implicit declaration of function 'device_create_diagonal' is invalid in C99 [-Wimplicit-function-declaration]
      device_create_diagonal(diag);
      ^
amg2013.c:207:7: warning: implicit declaration of function 'device_get_vector' is invalid in C99 [-Wimplicit-function-declaration]
      device_get_vector(rows, 0);
      ^
amg2013.c:211:9: warning: implicit declaration of function 'device_set_l1_norms' is invalid in C99 [-Wimplicit-function-declaration]
        device_set_l1_norms(diag, l1_norms[i]);
        ^
amg2013.c:216:9: warning: implicit declaration of function 'device_set_hyper_sparse' is invalid in C99 [-Wimplicit-function-declaration]
        device_set_hyper_sparse(diag, hypre_CSRMatrixNumRownnz(diag), hypre_CSRMatrixRownnz(diag));
        ^
amg2013.c:312:7: warning: implicit declaration of function 'device_report_memory' is invalid in C99 [-Wimplicit-function-declaration]
      device_report_memory();
      ^
amg2013.c:315:5: warning: implicit declaration of function 'device_createCublas' is invalid in C99 [-Wimplicit-function-declaration]
    device_createCublas();
    ^
9 warnings generated.
    1500-030: (I) INFORMATION: main: Additional optimization may be attained by recompiling and specifying MAXMEM option with a value greater than 8192.
Linking amg2013 ... 
mpicc  -o amg2013 amg2013.o -L. -L../parcsr_ls -L../parcsr_mv -L../IJ_mv -L../seq_mv -L../sstruct_mv -L../struct_mv -L../krylov -L../utilities -L../gpu -lamg_gpu -lparcsr_ls -lparcsr_mv -lseq_mv -lsstruct_mv -lIJ_mv -lHYPRE_struct_mv -lkrylov -lHYPRE_utilities -Xlinker -start-group -Xlinker -lamg_gpu -Xlinker -lseq_mv -Xlinker -end-group -lm -fopenmp -lcusparse -lcudart -lcublas -lnvToolsExt -L/sw/summit/cuda/10.1.243/bin/..//lib64 -lstdc++
make[1]: Leaving directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-AMG/Source/hypre-cusparse.V6/test'

