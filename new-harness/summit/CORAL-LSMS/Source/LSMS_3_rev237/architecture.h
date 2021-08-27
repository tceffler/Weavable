export USE_OPENMP=1
export USE_ACCELERATOR  = -DACCELERATOR_CUDA_C -DBUILDKKRMATRIX_GPU
export FUSE_ACCELERATOR = -DACCELERATOR_CUDA_C -DBUILDKKRMATRIX_GPU

export HDFLIBS = ${OLCF_HDF5_ROOT}/lib/libhdf5.a -ldl -lz
export LAPACK = $(TOP_DIR)/liblapack_.a

export BLAS = -L${OLCF_ESSL_ROOT}/lib64 -lessl -Wl,-rpath=${OLCF_ESSL_ROOT}/lib64  \
-L/sw/summit/xl/16.1.1-beta3/xlf/16.1.1/lib -lxlf90_r -lxlfmath -lxl -lxlopt -Wl,-rpath=/sw/summit/xl/16.1.1-beta3/lib
export FLIBS = -L/sw/summit/gcc/6.4.0/lib64/ -lgfortran \
               -Wl,-rpath=/sw/summit/gcc/6.4.0/lib64/

export LIBS +=
export ADD_LIBS += -L$(TOP_DIR)/CBLAS/lib -lcblas_LINUX -L${OLCF_CUDA_ROOT}/lib64 -lcublas -lcudart \
                  -Wl,-rpath=${OLCF_CUDA_ROOT}/lib64 ${BIND} ${HDFLIBS} ${LAPACK} ${BLAS} ${FLIBS} -lstdc++

export INC_PATH += -I$(TOP_DIR)/CBLAS/include -I$(CUDA_DIR)/include

export ADDITIONAL_TARGETS = CBLAS_target

export BOOST_ROOT=$(TOP_DIR)


ifdef USE_OPENMP
export CXX=mpic++ -g -std=c++11 -I$(BOOST_ROOT) $(USE_ACCELERATOR) -I${OLCF_CUDA_ROOT}/include -fopenmp -Ofast
export CC=mpicc -g $(USE_ACCELERATOR) -fopenmp -Ofast
export F77=mpifort -g $(FUSE_ACCELERATOR) -fopenmp -Ofast
export F90=mpifort -g $(FUSE_ACCELERATOR) -fopenmp -Ofast
export CUDA_CXX=nvcc -arch=sm_70 -O3  -I${OLCF_CUDA_ROOT}/include $(USE_ACCELERATOR) -I${OLCF_SPECTRUM_MPI_ROOT}/include -ccbin mpic++ -Xcompiler -fopenmp 
else
export CXX=mpic++ -g -std=c++11 -I$(BOOST_ROOT) $(USE_ACCELERATOR) -I${OLCF_CUDA_ROOT}/include 
export CC=mpicc -g $(USE_ACCELERATOR)
export F77=mpifort -g $(USE_ACCELERATOR)
export CUDA_CXX=nvcc -arch=sm_70 -I${OLCF_CUDA_ROOT}/include -I${OLCF_SPECTRUM_MPI_ROOT}/include $(USE_ACCELERATOR) -ccbin mpic++
endif

