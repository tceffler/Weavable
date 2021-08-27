export USE_OPENMP=1
export USE_ACCELERATOR  = -DACCELERATOR_CUDA_C -DBUILDKKRMATRIX_GPU
export FUSE_ACCELERATOR = -DACCELERATOR_CUDA_C -DBUILDKKRMATRIX_GPU

export HDFLIBS = /usr/workspace/wsa/walkup/hdf5-1.8.21/lib/libhdf5.a  -ldl -lz
export LAPACK =  /g/g14/walkup/lib/liblapack_.a

export BLAS = -L${OLCF_ESSL_ROOT}/lib64 -lessl -Wl,-rpath=${OLCF_ESSL_ROOT}/lib64  \
-L${OLCF_XLF_ROOT}/lib -lxlf90_r -lxlfmath -lxl -lxlopt -Wl,-rpath=${OLCF_XL_ROOT}/lib
export FLIBS = -L${OLCF_GFORTRAN_ROOT}/lib64 -lgfortran -Wl,-rpath=${OLCF_GFORTRAN_ROOT}/lib64 \
               /g/g14/walkup/codes/bind/bindthreads_gomp.o

export LIBS +=
export ADD_LIBS += -L$(TOP_DIR)/CBLAS/lib -lcblas_LINUX -L${OLCF_CUDA_ROOT}/lib64 -lcublas -lcudart \
                  -Wl,-rpath=${OLCF_CUDA_ROOT}/lib64 ${BIND} ${HDFLIBS} ${LAPACK} ${BLAS} ${FLIBS} -lstdc++

export INC_PATH += -I $(TOP_DIR)/CBLAS/include -I/usr/workspace/wsa/walkup/hdf5-1.8.21/include

export ADDITIONAL_TARGETS = CBLAS_target

export BOOST_ROOT=$(TOP_DIR)


ifdef USE_OPENMP
export CXX=mpig++ -g -std=c++11 -I$(BOOST_ROOT) $(USE_ACCELERATOR) -I${OLCF_CUDA_ROOT}/include -fopenmp -Ofast
export CC=mpigcc -g $(USE_ACCELERATOR) -fopenmp -Ofast
export F77=mpigfortran -g $(FUSE_ACCELERATOR) -fopenmp -Ofast
export F90=mpif90 -g $(FUSE_ACCELERATOR) -fopenmp -Ofast
export CUDA_CXX=nvcc -arch=sm_70 -O3  -I${OLCF_CUDA_ROOT}/include $(USE_ACCELERATOR) -I${OLCF_SPECTRUM_MPI_ROOT}/include -Xcompiler -fopenmp 
else
export CXX=mpig++ -g -std=c++11 -I$(BOOST_ROOT) $(USE_ACCELERATOR) -I${OLCF_CUDA_ROOT}/include 
export CC=mpigcc -g $(USE_ACCELERATOR)
export F77=mpigfortran -g $(USE_ACCELERATOR)
export CUDA_CXX=nvcc -arch=sm_70 -I${OLCF_CUDA_ROOT}/include -I${OLCF_SPECTRUM_MPI_ROOT}/include $(USE_ACCELERATOR)
endif

