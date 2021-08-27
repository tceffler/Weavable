#!/bin/bash -l
#
# Script to build the HACC benchmark
# 
# Author: Mark Berrill
#

echo "Building GPU version of HACC benchmark"


# Load the modules and enviornment
module unload xl gcc essl cuda spectrum-mpi
module load gcc
module load essl
module load cuda
module load spectrum-mpi
export root=$PWD

if [ -x bin/hacc_tpm ]; then
echo "Skipping build"
exit 0
fi

# Check that the enviornment loaded correctly
module list
echo MPI_ROOT=$OLCF_SPECTRUM_MPI_ROOT
which mpicc
mpicc --show


# Configure and build fftw
tar -xvf fftw-3.3.8.tar.gz
cd fftw-3.3.8
./configure --prefix=$PWD CC=gcc F77=gfortran OMPI_CC=gcc MPICC=mpicc PTHREAD_CC=gcc --enable-mpi --enable-fma --enable-vsx --enable-openmp
make -j 24
make install
make distclean
./configure --prefix=$PWD CC=gcc F77=gfortran OMPI_CC=gcc MPICC=mpicc PTHREAD_CC=gcc --enable-mpi --enable-fma --enable-vsx --enable-openmp --enable-single
make -j 24
make install
export FFTW_ROOT=$PWD
cd $root


# Build
echo Building HACC
mpicc --show
source ./ibm.env
make clean
cd cpu
make -j 4
cd $root
mkdir -p bin
cp cpu/ibm/hacc_tpm bin/.

