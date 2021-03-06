#!/bin/bash -l
#
# Script to build lulesh
# 
# Author: Mark Berrill
#


# Clear existing modules
module unload clang gnu pgi xl xlf cuda kokkos cmake
module purge


# Load modules
module load gcc
module load cmake
module load cuda/9.2.148
module load spectrum-mpi
CC=mpicc
CXX=mpic++
USE_CUDA=1
USE_OPENMP=1
export CXX=mpic++


# Build
export ROOT=$PWD
cd $ROOT/cuda
make
cd $ROOT/base
make


# Create symlinks
cd $ROOT
mkdir bin
cd bin
ln -s ../cuda/lulesh lulesh.cuda
ln -s ../base/lulesh lulesh.base
cd $ROOT


