#!/bin/bash

if [ -d bin ] ; then
    echo "bin dir found, not rebuilding"
    exit 0
fi

#-- Needed modules
module unload xl
module load gcc

#-- Set variables for convenience
set -o verbose

SW_BLDDIR=$PWD

PACKAGE=XSBench
VERSION=v11
SRCDIR=${PACKAGE}

#-- clean up old build (if exists)
rm -rf $SRCDIR bin

tar xf ${PACKAGE}_${VERSION}.tar.gz
cd ${SRCDIR}/src

#-- Fix up compiler
sed -i \
"s/GCC = gcc/GCC = mpicc/g; s/MPI       = no/MPI       = yes/g" \
Makefile

#-- Build XSBENCH
make
if [ $? -ne 0 ] ; then
  echo "XSBENCH: make failed"
  exit 1
fi

#-- 'Install' binaries
mkdir $SW_BLDDIR/bin
cp XSBench $SW_BLDDIR/bin/

cd $SW_BLDDIR

exit 0
