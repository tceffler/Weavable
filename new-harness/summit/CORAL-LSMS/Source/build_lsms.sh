#!/bin/bash -l

# Author: Veronica Vergara L.
# 
# This script builds the LSMS application using the source code optimized by
# IBM as part of the CORAL Benchmarks effort.
#

DATE=`date +"%Y%m%d_%H%M"`
HOST=`hostname | cut -d. -f1 | cut -d- -f1`

TOPLEVEL=`pwd`
LSMS3DIR=$TOPLEVEL/LSMS_3_rev237
BINDIR=$LSMS3DIR/bin
CBLASLIBDIR=$LSMS3DIR/CBLAS/lib
FINALBINDIR=$TOPLEVEL/bin

echo "TOPLEVEL=$TOPLEVEL"
echo "LSMS3DIR=$LSMS3DIR"
echo "BINDIR=$BINDIR"
echo "CBLASLIBDIR=$CBLASLIBDIR"
echo "FINALBINDIR=$FINALBINDIR"

mkdir -p $FINALBINDIR
mkdir -p $BINDIR
mkdir -p $CBLASLIBDIR

cd $LSMS3DIR

module list

if [[ $HOST =~ login* || $HOST =~ summitdev* || $HOST =~ tundra* || $HOST =~ batch* || $HOST =~ build* ]]
then
    echo "Building LSMS"
    module load gcc
    module load spectrum-mpi
    module load cuda
    module load essl
    module load hdf5
    module list
    
    #make clean
    make -j 1
fi

if [ -x $BINDIR/wl-lsms ]
then
  echo "wl-lsms built successfully!"
  cp $BINDIR/wl-lsms $FINALBINDIR/
  cp $BINDIR/lsms $FINALBINDIR/
  #cp $BINDIR/rewl-lsms $FINALBINDIR/
  exit 0
fi
