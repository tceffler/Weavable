#!/bin/bash -l
#
# Script to build the minisweep code on different systems.
# 
# Author: Veronica G. Vergara L. 
#

function usage {
  echo "Usage: ./build_umt.sh <type of build>"
  echo ""
  echo "  Default: compiles the code for current host"
  echo ""
  echo "  Possible build types values:" 
  echo "  - base"
  echo "  - opt"
  exit
}

if [[ $1 == "-h" || $1 == "--help" ]] 
then
  usage
  exit
fi

if [ $# -ne 1 ]
then
  echo "Type of build value is required."
  usage
  exit
fi

if [[ $1 == "base" || $1 == "opt" ]]
then
  BUILDTYPE=$1
else
  echo "Unsupported build type provided."
  usage
  exit
fi

if [ -d bin ] ; then
    echo "bin dir found, not rebuilding"
    exit 0
fi

DATE=`date +"%Y%m%d_%H%M"`
HOST=`hostname | cut -d. -f1 | cut -d- -f1`
BUILDDIR=`pwd`

if [[ $HOST =~ batch* || $HOST =~ login* || $HOST =~ build* ]]
then
  if [[ $BUILDTYPE == "base" ]]
  then
    module load pgi
    module load cuda
  elif [[ $BUILDTYPE == "opt" ]]
  then
    module load pgi/18.7
    module load cuda/9.1.85
    BUILDTLD=umt2016-managed-pgi18.7
  fi
fi

echo "Compiling UMT version $BUILDTYPE"
module list

EXECUTABLE=SuOlsonTest

cd $BUILDTLD

echo "Cleaning directory `pwd`..."
make clean

echo "Building ibmtimers..."
cd ibmtimers
make
cd ..

echo "Starting UMT make..."
#make VERBOSE=1 2>&1 | tee out_make.sh
make 2>&1 | tee out_make.sh

cd Teton
echo "Cleaning directory `pwd`..."
make clean
echo "Starting UMT Teton make..."
make $EXECUTABLE

if [ -x $EXECUTABLE ]
then
    echo "UMT $BUILDTYPE built successfully!"
    mkdir -p $BUILDDIR/bin
    cp $EXECUTABLE $BUILDDIR/bin/
else
    echo "UMT build failed!"
    exit 1
fi

cd $BUILDDIR

exit 0
