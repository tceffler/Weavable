#!/bin/bash -l
#
# Script to build the AMG CORAL benchmark on Summit.
# 
# Author: Veronica G. Vergara L. 
#

function usage {
  echo "Usage: ./build_amg.sh <type of build>"
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
  exit 1
fi

if [ $# -ne 1 ]
then
  echo "Type of build value is required."
  usage
  exit 1
fi

if [[ $1 == "base" || $1 == "opt" ]]
then
  BUILDTYPE=$1
else
  echo "Unsupported build type provided."
  usage
  exit 1
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
    module load xl
    module load cuda
    BUILDTLD=hypre-cusparse.V6
  fi
fi

echo "Compiling AMG version $BUILDTYPE"
module list

EXECUTABLE=amg2013

cd $BUILDTLD

echo "Cleaning directory `pwd`..."
#make clean

echo "Starting AMG make..."
#make VERBOSE=1 2>&1 | tee out_make.sh
make 2>&1 | tee out_make.sh

if [ -x test/$EXECUTABLE ]
then
    echo "AMG $BUILDTYPE built successfully!"
    mkdir -p $BUILDDIR/bin
    cp test/$EXECUTABLE $BUILDDIR/bin/
else
    echo "AMG build failed!"
    exit 1
fi

cd $BUILDDIR

exit 0
