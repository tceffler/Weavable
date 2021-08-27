#!/bin/bash
echo "Setting up modules environment for CORAL-FTQ (fwq threaded)"
module load xl
module list

bldir=$PWD
mkdir bin

echo "Building CORAL-FTQ (fwq threaded)"
cd ftqV110/ftq
make t_fwq && mv t_fwq $bldir/bin/
