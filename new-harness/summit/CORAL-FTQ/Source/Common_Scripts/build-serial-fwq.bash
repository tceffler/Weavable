#!/bin/bash
echo "Setting up modules environment for CORAL-FTQ (fwq serial)"
module load xl
module list

bldir=$PWD
mkdir bin

echo "Building CORAL-FTQ (fwq serial)"
cd ftqV110/ftq
make fwq && mv fwq $bldir/bin/
