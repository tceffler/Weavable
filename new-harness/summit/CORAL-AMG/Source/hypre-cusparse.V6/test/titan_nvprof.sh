#!/bin/bash
#PBS -A CSC122
#PBS -l nodes=8
#PBS -l walltime=00:10:00
cd /lustre/atlas/scratch/stencer/csc122/AMG/hypre-cusparse/test
source /opt/modules/default/init/sh
module load cudatoolkit
aprun -n 8 -N 1 ./nvprof.sh ./amg2013 -pooldist 1 -r 48 48 48 -P 1 1 1
