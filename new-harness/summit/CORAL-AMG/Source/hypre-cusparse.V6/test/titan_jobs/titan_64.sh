#!/bin/bash
#PBS -A CSC122
#PBS -l nodes=64
#PBS -l walltime=00:20:00

# setup env
source /opt/modules/default/init/sh
cd /lustre/atlas/scratch/stencer/csc122/AMG/hypre-cusparse/test

# verbose mode
set -x verbose

# create results folder
DATE=`date +%Y-%m-%d-%T`
echo $DATE
results_folder=results_mpi_$DATE
mkdir -p $results_folder

# run tests
for size in {12,24,36,48,60}
do
  px=2
  py=2
  pz=2

  proc=$((8*px*py*pz))
  file_name="./$results_folder/output_mpi_r${size}_n${proc}"
  touch $file_name
  aprun -n $proc -N 1 ./amg2013 -pooldist 1 -r $size $size $size -P $px $py $pz > $file_name 

done




