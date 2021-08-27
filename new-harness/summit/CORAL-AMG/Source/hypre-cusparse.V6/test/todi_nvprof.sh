#!/bin/bash -l
#SBATCH --job-name="amg" 
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1 
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00 
#======START=============================== 
#echo "On which nodes it executes" 
#echo $SLURM_JOB_NODELIST 
#echo "Now run the MPI tasks..." 
aprun -B ./nvprof.sh ./amg2013 -pooldist 1 -r 48 48 48 -P 1 1 1
#======END=================================

