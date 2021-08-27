#!/bin/bash -l
#SBATCH --job-name="amg" 
#SBATCH --nodes=64
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1 
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00 

# cd into test dir
cd ..

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
  aprun -B ./amg2013 -pooldist 1 -r $size $size $size -P $px $py $pz > $file_name
done

