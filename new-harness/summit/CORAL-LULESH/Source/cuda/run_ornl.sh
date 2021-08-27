#!/bin/bash 
nodes=$1
if [ "$nodes" -eq "188" ]
then
  nrs=1125
  #np=3375
elif [ "$nodes" -eq "96" ]
then
  nrs=576
  #np=1728
elif [ "$nodes" -eq "12" ]
then
  nrs=72
  #np=216
elif [ "$nodes" -eq "2" ]
then
  nrs=9
  #np=27
else
  echo "Invalid node count"
  exit
fi

#HELPER=/ccs/home/hfwen/bin/jsm_setup.sh
#--------------------------------------
#--------------------------------------
cat >batch.job <<EOF
#BSUB -P VEN201
#BSUB -o ornl_${nodes}_%J.out
#BSUB -e ornl_${nodes}_%J.err
#BSUB -nnodes ${nodes}
#BSUB -alloc_flags "gpumps"
##BSUB -q tested
##BSUB -U IBM_SSA
#BSUB -W 20
#---------------------------------
ulimit -s 10240

unset OMP_NUM_THREADS

export LD_PRELOAD=/ccs/home/walkup/mpitrace/spectrum_mpi/libmpitrace.so

jsrun --nrs ${nrs} --tasks_per_rs 3 --cpu_per_rs 6 --gpu_per_rs 1 -b packed:2 -d plane:3 ${HELPER} ./lulesh -s 128 -i 5000

#export CUDA_LAUNCH_BLOCKING=1
#export USE_MPS=yes
#export RANKS_PER_NODE=${ppn}
#jsrun --rs_per_host 1 --np ${np} --cpu_per_rs 42 --gpu_per_rs 6 -d plane:${ppn} -b proportional-packed:2 /ccs/home/hfwen/bin/h6.sh ./lulesh -s 128 -i 1000

EOF
#---------------------------------------
bsub <batch.job
