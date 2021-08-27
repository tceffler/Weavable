#!/bin/bash 
nodes=1
ppn=4
let nmpi=$nodes*$ppn
tstamp=`date +%m_%d_%H_%M_%S`
#--------------------------------------
cat >batch.job <<EOF
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -nnodes ${nodes}
##BSUB -x
#BSUB -q excl_6gpus
#BSUB -W 15
#---------------------------------------
ulimit -s 10240
ulimit -c 1000
export BIND_THREADS=yes
#export BIND_SLOTS=4
export USE_MPS=no
export OMP_NUM_THREADS=1
export USE_GOMP=yes 
export RANKS_PER_NODE=${ppn}
#export LD_PRELOAD=/home/walkup/mpitrace/spectrum_mpi/libmpitrace.so

/opt/ibm/spectrum_mpi/jsm_pmix/bin/jsrun --rs_per_host 1 --tasks_per_rs ${ppn} --cpu_per_rs 42 --gpu_per_rs 6 --nrs ${nodes} -d plane:${ppn} ./help4.sh ./snap 4rank.in mout.4rank.$tstamp

EOF
#---------------------------------------
bsub  <batch.job
