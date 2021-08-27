#!/bin/bash
let nodes=2048
let ppn=36
let nmpi=$ppn*$nodes
let ranks_per_socket=$ppn/2
let cores_per_socket=20
let cores_per_rank=$cores_per_socket/$ranks_per_socket
let num_sockets=$nodes*2
let cores=$nodes*40
let gpus_per_socket=2
tstamp=`date +%m_%d_%H_%M_%S`
#--------------------------------------
cat >batch.job <<EOF
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -nnodes ${nodes}
#BSUB -core_isolation 2
##BSUB -csm y 
##BSUB -R "1*{select[LN]} + ${cores}*{select[CN&&(hname!=sierra135)&&(hname!=sierra188)&&(hname!=sierra1719)&&(hname!=sierra4369)&&(type==any)]span[ptile=40]}"
#BSUB -G guests
#BSUB -q pibm 
#BSUB -W 60
#---------------------------------------
export SAVE_LIST=0
export OMP_NUM_THREADS=1
export BIND_THREADS=no
export SYSTEM_CORES=4
export RANKS_PER_NODE=${ppn}
export BIND_SLOTS=4
export USE_GOMP=yes
export USE_MPS=yes
export PAMI_IBV_ENABLE_DCT=1
export PAMI_PMIX_DATACACHE=1
export LD_PRELOAD=/g/g14/walkup/mpitrace/spectrum_mpi/libmpitrace.so

 jsrun --progress ${tstamp}.progress -X 1  \
  --nrs ${num_sockets}  --tasks_per_rs ${ranks_per_socket} --cpu_per_rs ${cores_per_socket} \
  --gpu_per_rs ${gpus_per_socket} --bind=proportional-packed:${cores_per_rank} -d plane:${ranks_per_socket}  \
  ./helper.sh  ../../../bin/wl-lsms -i i_lsms -align ${ppn} -mode 1d -size_lsms 1024 -num_lsms 2047 -num_steps 24564          
# ./helper.sh  ../../../bin/wl-lsms -i i_lsms -align ${ppn} -mode 1d -size_lsms 1024 -num_lsms 1023 -num_steps 12276          

EOF
#---------------------------------------
bsub  <batch.job

