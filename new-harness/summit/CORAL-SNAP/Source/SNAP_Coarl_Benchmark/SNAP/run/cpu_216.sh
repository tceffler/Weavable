#!/bin/bash
let nodes=216
let ppn=80
tstamp=`date +%m_%d_%H_%M_%S`
#--------------------------------------
cat >batch.job <<EOF
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -nnodes ${nodes}
#BSUB -core_isolation 2
#BSUB -G guests
#BSUB -q pbatch
#BSUB -W 35
#---------------------------------------
ulimit -s 10240
export OMP_NUM_THREADS=1

export BIND_THREADS=yes
export SYSTEM_CORES=4
export RANKS_PER_NODE=${ppn}
#export SAVE_LIST=0
export LD_PRELOAD=/g/g14/walkup/mpitrace/spectrum_mpi/libmpihpm.so 
export PAMI_ENABLE_STRIPING=1
export PAMI_IBV_ENABLE_OOO_AR=1
export PAMI_IBV_QP_SERVICE_LEVEL=8
export HPM_GROUP=1

cat >comm2d.in <<END
Ynodes 12
Znodes 18
Ylocal 10
Zlocal  8
END

# --smpiargs="-MXM"
# --progress ${tstamp}.progress

 jsrun  -X 1   \
 --nrs ${nodes}  --tasks_per_rs ${ppn} --cpu_per_rs 40 \
  -d plane:${ppn} --bind rs    \
  ../CPU/snap input.120x144_240x480x576  out.cpu.$tstamp


EOF
#---------------------------------------
bsub  <batch.job
