#!/bin/bash
let nodes=192
let ppn=42
let nmpi=$nodes*$ppn
let cores=42*${nodes}
tstamp=`date +%m_%d_%H_%M_%S`
#--------------------------------------
cat >batch.job <<EOF
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -nnodes ${nodes}
#BSUB -alloc_flags "smt4 cpublink"
#BSUB -P VEN201
#BSUB -q tested
#BSUB -W 35
#---------------------------------------
ulimit -c 0
ulimit -s unlimited
export OMP_NUM_THREADS=1

export BIND_SLOTS=4
export RANKS_PER_NODE=${ppn}
export BIND_THREADS=yes
export SYSTEM_CORES=2
#export SAVE_LIST=0
export DISABLE_ASLR=yes
export LD_PRELOAD=/ccs/home/walkup/mpitrace/spectrum_mpi/libmpihpm.so 
export PAMI_ENABLE_STRIPING=1
export PAMI_IBV_ENABLE_OOO_AR=1
export PAMI_IBV_QP_SERVICE_LEVEL=8
export OMPI_MCA_pml_pami_local_eager_limit=16001
export HPM_GROUP=1
#export HPM_EVENT_LIST="PM_LSU_DERAT_MISS,PM_DTLB_MISS,PM_RUN_INST_CMPL,PM_RUN_CYC"

cat >comm2d.in <<END
Ynodes 16
Znodes 12
Ylocal  6
Zlocal  7
END

# --smpiargs="-MXM"

 jsrun -X 1     \
 --nrs ${nodes}  --tasks_per_rs ${ppn} --cpu_per_rs 42 \
  -d plane:${ppn} --bind rs    \
  ./h6.sh  ../CPU/snap ./input.96x84_516x480x504  out.cpu.$tstamp
# ./h6.sh  ../CPU/snap ./input.120x128_480x480x512  out.cpu.$tstamp

# ./h6.sh  ../CPU/snap ./input.120x144_240x480x576  out.cpu.$tstamp
# ./h6.sh  ../CPU/snap ./input.90x96_258x540x480    out.cpu.$tstamp

# ../CPU/snap  input.16x20_496x64x80  out.cpu.$tstamp


EOF
#---------------------------------------
bsub  batch.job
