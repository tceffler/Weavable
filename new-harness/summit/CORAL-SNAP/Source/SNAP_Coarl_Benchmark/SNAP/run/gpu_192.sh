#!/bin/bash
let nodes=192
let ppn=4
let nmpi=$nodes*$ppn
let cores=40*${nodes}
tstamp=`date +%m_%d_%H_%M_%S`
#--------------------------------------
cat >batch.job <<EOF
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -nnodes ${nodes}
#BSUB -alloc_flags isolategpfs
#BSUB -core_isolation 2
#BSUB -P VEN201
#BSUB -q tested
#BSUB -W 35
#---------------------------------------


 jsrun -X 1    \
 --nrs ${nodes}  --tasks_per_rs ${ppn}  --cpu_per_rs 40  \
 --gpu_per_rs 6  --bind proportional-packed:10 -d plane:${ppn}   \
 ../opt/snap ./input.24x32_512x384x512  out.gpu.$tstamp


EOF
#---------------------------------------
bsub  <batch.job
