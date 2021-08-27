#!/bin/bash
let nodes=216
let ppn=4
let nmpi=$nodes*$ppn
let cores=42*${nodes}
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

 jsrun -X 1    \
 --nrs ${nodes}  --tasks_per_rs ${ppn}  --cpu_per_rs 40  \
 --gpu_per_rs 4  --bind proportional-packed:10 -d plane:${ppn}   \
 ../opt/snap ./input.27x32_256x432x512  out.gpu.$tstamp
#../opt/snap ./input.2x2_240x32x32      out.gpu.$tstamp


EOF
#---------------------------------------
bsub  <batch.job
