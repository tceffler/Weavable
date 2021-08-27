#!/bin/bash 
nodes=16
ppn=4
let nmpi=$nodes*$ppn
#--------------------------------------
cat >batch.job <<EOF
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -R "span[ptile=${ppn}]"
#BSUB -n ${nmpi}
#BSUB -x
#BSUB -G guests
#BSUB -q pbatch
#BSUB -W 15
#---------------------------------
ulimit -s 10240
export OMP_NUM_THREADS=5
export OMP_WAIT_POLICY=active
#mpirun --bind-to none -np ${nmpi} set_device_and_bind.sh ./amg2013 -pooldist 1 -r 81 81 81 -P 1 1 1
#mpirun --bind-to none -np ${nmpi} set_device_and_bind.sh ./amg2013 -pooldist 1 -r 72 72 72 -P 4 2 2
mpirun --bind-to none -np ${nmpi} set_device_and_bind.sh ./amg2013 -pooldist 1 -r 72 72 72 -P 2 2 2
EOF
#---------------------------------------
bsub  <batch.job
