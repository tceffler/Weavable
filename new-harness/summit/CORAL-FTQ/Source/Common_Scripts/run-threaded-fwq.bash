#!/bin/bash 

[ $# -ne 3 ] && { echo "USAGE ERROR: $0 exe_path num_nodes num_procs_per_node"; exit 1; }

fwq_exe=$1
nnodes=$2
ppn=$3

cpn=42
[ $ppn -gt $cpn ] &&  { echo "USAGE ERROR: num_procs_per_node must be <= $cpn, given $ppn"; exit 1; }

# two rs per node, with cores/tasks split across
rpn=2
nrs=$(( $nnodes * $rpn ))
cpr=$(( $cpn / $rpn ))
ppr=$(( $ppn / $rpn ))
nthr=$(( $cpn / $ppn ))
jsr_rs_args="-n $nrs -c $cpr -a $ppr -r $rpn --bind rs"

# use 2^16 steps per work cycle, measure work 10000 times, use $nthreads threads
fwq_args="-w 16 -n 10000 -t $nthr"

jsrun $jsr_rs_args $fwq_exe $fwq_args
