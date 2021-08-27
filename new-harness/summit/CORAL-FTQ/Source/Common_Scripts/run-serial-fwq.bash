#!/bin/bash 

[ $# -ne 3 ] && { echo "USAGE ERROR: $0 exe_path num_nodes num_procs_per_node"; exit 1; }

fwq_exe=$1
nnodes=$2
ppn=$3

cpn=42
[ $ppn -gt $cpn ] &&  { echo "USAGE ERROR: num_procs_per_node must be <= $cpn, given $ppn"; exit 1; }


# split cores/tasks across two resource sets per node
rpn=2
nrs=$(( $nnodes * $rpn ))
cpr=$(( $cpn / $rpn ))
ppr=$(( $ppn / $rpn ))
jsr_rs_args="-n $nrs -c $cpr -a $ppr -r $rpn"

# capture stdout/err per task
jsr_io_args="-e individual -o fwq_times.%h.%t -k fwq_err.%h.%t"

# send output to stdout, use 2^16 steps per work cycle, measure work 10000 times
fwq_args="-s -w 16 -n 10000"

echo "jsrun $jsr_rs_args $jsr_io_args $fwq_exe $fwq_args"
jsrun $jsr_rs_args $jsr_io_args $fwq_exe $fwq_args 1> stdout.txt 2> stderr.txt
