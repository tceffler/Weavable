#!/bin/bash
#wl-lsms requires an additional MPI rank for the WL master. I.e. if each LSMS instance uses
#K MPI ranks the total number of ranks required is 1+K*N.
#with N walkers and M WL steps/walker use the following command:
#
#mpirun -np 1+K*N $(LSMS_DIRECTORY)/bin/wl-lsms -i i_lsms -mode 1d -size_lsms 1024 -num_lsms N -num_steps N*M
#
#i.e for 15 walkers and 20 steps/walker and 32 MPI ranks/walker:
#
#mpirun -np 481 wl-lsms -i i_lsms -mode 1d -size_lsms 1024 -num_lsms 15 -num_steps 300

let num_atoms=1024
let ranks_per_walker=36
let num_walkers=1919
let steps_per_walker=12
let nmpi=1+$ranks_per_walker*$num_walkers
let num_steps=$num_walkers*$steps_per_walker
echo mpirun -np $nmpi ../../../bin/wl-lsms -i i_lsms -mode 1d -size_lsms $num_atoms  -num_lsms $num_walkers -num_steps $num_steps
