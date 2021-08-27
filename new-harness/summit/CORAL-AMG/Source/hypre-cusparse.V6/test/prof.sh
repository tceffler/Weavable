#!/bin/bash

let myrank=$OMPI_COMM_WORLD_RANK
let maxrank=$OMPI_COMM_WORLD_SIZE-1

if [ $myrank = $maxrank ]; then 
    nvprof -f -o timeline "$@"
else
    "$@"
fi
