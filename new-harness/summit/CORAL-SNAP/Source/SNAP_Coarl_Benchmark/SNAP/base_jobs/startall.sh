#! /usr/bin/env bash

export MAX_JOBS=2
export JOB_NUMBER=1

export JOB_STREAM=1
bsub runall.sh

#sleep 1
#
#export JOB_STREAM=2
#bsub runall.sh
