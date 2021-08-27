#!/bin/bash -l

echo "Starting $0 in `pwd`..."

REFVAL=24
nsuccess=`grep "FOM" */stdout.txt | wc -l`

if [ $nsuccess -eq ${REFVAL} ]
then
    echo "PASSED"
    exit 0
else
    echo "FAILED"
    exit 1
fi
