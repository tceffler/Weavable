#!/bin/bash -l

echo "Starting $0 in `pwd`..."

REFVAL=1
OUTFNAME=stdout.txt
nsuccess=`grep -c "FOM" $OUTFNAME`

if [ $nsuccess -eq ${REFVAL} ]
then
    echo "PASSED"
    exit 0
else
    echo "FAILED"
    exit 1
fi
