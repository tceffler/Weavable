#!/bin/bash -l

echo "Starting $0 in `pwd`..."

OUTFNAME=stdout.txt
nsuccess=`grep -c "ACCUMULATED STATS" $OUTFNAME`

if [ $nsuccess -eq 1 ]
then
    echo "PASSED"
    exit 0
else
    echo "FAILED"
    exit 1
fi
