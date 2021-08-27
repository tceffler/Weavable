#!/bin/bash -l

OUTFILE=stdout.txt
REPORTFILE=umt_opt_results.csv

fom=`grep merit $OUTFILE |  awk '{print $3}'`

echo "$fom" >> $REPORTFILE
