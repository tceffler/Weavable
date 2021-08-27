#!/bin/bash -l

OUTFILE=stdout.txt
REPORTFILE=amg_opt_results.csv

fom=`cat $OUTFILE | grep "Solve Phase" | cut -d: -f3 | sed 's/^[[:space:]]*//g'`

echo "$fom" >> $REPORTFILE
