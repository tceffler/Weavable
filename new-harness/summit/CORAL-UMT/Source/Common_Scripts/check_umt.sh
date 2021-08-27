#!/bin/bash -l

OUTFILE=stdout.txt

if [ $1 == "opt" ]
then
    coral_fom="9.17E+11"
    ibm_fom="6.938E+11"
    ornl_fom="6.9407E+11"
elif [ $1 == "base" ]
then
    coral_fom="3.46E+11"
    ibm_fom="1.992E+11"
    ornl_fom="1.96894e+11"
else
    echo "Incorrect type of test for UMT!"
    exit 1
fi

run_fom=`cat $OUTFILE | grep merit | cut -d= -f2 | sed 's/e/E/g;s/^[[:space:]]*//g'`

if [ "X$run_fom" = "X" ]
then
    echo "No FOM found in output!"
    exit 1
fi

echo "CORAL FOM is $coral_fom"
echo "IBM FOM is $ibm_fom"
echo "ORNL FOM is $ornl_fom"
echo "Run FOM is $run_fom"

meets_fom=`awk -v rf=$run_fom -v cf=$coral_fom 'BEGIN {print ( rf >= cf ? "yes" : "no")}'`
echo "Meets CORAL FOM? $meets_fom"
meets_ibm=`awk -v rf=$run_fom -v cf=$ibm_fom 'BEGIN {print ( rf >= cf ? "yes" : "no")}'`
echo "Meets IBM FOM? $meets_ibm"
meets_ornl=`awk -v rf=$run_fom -v cf=$ornl_fom 'BEGIN {print ( rf >= 0.7*cf ? "yes" : "no")}'`
echo "Meets .7*ORNL FOM? $meets_ornl"

[[ $meets_ornl == "no" ]] && { echo "PERF FAILURE: Run FOM $run_fom < 0.7 * ORNL FOM $ornl_fom"; exit 5; }

exit 0
