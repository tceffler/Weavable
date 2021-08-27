#!/bin/bash
nnodes=$1
echo "Checking CORAL-FTQ (fwq serial) for $nnodes nodes"

scriptdir=$(dirname $0)

module load gcc/4.8.5
module load python/2.7.12
module load py-numpy/1.11.2-py2
module list 2>&1

meets_fom=0

npass=0
nfail=0
for df in fwq_times.* ; do
    tail -n +2 $df > ${df}.dat && /bin/rm $df
    $scriptdir/check-fwq.py ${df}.dat
    if [[ $? -eq 0 ]]; then
	(( npass++ ))
    else
	(( nfail++ ))
    fi
done

fomrate=80
passrate=$(( (100 * npass) / (npass + nfail) ))
echo "CHECK RESULTS: pass rate is $passrate % (npass=$npass, nfail=$nfail)"

(( passrate >= $fomrate )) && meets_fom=1
[[ $meets_fom -eq 0 ]] && { echo "PERF FAILURE: pass rate % < $fomrate"; exit 2; }
exit 0 
