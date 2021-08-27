#!/bin/bash

CNS=$(bjobs -w $LSB_JOBID | tail -n1 | awk {'print $6'} | tr ":" "\n" | sort | uniq  | grep -v batch)

iLoop=0
for CN in $CNS
do
  ((iLoop++))
  ( ssh $CN 'pgrep jsmd  | tail -n1 | xargs gstack' &> jsmd.${CN}.${LSB_JOBID}.gstack ) &
  ( ssh $CN 'pgrep hacc  | xargs -n 1 gstack' &> hack.${CN}.${LSB_JOBID}.gstack ) &
  ( ssh $CN 'ps -A -o pid,state,command' &> ps.${CN}.${LSB_JOBID} ) &
  ( ssh $CN 'top -b -n1' &> top.${CN}.${LSB_JOBID} ) &
  if [ $(($iLoop % 100)) -eq 0 ]; then
    echo "waiting" `date`
    wait
  fi
done

