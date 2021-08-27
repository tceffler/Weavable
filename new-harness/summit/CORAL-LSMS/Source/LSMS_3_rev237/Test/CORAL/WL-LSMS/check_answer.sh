#!/bin/bash
if [ -z "$1" ]; then
  echo syntax: check_answer.sh output.file
  exit
fi
grep -A 2 "Energy mean" $1
#grep -B1 WL-LSMS $1
#grep -A1 WL-LSMS $1
