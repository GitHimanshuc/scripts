#!/bin/bash

match="_ng_"

while read job; do
  set -- $job
  if [[ $3 =~ $match ]]; then
    echo qdel "$1"
  fi
done < <(qstat)