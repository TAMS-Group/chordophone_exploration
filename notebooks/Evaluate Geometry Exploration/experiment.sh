#!/bin/bash 

if [[ -z "$1" ]]; then
  echo "usage: $0 <name of experiment> (datetime will be appended)"
  exit 1
fi

rosrun tams_pr2_guzheng iterate_ransac_experiment.sh "$1"
