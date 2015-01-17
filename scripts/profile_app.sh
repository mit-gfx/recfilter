#!/bin/bash

# Script to profile a single application

# profile each app for all image widths from min_w to max_w
min_w=64
max_w=4096
inc_w=64

if [ $# -eq 1 ]
then
    app=$1

    logfile="$app".ours.perflog
    rm -f $logfile

    for (( w=$min_w; w <= $max_w; w+=$inc_w ))
    do
        cmd="./cuda_profile.sh $app $w >> $logfile"
        echo $cmd
        eval $cmd
    done
else
    echo "Usaage: ./profile_app.sh [path to CUDA app]"
fi
