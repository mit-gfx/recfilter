#!/bin/bash

# Script to profile all applications

apps="\
./summed_table
./bicubic_filter
./biquintic_filter
./box_filter_1
./box_filter_3
./box_filter_6
./gaussian_filter_21
./gaussian_filter_3
./gaussian_filter_33
"

# profile each app for all image widths from min_w to max_w
min_w=64
max_w=4096
inc_w=64

for app in $apps
do
    logfile="$app".ours.perflog
    rm -f $logfile

    for (( w=$min_w; w <= $max_w; w+=$inc_w ))
    do
        cmd="./cuda_profile.sh $app $w >> $logfile"
        echo $cmd
        eval $cmd
    done
done
