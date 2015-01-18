#!/bin/bash

# Script to profile all applications

apps="\
./summed_table
./bicubic_filter
./box_filter_1
./box_filter_3
./box_filter_6
./biquintic_overlapped_filter
./biquintic_cascaded_filter
./gaussian_overlapped_filter
./gaussian_cascaded_filter
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

# profile CPU 1D filters
# HL_JIT_TARGET=x86-64 ./audio_higher_order -w 10000000 -t 1000 -iter 200
# HL_JIT_TARGET=x86-64 ./audio_biquads -w 10000000 -t 1000 -iter 200
