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
./gaussian_filter_1xy_1xy_1xy
./gaussian_filter_1xy_2x_2y
./gaussian_filter_1xy_2xy
./gaussian_filter_3x_3y
./gaussian_filter_3xy
./diff_gauss
"

# profile each app for all image widths from min_w to max_w
min_w=64
max_w=4096
inc_w=64

for app in $apps
do
    ./profile_app.sh $app
done

# profile CPU 1D filters
HL_JIT_TARGET=x86-64 ./audio_filter_higher_order -w 10000000 -t 1000 -iter 200
HL_JIT_TARGET=x86-64 ./audio_filter_biquads -w 10000000 -t 1000 -iter 200
