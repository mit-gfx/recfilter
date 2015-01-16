#/usr/bin/bash

# Script to profile a CUDA application using nvprof
#
# Assumptions:
# - name of each CUDA kernel starts with "kernel"
#
# Requirements:
# - nvprof (NVIDIA CUDA Toolkit)
# - awk
# - cat
# - bc

TFILE1=$( mktemp )

if [ -z "$TFILE1" ]
then
    TFILE1="$(basename $0).$RANDOM.tmp"
fi

if [ $# -eq 2 ]
then
    app=$1
    width=$2

    # run the application and redirect nvprof output to a temp file
    HL_JIT_TARGET=cuda_capability_35 \
        nvprof -u ms \
        $app -w $width  -t 32 -iter 1000 2> $TFILE1

    # average runtime per kernel execution is ms is in 4th column
    runtime=`cat $TFILE1 | awk '/kernel/' | awk '{ sum += $4 } END { print sum }'`

    # calculate throughput
    tput=`echo "scale=4;($width*$width*1000.0)/($runtime*1024.0*1024.0)" | bc`

    echo -e $width '\t' $runtime '\t' $tput

    rm -f $TFILE1

else
    echo 'Usage: ./cuda_profile [application] [image width]'
fi

