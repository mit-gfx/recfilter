#/usr/bin/bash

function mean_time_with_outlier_rejection {
    awk ' { if (NR!=1) print } ' $1 | awk '{ sum += $5 } END { print sum }'
}

TFILE1=$( mktemp )
TFILE2=$( mktemp )

if [ -z "$TFILE1" -o -z "$TFILE2" ]
then
    TFILE1="$(basename $0).$RANDOM.tmp"
    TFILE2="$(basename $0).$RANDOM.tmp"
fi

if [ $# -ge 1 ]
then
    HL_JIT_TARGET=opencl-gpu_debug $@ 2> $TFILE1

    if [ $? == 0 ]
    then
        echo 'Kernel Blocks Threads Shared_mem(KB) Time(ms)' > $TFILE2
        cat $TFILE1 | \
            sed '/clCreateKernel/d' | \
            sed '/halide_dev_run\|Time/!d' | \
            sed '/halide_dev_run/,+1!d' | \
            sed 's/CL.*entry: //g' | \
            sed ':a;N;$!ba;s/\n /;\t/g' | \
            sed 's/blocks: \|threads: \|shmem: \|Time: \|ms/;\t/g' | \
            sed 's/,\|)//g'  | \
            sed 's/ \|\t//g' | \
            sed 's/;/ /g' >> $TFILE2

        column -t < $TFILE2

        # calculate total time; extract last column, remove top row and sum
        echo ''
        echo ''
        echo 'Total tile for kernel execution in ms'
        mean_time_with_outlier_rejection $TFILE2

    else
        cat $TFILE1
    fi

    rm -f $TFILE1 $TFILE2

else
    echo 'Usage: ./opencl_profile [application_command_line]'
fi
