#/usr/bin/bash

TFILE=$( mktemp )

if [ -z "$TFILE" ]
then
    TFILE="$(basename $0).$$.tmp"
fi

if [ $# -ge 1 ]
then
    HL_JIT_TARGET=opencl-gpu_debug $@ 2> $TFILE

    if [ $? == 0 ]
    then
        echo -e 'Kernel name \t GPU tiles \t GPU threads \t Shared memory (KB) \t  Kernel execution time (ms)'
        cat $TFILE | \
            sed '/halide_dev_run\|Time/!d' | \
            sed '/halide_dev_run/,+1!d' | \
            sed 's/CL.*entry: //g' | \
            sed ':a;N;$!ba;s/\n /;\t/g' | \
            sed 's/blocks: \|threads: \|shmem: \|Time: \|ms/;\t/g' | \
            sed 's/,\|)//g'  | \
            sed 's/ \|\t//g' | \
            sed 's/;/\t/g'   | \
            sed 's/\t\t/\t/g'
    else
        cat $TFILE
    fi

    rm -f $TFILE

else
    echo 'Usage: ./opencl_profile [application_command_line]'
fi
