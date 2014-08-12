#/usr/bin/bash

if [ $# -ge 1 ]
then
    f=/tmp/$RANDOM
    HL_JIT_TARGET=cuda-gpu_debug $@ 2> $f

    if [ $? == 0 ]
    then
        cat $f | \
            sed '/halide_dev_run\|Time/!d' | \
            sed '/halide_dev_run/,+1!d' | \
            sed 's/CL.*entry: //g' | \
            sed ':a;N;$!ba;s/\n /;\t/g' | \
            sed 's/blocks: \|threads: \|shmem: \|Time: /;\t/g' | \
            sed 's/,\|)//g'
    else
        cat $f
    fi
else
    echo 'Usage: ./cuda_profile [application_command_line]'
fi
