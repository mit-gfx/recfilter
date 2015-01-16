#/usr/bin/bash

if [ $# -ge 1 ]
then
    HL_JIT_TARGET=opencl $@
else
    echo 'Usage: ./opencl_run [application_command_line]'
fi
