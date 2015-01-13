#/usr/bin/bash

TFILE=$( mktemp )

if [ -z "$TFILE1" ]
then
    TFILE="$(basename $0).$RANDOM.tmp"
fi

HALIDE_DIR=../halide
HL_PROFILER=$HALIDE_DIR/bin/HalideProf

if [ $# -ge 1 ]
then
    HL_PROFILE=1 HL_JIT_TARGET=x86-64 $@ 2> $TFILE

    $HL_PROFILER -overhead 0 -top 1000000 -sort t < $TFILE

    rm -f $TFILE

else
    echo 'Usage: ./cpu_profile [application_command_line]'
fi
