#/usr/bin/bash

TFILE1=$( mktemp )
TFILE2=$( mktemp )

if [ -z "$TFILE1" -o -z "$TFILE2" ]
then
    TFILE1="$(basename $0).$RANDOM.tmp"
    TFILE2="$(basename $0).$RANDOM.tmp"
fi

if [ $# -eq 3 ]
then
    app=$1
    width=$2

    HL_JIT_TARGET=cuda_capability_35 \
        nvprof -u ms \
        $app -w $width  -t 32 -iter 1000 > $TFILE1

    if [ $? == 0 ]
    then
        runtime=i`cat $TFILE1 | awk '/kernel/' | awk '{ sum += $5 } END { print sum }'`
        echo "$width \t $runtime"

    else
        cat $TFILE1
    fi

    rm -f $TFILE1

else
    echo 'Usage: ./cuda_profile [application] [image width]'
fi

