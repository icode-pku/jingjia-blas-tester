#!/bin/bash

# File Locations
SCRIPT_DIR=$(dirname "$0")
TEST="$SCRIPT_DIR/../../build/test"
EXE="tester"

# Start testing
echo "--------------------------------------------------------------------------------------------------------------------------"
echo "Testing the performance of gbmv"
echo "--------------------------------------------------------------------------------------------------------------------------"
cd $TEST
DIM=""
for m in 256 257 2048 2049; do
    for n in 256 257 2048 2049; do
        DIM+="$m*$n,"
    done
done
DIM=${DIM%,}

./$EXE --type=s,d,c,z --layout=r,c --trans=n,t,c --dim=$DIM --kl=0,10 --ku=0,10 --incx=-2,-1,1,2 --incy=-2,-1,1,2 --iscorrect=0 dev-gbmv 

