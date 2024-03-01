#!/bin/bash

# File Locations
SCRIPT_DIR=$(dirname "$0")
TEST="$SCRIPT_DIR/../../build/test"
EXE="tester"

# Start testing
echo "--------------------------------------------------------------------------------------------------------------------------"
echo "Testing the performance of gemm"
echo "--------------------------------------------------------------------------------------------------------------------------"
cd $TEST
DIM=""
for m in 256 257 2048 2049; do
    for n in 256 257 2048 2049; do
        for k in 256 257 2048 2049; do   
            DIM+="$m*$n*$k,"
        done
    done
done
DIM=${DIM%,}

./$EXE --type=s,d,c,z,h --layout=r,c --transA=n,t,c --transB=n,t,c --dim=$DIM --iscorrect=0 dev-gemm

