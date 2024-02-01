#!/bin/bash

# File Locations
SCRIPT_DIR=$(dirname "$0")
TEST="$SCRIPT_DIR/../../../build/test"
EXE="tester"

# Start testing
echo "--------------------------------------------------------------------------------------------------------------------------"
echo "Testing the correctness of gemv"
echo "--------------------------------------------------------------------------------------------------------------------------"
cd $TEST
echo "-----------------------------------------Testing the illegal input of gemv------------------------------------------------"
./$EXE --type=s,d,c,z --dim=1024 --testcase=0 dev-gemv 
echo "------------------------------------------Testing the legal input of gemv-------------------------------------------------"
DIM=""
for m in 256 257 2048 2049 4096 4097; do
    for n in 256 257 2048 2049 4096 4097; do
        DIM+="$m*$n,"
    done
done
DIM=${DIM%,}

./$EXE --type=s,d,c,z --layout=r,c --trans=n,t,c --dim=$DIM --incx=-2,-1,1,2 --incy=-2,-1,1,2 dev-gemv 