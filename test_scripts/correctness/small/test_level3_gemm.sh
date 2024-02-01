#!/bin/bash

# File Locations
SCRIPT_DIR=$(dirname "$0")
TEST="$SCRIPT_DIR/../../../build/test"
EXE="tester"

# Start testing
echo "--------------------------------------------------------------------------------------------------------------------------"
echo "Testing the correctness of gemm"
echo "--------------------------------------------------------------------------------------------------------------------------"
cd $TEST
echo "-----------------------------------------Testing the illegal input of gemm------------------------------------------------"
./$EXE --type=s,d,c,z --dim=1024 --testcase=0 dev-gemm 
echo "------------------------------------------Testing the legal input of gemm-------------------------------------------------"
DIM=""
for m in 256 257; do
    for n in 256 257; do
        for k in 256 257; do   
            DIM+="$m*$n*$k,"
        done
    done
done
DIM=${DIM%,}

./$EXE --type=s,d,c,z --layout=r,c --transA=n,t,c --transB=n,t,c --dim=$DIM dev-gemm

