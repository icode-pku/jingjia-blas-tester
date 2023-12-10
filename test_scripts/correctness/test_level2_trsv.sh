#!/bin/bash

# File Locations
SCRIPT_DIR=$(dirname "$0")
TEST="$SCRIPT_DIR/../../build/test"
EXE="tester"

# Start testing
echo "--------------------------------------------------------------------------------------------------------------------------"
echo "Testing the correctness of trsv"
echo "--------------------------------------------------------------------------------------------------------------------------"
cd $TEST
echo "-----------------------------------------Testing the illegal input of trsv------------------------------------------------"
./$EXE --type=s,d,c,z --dim=1024 --testcase=0 dev-trsv 
echo "------------------------------------------Testing the legal input of trsv-------------------------------------------------"
DIM=""
for m in 256 257 2048 2049; do
    for n in 256 257 2048 2049; do
        DIM+="$m*$n,"
    done
done
DIM=${DIM%,}

./$EXE --type=s,d,c,z --layout=r,c --uplo=l,u --trans=n,t,c --diag=n,u --dim=$DIM --incx=-2,-1,1,2 dev-trsv 

