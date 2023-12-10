#!/bin/bash

# File Locations
SCRIPT_DIR=$(dirname "$0")
TEST="$SCRIPT_DIR/../../build/test"
EXE="tester"

# Start testing
echo "--------------------------------------------------------------------------------------------------------------------------"
echo "Testing the correctness of trsmBatched"
echo "--------------------------------------------------------------------------------------------------------------------------"
cd $TEST
echo "-----------------------------------------Testing the illegal input of trsmBatched------------------------------------------------"
./$EXE --type=s,d,c,z --dim=1024 --testcase=0 dev-batch-trsm 
echo "------------------------------------------Testing the legal input of trsmBatched-------------------------------------------------"
DIM=""
for m in 256 257 2048 2049; do
    for n in 256 257 2048 2049; do
        DIM+="$m*$n,"
    done
done
DIM=${DIM%,}

./$EXE --type=s,d,c,z --layout=r,c --side=l,r --uplo=l,u --trans=n,t,c --diag=n,u --dim=$DIM --batch=1,10 dev-batch-trsm