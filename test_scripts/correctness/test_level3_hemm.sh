#!/bin/bash

# File Locations
SCRIPT_DIR=$(dirname "$0")
TEST="$SCRIPT_DIR/../../build/test"
EXE="tester"

# Start testing
echo "--------------------------------------------------------------------------------------------------------------------------"
echo "Testing the correctness of hemm"
echo "--------------------------------------------------------------------------------------------------------------------------"
cd $TEST
echo "-----------------------------------------Testing the illegal input of hemm------------------------------------------------"
./$EXE --type=s,d,c,z --dim=1024 --testcase=0 dev-hemm 
echo "------------------------------------------Testing the legal input of hemm-------------------------------------------------"
DIM=""
for m in 256 257 2048 2049; do
    for n in 256 257 2048 2049; do
        DIM+="$m*$n,"
    done
done
DIM=${DIM%,}

./$EXE --type=s,d,c,z --layout=r,c --side=l,r --uplo=l,u --dim=$DIM dev-hemm

