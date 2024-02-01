#!/bin/bash

# File Locations
SCRIPT_DIR=$(dirname "$0")
TEST="$SCRIPT_DIR/../../../build/test"
EXE="tester"

# Start testing
echo "--------------------------------------------------------------------------------------------------------------------------"
echo "Testing the correctness of trmm"
echo "--------------------------------------------------------------------------------------------------------------------------"
cd $TEST
echo "-----------------------------------------Testing the illegal input of trmm------------------------------------------------"
./$EXE --type=s,d,c,z --dim=1024 --testcase=0 dev-trmm 
echo "------------------------------------------Testing the legal input of trmm-------------------------------------------------"
DIM=""
for m in 256 257; do
    for n in 256 257; do
        DIM+="$m*$n,"
    done
done
DIM=${DIM%,}

./$EXE --type=s,d,c,z --layout=r,c --side=l,r --uplo=l,u --trans=n,t,c --diag=n,u --dim=$DIM dev-trmm