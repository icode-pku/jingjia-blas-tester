#!/bin/bash

# File Locations
SCRIPT_DIR=$(dirname "$0")
TEST="$SCRIPT_DIR/../../../build/test"
EXE="tester"

# Start testing
echo "--------------------------------------------------------------------------------------------------------------------------"
echo "Testing the correctness of herk"
echo "--------------------------------------------------------------------------------------------------------------------------"
cd $TEST
echo "-----------------------------------------Testing the illegal input of herk------------------------------------------------"
./$EXE --type=s,d,c,z --dim=1024 --testcase=0 dev-herk 
echo "------------------------------------------Testing the legal input of herk-------------------------------------------------"
DIM=""
for n in 256 257; do
    for k in 256 257; do
        DIM+="$n*$k,"
    done
done
DIM=${DIM%,}

./$EXE --type=s,d,c,z --layout=r,c --uplo=l,u --trans=n,c --dim=$DIM dev-herk

