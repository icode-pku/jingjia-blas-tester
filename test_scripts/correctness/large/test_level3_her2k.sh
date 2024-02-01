#!/bin/bash

# File Locations
SCRIPT_DIR=$(dirname "$0")
TEST="$SCRIPT_DIR/../../../build/test"
EXE="tester"

# Start testing
echo "--------------------------------------------------------------------------------------------------------------------------"
echo "Testing the correctness of her2k"
echo "--------------------------------------------------------------------------------------------------------------------------"
cd $TEST
echo "-----------------------------------------Testing the illegal input of her2k------------------------------------------------"
./$EXE --type=s,d,c,z --dim=1024 --testcase=0 dev-her2k 
echo "------------------------------------------Testing the legal input of her2k-------------------------------------------------"
DIM=""
for n in 256 257 2048 2049 4096; do
    for k in 256 257 2048 2049 4096; do
        DIM+="$n*$k,"
    done
done
DIM=${DIM%,}

./$EXE --type=s,d,c,z --layout=r,c --uplo=l,u --trans=n,c --dim=$DIM dev-her2k

