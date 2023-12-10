#!/bin/bash

# File Locations
SCRIPT_DIR=$(dirname "$0")
TEST="$SCRIPT_DIR/../../build/test"
EXE="tester"

# Start testing
echo "--------------------------------------------------------------------------------------------------------------------------"
echo "Testing the performance of syrk"
echo "--------------------------------------------------------------------------------------------------------------------------"
cd $TEST
DIM=""
for n in 256 257 2048 2049; do
    for k in 256 257 2048 2049; do
        DIM+="$n*$k,"
    done
done
DIM=${DIM%,}

./$EXE --type=s,d,c,z --layout=r,c --uplo=l,u --trans=n,t --dim=$DIM --iscorrect=0 dev-syrk

