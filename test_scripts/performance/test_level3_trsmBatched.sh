#!/bin/bash

# File Locations
SCRIPT_DIR=$(dirname "$0")
TEST="$SCRIPT_DIR/../../build/test"
EXE="tester"

# Start testing
echo "--------------------------------------------------------------------------------------------------------------------------"
echo "Testing the performance of trsmBatched"
echo "--------------------------------------------------------------------------------------------------------------------------"
cd $TEST
DIM=""
for m in 256 257 2048 2049; do
    for n in 256 257 2048 2049; do
        DIM+="$m*$n,"
    done
done
DIM=${DIM%,}

./$EXE --type=s,d,c,z --layout=r,c --side=l,r --uplo=l,u --trans=n,t,c --diag=n,u --dim=$DIM --batch=1,10 --iscorrect=0 dev-batch-trsm