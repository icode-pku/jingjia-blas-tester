#!/bin/bash

# File Locations
SCRIPT_DIR=$(dirname "$0")
TEST="$SCRIPT_DIR/../../build/test"
EXE="tester"

# Start testing
echo "--------------------------------------------------------------------------------------------------------------------------"
echo "Testing the performance of trsv"
echo "--------------------------------------------------------------------------------------------------------------------------"
cd $TEST
./$EXE --type=s,d,c,z --layout=r,c --uplo=l,u --trans=n,t,c --diag=n,u --dim=256,257,2048,2049 --incx=-2,-1,1,2 --iscorrect=0 dev-trsv 

