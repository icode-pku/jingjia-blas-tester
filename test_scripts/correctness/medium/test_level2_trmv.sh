#!/bin/bash

# File Locations
SCRIPT_DIR=$(dirname "$0")
TEST="$SCRIPT_DIR/../../../build/test"
EXE="tester"

# Start testing
echo "--------------------------------------------------------------------------------------------------------------------------"
echo "Testing the correctness of trmv"
echo "--------------------------------------------------------------------------------------------------------------------------"
cd $TEST
echo "-----------------------------------------Testing the illegal input of trmv------------------------------------------------"
./$EXE --type=s,d,c,z --dim=1024 --testcase=0 dev-trmv 
echo "------------------------------------------Testing the legal input of trmv-------------------------------------------------"
./$EXE --type=s,d,c,z --layout=r,c --uplo=l,u --trans=n,t,c --diag=n,u --dim=256,257,2048,2049 --incx=-2,-1,1,2 dev-trmv 


