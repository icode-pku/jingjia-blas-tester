#!/bin/bash

# File Locations
SCRIPT_DIR=$(dirname "$0")
TEST="$SCRIPT_DIR/../../../build/test"
EXE="tester"

# Start testing
echo "--------------------------------------------------------------------------------------------------------------------------"
echo "Testing the correctness of amax"
echo "--------------------------------------------------------------------------------------------------------------------------"
cd $TEST
echo "-----------------------------------------Testing the illegal input of amax------------------------------------------------"
./$EXE --type=s,d,c,z --dim=1024 --testcase=0 dev-amax 
echo "------------------------------------------Testing the legal input of amax-------------------------------------------------"
./$EXE --type=s,d,c,z --dim=4,5,1024,1025,524288,524289 --incx=1,2 dev-amax 

