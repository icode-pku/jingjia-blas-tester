#!/bin/bash

# File Locations
SCRIPT_DIR=$(dirname "$0")
TEST="$SCRIPT_DIR/../../build/test"
EXE="tester"

# Start testing
echo "--------------------------------------------------------------------------------------------------------------------------"
echo "Testing amax"
echo "--------------------------------------------------------------------------------------------------------------------------"
cd $TEST
echo "------------------------------------------Testing the legal input of amax-------------------------------------------------"
./$EXE --type=s,d,c,z --dim=1024 --incx=1,2 dev-amax 
echo "-----------------------------------------Testing the illegal input of amax------------------------------------------------"
./$EXE --type=s,d,c,z --dim=1024 --testcase=0 dev-amax 

