#!/bin/bash

# File Locations
SCRIPT_DIR=$(dirname "$0")
TEST="$SCRIPT_DIR/../../../build/test"
EXE="tester"

# Start testing
echo "--------------------------------------------------------------------------------------------------------------------------"
echo "Testing the correctness of swap"
echo "--------------------------------------------------------------------------------------------------------------------------"
cd $TEST
echo "-----------------------------------------Testing the illegal input of swap------------------------------------------------"
./$EXE --type=s,d,c,z --dim=1024 --testcase=0 dev-swap 
echo "------------------------------------------Testing the legal input of swap-------------------------------------------------"
./$EXE --type=s,d,c,z --dim=4,5,512,513 --incx=1,2 --incy=1,2 dev-swap 

