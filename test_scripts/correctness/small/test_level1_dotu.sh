#!/bin/bash

# File Locations
SCRIPT_DIR=$(dirname "$0")
TEST="$SCRIPT_DIR/../../../build/test"
EXE="tester"

# Start testing
echo "--------------------------------------------------------------------------------------------------------------------------"
echo "Testing the correctness of dotu"
echo "--------------------------------------------------------------------------------------------------------------------------"
cd $TEST
echo "-----------------------------------------Testing the illegal input of dotu------------------------------------------------"
./$EXE --type=s,d,c,z --dim=1024 --testcase=0 dev-dotu 
echo "------------------------------------------Testing the legal input of dotu-------------------------------------------------"
./$EXE --type=s,d,c,z --dim=4,5,512,513 --incx=-2,-1,1,2 --incy=-2,-1,1,2 dev-dotu 

