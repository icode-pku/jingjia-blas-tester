#!/bin/bash

# File Locations
SCRIPT_DIR=$(dirname "$0")
TEST="$SCRIPT_DIR/../../build/test"
EXE="tester"

# Start testing
echo "--------------------------------------------------------------------------------------------------------------------------"
echo "Testing the performance of dotu"
echo "--------------------------------------------------------------------------------------------------------------------------"
cd $TEST
./$EXE --type=s,d,c,z --dim=4,5,1024,1025,524288,524289 --incx=1,2 --incy=1,2 --iscorrect=0 dev-dotu 

