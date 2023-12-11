#!/bin/bash

# File Locations
SCRIPT_DIR=$(dirname "$0")
TEST="$SCRIPT_DIR/../../../build/test"
EXE="helper"

# Start testing
echo "--------------------------------------------------------------------------------------------------------------------------"
echo "Testing cublasGetLoggerCallback"
echo "--------------------------------------------------------------------------------------------------------------------------"
cd $TEST
./$EXE cublasgetloggercallback

