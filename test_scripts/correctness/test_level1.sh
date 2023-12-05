#!/bin/bash

 OUTPUT_FILE="correctness_amax_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level1_amax.sh 2>&1 | tee -a $OUTPUT_FILE
