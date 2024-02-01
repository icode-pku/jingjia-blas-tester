#!/bin/bash

OUTPUT_FILE2="correctness_gbmv_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level2_gbmv.sh 2>&1 | tee -a $OUTPUT_FILE2

OUTPUT_FILE2="correctness_gemv_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level2_gemv.sh 2>&1 | tee -a $OUTPUT_FILE2

OUTPUT_FILE2="correctness_trmv_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level2_trmv.sh 2>&1 | tee -a $OUTPUT_FILE2

OUTPUT_FILE2="correctness_trsv_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level2_trsv.sh 2>&1 | tee -a $OUTPUT_FILE2

