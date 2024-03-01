#!/bin/bash

OUTPUT_FILE="correctness_amax_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level1_amax.sh 2>&1 | tee -a $OUTPUT_FILE

OUTPUT_FILE="correctness_amin_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level1_amin.sh 2>&1 | tee -a $OUTPUT_FILE

OUTPUT_FILE="correctness_asum_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level1_asum.sh 2>&1 | tee -a $OUTPUT_FILE

OUTPUT_FILE="correctness_axpy_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level1_axpy.sh 2>&1 | tee -a $OUTPUT_FILE

OUTPUT_FILE="correctness_copy_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level1_copy.sh 2>&1 | tee -a $OUTPUT_FILE

OUTPUT_FILE="correctness_dot_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level1_dot.sh 2>&1 | tee -a $OUTPUT_FILE

OUTPUT_FILE="correctness_dotu_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level1_dotu.sh 2>&1 | tee -a $OUTPUT_FILE

OUTPUT_FILE="correctness_nrm2_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level1_nrm2.sh 2>&1 | tee -a $OUTPUT_FILE

OUTPUT_FILE="correctness_scal_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level1_scal.sh 2>&1 | tee -a $OUTPUT_FILE

OUTPUT_FILE="correctness_scalu_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level1_scalu.sh 2>&1 | tee -a $OUTPUT_FILE

OUTPUT_FILE="correctness_swap_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level1_swap.sh 2>&1 | tee -a $OUTPUT_FILE