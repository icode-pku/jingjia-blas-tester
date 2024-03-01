#!/bin/bash

OUTPUT_FILE="performance_amax_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level1_amax.sh 2>&1 | tee -a $OUTPUT_FILE

OUTPUT_FILE="performance_amin_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level1_amin.sh 2>&1 | tee -a $OUTPUT_FILE

OUTPUT_FILE="performance_asum_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level1_asum.sh 2>&1 | tee -a $OUTPUT_FILE

OUTPUT_FILE="performance_axpy_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level1_axpy.sh 2>&1 | tee -a $OUTPUT_FILE

OUTPUT_FILE="performance_copy_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level1_copy.sh 2>&1 | tee -a $OUTPUT_FILE

OUTPUT_FILE="performance_dot_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level1_dot.sh 2>&1 | tee -a $OUTPUT_FILE

OUTPUT_FILE="performance_dotu_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level1_dotu.sh 2>&1 | tee -a $OUTPUT_FILE

OUTPUT_FILE="performance_nrm2_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level1_nrm2.sh 2>&1 | tee -a $OUTPUT_FILE

OUTPUT_FILE="performance_scal_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level1_scal.sh 2>&1 | tee -a $OUTPUT_FILE

OUTPUT_FILE="performance_scalu_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level1_scalu.sh 2>&1 | tee -a $OUTPUT_FILE

OUTPUT_FILE="performance_swap_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level1_swap.sh 2>&1 | tee -a $OUTPUT_FILE