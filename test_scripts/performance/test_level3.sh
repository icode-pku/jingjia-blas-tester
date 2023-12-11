#!/bin/bash

OUTPUT_FILE3="performance_gemm_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level3_gemm.sh 2>&1 | tee -a $OUTPUT_FILE3

OUTPUT_FILE3="performance_gemmBatched_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level3_gemmBatched.sh 2>&1 | tee -a $OUTPUT_FILE3

OUTPUT_FILE3="performance_hemm_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level3_hemm.sh 2>&1 | tee -a $OUTPUT_FILE3

OUTPUT_FILE3="performance_herk_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level3_herk.sh 2>&1 | tee -a $OUTPUT_FILE3

OUTPUT_FILE3="performance_her2k_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level3_her2k.sh 2>&1 | tee -a $OUTPUT_FILE3

OUTPUT_FILE3="performance_symm_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level3_symm.sh 2>&1 | tee -a $OUTPUT_FILE3

OUTPUT_FILE3="performance_syrk_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level3_syrk.sh 2>&1 | tee -a $OUTPUT_FILE3

OUTPUT_FILE3="performance_syr2k_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level3_syr2k.sh 2>&1 | tee -a $OUTPUT_FILE3

OUTPUT_FILE3="performance_trmm_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level3_trmm.sh 2>&1 | tee -a $OUTPUT_FILE3

OUTPUT_FILE3="performance_trsm_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level3_trsm.sh 2>&1 | tee -a $OUTPUT_FILE3

OUTPUT_FILE3="performance_trsmBatched_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level3_trsmBatched.sh 2>&1 | tee -a $OUTPUT_FILE3