#!/bin/bash

OUTPUT_FILE3="correctness_gemm_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level3_gemm.sh 2>&1 | tee -a $OUTPUT_FILE3

OUTPUT_FILE3="correctness_gemmBatched_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level3_gemmBatched.sh 2>&1 | tee -a $OUTPUT_FILE3

OUTPUT_FILE3="correctness_hemm_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level3_hemm.sh 2>&1 | tee -a $OUTPUT_FILE3

OUTPUT_FILE3="correctness_herk_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level3_herk.sh 2>&1 | tee -a $OUTPUT_FILE3

OUTPUT_FILE3="correctness_her2k_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level3_her2k.sh 2>&1 | tee -a $OUTPUT_FILE3

OUTPUT_FILE3="correctness_symm_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level3_symm.sh 2>&1 | tee -a $OUTPUT_FILE3

OUTPUT_FILE3="correctness_syrk_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level3_syrk.sh 2>&1 | tee -a $OUTPUT_FILE3

OUTPUT_FILE3="correctness_syr2k_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level3_syr2k.sh 2>&1 | tee -a $OUTPUT_FILE3

OUTPUT_FILE3="correctness_trmm_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level3_trmm.sh 2>&1 | tee -a $OUTPUT_FILE3

OUTPUT_FILE3="correctness_trsm_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level3_trsm.sh 2>&1 | tee -a $OUTPUT_FILE3

OUTPUT_FILE3="correctness_trsmBatched_`date +%Y_%m_%d_%H_%M_%S`.txt"
. test_level3_trsmBatched.sh 2>&1 | tee -a $OUTPUT_FILE3