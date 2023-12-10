#!/bin/bash

HELPER_OUTPUT_FILE="helper_cublasgetloggercallback_`date +%Y_%m_%d_%H_%M_%S`.txt"
. helper_cublasgetloggercallback.sh 2>&1 | tee -a $HELPER_OUTPUT_FILE

HELPER_OUTPUT_FILE="helper_cublasgetmatrix_`date +%Y_%m_%d_%H_%M_%S`.txt"
. helper_cublasgetmatrix.sh 2>&1 | tee -a $HELPER_OUTPUT_FILE

HELPER_OUTPUT_FILE="helper_cublasgetmatrixasync_`date +%Y_%m_%d_%H_%M_%S`.txt"
. helper_cublasgetmatrixasync.sh 2>&1 | tee -a $HELPER_OUTPUT_FILE

HELPER_OUTPUT_FILE="helper_cublasgetpointermode_`date +%Y_%m_%d_%H_%M_%S`.txt"
. helper_cublasgetpointermode.sh 2>&1 | tee -a $HELPER_OUTPUT_FILE

HELPER_OUTPUT_FILE="helper_cublasgetstream_`date +%Y_%m_%d_%H_%M_%S`.txt"
. helper_cublasgetstream.sh 2>&1 | tee -a $HELPER_OUTPUT_FILE

HELPER_OUTPUT_FILE="helper_cublasgetvector_`date +%Y_%m_%d_%H_%M_%S`.txt"
. helper_cublasgetvector.sh 2>&1 | tee -a $HELPER_OUTPUT_FILE

HELPER_OUTPUT_FILE="helper_cublasgetvectorasync_`date +%Y_%m_%d_%H_%M_%S`.txt"
. helper_cublasgetvectorasync.sh 2>&1 | tee -a $HELPER_OUTPUT_FILE

HELPER_OUTPUT_FILE="helper_cublasloggeconfigure_`date +%Y_%m_%d_%H_%M_%S`.txt"
. helper_cublasloggerconfigure.sh 2>&1 | tee -a $HELPER_OUTPUT_FILE

HELPER_OUTPUT_FILE="helper_cublassetloggercallback_`date +%Y_%m_%d_%H_%M_%S`.txt"
. helper_cublassetloggercallback.sh 2>&1 | tee -a $HELPER_OUTPUT_FILE

HELPER_OUTPUT_FILE="helper_cublassetmatrix_`date +%Y_%m_%d_%H_%M_%S`.txt"
. helper_cublassetmatrix.sh 2>&1 | tee -a $HELPER_OUTPUT_FILE

HELPER_OUTPUT_FILE="helper_cublassetmatrixasync_`date +%Y_%m_%d_%H_%M_%S`.txt"
. helper_cublassetmatrixasync.sh 2>&1 | tee -a $HELPER_OUTPUT_FILE

HELPER_OUTPUT_FILE="helper_cublassetpointermode_`date +%Y_%m_%d_%H_%M_%S`.txt"
. helper_cublassetpointermode.sh 2>&1 | tee -a $HELPER_OUTPUT_FILE

HELPER_OUTPUT_FILE="helper_cublassetstream_`date +%Y_%m_%d_%H_%M_%S`.txt"
. helper_cublassetstream.sh 2>&1 | tee -a $HELPER_OUTPUT_FILE

HELPER_OUTPUT_FILE="helper_cublassetvector_`date +%Y_%m_%d_%H_%M_%S`.txt"
. helper_cublassetvector.sh 2>&1 | tee -a $HELPER_OUTPUT_FILE

HELPER_OUTPUT_FILE="helper_cublassetvectorasync_`date +%Y_%m_%d_%H_%M_%S`.txt"
. helper_cublassetvectorasync.sh 2>&1 | tee -a $HELPER_OUTPUT_FILE