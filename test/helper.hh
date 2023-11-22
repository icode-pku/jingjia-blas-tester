#ifndef HELPER_HH
#define HELPER_HH


#include "testsweeper.hh"
#include "blas.hh"

using llong = long long;

#include "cblas_wrappers.hh"
#include "lapack_wrappers.hh"
#include "blas/flops.hh"
#include "print_matrix.hh"
#include "check_gemm.hh"
#include  "../src/device_internal.hh"


void helper_cublasSetStream();
void helper_cublasGetStream();
void helper_cublasSetVector();



void helper_test1();
void helper_test2();


inline
bool check_return_status(cublasStatus_t device_status, const char* excpet_status, int &all_tests, int &passed_tests, int &failed_test)
{
    const char* device_status_name = blas::device_errorstatus_to_string(device_status);
    all_tests++;
    if(strcmp(device_status_name, excpet_status)==0){
        passed_tests++;
        return true;
    }
    //printf("API %s, Error status: %s Except error status: %s\n",api_name, device_status_name, excpet_status);
    failed_test++;
    return false;
}

#endif