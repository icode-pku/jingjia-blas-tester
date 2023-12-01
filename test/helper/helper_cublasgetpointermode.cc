#include "../helper.hh"

void helper_cublasGetPointerMode()
{
    TestId CaseId(2, std::string("cublasGetPointerMode(cublasHandle_t handle, cublasPointerMode_t *mode)"));
    CaseId.TestApiHeader();

    int All_tests = 0;
    int Passed_tests = 0;
    int Failed_tests = 0;
    bool status;
    cublasStatus_t stat;

    int64_t device = 0;
    if (blas::get_device_count() == 0) {
        printf("skipping: no GPU devices or no GPU support\n");
        return;
    }
    HelperSafeCall( cudaSetDevice( device ) );
    cudaStream_t stream;
    HelperSafeCall( cudaStreamCreate( &stream ) );

    cublasHandle_t handle;
    HelperSafeCall(cublasCreate( &handle ));
    
    HelperSafeCall(cublasSetStream( handle, stream));

    HelperSafeCall(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST) );
    
    //test case 1:Test parameter 2 is an invalid parameter  
    CaseId.TestProblemHeader(1, false, "NULL");
    stat = cublasGetPointerMode(handle, NULL);
    HelperTestCall("cublasGetPointerMode", check_return_status(stat, "CUBLAS_STATUS_INVALID_VALUE", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_INVALID_VALUE");
    //test case 2:This is a legal parameter input
    CaseId.TestProblemHeader(0, true);
    cublasPointerMode_t mode;
    stat = cublasGetPointerMode(handle, &mode);
    HelperTestCall("cublasGetPointerMode", check_return_status(stat, "CUBLAS_STATUS_SUCCESS", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_SUCCESS");
    if (mode != CUBLAS_POINTER_MODE_HOST){
        printf("cublasGetPointerMode error!\n");
    }

    printf("All test cases: %d Passed test cases: %d Failed test cases: %d\n", All_tests, Passed_tests, Failed_tests);
    
    cublasDestroy(handle);
    cudaStreamDestroy(stream);
}