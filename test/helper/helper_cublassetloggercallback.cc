#include "../helper.hh"

void helper_cublasSetLoggerCallback()
{
    TestId CaseId(1, std::string("cublasSetLoggerCallback(cublasLogCallback userCallback)"));
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

    cublasLogCallback calfunc;

    //test case 1: legal parameters
    CaseId.TestProblemHeader(0, true);
    stat = cublasSetLoggerCallback(calfunc);
    HelperTestCall("cublasSetLoggerCallback", check_return_status(stat, "CUBLAS_STATUS_SUCCESS", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_SUCCESS");

    printf("All test cases: %d Passed test cases: %d Failed test cases: %d\n", All_tests, Passed_tests, Failed_tests);
    

    cublasDestroy(handle);
    cudaStreamDestroy(stream);
}