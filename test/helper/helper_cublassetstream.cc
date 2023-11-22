#include "../helper.hh"

void helper_cublasSetStream()
{
    int All_tests = 0;
    int Passed_tests = 0;
    int Failed_tests = 0;
    bool status;
    cublasStatus_t stat;

    printf("This routine will test cublasSetStream API\n");
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

    //test case 1: Test the return status when cublas is not initialized
    //status = check_return_status("cublasSetStream", cublasSetStream( handle, ), "CUBLAS_STATUS_NOT_INITIALIZED", All_tests, Passed_tests, Failed_tests);

    //test case 2: This is a legal parameter input
    stat = cublasSetStream( handle, stream );
    HelperTestCall("cublasSetStream", check_return_status(stat, "CUBLAS_STATUS_SUCCESS", All_tests, Passed_tests, Failed_tests), stat);
    
    printf("All test cases: %d Passed test cases: %d Failed test cases: %d\n", All_tests, Passed_tests, Failed_tests);


    cublasDestroy(handle);
    cudaStreamDestroy(stream);
}