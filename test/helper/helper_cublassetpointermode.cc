#include "../helper.hh"

void helper_cublasSetPointerMode()
{
    TestId CaseId(3, std::string("cublasSetPointerMode(cublasHandle_t handle, cublasPointerMode_t mode)"));
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
    
    //test case 1:Test parameter 2 is an invalid parameter
    CaseId.TestProblemHeader(1, false, "-1");
    stat = cublasSetPointerMode(handle, (cublasPointerMode_t)-1);
    HelperTestCall("cublasSetPointerMode", check_return_status(stat, "CUBLAS_STATUS_INVALID_VALUE", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_INVALID_VALUE");
    //test case 2:This is a legal parameter input: mode is CUBLAS_POINTER_MODE_HOST
    CaseId.TestProblemHeader(0, true);
    stat = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST) ;
    HelperTestCall("cublasSetPointerMode", check_return_status(stat, "CUBLAS_STATUS_SUCCESS", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_SUCCESS");
    //test case 3:This is a legal parameter input: mode is CUBLAS_POINTER_MODE_DEVICE
    CaseId.TestProblemHeader(0, true);
    int64_t n = 100;
    int64_t incx = 1;
    size_t size_x = (n - 1) * std::abs(incx) + 1;
    float *x;
    x = new float[ size_x ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_x, x );
    float *dx;
    HelperSafeCall(cudaMalloc((float**)&dx, size_x*sizeof(float)) );   
    HelperSafeCall(cudaMemcpy(dx, x, size_x*sizeof(float),cudaMemcpyHostToDevice));
    
    float result_host;
    float result_device;
    float *result_h;

    result_host = cblas_asum(n, x, incx);
   
    HelperSafeCall(cudaMalloc((float**)&result_h, sizeof(float)) );
    HelperSafeCall(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE) );
    HelperSafeCall(
        cublasSasum(handle, n, dx, incx, result_h)
    );
    HelperSafeCall(cudaStreamSynchronize(stream));

    HelperSafeCall(cudaMemcpy(&result_device, result_h, sizeof(float), cudaMemcpyDeviceToHost));
    
    float error = std::abs( (result_host - result_device) / (n * result_host) );
    float u = 0.5 * std::numeric_limits< float >::epsilon();
    if(error >= u) {printf("cublasSetPointerMode error!\n");}
    
    HelperTestCall("cublasSetPointerMode", check_return_status(stat, "CUBLAS_STATUS_SUCCESS", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_SUCCESS");


    printf("All test cases: %d Passed test cases: %d Failed test cases: %d\n", All_tests, Passed_tests, Failed_tests);
    
    cublasDestroy(handle);
    cudaStreamDestroy(stream);
}