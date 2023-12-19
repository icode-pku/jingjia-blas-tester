#include "../helper.hh"

void helper_cublasSetVectorAsync()
{
    TestId CaseId(4, std::string("cublasSetVectorAsync(int n, int elemSize, const void *x, int incx, void *y, int incy, cudaStream_t stream)"));
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
    
    int64_t n = 100;
    int64_t incx = 1;
    size_t size_x = (n - 1) * std::abs(incx) + 1;


    float *x, *xcopy;
    x = new float[ size_x ];
    xcopy = new float[ size_x ];
    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    //init data
    lapack_larnv( idist, iseed, size_x, x );

    float *dx;
    HelperSafeCall(cudaMalloc((float**)&dx, size_x*sizeof(float)));

    //test case 1: legal parameters
    CaseId.TestProblemHeader(0, true);
    stat = cublasSetVectorAsync(size_x, sizeof(float), x, incx, dx, incx, stream);
    HelperTestCall("cublasSetVectorAsync", check_return_status(stat, "CUBLAS_STATUS_SUCCESS", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_SUCCESS");
    HelperSafeCall(cudaStreamSynchronize(stream));

    HelperSafeCall(cudaMemcpy(xcopy, dx, size_x*sizeof(float),cudaMemcpyDeviceToHost));
    if(strcmp(blas::device_errorstatus_to_string(stat),"CUBLAS_STATUS_SUCCESS")==0){
    for(int i=0; i<n; i++){
        if(x[i*incx]!=xcopy[i*incx]){
            printf("cublasSetVectorAsync error!\n");
            Passed_tests--;
            Failed_tests++;
            break;
        }
    }
    }
    //test case 2: Testing for illegal parameter elemSize
    CaseId.TestProblemHeader(1,false, "-1");
    stat = cublasSetVectorAsync(size_x, -1, x, incx, dx, incx, stream);
    HelperTestCall("cublasSetVectorAsync", check_return_status(stat , "CUBLAS_STATUS_INVALID_VALUE", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_INVALID_VALUE");
    HelperSafeCall(cudaStreamSynchronize(stream));

    //test case 3: Testing for illegal parameter incx
    CaseId.TestProblemHeader(3, false, "-1");
    stat = cublasSetVectorAsync(size_x, sizeof(float), x, -1, dx, incx, stream);
    HelperTestCall("cublasSetVectorAsync", check_return_status(stat, "CUBLAS_STATUS_INVALID_VALUE", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_INVALID_VALUE");
    HelperSafeCall(cudaStreamSynchronize(stream));

    //test case 4: Testing for illegal parameter incy
    CaseId.TestProblemHeader(5, false, "-1");
    stat = cublasSetVectorAsync(size_x, sizeof(float), x, incx, dx, -1, stream);
    HelperTestCall("cublasSetVectorAsync", check_return_status(stat, "CUBLAS_STATUS_INVALID_VALUE", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_INVALID_VALUE");
    HelperSafeCall(cudaStreamSynchronize(stream));

    printf("All test cases: %d Passed test cases: %d Failed test cases: %d\n", All_tests, Passed_tests, Failed_tests);
    

    cublasDestroy(handle);
    cudaStreamDestroy(stream);
}