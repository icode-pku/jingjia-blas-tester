#include "../helper.hh"

void helper_cublasGetVectorAsync()
{
    TestId CaseId(7, std::string("cublasGetVectorAsync(int n, int elemSize, const void *x, int incx, void *y, int incy, cudaStream_t stream)"));
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
    int64_t incx[2] = {1,2};
    int64_t incy[2] = {1,2};
    size_t Msize_x = (n - 1) * std::max(std::abs(incx[0]),std::abs(incx[1])) + 1;
    size_t Msize_y = (n - 1) * std::max(std::abs(incy[0]),std::abs(incy[1])) + 1;
    size_t size_x=1, size_y=1;


    float *x, *xcopy;
    x = new float[ Msize_x ];
    xcopy = new float[ Msize_y ];
    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    //init data
    lapack_larnv( idist, iseed, Msize_x, x );

    float *dx;
    HelperSafeCall(cudaMalloc((float**)&dx, Msize_x*sizeof(float)));



    //test case 1 to 4: legal parameters
    // incx =1  incy=1
    // incx =1  incy=2
    // incx =2  incy=1
    // incy =2  incy=2
    for(int i=0; i<2; i++){
        for(int j=0; j<2;j++){
            size_x = (n - 1) * std::abs(incx[i]) + 1;
            size_y = (n - 1) * std::abs(incy[j]) + 1;
            HelperSafeCall(cudaMemcpy(dx, x, size_x*sizeof(float),cudaMemcpyHostToDevice));

            //test case 1: legal parameters
            CaseId.TestProblemHeader(0, true);
            stat = cublasGetVectorAsync(n, sizeof(float), dx, incx[i], xcopy, incy[j], stream);
            HelperTestCall("cublasGetVectorAsync", check_return_status(stat, "CUBLAS_STATUS_SUCCESS", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_SUCCESS");
            HelperSafeCall(cudaStreamSynchronize(stream));
            if(strcmp(blas::device_errorstatus_to_string(stat),"CUBLAS_STATUS_SUCCESS")==0){
            for(int k=0; k<n; k++){
                if(x[k*incx[i]]!=xcopy[k*incy[j]]){
                    printf("cublasGetVectorAsync error!\n");
                    Passed_tests--;
                    Failed_tests++;
                    break;
                }
            }
            }
            //end for
        }
    }
    size_x = (n - 1) * std::abs(incx[0]) + 1;
    //test case 5: Testing for illegal parameter elemSize
    CaseId.TestProblemHeader(1,false, "-1");
    stat = cublasGetVectorAsync(size_x, -1, dx, incx[0], xcopy, incy[0], stream);
    HelperTestCall("cublasGetVectorAsync", check_return_status(stat , "CUBLAS_STATUS_INVALID_VALUE", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_INVALID_VALUE");
    HelperSafeCall(cudaStreamSynchronize(stream));

    //test case 6: Testing for illegal parameter incx
    CaseId.TestProblemHeader(3, false, "-1");
    stat = cublasGetVectorAsync(size_x, sizeof(float), dx, -1, xcopy, incy[0], stream);
    HelperTestCall("cublasGetVectorAsync", check_return_status(stat, "CUBLAS_STATUS_INVALID_VALUE", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_INVALID_VALUE");
    HelperSafeCall(cudaStreamSynchronize(stream));

    //test case 7: Testing for illegal parameter incy
    CaseId.TestProblemHeader(5, false, "-1");
    stat = cublasGetVectorAsync(size_x, sizeof(float), dx, incx[0], xcopy, -1, stream);
    HelperTestCall("cublasGetVectorAsync", check_return_status(stat, "CUBLAS_STATUS_INVALID_VALUE", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_INVALID_VALUE");
    HelperSafeCall(cudaStreamSynchronize(stream));

    printf("All test cases: %d Passed test cases: %d Failed test cases: %d\n", All_tests, Passed_tests, Failed_tests);
    

    cublasDestroy(handle);
    cudaStreamDestroy(stream);
}