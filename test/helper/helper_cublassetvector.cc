#include "../helper.hh"

void helper_cublasSetVector()
{
    TestId CaseId(7, std::string("cublasSetVector(int n, int elemSize, const void *x, int incx, void *y, int incy)"));
    CaseId.TestApiHeader();
    int All_tests = 0;
    int Passed_tests = 0;
    int Failed_tests = 0;
    bool status;
    cublasStatus_t stat;
    //printf("This routine will test cublasSetVector API\n");
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
    //malloc gpu memory
    float *dx;
    HelperSafeCall(cudaMalloc((float**)&dx, Msize_y*sizeof(float)));

    //test case 1 to 4: legal parameters
    // incx =1  incy=1
    // incx =1  incy=2
    // incx =2  incy=1
    // incy =2  incy=2
    for(int i=0; i<2; i++){
        for(int j=0; j<2;j++){
            CaseId.TestProblemHeader(0, true);

            size_x = (n - 1) * std::abs(incx[i]) + 1;
            size_y = (n - 1) * std::abs(incy[j]) + 1;
            stat = cublasSetVector(n, sizeof(float), x, incx[i], dx, incy[j]);
            HelperTestCall("cublasSetVector", check_return_status(stat, "CUBLAS_STATUS_SUCCESS", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_SUCCESS");
            HelperSafeCall(cudaMemcpy(xcopy, dx, size_y*sizeof(float),cudaMemcpyDeviceToHost));
            //HelperSafeCall(cublasGetVector(size_x, sizeof(float), dx, incx, xcopy, incx));
            if(strcmp(blas::device_errorstatus_to_string(stat),"CUBLAS_STATUS_SUCCESS")==0){
            for(int k=0; k<n; k++){
                if(x[k*incx[i]]!=xcopy[k*incy[j]]){
                    printf("cublasSetVector or cublasGetVector error\n");
                    Passed_tests--;
                    Failed_tests++;
                    break;
                }
            }
            }
        }
    }
    size_x = (n - 1) * std::abs(incx[0]) + 1;
    //test case 5: Testing for illegal parameter elemSize
    CaseId.TestProblemHeader(1, false, "-1");
    stat = cublasSetVector(size_x, -1, x, incx[0], dx, incy[0]);
    HelperTestCall("cublasSetVector", check_return_status(stat , "CUBLAS_STATUS_INVALID_VALUE", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_INVALID_VALUE");

    //test case 6: Testing for illegal parameter incx
    CaseId.TestProblemHeader(3, false, "-1");
    stat = cublasSetVector(size_x, sizeof(float), x, -1, dx, incy[0]);
    HelperTestCall("cublasSetVector", check_return_status(stat, "CUBLAS_STATUS_INVALID_VALUE", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_INVALID_VALUE");

    //test case 7: Testing for illegal parameter incy
    CaseId.TestProblemHeader(5, false, "-1");
    stat = cublasSetVector(size_x, sizeof(float), x, incx[0], dx, -1);
    HelperTestCall("cublasSetVector", check_return_status(stat, "CUBLAS_STATUS_INVALID_VALUE", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_INVALID_VALUE");


    printf("All test cases: %d Passed test cases: %d Failed test cases: %d\n", All_tests, Passed_tests, Failed_tests);
    

    cublasDestroy(handle);
    cudaStreamDestroy(stream);
}