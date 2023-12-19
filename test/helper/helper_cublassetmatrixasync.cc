#include "../helper.hh"

void helper_cublasSetMatrixAsync()
{
    TestId CaseId(6, std::string("cublasSetMatrixAsync(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb, cudaStream_t stream)"));
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

    int64_t rows = 100;
    int64_t cols = 100;
    int64_t elemSize = sizeof(float);
    int64_t lda = rows;
    int64_t ldb = rows;
    size_t size_A = rows * cols;

    float *A, *Acopy;
    A = new float[size_A];
    Acopy = new float[size_A];
    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    //init data
    lapack_larnv( idist, iseed, size_A, A );

    float *dA;
    HelperSafeCall(cudaMalloc((float**)&dA, size_A * sizeof(float)));

    //test case 1: legal parameters
    CaseId.TestProblemHeader(0, true);
    stat = cublasSetMatrixAsync(rows, cols, elemSize, A, lda, dA, ldb, stream);
    HelperTestCall("cublasSetMatrixAsync", check_return_status(stat, "CUBLAS_STATUS_SUCCESS", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_SUCCESS");
    HelperSafeCall(cudaStreamSynchronize(stream));

    HelperSafeCall(cudaMemcpy(Acopy, dA, size_A * sizeof(float), cudaMemcpyDeviceToHost));
    if(strcmp(blas::device_errorstatus_to_string(stat),"CUBLAS_STATUS_SUCCESS")==0){
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++) {
            if (A[i * lda + j] != Acopy[i * ldb + j]) {
                printf("cublasSetMatrix error\n");
                Passed_tests--;
                Failed_tests++;
                break;
            }
        }
    }
    }
    //test case 2: Testing for illegal parameter rows
    CaseId.TestProblemHeader(0, false, "-1");
    stat = cublasSetMatrixAsync(-1, cols, elemSize, A, lda, dA, ldb, stream);
    HelperTestCall("cublasSetMatrixAsync", check_return_status(stat , "CUBLAS_STATUS_INVALID_VALUE", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_INVALID_VALUE");
    HelperSafeCall(cudaStreamSynchronize(stream));

    //test case 3: Testing for illegal parameter cols
    CaseId.TestProblemHeader(1, false, "-1");
    stat = cublasSetMatrixAsync(rows, -1, elemSize, A, lda, dA, ldb, stream);
    HelperTestCall("cublasSetMatrixAsync", check_return_status(stat, "CUBLAS_STATUS_INVALID_VALUE", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_INVALID_VALUE");
    HelperSafeCall(cudaStreamSynchronize(stream));

    //test case 4: Testing for illegal parameter elemSize
    CaseId.TestProblemHeader(2, false, "-1");
    stat = cublasSetMatrixAsync(rows, cols, -1, A, lda, dA, ldb, stream);
    HelperTestCall("cublasSetMatrixAsync", check_return_status(stat, "CUBLAS_STATUS_INVALID_VALUE", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_INVALID_VALUE");
    HelperSafeCall(cudaStreamSynchronize(stream));

    //test case 5: Testing for illegal parameter lda
    CaseId.TestProblemHeader(4, false, "-1");
    stat = cublasSetMatrixAsync(rows, cols, elemSize, A, -1, dA, ldb, stream);
    HelperTestCall("cublasSetMatrixAsync", check_return_status(stat, "CUBLAS_STATUS_INVALID_VALUE", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_INVALID_VALUE");
    HelperSafeCall(cudaStreamSynchronize(stream));

    //test case 6: Testing for illegal parameter ldb
    CaseId.TestProblemHeader(6, false, "-1");
    stat = cublasSetMatrixAsync(rows, cols, elemSize, A, lda, dA, -1, stream);
    HelperTestCall("cublasSetMatrixAsync", check_return_status(stat, "CUBLAS_STATUS_INVALID_VALUE", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_INVALID_VALUE");
    HelperSafeCall(cudaStreamSynchronize(stream));

    printf("All test cases: %d Passed test cases: %d Failed test cases: %d\n", All_tests, Passed_tests, Failed_tests);

    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    cudaFree(dA);
    delete[] A;
    delete[] Acopy;
}