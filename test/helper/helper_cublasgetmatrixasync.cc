#include "../helper.hh"

void helper_cublasGetMatrixAsync()
{
    TestId CaseId(6, std::string("cublasGetMatrixAsync(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb, cudaStream_t stream)"));
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

    float *A, *Acopy;
    A = new float[rows * cols];
    Acopy = new float[rows * cols];
    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    //init data
    lapack_larnv( idist, iseed, rows * cols, A );

    float *dA;
    HelperSafeCall(cudaMalloc((float**)&dA, rows * cols * sizeof(float)));

    HelperSafeCall(cudaMemcpy(dA, A, rows * cols * sizeof(float),cudaMemcpyHostToDevice));

    //test case 1: legal parameters
    CaseId.TestProblemHeader(0, true);
    stat = cublasGetMatrixAsync(rows, cols, elemSize, dA, lda, Acopy, ldb, stream);
    HelperTestCall("cublasGetMatrixAsync", check_return_status(stat, "CUBLAS_STATUS_SUCCESS", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_SUCCESS");
    HelperSafeCall(cudaStreamSynchronize(stream));

    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            if(A[i*lda+j]!=Acopy[i*ldb+j]){
                printf("cublasGetMatrixAsync error!\n");
                Passed_tests--;
                Failed_tests++;
                break;
            }
        }
    }

    //test case 2: Testing for illegal parameter rows
    CaseId.TestProblemHeader(0, false, "-1");
    stat = cublasGetMatrixAsync(-1, cols, elemSize, dA, lda, Acopy, ldb, stream);
    HelperTestCall("cublasGetMatrixAsync", check_return_status(stat , "CUBLAS_STATUS_INVALID_VALUE", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_INVALID_VALUE");
    HelperSafeCall(cudaStreamSynchronize(stream));

    //test case 3: Testing for illegal parameter cols
    CaseId.TestProblemHeader(1, false, "-1");
    stat = cublasGetMatrixAsync(rows, -1, elemSize, dA, lda, Acopy, ldb, stream);
    HelperTestCall("cublasGetMatrixAsync", check_return_status(stat, "CUBLAS_STATUS_INVALID_VALUE", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_INVALID_VALUE");
    HelperSafeCall(cudaStreamSynchronize(stream));

    //test case 4: Testing for illegal parameter elemSize
    CaseId.TestProblemHeader(2, false, "-1");
    stat = cublasGetMatrixAsync(rows, cols, -1, dA, lda, Acopy, ldb, stream);
    HelperTestCall("cublasGetMatrixAsync", check_return_status(stat, "CUBLAS_STATUS_INVALID_VALUE", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_INVALID_VALUE");
    HelperSafeCall(cudaStreamSynchronize(stream));

    //test case 5: Testing for illegal parameter lda
    CaseId.TestProblemHeader(4, false, "-1");
    stat = cublasGetMatrixAsync(rows, cols, elemSize, dA, -1, Acopy, ldb, stream);
    HelperTestCall("cublasGetMatrixAsync", check_return_status(stat, "CUBLAS_STATUS_INVALID_VALUE", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_INVALID_VALUE");
    HelperSafeCall(cudaStreamSynchronize(stream));

    //test case 6: Testing for illegal parameter ldb
    CaseId.TestProblemHeader(6, false, "-1");
    stat = cublasGetMatrixAsync(rows, cols, elemSize, dA, lda, Acopy, -1, stream);
    HelperTestCall("cublasGetMatrixAsync", check_return_status(stat, "CUBLAS_STATUS_INVALID_VALUE", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_INVALID_VALUE");
    HelperSafeCall(cudaStreamSynchronize(stream));

    printf("All test cases: %d Passed test cases: %d Failed test cases: %d\n", All_tests, Passed_tests, Failed_tests);

    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    cudaFree(dA);
    delete[] A;
    delete[] Acopy;
}
