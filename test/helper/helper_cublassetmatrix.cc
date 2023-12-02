#include "../helper.hh"

void helper_cublasSetMatrix()
{
    TestId CaseId(6, std::string("cublasSetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb)"));
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

    HelperSafeCall(cublasSetStream( handle, stream ));

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

    //test case 1: legal parameters
    CaseId.TestProblemHeader(0, true);
    stat = cublasSetMatrix(rows, cols, elemSize, A, lda, dA, ldb);
    HelperTestCall("cublasSetMatrix", check_return_status(stat, "CUBLAS_STATUS_SUCCESS", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_SUCCESS");

    HelperSafeCall(cudaMemcpy(Acopy, dA, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (A[i * cols + j] != Acopy[i * cols + j]) {
                printf("cublasSetMatrix error\n");
                Passed_tests--;
                Failed_tests++;
                break;
            }
        }
        if (Failed_tests > 0) break;
    }

    //test case 2: Testing for illegal parameter rows
    CaseId.TestProblemHeader(0, false, "-1");
    stat = cublasSetMatrix(-1, cols, elemSize, A, lda, dA, ldb);
    HelperTestCall("cublasSetMatrix", check_return_status(stat, "CUBLAS_STATUS_INVALID_VALUE", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_INVALID_VALUE");

    //test case 3: Testing for illegal parameter cols
    CaseId.TestProblemHeader(1, false, "-1");
    stat = cublasSetMatrix(rows, -1, elemSize, A, lda, dA, ldb);
    HelperTestCall("cublasSetMatrix", check_return_status(stat, "CUBLAS_STATUS_INVALID_VALUE", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_INVALID_VALUE");

    //test case 4: Testing for illegal parameter elemSize
    CaseId.TestProblemHeader(2, false, "-1");
    stat = cublasSetMatrix(rows, cols, -1, A, lda, dA, ldb);
    HelperTestCall("cublasSetMatrix", check_return_status(stat, "CUBLAS_STATUS_INVALID_VALUE", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_INVALID_VALUE");

    //test case 5: Testing for illegal parameter lda
    CaseId.TestProblemHeader(4, false, "-1");
    stat = cublasSetMatrix(rows, cols, elemSize, A, -1, dA, ldb);
    HelperTestCall("cublasSetMatrix", check_return_status(stat, "CUBLAS_STATUS_INVALID_VALUE", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_INVALID_VALUE");

    //test case 6: Testing for illegal parameter ldb
    CaseId.TestProblemHeader(6, false, "-1");
    stat = cublasSetMatrix(rows, cols, elemSize, A, lda, dA, -1);
    HelperTestCall("cublasSetMatrix", check_return_status(stat, "CUBLAS_STATUS_INVALID_VALUE", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_INVALID_VALUE");

    printf("All test cases: %d Passed test cases: %d Failed test cases: %d\n", All_tests, Passed_tests, Failed_tests);

    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    delete[] A;
    delete[] Acopy;
}