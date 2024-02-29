#include "../helper.hh"

void helper_cublasSetMatrix()
{
    TestId CaseId(9, std::string("cublasSetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb)"));
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

    int64_t rows = 200;
    int64_t cols = 100;
    int64_t elemSize = sizeof(float);
    int64_t lda[2] = {rows, rows+10};
    int64_t ldb[2] = {rows, rows+10};
    size_t size_A = std::max(lda[0], lda[1]) * cols;
    size_t size_B = std::max(ldb[0], ldb[1]) * cols;

    float *A, *Acopy;
    A = new float[size_A];
    Acopy = new float[size_B];
    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    //init data
    lapack_larnv( idist, iseed, size_A, A );

    float *dA;
    HelperSafeCall(cudaMalloc((float**)&dA, size_B * sizeof(float)));

    //test case 1 to 4: legal parameters
    // lda = row  ldb=row
    // lda = row  ldb!=row
    // lda != row  ldb=row
    // lda != row  ldb !=row
    for(int aid=0; aid<2; aid++){
        for(int bid=0; bid<2; bid++){
            
            CaseId.TestProblemHeader(0, true);
            stat = cublasSetMatrix(rows, cols, elemSize, A, lda[aid], dA, ldb[bid]);
            HelperTestCall("cublasSetMatrix", check_return_status(stat, "CUBLAS_STATUS_SUCCESS", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_SUCCESS");

            HelperSafeCall(cudaMemcpy(Acopy, dA, size_B * sizeof(float), cudaMemcpyDeviceToHost));
            if(strcmp(blas::device_errorstatus_to_string(stat),"CUBLAS_STATUS_SUCCESS")==0){
            for (int i = 0; i < cols; i++) {
                for (int j = 0; j < rows; j++) {
                    if (A[i * lda[aid] + j] != Acopy[i * ldb[bid] + j]) {
                        printf("cublasSetMatrix error\n");
                        Passed_tests--;
                        Failed_tests++;
                        break;
                    }
                }
            }
            }
            //end for
        }
    }
    //test case 5: Testing for illegal parameter rows
    CaseId.TestProblemHeader(0, false, "-1");
    stat = cublasSetMatrix(-1, cols, elemSize, A, lda[0], dA, ldb[0]);
    HelperTestCall("cublasSetMatrix", check_return_status(stat, "CUBLAS_STATUS_INVALID_VALUE", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_INVALID_VALUE");

    //test case 6: Testing for illegal parameter cols
    CaseId.TestProblemHeader(1, false, "-1");
    stat = cublasSetMatrix(rows, -1, elemSize, A, lda[0], dA, ldb[0]);
    HelperTestCall("cublasSetMatrix", check_return_status(stat, "CUBLAS_STATUS_INVALID_VALUE", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_INVALID_VALUE");

    //test case 7: Testing for illegal parameter elemSize
    CaseId.TestProblemHeader(2, false, "-1");
    stat = cublasSetMatrix(rows, cols, -1, A, lda[0], dA, ldb[0]);
    HelperTestCall("cublasSetMatrix", check_return_status(stat, "CUBLAS_STATUS_INVALID_VALUE", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_INVALID_VALUE");

    //test case 8: Testing for illegal parameter lda
    CaseId.TestProblemHeader(4, false, "-1");
    stat = cublasSetMatrix(rows, cols, elemSize, A, -1, dA, ldb[0]);
    HelperTestCall("cublasSetMatrix", check_return_status(stat, "CUBLAS_STATUS_INVALID_VALUE", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_INVALID_VALUE");

    //test case 9: Testing for illegal parameter ldb
    CaseId.TestProblemHeader(6, false, "-1");
    stat = cublasSetMatrix(rows, cols, elemSize, A, lda[0], dA, -1);
    HelperTestCall("cublasSetMatrix", check_return_status(stat, "CUBLAS_STATUS_INVALID_VALUE", All_tests, Passed_tests, Failed_tests), stat, "CUBLAS_STATUS_INVALID_VALUE");

    printf("All test cases: %d Passed test cases: %d Failed test cases: %d\n", All_tests, Passed_tests, Failed_tests);

    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    cudaFree(dA);
    delete[] A;
    delete[] Acopy;
}