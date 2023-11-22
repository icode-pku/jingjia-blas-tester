#include "../helper.hh"

void helper_test2()
{
    using blas::Op;
    using blas::Layout;
    printf("The helper API for this routine test is as follows:  \n \
            cublasCreate  cublasDestroy \n \
            cublasSetStream cublasGetStream\n \
            cublasSetMatrix  cublasGetMatrix\n \
            cublasSetMatrixAsync cublasGetMatrixAsync\n \
            cublasGetVersion \n \
            cublasGetProperty(MAJOR_VERSION, MINOR_VERSION, PATCH_LEVEL)\n \
            cublasGetStatusName cublasGetStatusString\n");
    int64_t device = 0;
    if (blas::get_device_count() == 0) {
        printf("skipping: no GPU devices or no GPU support\n");
        return;
    }
    int64_t m = 4;
    int64_t n = 4;
    int64_t k = 4;
    cublasOperation_t transA = cublasOperation_t::CUBLAS_OP_N;
    cublasOperation_t transB = cublasOperation_t::CUBLAS_OP_T;
    const float alpha = 3.14;
    const float beta = 2.71;
    //test cublasCreate(), cublasDestroy(), cublasSetStream()
    cudaStream_t stream = blas::stream_create(device);
    cublasHandle_t handle;
    HelperSafeCall(cublasCreate( &handle ));
    HelperSafeCall(cublasSetStream( handle, stream ) );

    // Set the pointer mode  test cublasSetPointerMode
    // HelperSafeCall(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST) );

    // Get the pointer mode, test cublasGetPointerMode
    // cublasPointerMode_t mode;
    // HelperSafeCall(cublasGetPointerMode(handle, &mode));

    //test cublasGetStream()
    cudaStream_t test_stream;
    HelperSafeCall(cublasGetStream( handle, &test_stream));

    //test cublasGetVersion()
    int version;
    HelperSafeCall(cublasGetVersion(handle, &version));
    printf("cublas version is: %d\n",version);

    //test cublasGetProperty
    int value;
    HelperSafeCall(cublasGetProperty(MAJOR_VERSION, &value));
    printf("Test cublasGetProperty MAJOR_VERSION value is: %d\n", value);
    HelperSafeCall(cublasGetProperty(MINOR_VERSION, &value));
    printf("Test cublasGetProperty MINOR_VERSION value is: %d\n", value);
    HelperSafeCall(cublasGetProperty(PATCH_LEVEL, &value));
    printf("Test cublasGetProperty PATCH_LEVEL value is: %d\n", value);

    //test cublasGetStatusName
    printf("Test cublasGetStatusName: %s\n",cublasGetStatusName(cublasGetProperty(PATCH_LEVEL, &value)));
    //test cublasGetStatusString
    printf("Test cublasGetStatusString: %s\n",cublasGetStatusString(cublasGetProperty(PATCH_LEVEL, &value)));

    size_t workspace_size = 128;
    float *workspace;
    HelperSafeCall(cudaMalloc((float**)&workspace, workspace_size*sizeof(float)) );
    cublasStatus_t err = cublasSetWorkspace(handle, (float*)workspace, workspace_size*sizeof(float));
    HelperSafeCall(err);
    printf("Test cublasSetWorkspace: %s\n",cublasGetStatusName(err));
    
    size_t size_A = m * k;
    size_t size_B = n * k;
    size_t size_C = m * n;

    int64_t lda = m;
    int64_t ldb = n;
    int64_t ldc = m;
    //host
    float *A  = new float[size_A];
    float *Ap = new float[size_A];
    float *B  = new float[size_B];
    float *C  = new float[size_C];
    float *Cref = new float[size_C];

    //device 
    float *dA, *dB, *dC;
    float *dAy;
    HelperSafeCall(cudaMalloc((float**)&dA, size_A*sizeof(float)) );
    HelperSafeCall(cudaMalloc((float**)&dB, size_B*sizeof(float)) );
    HelperSafeCall(cudaMalloc((float**)&dC, size_C*sizeof(float)) );
    HelperSafeCall(cudaMalloc((float**)&dAy, size_A*sizeof(float)) );

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    //init data
    lapack_larnv( idist, iseed, size_A, A );
    lapack_larnv( idist, iseed, size_B, B );
    lapack_larnv( idist, iseed, size_C, C );
    lapack_lacpy( "g", m, n, C, ldc, Cref, ldc );

    // norms for error check
    float work[1];
    float Anorm = lapack_lange( "f", m, k, A, lda, work );
    float Bnorm = lapack_lange( "f", n, k, B, ldb, work );
    float Cnorm = lapack_lange( "f", m, n, C, ldc, work );
    
    // HelperSafeCall(cudaMemcpy(dA, A, size_A*sizeof(float),cudaMemcpyHostToDevice));
    // HelperSafeCall(cudaMemcpy(dB, B, size_B*sizeof(float),cudaMemcpyHostToDevice));

    HelperSafeCall(cublasSetMatrix(m, k, sizeof(float), A, lda, dA, lda));
    HelperSafeCall(cublasSetMatrix(n, k, sizeof(float), B, ldb, dB, ldb));
    HelperSafeCall(cublasSetMatrix(m, n, sizeof(float), C, ldc, dC, ldc));
    //test cublasGetVector
    HelperSafeCall(cublasGetMatrix(m, k, sizeof(float), dA, lda, Ap, lda));
    for(int i=0; i<m; i++){
        for(int j=0; j<k; j++){
            if(A[j*m + k]!=Ap[j*m + k]){
                printf("cublasSetMatrix or cublasGetMatrix error\n");
                return;
            }
        }
    }
    memset(Ap, 0, m*k*sizeof(float));

    //test cublasSetMatrixAsync
    HelperSafeCall(cublasSetMatrixAsync(m, k, sizeof(float), A, lda, dAy, lda, test_stream));
    HelperSafeCall(cudaStreamSynchronize(test_stream));

    //test cublasGetMatrixAsync
    HelperSafeCall(cublasGetMatrixAsync(m, k, sizeof(float), dAy, lda, Ap, lda, test_stream));
    HelperSafeCall(cudaStreamSynchronize(test_stream));
    for(int i=0; i<m; i++){
        for(int j=0; j<k; j++){
            if(A[j*m + k]!=Ap[j*m + k]){
                printf("cublasSetMatrixAsync or cublasGetMatrixAsync error\n");
                return;
            }
        }
    }
    //run cublas api
    HelperSafeCall(
        cublasSgemm(handle, transA, transB, m, n, k, &alpha, dA, 
                    lda, dB, ldb, &beta, dC, ldc);
    );
    HelperSafeCall(cudaStreamSynchronize(test_stream));
    HelperSafeCall(cudaMemcpy(C, dC, m*n*sizeof(float), cudaMemcpyDeviceToHost));

    //run cblas api
    cblas_gemm(CblasColMajor, CblasNoTrans, CblasTrans, m, n, k, 
               alpha, A, lda, B, ldb, beta, Cref, ldc);

    float error=0;
    bool okay;
    check_gemm( m, n, k, alpha, beta, Anorm, Bnorm, Cnorm,
                   Cref, ldc, C, ldc, false, &error, &okay );

    printf("Error is: %f  ", error);
    if(okay) {printf("Result pass\n");}
    else {printf("Result is error\n");}
    
    delete [] A;
    delete [] B;
    delete [] C;
    delete [] Cref;
    delete [] Ap;

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dAy);
    //test cublasDestroy
    HelperSafeCall(cublasDestroy(handle));
    HelperSafeCall(cudaStreamDestroy(test_stream));
    return;
}

