#include "../helper.hh"

// typedef void (*test_log_func_ptr)(void* userData, cublasStatus_t status, const char* msg);


// void print(void* userData, cublasStatus_t status, const char* msg) {
//     // 在这里处理日志信息
//     printf("logging: %s\n", msg);
// }


void helper_test1()
{
    printf("The helper API for this routine test is as follows:  \n \
            cublasCreate  cublasDestroy \n \
            cublasSetStream cublasGetStream\n \
            cublasSetPointerMode(Device) \n \
            cublasSetPointerMode(Host) \n \
            cublasGetPointerMode \n \
            cublasSetVector  cublasGetVector\n \
            cublasSetVectorAsync cublasGetVectorAsync\n \
            cublasLoggerConfigure \n \
            cublasSetLoggerCallback \n \
            cublasGetLoggerCallback\n ");
    int64_t device = 0;
    if (blas::get_device_count() == 0) {
        printf("skipping: no GPU devices or no GPU support\n");
        return;
    }
    int64_t n = 100;
    int64_t incx = 1;
    //test cublasCreate(), cublasDestroy(), cublasSetStream()
    cudaStream_t stream = blas::stream_create(device);
    cublasHandle_t handle;
    HelperSafeCall(cublasCreate( &handle ));
    HelperSafeCall( cublasSetStream( handle, stream ) );


    // Set the pointer mode  test cublasSetPointerMode
    HelperSafeCall(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST) );

    // Get the pointer mode, test cublasGetPointerMode
    cublasPointerMode_t mode;
    HelperSafeCall(cublasGetPointerMode(handle, &mode));

    //set log
    cublasLogCallback calfunc;
    HelperSafeCall(cublasLoggerConfigure(1, 0, 0, "helper.log"));
    HelperSafeCall(cublasSetLoggerCallback(NULL));
    HelperSafeCall(cublasGetLoggerCallback(&calfunc));


    //test cublasGetStream()
    cudaStream_t test_stream;
    HelperSafeCall(cublasGetStream( handle, &test_stream));

    //set a queue
    //blas::Queue queue(device, handle);

    size_t size_x = (n - 1) * std::abs(incx) + 1;



    float *x, *xcopy, *xasync;
    x = new float[ size_x ];
    xcopy = new float[ size_x ];
    xasync = new float[ size_x ];

    //HelperSafeCall(cudaMallocManaged((float**)&x, size_x*sizeof(float)) );
    //HelperSafeCall(cudaMallocManaged((float**)&xcopy, size_x*sizeof(float)) );
    //HelperSafeCall(cudaMallocManaged((float**)&xasync, size_x*sizeof(float)) );
    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    //init data
    lapack_larnv( idist, iseed, size_x, x );

    float *dx, *dxp, *dasync, *dm;


    HelperSafeCall(cudaMalloc((float**)&dx, size_x*sizeof(float)) );
    HelperSafeCall(cudaMalloc((float**)&dxp, size_x*sizeof(float)) );
    HelperSafeCall(cudaMalloc((float**)&dasync, size_x*sizeof(float)) );


    
    HelperSafeCall(cudaMemcpy(dx, x, size_x*sizeof(float),cudaMemcpyHostToDevice));

    HelperSafeCall(cublasSetVector(size_x, sizeof(float), x, incx, dxp, incx));
    //test cublasGetVector
    HelperSafeCall(cublasGetVector(size_x, sizeof(float), dxp, incx, xcopy, incx));
    for(int i=0; i<n; i++){
        if(x[i*incx]!=xcopy[i*incx]){
            printf("cublasSetVector or cublasGetVector error\n");
            return;
        }
    }


    //test cublasSetVectorAsync
    HelperSafeCall(cublasSetVectorAsync(size_x, sizeof(float), x, incx, dasync, incx, test_stream));
    //queue.sync();
    HelperSafeCall(cudaStreamSynchronize(test_stream));
    //test cublasGetVectorAsync
    HelperSafeCall(cublasGetVectorAsync(size_x, sizeof(float), dasync, incx, xasync, incx, test_stream));
    HelperSafeCall(cudaStreamSynchronize(test_stream));
    //queue.sync();
    for(int i=0; i<n; i++){
        if(x[i*incx]!=xasync[i*incx]){
            printf("cublasSetVectorAsync or cublasGetVectorAsync error\n");
            return;
        }
    }

    float result_host;
    float result_device;
    //run cublas api
    // blas::asum( n, dx, incx, &result_device, queue);
    // queue.sync();
    HelperSafeCall(
        cublasSasum(handle, n, dx, incx, &result_device)
    );
    HelperSafeCall(cudaStreamSynchronize(test_stream));

    //run cblas api
    result_host = cblas_asum(n, x, incx);
    
    float error = std::abs( (result_host - result_device) / (n * result_host) );
    float u = 0.5 * std::numeric_limits< float >::epsilon();

    printf("Host pointer: cblas result is : %f  cublas result is: %f error is: %f  ",result_host,result_device, error);
    if(error < u) {printf("Result pass\n");}
    else {printf("Result is error\n");}
    
    //--------------------------------------DEVICE--------------
    float *result_h;
    HelperSafeCall(cudaMalloc((float**)&result_h, sizeof(float)) );
    // Set the pointer mode  test cublasSetPointerMode
    HelperSafeCall(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE) );
    HelperSafeCall(cublasGetStream( handle, &test_stream));
    HelperSafeCall(
        cublasSasum(handle, n, dx, incx, result_h)
    );
    HelperSafeCall(cudaStreamSynchronize(test_stream));

    HelperSafeCall(cudaMemcpy(&result_device, result_h, sizeof(float),cudaMemcpyDeviceToHost));
    
    float error2 = std::abs( (result_host - result_device) / (n * result_host) );
    //float u = 0.5 * std::numeric_limits< float >::epsilon();

    printf("Device pointer: cblas result is : %f  cublas result is: %f error is: %f ",result_host,result_device, error2);
    if(error < u) {printf("Result pass\n All helper api tests passed \n");}
    else {printf("Result is error\n");}
    //---------------------------------------END-----------------
    delete [] x;
    delete [] xcopy;
    delete [] xasync;

    cudaFree(dasync);
    cudaFree(dx);
    cudaFree(dxp);
    cudaFree(result_h);
    // blas::device_free(dasync, queue);
    // blas::device_free(dx, queue);
    //test cublasDestroy
    HelperSafeCall(cublasDestroy(handle));
    HelperSafeCall(cudaStreamDestroy(test_stream));
    return;
}

