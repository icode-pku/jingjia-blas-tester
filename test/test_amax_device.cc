#include "test.hh"
#include "cblas_wrappers.hh"
#include "lapack_wrappers.hh"
#include "blas/flops.hh"
#include "print_matrix.hh"

// -----------------------------------------------------------------------------
template <typename T>
void test_amax_device_work( Params& params, bool run )
{
    using namespace testsweeper;
    using real_t   = blas::real_type< T >;

    // get & mark input values
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx();
    int64_t verbose = params.verbose();
    int64_t device  = params.device();
    int64_t testcase    = params.testcase();
    // mark non-standard output values
    params.gflops();
    params.gbytes();
    params.ref_time();
    params.ref_gflops();
    params.ref_gbytes();
    params.runs();
    params.iscorrect();

    // adjust header to msec
    params.time.name( "time (ms)" );
    params.ref_time.name( "ref time (ms)" );
    params.ref_time.width( 13 );

    if (! run)
        return;

    if(1==params.iscorrect()){
        params.gflops.used(false);
        params.gbytes.used(false);
        params.ref_time.used(false);
        params.ref_gflops.used(false);
        params.ref_gbytes.used(false);
        params.time.used(false);
        params.runs.used(false);
    }
    else{
        params.okay.used(false);
        params.error.used(false);
    }

    if (blas::get_device_count() == 0) {
        params.msg() = "skipping: no GPU devices or no GPU support";
        return;
    }
    // setup
    size_t size_x = (n - 1) * std::abs(incx) + 1;
    T* x = new T[ size_x ];

    blas::Queue queue(device);
    T* dx;
    dx = blas::device_malloc<T>(size_x, queue);
    int64_t result=0;
    //init data
    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_x, x );
    blas::device_copy_vector(n, x, std::abs(incx), dx, std::abs(incx), queue);
    queue.sync();

    // test error exits
    if(testcase==0){
        char *error_name = (char *)malloc(sizeof(char)*35);
        int all_testcase = 0;
        int passed_testcase = 0;
        int failed_testcase = 0;
        //Test case 1: Test result is an nullptr
        blas::amax( n, dx, incx, nullptr, queue, testcase, error_name);
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //Test case 2: Test n is 0
        blas::amax( 0, dx, incx, &result, queue, testcase, error_name);
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_SUCCESS", all_testcase, passed_testcase, failed_testcase)&&result==0, error_name);
        //Test case 3: Test n is -1
        blas::amax( -1, dx, incx, &result, queue, testcase, error_name);
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_SUCCESS", all_testcase, passed_testcase, failed_testcase)&&result==0, error_name);
        //Test case 4: Test incx is 0
        blas::amax( n, dx, 0, &result, queue, testcase, error_name);
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_SUCCESS", all_testcase, passed_testcase, failed_testcase)&&result==0, error_name);
        //Test case 5: Test incx is -1
        blas::amax( n, dx, -1, &result, queue, testcase, error_name);
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_SUCCESS", all_testcase, passed_testcase, failed_testcase)&&result==0, error_name);

        queue.sync();
        params.Totalcase()+=all_testcase;
        params.Passedcase()+=passed_testcase;
        params.Failedcase()+=failed_testcase;
        //printf("All Test Cases: %d  Passed Cases: %d  Failed Cases: %d\n",all_testcase, passed_testcase, failed_testcase);
        free(error_name);
    }
    else{
        if (verbose >= 1) {
            printf( "\n"
                    "x n=%5lld, inc=%5lld, size=%10lld\n",
                    llong( n ), llong( incx ), llong( size_x ) );
        }
        if (verbose >= 2) {
            printf( "x = " ); print_vector( n, x, incx );
        }

        // run test
        testsweeper::flush_cache( params.cache() );
        blas::amax( n, dx, incx, &result, queue);
        queue.sync();
        // blas::device_copy_vector(1, &result, 1, &result_host, 1, queue);
        // queue.sync();

        double gflop = blas::Gflop< T >::iamax( n );
        double gbyte = blas::Gbyte< T >::iamax( n );

        if (verbose >= 1) {
            printf( "result = %5lld\n", llong( result ) );
        }
        double time;
        if (params.check() == 'y') {
            // run reference
            testsweeper::flush_cache( params.cache() );
            time = get_wtime();
            int64_t ref = cblas_iamax( n, x, incx );
            time = get_wtime() - time;
            if(n>0 && incx>0) ref += 1;
            //printf( "result_dev = %5lld cblas= %5lld\n", llong( result ),llong(ref) );
            if(params.iscorrect()==0){
                params.ref_time()   = time * 1000;  // msec
                params.ref_gflops() = gflop / time;
                params.ref_gbytes() = gbyte / time;
            }
            if (verbose >= 1) {
                printf( "ref    = %5lld\n", llong( ref ) );
            }

            if(params.iscorrect()==1){
                // error = |ref - result|
                real_t error = std::abs( ref - result );
                params.error() = error;

                // iamax must be exact!
                params.okay() = (error == 0);
            }
        }
        if(params.iscorrect()==0){
            int runs = params.runs();
            double stime;
            double all_time=0.0f;
            for(int i = 0; i < runs; i++){
                testsweeper::flush_cache( params.cache() );
                stime = get_wtime();
                blas::amax( n, dx, incx, &result, queue);
                queue.sync();
                all_time += (get_wtime() - stime);
            }
            all_time/=(double)runs;
            params.time()   = all_time * 1000;  // msec
            params.gflops() = gflop / all_time;
            params.gbytes() = gbyte / all_time;
        }
    }
    delete[] x;
    blas::device_free(dx, queue);
}

// -----------------------------------------------------------------------------
void test_amax_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_amax_device_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_amax_device_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_amax_device_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_amax_device_work< std::complex<double> >( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
