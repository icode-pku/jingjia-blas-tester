#include "test.hh"
#include "cblas_wrappers.hh"
#include "lapack_wrappers.hh"
#include "blas/flops.hh"
#include "print_matrix.hh"
#include "check_gemm.hh"

// -----------------------------------------------------------------------------
template <typename TA, typename TX>
void test_trmv_device_work( Params& params, bool run )
{

    using namespace testsweeper;
    using blas::Uplo;
    using blas::Op;
    using blas::Layout;
    using blas::Diag;
    using scalar_t = blas::scalar_type< TA, TX >;
    using real_t   = blas::real_type< scalar_t >;

    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Uplo uplo = params.uplo();
    blas::Op trans  = params.trans();
    blas::Diag diag = params.diag();
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx();
    int64_t device  = params.device();
    int64_t align   = params.align();
    int64_t verbose = params.verbose();
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

    // ----------
    // setup
    int64_t lda = roundup( n, align );
    size_t size_A = size_t(lda)*n;
    size_t size_x = (n - 1) * std::abs(incx) + 1;
    TA* A    = new TA[ size_A ];
    TX* x    = new TX[ size_x ];
    TX* xref = new TX[ size_x ];

    // device specifics
    blas::Queue queue( device );
    TA* dA;
    TX* dx;

    dA = blas::device_malloc<TA>( size_A, queue );
    dx = blas::device_malloc<TX>( size_x, queue );

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_A, A );  // TODO: generate
    lapack_larnv( idist, iseed, size_x, x );  // TODO
    cblas_copy( n, x, incx, xref, incx );

    blas::device_copy_matrix(n, n, A, lda, dA, lda, queue);
    blas::device_copy_vector(n, x, std::abs(incx), dx, std::abs(incx), queue);
    queue.sync();

    // norms for error check
    real_t work[1];
    real_t Anorm = lapack_lantr( "f", uplo2str(uplo), diag2str(diag),
                                 n, n, A, lda, work );
    real_t Xnorm = cblas_nrm2( n, x, std::abs(incx) );

    // test error exits
    if(testcase == 0){
        char *error_name = (char *)malloc(sizeof(char)*35);
        int all_testcase = 0;
        int passed_testcase = 0;
        int failed_testcase = 0;
        //case 1: Test uplo is an illegal value
        blas::trmv( layout,    Uplo(0), trans, diag,     n, dA, lda, dx, incx, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 2: Test trans is an illegal value
        blas::trmv( layout, uplo, Op(0), diag, n, dA, lda, dx, incx, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 3: Test diag is an illegal value
        blas::trmv( layout, uplo, trans, Diag(0), n, dA, lda, dx, incx, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 4: Test the return value when n is an illegal value
        blas::trmv( layout, uplo, trans, diag, -1, dA, lda, dx, incx, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        
        //case 5: Test the return value when lda is an illegal value and trans is NoTrans
        blas::trmv( Layout::ColMajor, uplo, Op::NoTrans, diag, n, dA, n-1, dx, incx, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 6: Test the return value when lda is an illegal value and trans is Trans
        blas::trmv( Layout::ColMajor, uplo, Op::Trans, diag, n, dA, n-1, dx, incx, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 7: Test the return value when lda is an illegal value and trans is ConjTrans
        blas::trmv( Layout::ColMajor, uplo, Op::NoTrans, diag, n, dA, n-1, dx, incx, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
    
        //case 8: Test the return value when incx is an illegal value
        blas::trmv( layout, uplo, trans, diag, n, dA, lda, dx, 0, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
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
                    "A n=%5lld, lda=%5lld, size=%10lld, norm=%.2e\n"
                    "x n=%5lld, inc=%5lld, size=%10lld, norm=%.2e\n",
                    llong( n ), llong( lda ),  llong( size_A ), Anorm,
                    llong( n ), llong( incx ), llong( size_x ), Xnorm );
        }
        if (verbose >= 2) {
            printf( "A = [];\n"    ); print_matrix( n, n, A, lda );
            printf( "x    = [];\n" ); print_vector( n, x, incx );
        }

        // run test
        testsweeper::flush_cache( params.cache() );
        //double time = get_wtime();
        blas::trmv( layout, uplo, trans, diag, n, dA, lda, dx, incx, queue );
        queue.sync();
        //time = get_wtime() - time;

        double gflop = blas::Gflop< scalar_t >::trmv( n );
        double gbyte = blas::Gbyte< scalar_t >::trmv( n );
        //params.time()   = time * 1000;  // msec
        //params.gflops() = gflop / time;
        //params.gbytes() = gbyte / time;
        blas::device_copy_vector(n, dx, std::abs(incx), x, std::abs(incx), queue);
        queue.sync();

        if (verbose >= 2) {
            printf( "x2   = [];\n" ); print_vector( n, x, incx );
        }

        double time;
        if (params.check() == 'y') {
            // run reference
            testsweeper::flush_cache( params.cache() );
            time = get_wtime();
            cblas_trmv( cblas_layout_const(layout),
                        cblas_uplo_const(uplo),
                        cblas_trans_const(trans),
                        cblas_diag_const(diag),
                        n, A, lda, xref, incx );
            time = get_wtime() - time;

            if(params.iscorrect()==0){
                params.ref_time()   = time * 1000;  // msec
                params.ref_gflops() = gflop / time;
                params.ref_gbytes() = gbyte / time;
            }

            if (verbose >= 2) {
                printf( "xref = [];\n" ); print_vector( n, xref, incx );
            }

            if(params.iscorrect()==1){
                // check error compared to reference
                // treat x as 1 x n matrix with ld = incx; k = n is reduction dimension
                // alpha = 1, beta = 0.
                real_t error;
                bool okay;
                check_gemm( 1, n, n, scalar_t(1), scalar_t(0), Anorm, Xnorm, real_t(0),
                            xref, std::abs(incx), x, std::abs(incx), verbose, &error, &okay );
                params.error() = error;
                params.okay() = okay;
            }
        }

        if(params.iscorrect()==0){
            int runs = params.runs();
            double stime;
            double all_time=0.0f;
            for(int i = 0; i < runs; i++){
                testsweeper::flush_cache( params.cache() );
                stime = get_wtime();
                blas::trmv( layout, uplo, trans, diag, n, dA, lda, dx, incx, queue );
                queue.sync();
                all_time += (get_wtime() - stime);
            }
            all_time/=(double)runs;
            params.time()   = all_time * 1000;  //msec
            params.gflops() = gflop / all_time;
            params.gbytes() = gbyte / all_time;
        }
    }

    delete[] A;
    delete[] x;
    delete[] xref;

    blas::device_free( dA, queue );
    blas::device_free( dx, queue );
}

// -----------------------------------------------------------------------------
void test_trmv_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_trmv_device_work< float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_trmv_device_work< double, double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_trmv_device_work< std::complex<float>, std::complex<float> >
                ( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_trmv_device_work< std::complex<double>, std::complex<double> >
                ( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
