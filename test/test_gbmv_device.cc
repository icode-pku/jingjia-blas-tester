#include "test.hh"
#include "cblas_wrappers.hh"
#include "lapack_wrappers.hh"
#include "blas/flops.hh"
#include "print_matrix.hh"
#include "check_gemm.hh"

// -----------------------------------------------------------------------------
template <typename TA, typename TX, typename TY>
void test_gbmv_device_work( Params& params, bool run )
{
    using namespace testsweeper;
    using std::real;
    using std::imag;
    using blas::Op;
    using blas::Layout;
    using scalar_t = blas::scalar_type< TA, TX, TY >;
    using real_t   = blas::real_type< scalar_t >;

    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Op trans  = params.trans();
    scalar_t alpha  = params.alpha();
    scalar_t beta   = params.beta();
    int64_t m       = params.dim.m();
    int64_t n       = params.dim.n();
    int64_t kl      = params.kl();
    int64_t ku      = params.ku();   
    int64_t incx    = params.incx();
    int64_t incy    = params.incy();
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

    // adjust header to msec
    params.time.name( "time (ms)" );
    params.ref_time.name( "ref time (ms)" );
    params.ref_time.width( 13 );

    if (! run)
        return;

    if (blas::get_device_count() == 0) {
        params.msg() = "skipping: no GPU devices or no GPU support";
        return;
    }

    // setup
    int64_t Am = (layout == Layout::ColMajor ? m : n);
    int64_t An = (layout == Layout::ColMajor ? n : m);
    int64_t lda = roundup( Am, align );
    int64_t Xm = (trans == Op::NoTrans ? n : m);
    int64_t Ym = (trans == Op::NoTrans ? m : n);
    size_t size_A = size_t(lda)*An;
    size_t size_x = (Xm - 1) * std::abs(incx) + 1;
    size_t size_y = (Ym - 1) * std::abs(incy) + 1;
    TA* A    = new TA[ size_A ];
    TX* x    = new TX[ size_x ];
    TY* y    = new TY[ size_y ];
    TY* yref = new TY[ size_y ];

    // device specifics
    blas::Queue queue( device );
    TA* dA;
    TX* dx;
    TY* dy;

    dA = blas::device_malloc<TA>( size_A, queue );
    dx = blas::device_malloc<TX>( size_x, queue );
    dy = blas::device_malloc<TY>( size_y, queue );

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_A, A );
    lapack_larnv( idist, iseed, size_x, x );
    lapack_larnv( idist, iseed, size_y, y );
    cblas_copy( Ym, y, incy, yref, incy );

    blas::device_copy_matrix(Am, An, A, lda, dA, lda, queue);
    blas::device_copy_vector(Xm, x, std::abs(incx), dx, std::abs(incx), queue);
    blas::device_copy_vector(Ym, y, std::abs(incy), dy, std::abs(incy), queue);
    queue.sync();

    // norms for error check
    real_t work[1];
    real_t Anorm = lapack_lange( "f", Am, An, A, lda, work );
    real_t Xnorm = cblas_nrm2( Xm, x, std::abs(incx) );
    real_t Ynorm = cblas_nrm2( Ym, y, std::abs(incy) );

    // test error exits
    assert_throw( blas::gbmv( Layout(0), trans,  m,  n, kl, ku, alpha, dA, lda, dx, incx, beta, dy, incy, queue ), blas::Error );
    assert_throw( blas::gbmv( layout,    Op(0),  m,  n, kl, ku, alpha, dA, lda, dx, incx, beta, dy, incy, queue ), blas::Error );
    assert_throw( blas::gbmv( layout,    trans, -1,  n, kl, ku, alpha, dA, lda, dx, incx, beta, dy, incy, queue ), blas::Error );
    assert_throw( blas::gbmv( layout,    trans,  m, -1, kl, ku, alpha, dA, lda, dx, incx, beta, dy, incy, queue ), blas::Error );
    assert_throw( blas::gbmv( layout,    trans,  m,  n, -1, ku, alpha, dA, lda, dx, incx, beta, dy, incy, queue ), blas::Error );
    assert_throw( blas::gbmv( layout,    trans,  m,  n, kl, -1, alpha, dA, lda, dx, incx, beta, dy, incy, queue ), blas::Error );

    assert_throw( blas::gbmv( layout, trans,  m,  n, kl, ku, alpha, dA, kl + ku, dx, incx, beta, dy, incy, queue ), blas::Error );

    assert_throw( blas::gbmv( layout,    trans,  m,  n, kl, ku, alpha, dA, lda, dx, 0,    beta, dy, incy, queue ), blas::Error );
    assert_throw( blas::gbmv( layout,    trans,  m,  n, kl, ku, alpha, dA, lda, dx, incx, beta, dy, 0,  queue   ), blas::Error );
    if(testcase == 0){
        char *error_name = (char *)malloc(sizeof(char)*35);
        int all_testcase = 0;
        int passed_testcase = 0;
        int failed_testcase = 0;
        //case 1: Test the return when trans is an illegal value
        blas::gbmv( layout, Op(0),  m,  n, kl, ku, alpha, dA, lda, dx, incx, beta, dy, incy, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 2: Test the return value when m is an illegal value
        blas::gbmv( layout, trans, -1,  n, kl, ku, alpha, dA, lda, dx, incx, beta, dy, incy, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 3: Test the return value when n is an illegal value
        blas::gbmv( layout, trans,  m, -1, kl, ku, alpha, dA, lda, dx, incx, beta, dy, incy, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 4: Test the return value when kl is an illegal value
        blas::gbmv( layout, trans,  m,  n, -1, ku, alpha, dA, lda, dx, incx, beta, dy, incy, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 5: Test the return value when ku is an illegal value
        blas::gbmv( layout, trans,  m,  n, kl, -1, alpha, dA, lda, dx, incx, beta, dy, incy, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        
        
        //case 6: Test the return value when lda is an illegal value and transA is NoTrans
        blas::gbmv( Layout::ColMajor, Op::NoTrans,  m,  n, kl, ku, alpha, dA, kl + ku, dx, incx, beta, dy, incy, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 7: Test the return value when lda is an illegal value and transA is Trans
        blas::gbmv( Layout::ColMajor, Op::Trans,  m,  n, kl, ku, alpha, dA, kl + ku, dx, incx, beta, dy, incy, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 8: Test the return value when lda is an illegal value and tansA is ConjTrans
        blas::gbmv( Layout::ColMajor, Op::ConjTrans,  m,  n, kl, ku, alpha, dA, kl + ku, dx, incx, beta, dy, incy, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);

        //case 9: Test the return value when row-major order, tansA=NoTrans, and lda is an illegal value
        blas::gbmv( Layout::RowMajor, Op::NoTrans,  m,  n, kl, ku, alpha, dA, kl + ku, dx, incx, beta, dy, incy, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 10: Test the return value when row-major order, tansA=Trans, and lda is an illegal value
        blas::gbmv( Layout::RowMajor, Op::Trans,  m,  n, kl, ku, alpha, dA, kl + ku, dx, incx, beta, dy, incy, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 11: Test the return value when row-major order, tansA=ConjTrans, and lda is an illegal value
        blas::gbmv( Layout::RowMajor, Op::ConjTrans,  m,  n, kl, ku, alpha, dA, kl + ku, dx, incx, beta, dy, incy, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);

        //case 12: Test the return value when incx is an illegal value
        blas::gbmv( layout, trans, m, n, kl, ku, alpha, dA, lda, dx, 0,    beta, dy, incy, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 13: Test the return value when incy is an illegal value
        blas::gbmv( layout, trans, m, n, kl, ku, alpha, dA, lda, dx, incx, beta, dy, 0,    queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        queue.sync();

        printf("All Test Cases: %d  Passed Cases: %d  Failed Cases: %d\n",all_testcase, passed_testcase, failed_testcase);

        free(error_name);
    }
    else{
        if (verbose >= 1) {
            printf( "\n"
                    "A Am=%5lld, An=%5lld, lda=%5lld, size=%10lld, norm=%.2e\n"
                    "x Xm=%5lld, inc=%5lld,           size=%10lld, norm=%.2e\n"
                    "y Ym=%5lld, inc=%5lld,           size=%10lld, norm=%.2e\n",
                    llong( Am ), llong( An ), llong( lda ), llong( size_A ), Anorm,
                    llong( Xm ), llong( incx ), llong( size_x ), Xnorm,
                    llong( Ym ), llong( incy ), llong( size_y ), Ynorm );
        }
        if (verbose >= 2) {
            printf( "alpha = %.4e + %.4ei; beta = %.4e + %.4ei;\n",
                    real(alpha), imag(alpha),
                    real(beta),  imag(beta) );
            printf( "A = "    ); print_matrix( m, n, A, lda );
            printf( "x    = " ); print_vector( Xm, x, incx );
            printf( "y    = " ); print_vector( Ym, y, incy );
        }

        // run test
        testsweeper::flush_cache( params.cache() );
        //double time = get_wtime();
        blas::gbmv( layout, trans, m, n, kl, ku, alpha, dA, lda, 
                    dx, incx, beta, dy, incy, queue);
        queue.sync();
        //time = get_wtime() - time;

        double gflop = blas::Gflop< scalar_t >::gbmv( m, n, kl, ku );
        double gbyte = blas::Gbyte< scalar_t >::gbmv( layout, m, n, kl, ku);
        //params.time()   = time * 1000;  // msec
        //params.gflops() = gflop / time;
        //params.gbytes() = gbyte / time;
        blas::device_copy_vector(Ym, dy, std::abs(incy), y, std::abs(incy), queue);
        queue.sync();

        if (verbose >= 2) {
            printf( "y2   = " ); print_vector( Ym, y, incy );
        }

        double time;
        if (params.ref() == 'y' || params.check() == 'y') {
            // run reference
            testsweeper::flush_cache( params.cache() );
            time = get_wtime();
            cblas_gbmv( cblas_layout_const(layout), cblas_trans_const(trans), m, n,
                        kl, ku, alpha, A, lda, x, incx, beta, yref, incy );
            time = get_wtime() - time;

            params.ref_time()   = time * 1000;  // msec
            params.ref_gflops() = gflop / time;
            params.ref_gbytes() = gbyte / time;

            if (verbose >= 2) {
                printf( "yref = " ); print_vector( Ym, yref, incy );
            }

            // check error compared to reference
            // treat y as 1 x Ym matrix with ld = incy; k = Xm is reduction dimension
            real_t error;
            bool okay;
            check_gemm( 1, Ym, Xm, alpha, beta, Anorm, Xnorm, Ynorm,
                        yref, std::abs(incy), y, std::abs(incy), verbose, &error, &okay );
            params.error() = error;
            params.okay() = okay;
        }

        int runs = params.runs();
        double stime;
        double all_time=0.0f;
        for(int i = 0; i < runs; i++){
            testsweeper::flush_cache( params.cache() );
            stime = get_wtime();
            blas::gbmv( layout, trans, m, n, kl, ku, alpha, dA, lda, 
                dx, incx, beta, dy, incy, queue);
            queue.sync();
            all_time += (get_wtime() - stime);
        }
        all_time/=(double)runs;
        params.time()   = all_time * 1000;  // msec
        params.gflops() = gflop / all_time;
        params.gbytes() = gbyte / all_time;
    }

    delete[] A;
    delete[] x;
    delete[] y;
    delete[] yref;

    blas::device_free( dA, queue );
    blas::device_free( dx, queue );
    blas::device_free( dy, queue );
}

// -----------------------------------------------------------------------------
void test_gbmv_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_gbmv_device_work< float, float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_gbmv_device_work< double, double, double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_gbmv_device_work< std::complex<float>, std::complex<float>,
                            std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_gbmv_device_work< std::complex<double>, std::complex<double>,
                            std::complex<double> >( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
