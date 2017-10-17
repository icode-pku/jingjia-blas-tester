#include <omp.h>

#include "test.hh"
#include "cblas.hh"
#include "lapack.hh"
#include "flops.hh"
#include "print_matrix.hh"
#include "check_gemm.hh"

#include "copy.hh"
#include "rot.hh"

// -----------------------------------------------------------------------------
// TX is data [x, y]
// TS is for sine, which can be real (zdrot) or complex (zrot)
// cosine is always real
template< typename TX, typename TS >
void test_rot_work( Params& params, bool run )
{
    using namespace blas;
    typedef typename traits< TX >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx.value();
    int64_t incy    = params.incy.value();
    int64_t verbose = params.verbose.value();

    // mark non-standard output values
    params.ref_time.value();
    params.ref_gflops.value();

    // adjust header names
    params.time.name( "SLATE\ntime (ms)" );
    params.ref_time.name( "Ref.\ntime (ms)" );

    if ( ! run)
        return;

    // setup
    size_t size_x = (n - 1) * abs(incx) + 1;
    size_t size_y = (n - 1) * abs(incy) + 1;
    TX* x    = new TX[ size_x ];
    TX* xref = new TX[ size_x ];
    TX* y    = new TX[ size_y ];
    TX* yref = new TX[ size_y ];
    TX s = rand() / double(RAND_MAX);    // todo: imag
    real_t c = sqrt( 1 - real(s*conj(s)) );  // real

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_x, x );
    lapack_larnv( idist, iseed, size_y, y );
    cblas_copy( n, x, incx, xref, incx );
    cblas_copy( n, y, incy, yref, incy );

    // norms for error check
    real_t Xnorm = cblas_nrm2( n, x, abs(incx) );
    real_t Ynorm = cblas_nrm2( n, y, abs(incy) );
    real_t Anorm = sqrt( Xnorm*Xnorm + Ynorm*Ynorm ); // || [x y] ||_F

    // test error exits
    assert_throw( blas::rot( -1, x, incx, y, incy, c, s ), blas::Error );
    assert_throw( blas::rot(  n, x,    0, y, incy, c, s ), blas::Error );
    assert_throw( blas::rot(  n, x, incx, y,    0, c, s ), blas::Error );

    if (verbose >= 1) {
        printf( "x n=%5lld, inc=%5lld, size=%5lld\n"
                "y n=%5lld, inc=%5lld, size=%5lld\n",
                (lld) n, (lld) incx, (lld) size_x,
                (lld) n, (lld) incy, (lld) size_y );
    }
    if (verbose >= 2) {
        printf( "x    = " ); print_vector( n, x, incx );
        printf( "y    = " ); print_vector( n, y, incy );
    }

    // run test
    libtest::flush_cache( params.cache.value() );
    double time = omp_get_wtime();
    blas::rot( n, x, incx, y, incy, c, s );
    time = omp_get_wtime() - time;

    double gflop = gflop_dot( n, x );
    params.time.value()   = time * 1000;  // msec
    params.gflops.value() = gflop / time;

    if (verbose >= 1) {
        printf( "x2   = " ); print_vector( n, x, incx );
        printf( "y2   = " ); print_vector( n, y, incy );
    }

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        cblas_rot( n, xref, incx, yref, incy, c, s );
        time = omp_get_wtime() - time;

        params.ref_time.value()   = time * 1000;  // msec
        params.ref_gflops.value() = gflop / time;

        if (verbose >= 1) {
            printf( "xref = " ); print_vector( n, xref, incx );
            printf( "yref = " ); print_vector( n, yref, incy );
        }

        // check error compared to reference
        // C = [x y] * R for n x 2 matrix C and 2 x 2 rotation R
        // alpha=1, beta=0, C0norm=0
        TX* C    = new TX[ 2*n ];
        TX* Cref = new TX[ 2*n ];
        blas::copy( n, x,    incx, &C[0],    1 );
        blas::copy( n, y,    incy, &C[n],    1 );
        blas::copy( n, xref, incx, &Cref[0], 1 );
        blas::copy( n, yref, incy, &Cref[n], 1 );
        real_t Rnorm = sqrt(2);  // ||R||_F
        real_t error;
        int64_t okay;
        check_gemm( n, 2, 2, TX(1), TX(0), Anorm, Rnorm, real_t(0),
                    Cref, n, C, n, &error, &okay );
        params.error.value() = error;
        params.okay.value() = okay;

        delete[] C;
        delete[] Cref;
    }

    delete[] x;
    delete[] y;
    delete[] xref;
    delete[] yref;
}

// -----------------------------------------------------------------------------
void test_rot( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            //test_rot_work< int64_t >( params, run );  // todo: generic implementation
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_rot_work< float, float >( params, run );
            break;

        case libtest::DataType::Double:
            test_rot_work< double, double >( params, run );
            break;

        // // real sine
        // case libtest::DataType::SingleComplex:
        //     //test_rot_work< std::complex<float>, float >
        //     //    ( params, run );
        //     throw std::exception();
        //     break;
        //
        // // real sine
        // case libtest::DataType::DoubleComplex:
        //     //test_rot_work< std::complex<double>, double >
        //     //    ( params, run );
        //     throw std::exception();
        //     break;

        // complex sine
        case libtest::DataType::SingleComplex:
            //test_rot_work< std::complex<float>, std::complex<float> >
            //    ( params, run );
            throw std::exception();
            break;

        // complex sine
        case libtest::DataType::DoubleComplex:
            //test_rot_work< std::complex<double>, std::complex<double> >
            //    ( params, run );
            throw std::exception();
            break;
    }
}
