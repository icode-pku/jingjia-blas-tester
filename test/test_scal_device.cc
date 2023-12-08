// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"
#include "cblas_wrappers.hh"
#include "lapack_wrappers.hh"
#include "blas/flops.hh"
#include "print_matrix.hh"

// -----------------------------------------------------------------------------
template <typename T>
void test_scal_device_work( Params& params, bool run )
{
    using namespace testsweeper;
    using std::real;
    using std::imag;
    using real_t = blas::real_type< T >;

    // get & mark input values
    T alpha         = params.alpha();
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx();
    int64_t device  = params.device();
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
    size_t size_x = (n - 1) * std::abs(incx) + 1;
    T* x    = new T[ size_x ];
    T* xref = new T[ size_x ];

    // device specifics
    blas::Queue queue( device );
    T* dx;

    dx = blas::device_malloc<T>(size_x, queue);

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_x, x );
    cblas_copy( n, x, incx, xref, incx );

    blas::device_copy_vector(n, x, std::abs(incx), dx, std::abs(incx), queue);
    queue.sync();

    // test error exits
    if(testcase == 0){
        char *error_name = (char *)malloc(sizeof(char)*35);
        int all_testcase = 0;
        int passed_testcase = 0;
        int failed_testcase = 0;
        //Test case 1: Test the return when n is 0
        blas::scal(  0, alpha, dx, incx, queue, testcase, error_name);
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_SUCCESS", all_testcase, passed_testcase, failed_testcase), error_name);
        //Test case 2: Test the return when n is -1
        blas::scal( -1, alpha, dx, incx, queue, testcase, error_name);
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_SUCCESS", all_testcase, passed_testcase, failed_testcase), error_name);
        //Test case 3: Test the return when incx is 0
        blas::scal(  n, alpha, dx,    0, queue, testcase, error_name);
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_SUCCESS", all_testcase, passed_testcase, failed_testcase), error_name);
        //Test case 4: Test the return when incx is -1
        blas::scal(  n, alpha, dx,   -1, queue, testcase, error_name);
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_SUCCESS", all_testcase, passed_testcase, failed_testcase), error_name);

        queue.sync();
        printf("All Test Cases: %d  Passed Cases: %d  Failed Cases: %d\n",all_testcase, passed_testcase, failed_testcase);
        free(error_name);
    }
    else{
        if (verbose >= 1) {
            printf( "\n"
                    "x n=%5lld, inc=%5lld, size=%10lld\n",
                    llong( n ), llong( incx ), llong( size_x ) );
        }
        if (verbose >= 2) {
            printf( "alpha = %.4e + %.4ei;\n",
                    real(alpha), imag(alpha) );
            printf( "x    = " ); print_vector( n, x, incx );
        }

        // run test
        testsweeper::flush_cache( params.cache() );
        blas::scal( n, alpha, dx, incx, queue );
        queue.sync();

        double gflop = blas::Gflop< T >::scal( n );
        double gbyte = blas::Gbyte< T >::scal( n );

        blas::device_copy_vector(n, dx, std::abs(incx), x, std::abs(incx), queue);
        queue.sync();

        if (verbose >= 2) {
            printf( "x2   = " ); print_vector( n, x, incx );
        }
        double time;
        if (params.check() == 'y') {
            // run reference
            testsweeper::flush_cache( params.cache() );
            time = get_wtime();
            cblas_scal( n, alpha, xref, incx );
            time = get_wtime() - time;

            params.ref_time()   = time * 1000;  // msec
            params.ref_gflops() = gflop / time;
            params.ref_gbytes() = gbyte / time;

            if (verbose >= 2) {
                printf( "xref = " ); print_vector( n, xref, incx );
            }

            // maximum component-wise forward error:
            // | fl(xi) - xi | / | xi |
            real_t error = 0;
            int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
            for (int64_t i = 0; i < n; ++i) {
                error = std::max( error, std::abs( (xref[ix] - x[ix]) / xref[ix] ));
                ix += incx;
            }
            params.error() = error;

            // complex needs extra factor; see Higham, 2002, sec. 3.6.
            if (blas::is_complex<T>::value) {
                error /= 2*sqrt(2);
            }

            real_t u = 0.5 * std::numeric_limits< real_t >::epsilon();
            params.okay() = (error < u);
        }

        int runs = params.runs();
        double stime;
        double all_time=0.0f;
        for(int i = 0; i < runs; i++){
            testsweeper::flush_cache( params.cache() );
            stime = get_wtime();
            blas::scal( n, alpha, dx, incx, queue );
            queue.sync();
            all_time += (get_wtime() - stime);
        }
        all_time/=(double)runs;
        params.time()   = all_time * 1000;  // msec
        params.gflops() = gflop / all_time;
        params.gbytes() = gbyte / all_time;
    }
    
    delete[] x;
    delete[] xref;

    blas::device_free( dx, queue );
}

// -----------------------------------------------------------------------------
void test_scal_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_scal_device_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_scal_device_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_scal_device_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_scal_device_work< std::complex<double> >( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
