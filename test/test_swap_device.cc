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
template <typename TX, typename TY>
void test_swap_device_work( Params& params, bool run )
{
    using namespace testsweeper;
    using scalar_t = blas::scalar_type< TX, TY >;
    using real_t   = blas::real_type< scalar_t >;

    // get & mark input values
    int64_t n          = params.dim.n();
    int64_t incx       = params.incx();
    int64_t incy       = params.incy();
    int64_t device     = params.device();
    int64_t verbose    = params.verbose();
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
    size_t size_y = (n - 1) * std::abs(incy) + 1;
    TX* x    = new TX[ size_x ];
    TX* xref = new TX[ size_x ];
    TY* y    = new TY[ size_y ];
    TY* yref = new TY[ size_y ];

    // device specifics
    blas::Queue queue( device );
    TX* dx;
    TY* dy;

    dx = blas::device_malloc<TX>( size_x, queue );
    dy = blas::device_malloc<TY>( size_y, queue );

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_x, x );
    lapack_larnv( idist, iseed, size_y, y );
    cblas_copy( n, x, incx, xref, incx );
    cblas_copy( n, y, incy, yref, incy );

    // todo: should we have different incdx and incdy
    blas::device_copy_vector(n, x, std::abs(incx), dx, std::abs(incx), queue);
    blas::device_copy_vector(n, y, std::abs(incy), dy, std::abs(incy), queue);
    queue.sync();

    // test error exits
    if(testcase == 0){
        //The number of test cases is 0
        int all_testcase = 0;
        int passed_testcase = 0;
        int failed_testcase = 0;
        printf("All Test Cases: %d  Passed Cases: %d  Failed Cases: %d\n",all_testcase, passed_testcase, failed_testcase);
    }
    else{
        if (verbose >= 1) {
            printf( "\n"
                    "x n=%5lld, inc=%5lld, size=%10lld\n"
                    "y n=%5lld, inc=%5lld, size=%10lld\n",
                    llong( n ), llong( incx ), llong( size_x ),
                    llong( n ), llong( incy ), llong( size_y ) );
        }
        if (verbose >= 2) {
            printf( "x    = " ); print_vector( n, x, incx );
            printf( "y    = " ); print_vector( n, y, incy );
        }

        // run test
        testsweeper::flush_cache( params.cache() );
        blas::swap( n, dx, incx, dy, incy, queue );
        queue.sync();

        double gflop = blas::Gflop< scalar_t >::swap( n );
        double gbyte = blas::Gbyte< scalar_t >::swap( n );

        // todo: should we have different incdx and incdy
        blas::device_copy_vector(n, dx, std::abs(incx), x, std::abs(incx), queue);
        blas::device_copy_vector(n, dy, std::abs(incy), y, std::abs(incy), queue);
        queue.sync();

        if (verbose >= 2) {
            printf( "x2   = " ); print_vector( n, x, incx );
            printf( "y2   = " ); print_vector( n, y, incy );
        }
        double time;
        if (params.check() == 'y') {
            // run reference
            testsweeper::flush_cache( params.cache() );
            time = get_wtime();
            cblas_swap( n, xref, incx, yref, incy );
            time = get_wtime() - time;
            if (verbose >= 2) {
                printf( "xref = " ); print_vector( n, xref, incx );
                printf( "yref = " ); print_vector( n, yref, incy );
            }

            params.ref_time()   = time * 1000;  // msec
            params.ref_gflops() = gflop / time;
            params.ref_gbytes() = gbyte / time;

            // error = ||xref - x|| + ||yref - y||
            cblas_axpy( n, -1.0, x, incx, xref, incx );
            cblas_axpy( n, -1.0, y, incy, yref, incy );
            real_t error = cblas_nrm2( n, xref, std::abs(incx) )
                        + cblas_nrm2( n, yref, std::abs(incy) );
            params.error() = error;

            // swap must be exact!
            params.okay() = (error == 0);
        }
        //test time
        int runs = params.runs();
        double stime;
        double all_time=0.0f;
        for(int i = 0; i < runs; i++){
            testsweeper::flush_cache( params.cache() );
            stime = get_wtime();
            blas::swap( n, dx, incx, dy, incy, queue );
            queue.sync();
            all_time += (get_wtime() - stime);
        }
        all_time/=(double)runs;
        params.time()   = all_time * 1000;  // msec
        params.gflops() = gflop / all_time;
        params.gbytes() = gbyte / all_time;
    }

    delete[] x;
    delete[] y;
    delete[] xref;
    delete[] yref;

    blas::device_free( dx, queue );
    blas::device_free( dy, queue );
}

// -----------------------------------------------------------------------------
void test_swap_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_swap_device_work< float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_swap_device_work< double, double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_swap_device_work< std::complex<float>, std::complex<float> >
                ( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_swap_device_work< std::complex<double>, std::complex<double> >
                ( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
