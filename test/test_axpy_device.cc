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
template <typename Tx, typename Ty>
void test_axpy_device_work( Params& params, bool run )
{
    using namespace testsweeper;
    using std::real;
    using std::imag;
    using scalar_t = blas::scalar_type< Tx, Ty >;
    using real_t   = blas::real_type< scalar_t >;

    // get & mark input values
    scalar_t alpha  = params.alpha();
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx();
    int64_t incy    = params.incy();
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
    size_t size_y = (n - 1) * std::abs(incy) + 1;
    Tx* x    = new Tx[ size_x ];
    Ty* y    = new Ty[ size_y ];
    Ty* yref = new Ty[ size_y ];
    Ty* y0   = new Ty[ size_y ];

    // device specifics
    blas::Queue queue( device );
    Tx* dx;
    Ty* dy;

    dx = blas::device_malloc<Tx>(size_x, queue);
    dy = blas::device_malloc<Ty>(size_y, queue);

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_y, y );
    cblas_copy( n, y, incy, yref, incy );
    cblas_copy( n, y, incy,   y0, incy );

    blas::device_copy_vector(n, x, std::abs(incx), dx, std::abs(incx), queue);
    blas::device_copy_vector(n, y, std::abs(incy), dy, std::abs(incy), queue);
    queue.sync();

    // test error exits
    if(testcase == 0){
        //The number of test cases is 2
        char *error_name = (char *)malloc(sizeof(char)*35);
        int all_testcase = 0;
        int passed_testcase = 0;
        int failed_testcase = 0;
        //case1: test n is 0
        blas::axpy( 0, alpha, dx, incx, dy, incy, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_SUCCESS", all_testcase, passed_testcase, failed_testcase), error_name);
        //case2: test n is -1
        blas::axpy( -1, alpha, dx, incx, dy, incy, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_SUCCESS", all_testcase, passed_testcase, failed_testcase), error_name);
        //case3: test incx is 0
        blas::axpy( n, alpha, dx, 0, dy, incy, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_SUCCESS", all_testcase, passed_testcase, failed_testcase), error_name);
        //case4: test incx is -1
        blas::axpy( n, alpha, dx, -1, dy, incy, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_SUCCESS", all_testcase, passed_testcase, failed_testcase), error_name);

        //case5: test incy is 0
        blas::axpy( n, alpha, dx, incx, dy, 0, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_SUCCESS", all_testcase, passed_testcase, failed_testcase), error_name);

        //case6: test incy is -1
        blas::axpy( n, alpha, dx, incx, dy, -1, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_SUCCESS", all_testcase, passed_testcase, failed_testcase), error_name);

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
                    "n=%5lld, incx=%5lld, sizex=%10lld, incy=%5lld, sizey=%10lld\n",
                    llong( n ), llong( incx ), llong( size_x ), llong( incy ), llong( size_y ) );
        }
        if (verbose >= 2) {
            printf( "alpha = %.4e + %.4ei;\n",
                    real(alpha), imag(alpha) );
            printf( "y    = " ); print_vector( n, y, incy );
        }

        // run test
        testsweeper::flush_cache( params.cache() );
        blas::axpy( n, alpha, dx, incx, dy, incy, queue );
        queue.sync();

        double gflop = blas::Gflop< Ty >::axpy( n );
        double gbyte = blas::Gbyte< Ty >::axpy( n );

        blas::device_copy_vector(n, dy, std::abs(incy), y, std::abs(incy), queue);
        queue.sync();

        if (verbose >= 2) {
            printf( "y2   = " ); print_vector( n, y, incy );
        }
        double time;
        if (params.check() == 'y') {
            // run reference
            testsweeper::flush_cache( params.cache() );
            time = get_wtime();
            cblas_axpy( n, alpha, x, incx, yref, incy );
            time = get_wtime() - time;
            if(params.iscorrect()==0){
                params.ref_time()   = time * 1000;  // msec
                params.ref_gflops() = gflop / time;
                params.ref_gbytes() = gbyte / time;
            }
            if (verbose >= 2) {
                printf( "yref = " ); print_vector( n, yref, incy );
            }

            if(params.iscorrect()==1){
                // maximum component-wise forward error:
                // | fl(yi) - yi | / | yi |
                real_t error = 0;
                int64_t iy = (incy > 0 ? 0 : (-n + 1)*incy);
                int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
                for (int64_t i = 0; i < n; ++i) {
                    y[iy] = std::abs( y[iy] - yref[iy] )
                        / (2*(std::abs( alpha * x[ix] ) + std::abs( y0[iy] )));
                    ix += incx;
                    iy += incy;
                }
                params.error() = error;

                if (verbose >= 2) {
                    printf( "err  = " ); print_vector( n, y, incy, "%9.2e" );
                }

                // complex needs extra factor; see Higham, 2002, sec. 3.6.
                if (blas::is_complex<scalar_t>::value) {
                    error /= 2*sqrt(2);
                }

                real_t u = 0.5 * std::numeric_limits< real_t >::epsilon();
                params.error() = error;
                params.okay() = (error < u);
            }

        }

        if(params.iscorrect()==0){
            int runs = params.runs();
            double stime;
            double all_time=0.0f;
            for(int i = 0; i < runs; i++){
                testsweeper::flush_cache( params.cache() );
                stime = get_wtime();
                blas::axpy( n, alpha, dx, incx, dy, incy, queue );
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
    delete[] y;
    delete[] yref;
    delete[] y0;

    blas::device_free( dx, queue );
    blas::device_free( dy, queue );
}

// -----------------------------------------------------------------------------
void test_axpy_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_axpy_device_work< float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_axpy_device_work< double, double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_axpy_device_work< std::complex<float>, std::complex<float> >
                ( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_axpy_device_work< std::complex<double>, std::complex<double> >
                ( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
