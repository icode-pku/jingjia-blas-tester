// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"
#include "cblas_wrappers.hh"
#include "lapack_wrappers.hh"
#include "blas/flops.hh"
#include "print_matrix.hh"
#include "check_gemm.hh"
#include "blas/util.hh"

// -----------------------------------------------------------------------------
template <typename TX, typename TY>
void test_dot_device_work( Params& params, bool run )
{
    using namespace testsweeper;
    using std::real;
    using std::imag;
    using scalar_t = blas::scalar_type< TX, TY >;
    using real_t   = blas::real_type< scalar_t >;

    // get & mark input values
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx();
    int64_t incy    = params.incy();
    int64_t verbose = params.verbose();
    int64_t device  = params.device();
    char mode       = params.pointer_mode();
    int64_t testcase    = params.testcase();

    scalar_t  result_host;
    scalar_t* result = &result_host;

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
    TX* x = new TX[ size_x ];
    TY* y = new TY[ size_y ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_x, x );
    lapack_larnv( idist, iseed, size_y, y );

    // norms for error check
    real_t Xnorm = cblas_nrm2( n, x, std::abs(incx) );
    real_t Ynorm = cblas_nrm2( n, y, std::abs(incy) );

    // device specifics
    blas::Queue queue( device );
    TX* dx;
    TY* dy;

    dx = blas::device_malloc<TX>( size_x, queue );
    dy = blas::device_malloc<TY>( size_y, queue );

    blas::device_copy_vector( n, x, std::abs(incx), dx, std::abs(incx), queue );
    blas::device_copy_vector( n, y, std::abs(incy), dy, std::abs(incy), queue );
    queue.sync();

    if (mode == 'd') {
        result = blas::device_malloc<scalar_t>( 1, queue );
        #if defined( BLAS_HAVE_CUBLAS )
        cublasSetPointerMode( queue.handle(), CUBLAS_POINTER_MODE_DEVICE );
        #elif defined( BLAS_HAVE_ROCBLAS )
        rocblas_set_pointer_mode( queue.handle(), rocblas_pointer_mode_device );
        #endif
    }



    // test error exits
    if(testcase == 0){
        char *error_name = (char *)malloc(sizeof(char)*35);
        int all_testcase = 0;
        int passed_testcase = 0;
        int failed_testcase = 0;
        //Test case 1: Test the return when n is 0
        blas::dot( 0, dx, incx, dy, incy, result, queue, testcase, error_name);
        queue.sync();
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_SUCCESS", all_testcase, passed_testcase, failed_testcase, blas::isEqualToZero(result_host)), error_name);
        //Test case 2: Test the return when n is -1
        blas::dot( -1, dx, incx, dy, incy, result, queue, testcase, error_name);
        queue.sync();
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_SUCCESS", all_testcase, passed_testcase, failed_testcase, blas::isEqualToZero(result_host)), error_name);
        //Test case 3: Test the return when incx is 0
        blas::dot(  n, dx,    0, dy, incy, result, queue, testcase, error_name);
        queue.sync();
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_SUCCESS", all_testcase, passed_testcase, failed_testcase), error_name);
        //Test case 4: Test the return when incx is -1
        blas::dot(  n, dx,   -1, dy, incy, result, queue, testcase, error_name);
        queue.sync();
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_SUCCESS", all_testcase, passed_testcase, failed_testcase), error_name);
        //Test case 5: Test the return when incy is 0
        blas::dot(  n, dx, incx, dy,    0, result, queue, testcase, error_name);
        queue.sync();
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_SUCCESS", all_testcase, passed_testcase, failed_testcase), error_name);
        //Test case 6: Test the return when incy is -1
        blas::dot(  n, dx, incx, dy,   -1, result, queue, testcase, error_name);
        queue.sync();
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_SUCCESS", all_testcase, passed_testcase, failed_testcase), error_name);
        
        params.Totalcase()+=all_testcase;
        params.Passedcase()+=passed_testcase;
        params.Failedcase()+=failed_testcase;
        //printf("All Test Cases: %d  Passed Cases: %d  Failed Cases: %d\n",all_testcase, passed_testcase, failed_testcase);
        free(error_name);
    }
    else{
        if (verbose >= 1) {
            printf( "\n"
                    "x n=%5lld, inc=%5lld, size=%10lld, norm %.2e\n"
                    "y n=%5lld, inc=%5lld, size=%10lld, norm %.2e\n",
                    llong( n ), llong( incx ), llong( size_x ), Xnorm,
                    llong( n ), llong( incy ), llong( size_y ), Ynorm );
        }
        if (verbose >= 2) {
            printf( "x = " ); print_vector( n, x, incx );
            printf( "y = " ); print_vector( n, y, incy );
        }

        // run test
        testsweeper::flush_cache( params.cache() );
        blas::dot( n, dx, incx, dy, incy, result, queue );
        queue.sync();

        if (mode == 'd') {
            device_memcpy( &result_host, result, 1, queue );
        }

        double gflop = blas::Gflop<scalar_t>::dot( n );
        double gbyte = blas::Gbyte<scalar_t>::dot( n );

        if (verbose >= 1) {
            printf( "dot = %.4e + %.4ei\n", real(result_host), imag(result_host) );
        }
        double time;
        if (params.check() == 'y') {
            // run reference
            testsweeper::flush_cache( params.cache() );
            time = get_wtime();
            scalar_t ref = cblas_dot( n, x, incx, y, incy );
            time = get_wtime() - time;
            if(params.iscorrect()==0){
                params.ref_time()   = time * 1000;  // msec
                params.ref_gflops() = gflop / time;
                params.ref_gbytes() = gbyte / time;
            }

            if (verbose >= 1) {
                printf( "ref = %.4e + %.4ei\n", real(ref), imag(ref) );
            }

            if(params.iscorrect()==1){
                // check error compared to reference
                // treat result as 1 x 1 matrix; k = n is reduction dimension
                // alpha=1, beta=0, Cnorm=0
                real_t error;
                bool okay;
                check_gemm( 1, 1, n, scalar_t(1), scalar_t(0), Xnorm, Ynorm, real_t(0),
                            &ref, 1, &result_host, 1, verbose, &error, &okay );
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
                blas::dot( n, dx, incx, dy, incy, result, queue );
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

    blas::device_free( dx, queue );
    blas::device_free( dy, queue );
    if (mode == 'd')
        blas::device_free( result, queue );
}

// -----------------------------------------------------------------------------
void test_dot_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_dot_device_work< float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_dot_device_work< double, double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_dot_device_work< std::complex<float>, std::complex<float> >
                ( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_dot_device_work< std::complex<double>, std::complex<double> >
                ( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}