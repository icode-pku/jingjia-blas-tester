// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"
#include "cblas_wrappers.hh"
#include "lapack_wrappers.hh"
#include "blas/flops.hh"
#include "print_matrix.hh"
#include "blas/util.hh"

// -----------------------------------------------------------------------------
template <typename Tx>
void test_nrm2_device_work( Params& params, bool run )
{
    using namespace testsweeper;
    using scalar_t = blas::scalar_type<Tx>;
    using real_t   = blas::real_type<scalar_t>;

    // get & mark input values
    char mode       = params.pointer_mode();
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx();
    int64_t device  = params.device();
    int64_t verbose = params.verbose();
    int64_t testcase    = params.testcase();

    real_t  result_host;
    real_t* result = &result_host;
    real_t  result_cblas;

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
    Tx* x    = new Tx[ size_x ];
    Tx* xref = new Tx[ size_x ];

    // device specifics
    blas::Queue queue( device );
    Tx* dx;

    dx = blas::device_malloc<Tx>(size_x, queue);
    if (mode == 'd') {
        result = blas::device_malloc<real_t>(1, queue);
        #if defined( BLAS_HAVE_CUBLAS )
        cublasSetPointerMode(queue.handle(), CUBLAS_POINTER_MODE_DEVICE);
        #elif defined( BLAS_HAVE_ROCBLAS )
        rocblas_set_pointer_mode( queue.handle(), rocblas_pointer_mode_device );
        #endif
    }

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
        //Test case 1: Test the return when result is an nullptr
        blas::nrm2(  n, dx, incx, nullptr, queue, testcase, error_name);
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //Test case 2: Test the return when n is 0
        blas::nrm2(  0, dx, incx, result, queue, testcase, error_name);
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_SUCCESS", all_testcase, passed_testcase, failed_testcase)&&blas::isEqualToZero(result_host), error_name);
        //Test case 3: Test the return when  n is -1
        blas::nrm2( -1, dx, incx, result, queue, testcase, error_name);
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_SUCCESS", all_testcase, passed_testcase, failed_testcase)&&blas::isEqualToZero(result_host), error_name);
        //Test case 4: Test the return when incx is 0
        blas::nrm2(  n, dx,    0, result, queue, testcase, error_name);
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_SUCCESS", all_testcase, passed_testcase, failed_testcase)&&blas::isEqualToZero(result_host), error_name);
        //Test case 5: Test the return when incx is -1
        blas::nrm2(  n, dx,   -1, result, queue, testcase, error_name);
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_SUCCESS", all_testcase, passed_testcase, failed_testcase)&&blas::isEqualToZero(result_host), error_name);

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
                    "n=%5lld, incx=%5lld, sizex=%10lld\n",
                    llong( n ), llong( incx ), llong( size_x ) );
        }
        if (verbose >= 2) {
            printf( "x    = " ); print_vector( n, x, incx );
        }

        // run test
        testsweeper::flush_cache( params.cache() );
        blas::nrm2( n, dx, incx, result, queue );
        queue.sync();

        if (mode == 'd') {
            device_memcpy( &result_host, result, 1, queue );
        }

        double gflop = blas::Gflop< Tx >::nrm2( n );
        double gbyte = blas::Gbyte< Tx >::nrm2( n );

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
            result_cblas = cblas_nrm2( n, xref, incx );
            time = get_wtime() - time;

            if(params.iscorrect()==0){
                params.ref_time()   = time * 1000;  // msec
                params.ref_gflops() = gflop / time;
                params.ref_gbytes() = gbyte / time;
            }
            if (verbose >= 2) {
                printf( "result0 = %.2e\n", result_cblas );
            }

            if(params.iscorrect()==1){
                // relative forward error:
                real_t error = std::abs( (result_cblas - result_host)
                                / (sqrt(n+1) * result_cblas) );
                params.error() = error;


                if (verbose >= 2) {
                    printf( "err  = " ); print_vector( n, x, incx, "%9.2e" );
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
                blas::nrm2( n, dx, incx, result, queue );
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
    delete[] xref;

    blas::device_free( dx, queue );
    if (mode == 'd')
        blas::device_free( result, queue );
}

// -----------------------------------------------------------------------------
void test_nrm2_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_nrm2_device_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_nrm2_device_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_nrm2_device_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_nrm2_device_work< std::complex<double> >( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
