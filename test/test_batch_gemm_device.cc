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

// -----------------------------------------------------------------------------
template <typename TA, typename TB, typename TC>
void test_device_batch_gemm_work( Params& params, bool run )
{
    using namespace testsweeper;
    using namespace blas::batch;
    using blas::Op;
    using blas::Layout;
    using scalar_t = blas::scalar_type< TA, TB, TC >;
    using real_t   = blas::real_type< scalar_t >;

    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Op transA_    = params.transA();
    blas::Op transB_    = params.transB();
    scalar_t alpha_     = params.alpha();
    scalar_t beta_      = params.beta();
    int64_t m_          = params.dim.m();
    int64_t n_          = params.dim.n();
    int64_t k_          = params.dim.k();
    size_t  batch       = params.batch();
    int64_t device      = params.device();
    int64_t align       = params.align();
    int64_t verbose     = params.verbose();
    int64_t testcase    = params.testcase();
    // mark non-standard output values
    params.gflops();
    params.ref_time();
    params.ref_gflops();
    params.runs();
    params.iscorrect();

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
    int64_t Am = (transA_ == Op::NoTrans ? m_ : k_);
    int64_t An = (transA_ == Op::NoTrans ? k_ : m_);
    int64_t Bm = (transB_ == Op::NoTrans ? k_ : n_);
    int64_t Bn = (transB_ == Op::NoTrans ? n_ : k_);
    int64_t Cm = m_;
    int64_t Cn = n_;
    if (layout == Layout::RowMajor) {
        std::swap( Am, An );
        std::swap( Bm, Bn );
        std::swap( Cm, Cn );
    }

    int64_t lda_ = roundup( Am, align );
    int64_t ldb_ = roundup( Bm, align );
    int64_t ldc_ = roundup( Cm, align );
    size_t size_A = size_t(lda_)*An;
    size_t size_B = size_t(ldb_)*Bn;
    size_t size_C = size_t(ldc_)*Cn;
    TA* A    = new TA[ batch * size_A ];
    TB* B    = new TB[ batch * size_B ];
    TC* C    = new TC[ batch * size_C ];
    TC* Cref = new TC[ batch * size_C ];

    // device specifics
    blas::Queue queue( device );
    TA* dA = blas::device_malloc<TA>( batch * size_A, queue );
    TB* dB = blas::device_malloc<TB>( batch * size_B, queue );
    TC* dC = blas::device_malloc<TC>( batch * size_C, queue );
    queue.sync();

    // pointer arrays
    std::vector<TA*>    Aarray( batch );
    std::vector<TB*>    Barray( batch );
    std::vector<TC*>    Carray( batch );
    std::vector<TC*> Crefarray( batch );
    std::vector<TA*>   dAarray( batch );
    std::vector<TB*>   dBarray( batch );
    std::vector<TC*>   dCarray( batch );

    for (size_t s = 0; s < batch; ++s) {
         Aarray[s]   =  A   + s * size_A;
         Barray[s]   =  B   + s * size_B;
         Carray[s]   =  C   + s * size_C;
        Crefarray[s] = Cref + s * size_C;
        dAarray[s]   = dA   + s * size_A;
        dBarray[s]   = dB   + s * size_B;
        dCarray[s]   = dC   + s * size_C;
    }

    // info
    std::vector<int64_t> info( batch );

    // wrap scalar arguments in std::vector
    std::vector<blas::Op> transA(1, transA_);
    std::vector<blas::Op> transB(1, transB_);
    std::vector<int64_t>  m(1, m_);
    std::vector<int64_t>  n(1, n_);
    std::vector<int64_t>  k(1, k_);
    std::vector<int64_t>  ldda(1, lda_);
    std::vector<int64_t>  lddb(1, ldb_);
    std::vector<int64_t>  lddc(1, ldc_);
    std::vector<scalar_t> alpha(1, alpha_);
    std::vector<scalar_t> beta(1, beta_);

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, batch * size_A, A );
    lapack_larnv( idist, iseed, batch * size_B, B );
    lapack_larnv( idist, iseed, batch * size_C, C );
    lapack_lacpy( "g", Cm, batch * Cn, C, ldc_, Cref, ldc_ );

    blas::device_copy_matrix(Am, batch * An, A, lda_, dA, lda_, queue);
    blas::device_copy_matrix(Bm, batch * Bn, B, ldb_, dB, ldb_, queue);
    blas::device_copy_matrix(Cm, batch * Cn, C, ldc_, dC, ldc_, queue);
    queue.sync();

    // norms for error check
    real_t work[1];
    real_t* Anorm = new real_t[ batch ];
    real_t* Bnorm = new real_t[ batch ];
    real_t* Cnorm = new real_t[ batch ];

    for (size_t s = 0; s < batch; ++s) {
        Anorm[s] = lapack_lange( "f", Am, An, Aarray[s], lda_, work );
        Bnorm[s] = lapack_lange( "f", Bm, Bn, Barray[s], ldb_, work );
        Cnorm[s] = lapack_lange( "f", Cm, Cn, Carray[s], ldc_, work );
    }
    if(testcase == 0){
        char *error_name = (char *)malloc(sizeof(char)*35);
        int all_testcase = 0;
        int passed_testcase = 0;
        int failed_testcase = 0;
        std::vector<blas::Op>N(1, Op::NoTrans);
        std::vector<blas::Op>T(1, Op::Trans);
        std::vector<blas::Op>C(1, Op::ConjTrans);
        //case 1: Test transA is an illegal value
        blas::batch::gemm( layout, std::vector<blas::Op>{Op(0)}, transB, m, n, k, alpha, dAarray, ldda, dBarray, lddb, beta, dCarray, lddc, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 2: Test transB is an illegal value
        blas::batch::gemm( layout, transA, std::vector<blas::Op>{Op(0)}, m, n, k, alpha, dAarray, ldda, dBarray, lddb, beta, dCarray, lddc, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 3: Test the return value when m is an illegal value
        blas::batch::gemm( layout, transA, transB, std::vector<int64_t>{-1}, n, k, alpha, dAarray, ldda, dBarray, lddb, beta, dCarray, lddc, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 4: Test the return value when n is an illegal value
        blas::batch::gemm( layout, transA, transB, m, std::vector<int64_t>{-1}, k, alpha, dAarray, ldda, dBarray, lddb, beta, dCarray, lddc, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 5: Test the return value when k is an illegal value
        blas::batch::gemm( layout, transA, transB, m, n, std::vector<int64_t>{-1}, alpha, dAarray, ldda, dBarray, lddb, beta, dCarray, lddc, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 6: Test the return value when lda is an illegal value and tansA is NoTrans
        blas::batch::gemm( Layout::ColMajor, N, N, m, n, k, alpha, dAarray, std::vector<int64_t>{m_-1}, dBarray, lddb, beta, dCarray, lddc, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 7: Test the return value when lda is an illegal value and tansA is Trans
        blas::batch::gemm( Layout::ColMajor, T, N, m, n, k, alpha, dAarray, std::vector<int64_t>{k_-1}, dBarray, lddb, beta, dCarray, lddc, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 8: Test the return value when lda is an illegal value and tansA is ConjTrans
        blas::batch::gemm( Layout::ColMajor, C, N, m, n, k, alpha, dAarray, std::vector<int64_t>{k_-1}, dBarray, lddb, beta, dCarray, lddc, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);

        //case 9: Test the return value when row-major order, tansA=NoTrans, and lda is an illegal value
        blas::batch::gemm( Layout::RowMajor,N, N, m, n, k, alpha, dAarray, std::vector<int64_t>{k_-1}, dBarray, lddb, beta, dCarray, lddc, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 10: Test the return value when row-major order, tansA=Trans, and lda is an illegal value
        blas::batch::gemm( Layout::RowMajor,T, N, m, n, k, alpha, dAarray, std::vector<int64_t>{m_-1}, dBarray, lddb, beta, dCarray, lddc, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 11: Test the return value when row-major order, tansA=ConjTrans, and lda is an illegal value
        blas::batch::gemm( Layout::RowMajor,C, N, m, n, k, alpha, dAarray, std::vector<int64_t>{m_-1}, dBarray, lddb, beta, dCarray, lddc, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);

        //case 12: Test the return value when col-major order, tansB=NoTrans, ldb is an illegal value
        blas::batch::gemm( Layout::ColMajor, N, N, m, n, k, alpha, dAarray, ldda, dBarray, std::vector<int64_t>{k_-1}, beta, dCarray, lddc, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 13: Test the return value when col-major order, tansB=Trans, ldb is an illegal value
        blas::batch::gemm( Layout::ColMajor, N, T, m, n, k, alpha, dAarray, ldda, dBarray, std::vector<int64_t>{n_-1}, beta, dCarray, lddc, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 14: Test the return value when col-major order, tansB=ConjTrans, ldb is an illegal value
        blas::batch::gemm( Layout::ColMajor, N, C, m, n, k, alpha, dAarray, ldda, dBarray, std::vector<int64_t>{n_-1}, beta, dCarray, lddc, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);

        //case 15: Test the return value when row-major order, tansB=NoTrans, ldb is an illegal value
        blas::batch::gemm( Layout::RowMajor, N, N, m, n, k, alpha, dAarray, ldda, dBarray, std::vector<int64_t>{n_-1}, beta, dCarray, lddc, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 16: Test the return value when row-major order, tansB=Trans, ldb is an illegal value
        blas::batch::gemm( Layout::RowMajor, N, T, m, n, k, alpha, dAarray, ldda, dBarray, std::vector<int64_t>{k_-1}, beta, dCarray, lddc, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 17: Test the return value when row-major order, tansB=ConjTrans, ldb is an illegal value
        blas::batch::gemm( Layout::RowMajor, N, C, m, n, k, alpha, dAarray, ldda, dBarray, std::vector<int64_t>{k_-1}, beta, dCarray, lddc, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);

        //case 18: Test the return value when col-major order, ldc is an illegal value
        blas::batch::gemm( Layout::ColMajor, transA, transB, m, n, k, alpha, dAarray, ldda, dBarray, lddb, beta, dCarray, std::vector<int64_t>{m_-1}, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name );
        //case 19: Test the return value when row-major order, ldc is an illegal value
        blas::batch::gemm( Layout::RowMajor, transA, transB, m, n, k, alpha, dAarray, ldda, dBarray, lddb, beta, dCarray, std::vector<int64_t>{n_-1}, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name );
        queue.sync();

        params.Totalcase()+=all_testcase;
        params.Passedcase()+=passed_testcase;
        params.Failedcase()+=failed_testcase;

        //printf("All Test Cases: %d  Passed Cases: %d  Failed Cases: %d\n",all_testcase, passed_testcase, failed_testcase);

        N.clear();
        N.shrink_to_fit();
        T.clear();
        T.shrink_to_fit();
        C.clear();
        C.shrink_to_fit();
        free(error_name);
    }
    else{
    // decide error checking mode
    info.resize( 0 );
    // run test
    testsweeper::flush_cache( params.cache() );
    blas::batch::gemm( layout, transA, transB, m, n, k,
                       alpha, dAarray, ldda, dBarray, lddb, beta, dCarray, lddc,
                       batch, info, queue );
    queue.sync();

    double gflop = batch * blas::Gflop< scalar_t >::gemm( m_, n_, k_ );
    blas::device_copy_matrix(Cm, batch * Cn, dC, ldc_, C, ldc_, queue);
    queue.sync();
    double time;
    if (params.ref() == 'y' || params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        for (size_t s = 0; s < batch; ++s) {
            cblas_gemm( cblas_layout_const(layout),
                        cblas_trans_const(transA_),
                        cblas_trans_const(transB_),
                        m_, n_, k_, alpha_, Aarray[s], lda_, Barray[s], ldb_, beta_, Crefarray[s], ldc_ );
        }
        time = get_wtime() - time;
        if(params.iscorrect()==0){
            params.ref_time()   = time;
            params.ref_gflops() = gflop / time;
        }

        if(params.iscorrect()==1){
            // check error compared to reference
            real_t err, error = 0;
            bool ok, okay = true;
            for (size_t s = 0; s < batch; ++s) {
                check_gemm( Cm, Cn, k_, alpha_, beta_, Anorm[s], Bnorm[s], Cnorm[s],
                            Crefarray[s], ldc_, Carray[s], ldc_, verbose, &err, &ok );
                error = std::max( error, err );
                okay &= ok;
            }
            params.error() = error;
            params.okay() = okay;
        }
    }

    if(params.iscorrect()==0){
        int runs = params.runs();
        double stime;
        double all_time=0.0f;
        scalar_t alpha_r = 0.01 * params.alpha();
        scalar_t beta_r = 0.01 * params.beta();
        std::vector<scalar_t> alpha2(1, alpha_r);
        std::vector<scalar_t> beta2(1, beta_r);
        for(int i = 0; i < runs; i++){
            info.resize(0);
            testsweeper::flush_cache( params.cache() );
            stime = get_wtime();
            blas::batch::gemm( layout, transA, transB, m, n, k,
                        alpha2, dAarray, ldda, dBarray, lddb, beta2, dCarray, lddc,
                        batch, info, queue );
            queue.sync();
            all_time += (get_wtime() - stime);
        }
        all_time/=(double)runs;
        params.time()   = all_time;  // s
        params.gflops() = gflop / all_time;
    }
    }

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] Cref;
    delete[] Anorm;
    delete[] Bnorm;
    delete[] Cnorm;

    blas::device_free( dA, queue );
    blas::device_free( dB, queue );
    blas::device_free( dC, queue );
}


// -----------------------------------------------------------------------------
template <>
void test_device_batch_gemm_work<Half_t, Half_t, Half_t>( Params& params, bool run )
{
    using namespace testsweeper;
    using namespace blas::batch;
    using blas::Op;
    using blas::Layout;
    using scalar_t = float;//blas::scalar_type< TA, TB, TC >;
    using real_t   = float;//blas::real_type< scalar_t >;

    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Op transA_    = params.transA();
    blas::Op transB_    = params.transB();
    scalar_t alpha_     = params.alpha();
    scalar_t beta_      = params.beta();
    int64_t m_          = params.dim.m();
    int64_t n_          = params.dim.n();
    int64_t k_          = params.dim.k();
    size_t  batch       = params.batch();
    int64_t device      = params.device();
    int64_t align       = params.align();
    int64_t verbose     = params.verbose();
    int64_t testcase    = params.testcase();
    // mark non-standard output values
    params.gflops();
    params.ref_time();
    params.ref_gflops();
    params.runs();
    params.iscorrect();

    params.fthreshold() = blas::paramspace::correct_threshld<Half_t>;
    params.fabsolute() = blas::paramspace::absolute<Half_t>;
    params.frelative() = blas::paramspace::relative<Half_t>;

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
        params.fthreshold.used(false);
        params.fabsolute.used(false);
        params.frelative.used(false);
    }


    if (blas::get_device_count() == 0) {
        params.msg() = "skipping: no GPU devices or no GPU support";
        return;
    }

    // setup
    int64_t Am = (transA_ == Op::NoTrans ? m_ : k_);
    int64_t An = (transA_ == Op::NoTrans ? k_ : m_);
    int64_t Bm = (transB_ == Op::NoTrans ? k_ : n_);
    int64_t Bn = (transB_ == Op::NoTrans ? n_ : k_);
    int64_t Cm = m_;
    int64_t Cn = n_;
    if (layout == Layout::RowMajor) {
        std::swap( Am, An );
        std::swap( Bm, Bn );
        std::swap( Cm, Cn );
    }

    int64_t lda_ = roundup( Am, align );
    int64_t ldb_ = roundup( Bm, align );
    int64_t ldc_ = roundup( Cm, align );
    size_t size_A = size_t(lda_)*An;
    size_t size_B = size_t(ldb_)*Bn;
    size_t size_C = size_t(ldc_)*Cn;
    float* A    = new float[ batch * size_A ];
    float* B    = new float[ batch * size_B ];
    float* C    = new float[ batch * size_C ];
    float* Cref = new float[ batch * size_C ];

    half* hA    = new half[ batch * size_A ];
    half* hB    = new half[ batch * size_B ];
    half* hC    = new half[ batch * size_C ];

    // device specifics
    blas::Queue queue( device );
    half* dA = blas::device_malloc<half>( batch * size_A, queue );
    half* dB = blas::device_malloc<half>( batch * size_B, queue );
    half* dC = blas::device_malloc<half>( batch * size_C, queue );
    queue.sync();

    // pointer arrays
    std::vector<float*>    Aarray( batch );
    std::vector<float*>    Barray( batch );
    std::vector<float*>    Carray( batch );
    std::vector<float*> Crefarray( batch );

    std::vector<half*>   dAarray( batch );
    std::vector<half*>   dBarray( batch );
    std::vector<half*>   dCarray( batch );

    for (size_t s = 0; s < batch; ++s) {
         Aarray[s]   =  A   + s * size_A;
         Barray[s]   =  B   + s * size_B;
         Carray[s]   =  C   + s * size_C;
        Crefarray[s] = Cref + s * size_C;

        dAarray[s]   = dA   + s * size_A;
        dBarray[s]   = dB   + s * size_B;
        dCarray[s]   = dC   + s * size_C;
    }

    // info
    std::vector<int64_t> info( batch );

    // wrap scalar arguments in std::vector
    std::vector<blas::Op> transA(1, transA_);
    std::vector<blas::Op> transB(1, transB_);
    std::vector<int64_t>  m(1, m_);
    std::vector<int64_t>  n(1, n_);
    std::vector<int64_t>  k(1, k_);
    std::vector<int64_t>  ldda(1, lda_);
    std::vector<int64_t>  lddb(1, ldb_);
    std::vector<int64_t>  lddc(1, ldc_);
    std::vector<scalar_t> alpha(1, alpha_);
    std::vector<scalar_t> beta(1, beta_);

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, batch * size_A, A );
    lapack_larnv( idist, iseed, batch * size_B, B );
    lapack_larnv( idist, iseed, batch * size_C, C );

    //from float to half then, half to float
    blas::copydatafloat2float(batch*size_A, A, A);
    blas::copydatafloat2float(batch*size_B, B, B);
    blas::copydatafloat2float(batch*size_C, C, C);
    lapack_lacpy( "g", Cm, batch * Cn, C, ldc_, Cref, ldc_ );

    //from float to half
    blas::copydatafloat2half(batch*size_A, A, hA);
    blas::copydatafloat2half(batch*size_B, B, hB);
    blas::copydatafloat2half(batch*size_C, C, hC);

    blas::device_copy_matrix(Am, batch * An, hA, lda_, dA, lda_, queue);
    blas::device_copy_matrix(Bm, batch * Bn, hB, ldb_, dB, ldb_, queue);
    blas::device_copy_matrix(Cm, batch * Cn, hC, ldc_, dC, ldc_, queue);
    queue.sync();

    // norms for error check
    real_t work[1];
    real_t* Anorm = new real_t[ batch ];
    real_t* Bnorm = new real_t[ batch ];
    real_t* Cnorm = new real_t[ batch ];

    for (size_t s = 0; s < batch; ++s) {
        Anorm[s] = lapack_lange( "f", Am, An, Aarray[s], lda_, work );
        Bnorm[s] = lapack_lange( "f", Bm, Bn, Barray[s], ldb_, work );
        Cnorm[s] = lapack_lange( "f", Cm, Cn, Carray[s], ldc_, work );
    }
    if(testcase == 0){
        char *error_name = (char *)malloc(sizeof(char)*35);
        int all_testcase = 0;
        int passed_testcase = 0;
        int failed_testcase = 0;
        std::vector<blas::Op>N(1, Op::NoTrans);
        std::vector<blas::Op>T(1, Op::Trans);
        std::vector<blas::Op>C(1, Op::ConjTrans);
        //case 1: Test transA is an illegal value
        blas::batch::gemm( layout, std::vector<blas::Op>{Op(0)}, transB, m, n, k, alpha, dAarray, ldda, dBarray, lddb, beta, dCarray, lddc, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 2: Test transB is an illegal value
        blas::batch::gemm( layout, transA, std::vector<blas::Op>{Op(0)}, m, n, k, alpha, dAarray, ldda, dBarray, lddb, beta, dCarray, lddc, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 3: Test the return value when m is an illegal value
        blas::batch::gemm( layout, transA, transB, std::vector<int64_t>{-1}, n, k, alpha, dAarray, ldda, dBarray, lddb, beta, dCarray, lddc, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 4: Test the return value when n is an illegal value
        blas::batch::gemm( layout, transA, transB, m, std::vector<int64_t>{-1}, k, alpha, dAarray, ldda, dBarray, lddb, beta, dCarray, lddc, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 5: Test the return value when k is an illegal value
        blas::batch::gemm( layout, transA, transB, m, n, std::vector<int64_t>{-1}, alpha, dAarray, ldda, dBarray, lddb, beta, dCarray, lddc, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 6: Test the return value when lda is an illegal value and tansA is NoTrans
        blas::batch::gemm( Layout::ColMajor, N, N, m, n, k, alpha, dAarray, std::vector<int64_t>{m_-1}, dBarray, lddb, beta, dCarray, lddc, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 7: Test the return value when lda is an illegal value and tansA is Trans
        blas::batch::gemm( Layout::ColMajor, T, N, m, n, k, alpha, dAarray, std::vector<int64_t>{k_-1}, dBarray, lddb, beta, dCarray, lddc, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 8: Test the return value when lda is an illegal value and tansA is ConjTrans
        blas::batch::gemm( Layout::ColMajor, C, N, m, n, k, alpha, dAarray, std::vector<int64_t>{k_-1}, dBarray, lddb, beta, dCarray, lddc, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);

        //case 9: Test the return value when row-major order, tansA=NoTrans, and lda is an illegal value
        blas::batch::gemm( Layout::RowMajor,N, N, m, n, k, alpha, dAarray, std::vector<int64_t>{k_-1}, dBarray, lddb, beta, dCarray, lddc, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 10: Test the return value when row-major order, tansA=Trans, and lda is an illegal value
        blas::batch::gemm( Layout::RowMajor,T, N, m, n, k, alpha, dAarray, std::vector<int64_t>{m_-1}, dBarray, lddb, beta, dCarray, lddc, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 11: Test the return value when row-major order, tansA=ConjTrans, and lda is an illegal value
        blas::batch::gemm( Layout::RowMajor,C, N, m, n, k, alpha, dAarray, std::vector<int64_t>{m_-1}, dBarray, lddb, beta, dCarray, lddc, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);

        //case 12: Test the return value when col-major order, tansB=NoTrans, ldb is an illegal value
        blas::batch::gemm( Layout::ColMajor, N, N, m, n, k, alpha, dAarray, ldda, dBarray, std::vector<int64_t>{k_-1}, beta, dCarray, lddc, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 13: Test the return value when col-major order, tansB=Trans, ldb is an illegal value
        blas::batch::gemm( Layout::ColMajor, N, T, m, n, k, alpha, dAarray, ldda, dBarray, std::vector<int64_t>{n_-1}, beta, dCarray, lddc, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 14: Test the return value when col-major order, tansB=ConjTrans, ldb is an illegal value
        blas::batch::gemm( Layout::ColMajor, N, C, m, n, k, alpha, dAarray, ldda, dBarray, std::vector<int64_t>{n_-1}, beta, dCarray, lddc, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);

        //case 15: Test the return value when row-major order, tansB=NoTrans, ldb is an illegal value
        blas::batch::gemm( Layout::RowMajor, N, N, m, n, k, alpha, dAarray, ldda, dBarray, std::vector<int64_t>{n_-1}, beta, dCarray, lddc, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 16: Test the return value when row-major order, tansB=Trans, ldb is an illegal value
        blas::batch::gemm( Layout::RowMajor, N, T, m, n, k, alpha, dAarray, ldda, dBarray, std::vector<int64_t>{k_-1}, beta, dCarray, lddc, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);
        //case 17: Test the return value when row-major order, tansB=ConjTrans, ldb is an illegal value
        blas::batch::gemm( Layout::RowMajor, N, C, m, n, k, alpha, dAarray, ldda, dBarray, std::vector<int64_t>{k_-1}, beta, dCarray, lddc, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name);

        //case 18: Test the return value when col-major order, ldc is an illegal value
        blas::batch::gemm( Layout::ColMajor, transA, transB, m, n, k, alpha, dAarray, ldda, dBarray, lddb, beta, dCarray, std::vector<int64_t>{m_-1}, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name );
        //case 19: Test the return value when row-major order, ldc is an illegal value
        blas::batch::gemm( Layout::RowMajor, transA, transB, m, n, k, alpha, dAarray, ldda, dBarray, lddb, beta, dCarray, std::vector<int64_t>{n_-1}, batch, info, queue, testcase, error_name );
        Blas_Match_Call( result_match(error_name, "CUBLAS_STATUS_INVALID_VALUE", all_testcase, passed_testcase, failed_testcase), error_name );
        queue.sync();

        params.Totalcase()+=all_testcase;
        params.Passedcase()+=passed_testcase;
        params.Failedcase()+=failed_testcase;

        //printf("All Test Cases: %d  Passed Cases: %d  Failed Cases: %d\n",all_testcase, passed_testcase, failed_testcase);

        N.clear();
        N.shrink_to_fit();
        T.clear();
        T.shrink_to_fit();
        C.clear();
        C.shrink_to_fit();
        free(error_name);
    }
    else{
    // decide error checking mode
    info.resize( 0 );
    // run test
    testsweeper::flush_cache( params.cache() );
    blas::batch::gemm( layout, transA, transB, m, n, k,
                       alpha, dAarray, ldda, dBarray, lddb, beta, dCarray, lddc,
                       batch, info, queue );
    queue.sync();

    double gflop = batch * blas::Gflop< scalar_t >::gemm( m_, n_, k_ );
    blas::device_copy_matrix(Cm, batch * Cn, dC, ldc_, hC, ldc_, queue);
    queue.sync();

    //convert half to float
    blas::copydatahalf2float(batch*size_C, hC, C);

    double time;
    if (params.ref() == 'y' || params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        for (size_t s = 0; s < batch; ++s) {
            cblas_gemm( cblas_layout_const(layout),
                        cblas_trans_const(transA_),
                        cblas_trans_const(transB_),
                        m_, n_, k_, alpha_, Aarray[s], lda_, Barray[s], ldb_, beta_, Crefarray[s], ldc_ );
        }
        time = get_wtime() - time;
        if(params.iscorrect()==0){
            params.ref_time()   = time;
            params.ref_gflops() = gflop / time;
        }

        if(params.iscorrect()==1){
            // check error compared to reference
            real_t err, error = 0;
            bool ok, okay = true;
            for (size_t s = 0; s < batch; ++s) {
                check_gemm( Cm, Cn, k_, alpha_, beta_, Anorm[s], Bnorm[s], Cnorm[s],
                            Crefarray[s], ldc_, Carray[s], ldc_, verbose, &err, &ok , true);
                error = std::max( error, err );
                okay &= ok;
            }
            params.error() = error;
            params.okay() = okay;
        }
    }

    if(params.iscorrect()==0){
        int runs = params.runs();
        double stime;
        double all_time=0.0f;
        scalar_t alpha_r = 0.01 * params.alpha();
        scalar_t beta_r = 0.01 * params.beta();
        std::vector<scalar_t> alpha2(1, alpha_r);
        std::vector<scalar_t> beta2(1, beta_r);
        for(int i = 0; i < runs; i++){
            info.resize(0);
            testsweeper::flush_cache( params.cache() );
            stime = get_wtime();
            blas::batch::gemm( layout, transA, transB, m, n, k,
                        alpha2, dAarray, ldda, dBarray, lddb, beta2, dCarray, lddc,
                        batch, info, queue );
            queue.sync();
            all_time += (get_wtime() - stime);
        }
        all_time/=(double)runs;
        params.time()   = all_time;  // s
        params.gflops() = gflop / all_time;
    }
    }

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] Cref;
    delete[] Anorm;
    delete[] Bnorm;
    delete[] Cnorm;

    delete[] hA;
    delete[] hB;
    delete[] hC;

    blas::device_free( dA, queue );
    blas::device_free( dB, queue );
    blas::device_free( dC, queue );
}

// -----------------------------------------------------------------------------
void test_batch_gemm_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_device_batch_gemm_work< float, float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_device_batch_gemm_work< double, double, double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_device_batch_gemm_work< std::complex<float>, std::complex<float>,
                            std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_device_batch_gemm_work< std::complex<double>, std::complex<double>,
                            std::complex<double> >( params, run );
            break;
        case testsweeper::DataType::Half:
            test_device_batch_gemm_work< Half_t, Half_t, Half_t >( params, run );
            break;
        default:
            throw std::exception();
            break;
    }
}
