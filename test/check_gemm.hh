// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef CHECK_GEMM_HH
#define CHECK_GEMM_HH

#include "blas/util.hh"

// Test headers.
#include "lapack_wrappers.hh"

#include <limits>
#include <assert.h>
#include <complex>
#include <cmath>
using testsweeper::ansi_bold;
using testsweeper::ansi_red;
using testsweeper::ansi_normal;
// -----------------------------------------------------------------------------
// Computes error for multiplication with general matrix result.
// Covers dot, gemv, ger, geru, gemm, symv, hemv, symm, trmv, trsv?, trmm, trsm?.
// Cnorm is norm of original C, before multiplication operation.
template <typename T>
void print_value( T *C, int64_t ldc, T const* Cref, int64_t ldcref, int64_t i, int64_t j)
{
    #define    C(i_, j_)    C[ (i_) + (j_)*ldc ]
    #define Cref(i_, j_) Cref[ (i_) + (j_)*ldcref ]
    printf("%s%sIn (%ld, %ld), Device result is %f, Host(ref) result is %f%s\n",ansi_bold, ansi_red, i, j, C(i, j), Cref(i, j),ansi_normal);
    #undef C
    #undef Cref
}   

template <typename T>
void print_value( std::complex<T> *C, int64_t ldc, std::complex<T> const* Cref, int64_t ldcref, int64_t i, int64_t j)
{
    #define    C(i_, j_)    C[ (i_) + (j_)*ldc ]
    #define Cref(i_, j_) Cref[ (i_) + (j_)*ldcref ]
    printf("%s%sIn (%ld, %ld), Device result is (%fi + %f), Host(ref) result is (%fi + %f)%s\n",ansi_bold, ansi_red,i, j, real(C(i, j)), imag(C(i, j)), real(Cref(i, j)), imag(Cref(i, j)),ansi_normal);
    #undef C
    #undef Cref
}   

// Error margin: numbers beyond this value are considered equal to inf or NaN
template <typename T>
T getAlmostInfNumber() {
  return static_cast<T>(1e35); // used for correctness testing of TRSV and TRSM routines
}


template<typename T>
inline bool check(const T val1, const T val2, bool ishalf = false){
    const auto error_margin_relative = ishalf ? static_cast<float>(blas::paramspace::relative<Half_t>) : static_cast<T>(blas::paramspace::relative<T>);
    const auto error_margin_absolute = ishalf ? static_cast<float>(blas::paramspace::absolute<Half_t>) : static_cast<T>(blas::paramspace::absolute<T>);

    const auto difference = std::fabs(val1- val2);
    if(val1 == val2) return true;
    else if ((std::isnan(val1) && std::isnan(val2)) || (std::isinf(val1) && std::isinf(val2))) {
        return true;
    }
    else if ((std::isnan(val1) && std::isinf(val2)) || (std::isinf(val1) && std::isnan(val2))) {
        return true;
    }
    else if ((std::abs(val1) > getAlmostInfNumber<T>() && (std::isinf(val2) || std::isnan(val2))) ||
           (std::abs(val2) > getAlmostInfNumber<T>() && (std::isinf(val1) || std::isnan(val1)))) {
        return true;
    }
    // The values are zero or very small: the relative error is less meaningful
    else if (val1 == 0 || val2 == 0 || difference < error_margin_absolute) {
        return (difference < error_margin_absolute);
    }
    else if(std::isnan(val2) || std::isnan(val1) || std::isinf(val2) || std::isinf(val1)){
      return false;
    }
    else {
        const auto absolute_sum = std::fabs(val1) + std::fabs(val2);
        return (difference / absolute_sum) < error_margin_relative;
    }
}
// template bool check<float>(const float, const float);
// template bool check<double>(const double, const double);

template <>
inline bool check(const std::complex<float> val1, const std::complex<float> val2, bool ishalf) {
  const auto realf = check(real(val1), real(val2));
  const auto imagf = check(imag(val1), imag(val2));
  if (realf && imagf) { return true; }
  // also OK if one is good and the combined is good (indicates a big diff between real & imag)
  if (realf || imagf) { return check(real(val1) + imag(val1), real(val2) + imag(val2)); }
  return false; // neither real nor imag is good, return false
}

template <>
inline bool check(const std::complex<double> val1, const std::complex<double> val2, bool ishalf) {
  const auto reald = check(real(val1), real(val2));
  const auto imagd = check(imag(val1), imag(val2));
  if (reald && imagd) { return true; }
  // also OK if one is good and the combined is good (indicates a big diff between real & imag)
  if (reald || imagd) { return check(real(val1) + imag(val1), real(val2) + imag(val2)); }
  return false; // neither real nor imag is good, return false
}



template <typename T>
void check_gemm(
    int64_t m, int64_t n, int64_t k,
    T alpha,
    T beta,
    blas::real_type<T> Anorm,
    blas::real_type<T> Bnorm,
    blas::real_type<T> Cnorm,
    T const* Cref, int64_t ldcref,
    T* C, int64_t ldc,
    bool verbose,
    blas::real_type<T> error[1],
    bool* okay , bool ishalf = false)
{
    #define    C(i_, j_)    C[ (i_) + (j_)*ldc ]
    #define    Cd(i_, j_)    Cd[ (i_) + (j_)*ldc ]
    #define Cref(i_, j_) Cref[ (i_) + (j_)*ldcref ]

    typedef blas::real_type<T> real_t;

    assert( m >= 0 );
    assert( n >= 0 );
    assert( k >= 0 );
    assert( ldc >= m );
    assert( ldcref >= m );
    // if(std::is_same<T, float>::value||std::is_same<T, double>::value){
    //     C(0,0)+=5.0;
    // }
    // else{
    //     C(0,0)-=Cref(0,0);
    //     C(0,1)-=Cref(0,1);
    // }
    T* Cd = (T*)malloc(sizeof(T)*ldc*n);
    // C -= Cref
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < m; ++i) {
            Cd(i, j) = C(i, j);
            C(i, j) -= Cref(i, j);
        }
    }

    real_t work[1], Cout_norm;
    Cout_norm = lapack_lange( "f", m, n, C, ldc, work );
    error[0] = Cout_norm
             / (sqrt(real_t(k)+2)*std::abs(alpha)*Anorm*Bnorm
                 + 2*std::abs(beta)*Cnorm);
    if (verbose) {
        printf( "error: ||Cout||=%.2e / (sqrt(k=%lld + 2)"
                " * |alpha|=%.2e * ||A||=%.2e * ||B||=%.2e"
                " + 2 * |beta|=%.2e * ||C||=%.2e) = %.2e\n",
                Cout_norm, llong( k ),
                std::abs( alpha ), Anorm, Bnorm,
                std::abs( beta ), Cnorm, error[0] );
    }

    // complex needs extra factor; see Higham, 2002, sec. 3.6.
    if (blas::is_complex<T>::value) {
        error[0] /= 2*sqrt(2);
    }
    real_t u = ishalf ? blas::paramspace::correct_threshld< Half_t > : blas::paramspace::correct_threshld< real_t >;
    *okay = (error[0] < u);
    bool hasprint = false;
    if(!(*okay)){
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = 0; i < m; ++i) {
                if( ! check(Cd(i,j), Cref(i,j), true) ){
                    print_value(Cd, ldc, Cref, ldcref, i, j);
                    hasprint = true;
                }
            }
        }
    }
    //Two standards, just meet one of them
    *okay = *okay | (!hasprint);
    #undef C
    #undef Cref
    #undef Cd
    free(Cd);
}

// -----------------------------------------------------------------------------
// Computes error for multiplication with symmetric or Hermitian matrix result.
// Covers syr, syr2, syrk, syr2k, her, her2, herk, her2k.
// Cnorm is norm of original C, before multiplication operation.
//
// alpha and beta are either real or complex, depending on routine:
//          zher    zher2   zherk   zher2k  zsyr    zsyr2   zsyrk   zsyr2k
// alpha    real    complex real    complex complex complex complex complex
// beta     --      --      real    real    --      --      complex complex
// zsyr2 doesn't exist in standard BLAS or LAPACK.
template <typename TA, typename TB, typename T>
void check_herk(
    blas::Uplo uplo,
    int64_t n, int64_t k,
    TA alpha,
    TB beta,
    blas::real_type<T> Anorm,
    blas::real_type<T> Bnorm,
    blas::real_type<T> Cnorm,
    T const* Cref, int64_t ldcref,
    T* C, int64_t ldc,
    bool verbose,
    blas::real_type<T> error[1],
    bool* okay )
{
    #define    C(i_, j_)    C[ (i_) + (j_)*ldc ]
    #define    Cd(i_, j_)    Cd[ (i_) + (j_)*ldc ]
    #define Cref(i_, j_) Cref[ (i_) + (j_)*ldcref ]

    typedef blas::real_type<T> real_t;

    assert( n >= 0 );
    assert( k >= 0 );
    assert( ldc >= n );
    assert( ldcref >= n );
    // if(std::is_same<T, float>::value||std::is_same<T, double>::value){
    //     C(0,0)+=5.0;
    // }
    // else{
    //     C(0,0)-=Cref(0,0);
    //     C(0,1)-=Cref(0,1);
    // }
    T* Cd = (T*)malloc(sizeof(T)*ldc*n);
    // C -= Cref
    if (uplo == blas::Uplo::Lower) {
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = j; i < n; ++i) {
                Cd(i, j) = C(i, j);
                C(i, j) -= Cref(i, j);
            }
        }
    }
    else {
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = 0; i <= j; ++i) {
                Cd(i, j) = C(i, j);
                C(i, j) -= Cref(i, j);
            }
        }
    }

    // For a rank-2k update, this should be
    // sqrt(k+3) |alpha| (norm(A)*norm(B^T) + norm(B)*norm(A^T))
    //     + 3 |beta| norm(C)
    // However, so far using the same bound as rank-k works fine.
    real_t work[1], Cout_norm;
    Cout_norm = lapack_lanhe( "f", uplo2str(uplo), n, C, ldc, work );
    error[0] = Cout_norm
             / (sqrt(real_t(k)+2)*std::abs(alpha)*Anorm*Bnorm
                 + 2*std::abs(beta)*Cnorm);
    if (verbose) {
        printf( "error: ||Cout||=%.2e / (sqrt(k=%lld + 2)"
                " * |alpha|=%.2e * ||A||=%.2e * ||B||=%.2e"
                " + 2 * |beta|=%.2e * ||C||=%.2e) = %.2e\n",
                Cout_norm, llong( k ),
                std::abs( alpha ), Anorm, Bnorm,
                std::abs( beta ), Cnorm, error[0] );
    }

    // complex needs extra factor; see Higham, 2002, sec. 3.6.
    if (blas::is_complex<T>::value) {
        error[0] /= 2*sqrt(2);
    }

    real_t u = blas::paramspace::correct_threshld< real_t >;//std::numeric_limits< real_t >::epsilon();
    *okay = (error[0] < u);
    if(!(*okay)){
        if (uplo == blas::Uplo::Lower) {
            for (int64_t j = 0; j < n; ++j) {
                for (int64_t i = j; i < n; ++i) {
                    if( !check(Cd(i,j), Cref(i,j)) ){
                        print_value(Cd, ldc, Cref, ldcref, i, j);
                    }
                }
            }
        }
        else {
            for (int64_t j = 0; j < n; ++j) {
                for (int64_t i = 0; i <= j; ++i) {
                    if( !check(Cd(i,j), Cref(i,j)) ){
                        print_value(Cd, ldc, Cref, ldcref, i, j);
                    }
                }
            }
        }
    }
    #undef C
    #undef Cref
    #undef Cd
    free(Cd);
}

#endif        //  #ifndef CHECK_GEMM_HH
