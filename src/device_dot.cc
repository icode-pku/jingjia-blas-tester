// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/device_blas.hh"

#include "device_internal.hh"

#include <limits>

// -----------------------------------------------------------------------------
namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
/// @ingroup dot
void dot(
    int64_t n,
    float const *dx, int64_t incdx,
    float const *dy, int64_t incdy,
    float *result,
    blas::Queue& queue)
{
    // check arguments
    blas_error_if( n < 0 );       // standard BLAS returns, doesn't fail
    blas_error_if( incdx == 0 );  // standard BLAS doesn't detect incdx == 0
    blas_error_if( incdy == 0 );  // standard BLAS doesn't detect incdy == 0

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(device_blas_int)) {
        blas_error_if( n     > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( incdx > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( incdy > std::numeric_limits<device_blas_int>::max() );
    }

    device_blas_int n_     = (device_blas_int) n;
    device_blas_int incdx_ = (device_blas_int) incdx;
    device_blas_int incdy_ = (device_blas_int) incdy;

    blas::internal_set_device( queue.device() );

    device::sdot( queue, n_, dx, incdx_, dy, incdy_, result );
}

// -----------------------------------------------------------------------------
/// @ingroup dot
void dot(
    int64_t n,
    double const *dx, int64_t incdx,
    double const *dy, int64_t incdy,
    double *result,
    blas::Queue& queue)
{
    // check arguments
    blas_error_if( n < 0 );       // standard BLAS returns, doesn't fail
    blas_error_if( incdx == 0 );  // standard BLAS doesn't detect inc[dx] == 0
    blas_error_if( incdy == 0 );  // standard BLAS doesn't detect inc[dy] == 0

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(device_blas_int)) {
        blas_error_if( n     > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( incdx > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( incdy > std::numeric_limits<device_blas_int>::max() );
    }

    device_blas_int n_     = (device_blas_int) n;
    device_blas_int incdx_ = (device_blas_int) incdx;
    device_blas_int incdy_ = (device_blas_int) incdy;

    blas::internal_set_device( queue.device() );

    device::ddot( queue, n_, dx, incdx_, dy, incdy_, result );
}

// -----------------------------------------------------------------------------
/// @ingroup dot
void dot(
    int64_t n,
    std::complex<float> const *dx, int64_t incdx,
    std::complex<float> const *dy, int64_t incdy,
    std::complex<float> *result,
    blas::Queue& queue)
{
    // check arguments
    blas_error_if( n < 0 );       // standard BLAS returns, doesn't fail
    blas_error_if( incdx == 0 );  // standard BLAS doesn't detect inc[dx] == 0
    blas_error_if( incdy == 0 );  // standard BLAS doesn't detect inc[dy] == 0

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(device_blas_int)) {
        blas_error_if( n     > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( incdx > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( incdy > std::numeric_limits<device_blas_int>::max() );
    }

    device_blas_int n_     = (device_blas_int) n;
    device_blas_int incdx_ = (device_blas_int) incdx;
    device_blas_int incdy_ = (device_blas_int) incdy;

    blas::internal_set_device( queue.device() );

    device::cdotc( queue, n_, dx, incdx_, dy, incdy_, result );
}

// -----------------------------------------------------------------------------
/// @ingroup dot
void dot(
    int64_t n,
    std::complex<double> const *dx, int64_t incdx,
    std::complex<double> const *dy, int64_t incdy,
    std::complex<double> *result,
    blas::Queue& queue)
{
    // check arguments
    blas_error_if( n < 0 );       // standard BLAS returns, doesn't fail
    blas_error_if( incdx == 0 );  // standard BLAS doesn't detect inc[dx] == 0
    blas_error_if( incdy == 0 );  // standard BLAS doesn't detect inc[dy] == 0

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(device_blas_int)) {
        blas_error_if( n     > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( incdx > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( incdy > std::numeric_limits<device_blas_int>::max() );
    }

    device_blas_int n_     = (device_blas_int) n;
    device_blas_int incdx_ = (device_blas_int) incdx;
    device_blas_int incdy_ = (device_blas_int) incdy;

    blas::internal_set_device( queue.device() );

    device::zdotc( queue, n_, dx, incdx_, dy, incdy_, result );
}

// =============================================================================
// Unconjugated version, x^T y

// -----------------------------------------------------------------------------
/// @ingroup dotu
void dotu(
    int64_t n,
    float const *dx, int64_t incdx,
    float const *dy, int64_t incdy,
    float *result,
    blas::Queue& queue)
{
    dot( n, dx, incdx, dy, incdy, result, queue );
}

// -----------------------------------------------------------------------------
/// @ingroup dotu
void dotu(
    int64_t n,
    double const *dx, int64_t incdx,
    double const *dy, int64_t incdy,
    double *result,
    blas::Queue& queue)
{
    dot( n, dx, incdx, dy, incdy, result, queue );
}

// -----------------------------------------------------------------------------
/// @ingroup dotu
void dotu(
    int64_t n,
    std::complex<float> const *dx, int64_t incdx,
    std::complex<float> const *dy, int64_t incdy,
    std::complex<float> *result,
    blas::Queue& queue)
{
    // check arguments
    blas_error_if( n < 0 );       // standard BLAS returns, doesn't fail
    blas_error_if( incdx == 0 );  // standard BLAS doesn't detect inc[dx] == 0
    blas_error_if( incdy == 0 );  // standard BLAS doesn't detect inc[dy] == 0

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(device_blas_int)) {
          blas_error_if( n     > std::numeric_limits<device_blas_int>::max() );
          blas_error_if( incdx > std::numeric_limits<device_blas_int>::max() );
          blas_error_if( incdy > std::numeric_limits<device_blas_int>::max() );
    }

    device_blas_int n_     = (device_blas_int) n;
    device_blas_int incdx_ = (device_blas_int) incdx;
    device_blas_int incdy_ = (device_blas_int) incdy;

    blas::internal_set_device( queue.device() );

    device::cdotu( queue, n_, dx, incdx_, dy, incdy_, result );
}

// -----------------------------------------------------------------------------
/// @ingroup dotu
void dotu(
    int64_t n,
    std::complex<double> const *dx, int64_t incdx,
    std::complex<double> const *dy, int64_t incdy,
    std::complex<double> *result,
    blas::Queue& queue)
{
    // check arguments
    blas_error_if( n < 0 );       // standard BLAS returns, doesn't fail
    blas_error_if( incdx == 0 );  // standard BLAS doesn't detect inc[dx] == 0
    blas_error_if( incdy == 0 );  // standard BLAS doesn't detect inc[dy] == 0

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(device_blas_int)) {
          blas_error_if( n     > std::numeric_limits<device_blas_int>::max() );
          blas_error_if( incdx > std::numeric_limits<device_blas_int>::max() );
          blas_error_if( incdy > std::numeric_limits<device_blas_int>::max() );
    }

    device_blas_int n_     = (device_blas_int) n;
    device_blas_int incdx_ = (device_blas_int) incdx;
    device_blas_int incdy_ = (device_blas_int) incdy;

    blas::internal_set_device( queue.device() );

    device::zdotu( queue, n_, dx, incdx_, dy, incdy_, result );
}
} // namespace blas
