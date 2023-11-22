#include "blas/device_blas.hh"

#include "device_internal.hh"

#include <limits>

namespace blas {

//==============================================================================
namespace impl {

//------------------------------------------------------------------------------
/// Mid-level templated wrapper checks and converts arguments,
/// then calls low-level wrapper.
/// @ingroup trmv_internal
///
template <typename scalar_t>
void trmv(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t n,
    scalar_t const* A, int64_t lda,
    scalar_t*       x, int64_t incx,
    blas::Queue& queue )
{
#ifndef BLAS_HAVE_DEVICE
    throw blas::Error( "device BLAS not available", __func__ );
#else
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans &&
                   trans != Op::ConjTrans );
    blas_error_if( diag != Diag::NonUnit &&
                   diag != Diag::Unit );
    blas_error_if( n < 0 );
    blas_error_if( incx == 0 );
    blas_error_if( lda < n );

    // convert arguments
    device_blas_int n_   = to_device_blas_int( n );
    device_blas_int lda_ = to_device_blas_int( lda );
    device_blas_int incx_ = to_device_blas_int( incx );

    blas::Op trans2 = trans;
    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        // A => A^T; A^T => A; A^H => A + conj
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        trans2 = (trans == Op::NoTrans ? Op::Trans : Op::NoTrans);

        if constexpr (is_complex<scalar_t>::value) {
            if (trans == Op::ConjTrans) {
                // conjugate x (in-place)
                // int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
                // for (int64_t i = 0; i < n; ++i) {
                //     x[ ix ] = conj( x[ ix ] );
                //     ix += incx;
                // }
                scalar_t* x_host = new scalar_t[ n ];
                blas::device_copy_vector(n, x, std::abs(incx), x_host, std::abs(incx),queue);
                queue.sync();
                int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
                for (int64_t i = 0; i < n; ++i) {
                    x_host[ ix ] = conj( x_host[ ix ]);
                    ix += incx;
                }
                blas::device_copy_vector(n, x_host, std::abs(incx), x, std::abs(incx), queue);
                queue.sync();
                delete[] x_host;
            }
        }
    }

    blas::internal_set_device( queue.device() );
    
    // call low-level wrapper
    internal::trmv( uplo, trans2, diag, n_,
                    A, lda_, x, incx_, queue );

    if constexpr (is_complex<scalar_t>::value) {
    if (layout == Layout::RowMajor && trans == Op::ConjTrans) {
        scalar_t* x_host = new scalar_t[ n ];
        blas::device_copy_vector(n, x, std::abs(incx), x_host, std::abs(incx),queue);
        queue.sync();
        int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
        for (int64_t i = 0; i < n; ++i) {
            x_host[ ix ] = conj( x_host[ ix ]);
            ix += incx;
        }
        blas::device_copy_vector(n, x_host, std::abs(incx), x, std::abs(incx), queue);
        queue.sync();
        delete[] x_host;
    }
}
#endif
}

}  // namespace impl

//==============================================================================
// High-level overloaded wrappers call mid-level templated wrapper.

//------------------------------------------------------------------------------
/// GPU device, float version.
/// @ingroup trmv
void trmv(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t n,
    float const* A, int64_t lda,
    float*       x, int64_t incx,
    blas::Queue& queue )
{
    impl::trmv( layout, uplo, trans, diag, n,
                A, lda, x, incx, queue );
}

//------------------------------------------------------------------------------
/// GPU device, double version.
/// @ingroup trmv
void trmv(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t n,
    double const* A, int64_t lda,
    double*       x, int64_t incx,
    blas::Queue& queue )
{
    impl::trmv( layout, uplo, trans, diag, n,
                A, lda, x, incx, queue );
}

//------------------------------------------------------------------------------
/// GPU device, complex<float> version.
/// @ingroup trmv
void trmv(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t n,
    std::complex<float> const* A, int64_t lda,
    std::complex<float>*       x, int64_t incx,
    blas::Queue& queue )
{
    impl::trmv( layout, uplo, trans, diag, n,
                A, lda, x, incx, queue );
}

//------------------------------------------------------------------------------
/// GPU device, complex<double> version.
/// @ingroup trmv
void trmv(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t n,
    std::complex<double> const* A, int64_t lda,
    std::complex<double>*       x, int64_t incx,
    blas::Queue& queue )
{
    impl::trmv( layout, uplo, trans, diag, n,
                A, lda, x, incx, queue );
}

}  // namespace blas
