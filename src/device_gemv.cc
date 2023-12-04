#include "blas/device_blas.hh"

#include "device_internal.hh"

#include <limits>

namespace blas {

//==============================================================================
namespace impl {

//------------------------------------------------------------------------------
/// Mid-level templated wrapper checks and converts arguments,
/// then calls low-level wrapper.
/// @ingroup gemv_internal
///
template <typename scalar_t>
void gemv(
    blas::Layout layout,
    blas::Op trans,
    int64_t m, int64_t n,
    scalar_t alpha,
    scalar_t const* A, int64_t lda,
    scalar_t const* x, int64_t incx,
    scalar_t beta,
    scalar_t* y, int64_t incy,
    blas::Queue& queue, int64_t testcase = 1, char *errname = nullptr )
{
#ifndef BLAS_HAVE_DEVICE
    throw blas::Error( "device BLAS not available", __func__ );
#else
    // check arguments
    if(testcase == 1){
        blas_error_if( layout != Layout::ColMajor &&
                    layout != Layout::RowMajor );
        blas_error_if( trans != Op::NoTrans &&
                    trans != Op::Trans &&
                    trans != Op::ConjTrans );
        blas_error_if( m < 0 );
        blas_error_if( n < 0 );
        
        if (layout == Layout::ColMajor) {
            // if (trans == Op::NoTrans)
                blas_error_if( lda < m );
            // else
            //     blas_error_if( lda < m );
        }
        else {
            // if (trans != Op::NoTrans)
                blas_error_if( lda < n );
            // else
            //     blas_error_if( lda < n );
        }
        blas_error_if( incx == 0 );
        blas_error_if( incy == 0 );
    }

    // convert arguments
    device_blas_int m_   = to_device_blas_int( m );
    device_blas_int n_   = to_device_blas_int( n );
    device_blas_int lda_ = to_device_blas_int( lda );
    device_blas_int incx_ = to_device_blas_int( incx );
    device_blas_int incy_ = to_device_blas_int( incy );
    device_blas_int testcase_ = to_device_blas_int( testcase );

    blas::internal_set_device( queue.device() );

    //call low-level wrapper
    // if (layout == Layout::RowMajor) {
    //     // swap m <=> n
    //     blas::Op trans2 = (trans == Op::NoTrans ? Op::Trans : Op::NoTrans);
    //     internal::gemv( trans2, n_, m_, alpha, A, lda_, 
    //                     x, incx_, beta, y, incy_, queue );
    // }
    // else {
    //     internal::gemv( trans, m_, n_, alpha, A, lda_, 
    //                     x, incx_, beta, y, incy_, queue );
    // }
    
    // Deal with layout. RowMajor ConjTrans needs copy of x in x2;
    // in other cases, x2 == x.
    scalar_t* x2 = const_cast< scalar_t* >( x );
    Op trans2 = trans;
    if (layout == Layout::RowMajor) {
        if constexpr (is_complex<scalar_t>::value) {
            if (trans == Op::ConjTrans) {
                alpha = conj( alpha );
                beta  = conj( beta );

                //x2 = new scalar_t[ m ];
                scalar_t* x_host = new scalar_t[ m ];
                blas::device_copy_vector(m, x, std::abs(incx), x_host, std::abs(incx),queue);
                queue.sync();
                scalar_t* x2_host = new scalar_t[ m ];
                x2 = blas::device_malloc<scalar_t>( m, queue );
                int64_t ix = (incx > 0 ? 0 : (-m + 1)*incx);
                for (int64_t i = 0; i < m; ++i) {
                    x2_host[ i ] = conj( x_host [ ix ]);
                    ix += incx;
                    //x2[ i ] = conj( x[ ix ] );
                }
                incx_ = 1;
                blas::device_copy_vector(m, x2_host, std::abs(incx), x2, std::abs(incx), queue);
                queue.sync();
                delete[] x_host;
                delete[] x2_host;

                scalar_t* y1_host = new scalar_t[ n ];
                blas::device_copy_vector(n, y, std::abs(incy), y1_host, std::abs(incy),queue);
                queue.sync();
                int64_t iy = (incy > 0 ? 0 : (-n + 1)*incy);
                for (int64_t i = 0; i < n; ++i) {
                    //y[ iy ] = conj( y[ iy ] );
                    y1_host[ iy ] = conj( y1_host[ iy ] );
                    iy += incy;
                }
                blas::device_copy_vector(n, y1_host, std::abs(incy), y, std::abs(incy),queue);
                queue.sync();
                delete[] y1_host;
            }
        }
        // A => A^T; A^T => A; A^H => A + conj
        std::swap( m_, n_ );
        trans2 = (trans == Op::NoTrans ? Op::Trans : Op::NoTrans);
    }
    //char trans_ = op2char( trans2 );

    //call low-level wrapper
    internal::gemv( trans2, m_, n_,
                    alpha, A, lda_, x2, incx_, beta, y, incy_, queue, testcase_, errname );

    if constexpr (is_complex<scalar_t>::value) {
        if (x2 != x) {  // RowMajor ConjTrans
            // y = conj( y )
            scalar_t* y2_host = new scalar_t[ n ];
            blas::device_copy_vector(n, y, std::abs(incy), y2_host, std::abs(incy),queue);
            queue.sync();
            int64_t iy = (incy > 0 ? 0 : (-n + 1)*incy);
            for (int64_t i = 0; i < n; ++i) {
                y2_host[ iy ] = conj( y2_host[ iy ] );
                iy += incy;
            }
            blas::device_copy_vector(n, y2_host, std::abs(incy), y, std::abs(incy),queue);
            queue.sync();
            //delete[] x2;
            delete[] y2_host;
            blas::device_free( x2, queue );
        }
    }
#endif
}

}  // namespace impl

//==============================================================================
// High-level overloaded wrappers call mid-level templated wrapper.

//------------------------------------------------------------------------------
/// GPU device, float version.
/// @ingroup gemv
void gemv(
    blas::Layout layout,
    blas::Op trans,
    int64_t m, int64_t n,
    float alpha,
    float const* A, int64_t lda,
    float const* x, int64_t incx,
    float beta,
    float* y, int64_t incy,
    blas::Queue& queue, int64_t testcase, char *errname )
{
    impl::gemv( layout, trans, m, n, alpha, A, lda,
                x, incx, beta, y, incy, queue, testcase, errname );
}

//------------------------------------------------------------------------------
/// GPU device, double version.
/// @ingroup gemv
void gemv(
    blas::Layout layout,
    blas::Op trans,
    int64_t m, int64_t n,
    double alpha,
    double const* A, int64_t lda,
    double const* x, int64_t incx,
    double beta,
    double* y, int64_t incy,
    blas::Queue& queue, int64_t testcase, char *errname )
{
    impl::gemv( layout, trans, m, n, alpha, A, lda,
                x, incx, beta, y, incy, queue, testcase, errname );
}

//------------------------------------------------------------------------------
/// GPU device, complex<float> version.
/// @ingroup gemv
void gemv(
    blas::Layout layout,
    blas::Op trans,
    int64_t m, int64_t n,
    std::complex<float> alpha,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* x, int64_t incx,
    std::complex<float> beta,
    std::complex<float>* y, int64_t incy,
    blas::Queue& queue, int64_t testcase, char *errname )
{
    impl::gemv( layout, trans, m, n, alpha, A, lda,
                x, incx, beta, y, incy, queue, testcase, errname );
}

//------------------------------------------------------------------------------
/// GPU device, complex<double> version.
/// @ingroup gemv
void gemv(
    blas::Layout layout,
    blas::Op trans,
    int64_t m, int64_t n,
    std::complex<double> alpha,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* x, int64_t incx,
    std::complex<double> beta,
    std::complex<double>* y, int64_t incy,
    blas::Queue& queue, int64_t testcase, char *errname )
{
    impl::gemv( layout, trans, m, n, alpha, A, lda,
                x, incx, beta, y, incy, queue, testcase, errname );
}

}  // namespace blas
