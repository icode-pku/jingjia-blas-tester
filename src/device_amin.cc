#include "blas/device_blas.hh"
#include "device_internal.hh"

#include <limits>

namespace blas {

//==============================================================================
namespace impl {

template <typename scalar_t>
void amin(
    int64_t n,
    scalar_t const *x, int64_t incx,
    int64_t *result,
    blas::Queue &queue, int64_t testcase = 1, char *errname = nullptr )
{
#ifndef BLAS_HAVE_DEVICE
    throw blas::Error( "device BLAS not available", __func__ );
#else
    if(testcase == 1){
        blas_error_if(n < 0);
        blas_error_if(incx == 0);
    }
    //check param
    device_blas_int n_ = to_device_blas_int(n);
    device_blas_int incx_ = to_device_blas_int(incx);
    device_blas_int *result_ = to_device_blas_intp(result);
    device_blas_int testcase_ = to_device_blas_int( testcase );

    blas::internal_set_device( queue.device() );
    #if defined( BLAS_HAVE_SYCL )
        return;
    #else
        internal::amin(n_, x, incx_, result_, queue, testcase_, errname );
    #endif
#endif
}


}//namespace impl


///@ingroup amin
void amin(    
    int64_t n,
    float const* dx, int64_t incdx,
    int64_t *result, 
    blas::Queue& queue, int64_t testcase, char *errname ){
        impl::amin(n, dx, incdx, result, queue, testcase, errname );
    }
//---------------------------------------------------------
///@ingroup amin
void amin(    
    int64_t n,
    double const* dx, int64_t incdx,
    int64_t *result, 
    blas::Queue& queue, int64_t testcase, char *errname ){
        impl::amin(n, dx, incdx, result, queue, testcase, errname );
    }
//---------------------------------------------------------
///@ingroup amin
void amin(    
    int64_t n,
    std::complex<float> const* dx, int64_t incdx,
    int64_t *result, 
    blas::Queue& queue, int64_t testcase, char *errname ){
        impl::amin(n, dx, incdx, result, queue, testcase, errname );
    }
//---------------------------------------------------------
///@ingroup amin
void amin(    
    int64_t n,
    std::complex<double> const* dx, int64_t incdx,
    int64_t *result, 
    blas::Queue& queue, int64_t testcase, char *errname ){
        impl::amin(n, dx, incdx, result, queue, testcase, errname );
    }
//----------------------------------------------------------




}//namespace blas