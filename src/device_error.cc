// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/device.hh"

#include "device_internal.hh"

#ifdef BLAS_HAVE_CUBLAS

// -----------------------------------------------------------------------------
// return string of blas error
const char* blas::device_error_string( cublasStatus_t error )
{
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "cublas: success";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "cublas: not initialized";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "cublas: out of memory";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "cublas: invalid value";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "cublas: architecture mismatch";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "cublas: memory mapping error";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "cublas: execution failed";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "cublas: internal error";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "cublas: functionality not supported";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "cublas: license error";

        default:
            return "cublas: unknown error code";
    }
}

const char* blas::device_errorstatus_to_string(cublasStatus_t error)
{
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";

        default:
            return "CUBLAS_UNKNOWN_ERROR";
    }

}

#endif  // HAVE_CUBLAS
