#pragma once

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

    typedef enum {
        vector_math_add,
        vector_math_substract,
        vector_math_divide,
        vector_math_multiply,
        vector_math_scalar_multiply,
        vector_math_scalar_add,
        vector_math_substact_scalar_multiply,
        vector_math_add_scalar_multiply,
        vector_math_sqrt,
    } vector_math_op_t;

    typedef enum {
        vector_forward_tanh,
        vector_forward_tanf,
        vector_forward_sigmoid,
    } vector_forward_op_t;

    typedef enum {
        vector_backward_tanh,
        vector_backward_tanf,
        vector_backward_sigmoid,
    } vector_backward_op_t;

    // Helper function for using CUDA to add vectors in parallel.
    cudaError_t cuda_inplace_vectors_math_op(double* a, const double* b, unsigned int size, double scalar, vector_math_op_t operation);
    cudaError_t cuda_forward_vectors_math_op(double* y, const double* x, unsigned int size, vector_forward_op_t operation);
    cudaError_t cuda_backward_vectors_math_op(double* dx, const double* y, const double* dy, unsigned int size, vector_backward_op_t operation);

#ifdef __cplusplus
}
#endif // __cplusplus

