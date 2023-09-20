//
//  layers.c
//  recurrent-neural-net
//
//  Created by Stan Miasnikov on 10/11/22.
//

/*
 * Dealing with FC layers, forward and backward
 *
 * ==================== NOTE: ====================
 *   The caller should have thought about the memory
 *    allocation, these functions assumes that
 *           everything is OK.
 * =================================================
 *
 */

#include "layers.h"

#ifdef _WIN32
#include <stdio.h>
#endif

#ifdef _BUILD_FOR_CUDA
#include "cuda_utils.cuh"

#define SHORT_VECTOR    50

#endif // _BUILD_FOR_CUDA

//    Y = AX + b        &Y,      A,       X,    B,     Rows (for A), Columns (for A)
void  fully_connected_forward(numeric_t* Y, const numeric_t* A, const numeric_t* X, const numeric_t* b, int R, int C)
{
    int i, n;
    for (i = 0; i < R; i++)
    {
        Y[i] = b[i];
        n = 0;
        for (n = 0; n < C; n++) 
        {
            Y[i] += A[i * C + n] * X[n];
        }
    }
}
//    Y = AX + b        dldY,       A,     X,        &dldA,    &dldX,    &dldb   Rows (A), Columns (A)
void fully_connected_backward(const numeric_t* dldY, const numeric_t* A, const numeric_t* X,
    numeric_t* dldA, numeric_t* dldX, numeric_t* dldb, int R, int C)
{
    int i, n;
    for (i = 0; i < R; i++)
    {
        for (n = 0; n < C; n++)
        {
            dldA[i * C + n] = dldY[i] * X[n];
        }
    }
    
    // computing dldb (easy peasy)
    for (i = 0; i < R; i++) 
    {
        dldb[i] = dldY[i];
    }
    
    // computing dldX
    for (i = 0; i < C; i++) 
    {
        dldX[i] = 0.0;
        for (n = 0; n < R; n++) 
        {
            dldX[i] += A[n * C + i] * dldY[n];
        }
    }
}

numeric_t cross_entropy(const numeric_t* probabilities, int correct)
{
    return -log(probabilities[correct]);
}

// Dealing with softmax layer, forward and backward
//                &P,   Y,    features
void  softmax_layers_forward(numeric_t* P, const numeric_t* Y, int F, numeric_t temperature)
{
    int f = 0;
    numeric_t sum = 0;
#ifdef _WIN32
    // MSVC is not a C99 compiler, and does not support variable length arrays
    // MSVC is documented as conforming to C90
    numeric_t *cache = malloc(sizeof(numeric_t)*F);
    if ( cache == NULL )
    {
        fprintf(stderr, "%s.%s.%d malloc(%zu) failed\r\n",
                __FILE__, __func__, __LINE__, sizeof(numeric_t)*F);
        exit(1);
    }
#else
    numeric_t cache[F];
#endif
    
    while ( f < F ) 
    {
        cache[f] = exp(Y[f] / temperature);
        sum += cache[f];
        ++f;
    }
    
    f = 0;
    while ( f < F ) 
    {
        P[f] = cache[f] / sum;
        ++f;
    }
    
#ifdef _WIN32
    free(cache);
#endif
}
//                    P,    c,  &dldh, rows
void  softmax_loss_layer_backward(const numeric_t* P, int c, numeric_t* dldh, int R)
{
    int r = 0;
    
    while ( r < R ) 
    {
        dldh[r] = P[r];
        ++r;
    }
    
    dldh[c] -= 1.0;
}
// Other layers used: sigmoid and tanh
//
//    Y = sigmoid(X), &Y, X, length
void  sigmoid_forward(numeric_t* Y, const numeric_t* X, int L)
{
#ifdef _BUILD_FOR_CUDA
    if (L > SHORT_VECTOR)
    {
        if (cudaSuccess == cuda_forward_vectors_math_op(Y, X, L, vector_forward_sigmoid))
            return;
    }
#endif // _BUILD_FOR_CUDA
    int l = 0;
    while (l < L)
    {
        Y[l] = 1.0 / (1.0 + exp(-X[l]));
        ++l;
    }
}
//    Y = sigmoid(X), dldY, Y, &dldX, length
void  sigmoid_backward(const numeric_t* dldY, const numeric_t* Y, numeric_t* dldX, int L)
{
#ifdef _BUILD_FOR_CUDA
    if (L > SHORT_VECTOR)
    {
        if (cudaSuccess == cuda_backward_vectors_math_op(dldX, Y, dldY, L, vector_backward_sigmoid))
            return;
    }
#endif // _BUILD_FOR_CUDA
    int l = 0;
    while (l < L)
    {
        dldX[l] = (1.0 - Y[l]) * Y[l] * dldY[l];
        ++l;
    }
}

//    Y = tanh(X), &Y, X, length
void  tanh_forward(numeric_t* Y, const numeric_t* X, int L)
{
#ifdef _BUILD_FOR_CUDA
    if (L > SHORT_VECTOR)
    {
        if (cudaSuccess == cuda_forward_vectors_math_op(Y, X, L, vector_forward_tanh))
            return;
    }
#endif // _BUILD_FOR_CUDA
    int l = 0;
    while (l < L)
    {
        Y[l] = tanh(X[l]);
        ++l;
    }
}

//    Y = tanh(X), dldY, Y, &dldX, length
void  tanh_backward(const numeric_t* dldY, const numeric_t* Y, numeric_t* dldX, int L)
{
#ifdef _BUILD_FOR_CUDA
    if (L > SHORT_VECTOR)
    {
        if (cudaSuccess == cuda_backward_vectors_math_op(dldX, Y, dldY, L, vector_backward_tanh))
            return;
    }
#endif // _BUILD_FOR_CUDA
    int l = 0;
    while (l < L)
    {
        dldX[l] = (1.0 - Y[l] * Y[l]) * dldY[l];
        ++l;
    }
}

//    Y = tanf(X), &Y, X, length
void  tanf_forward(numeric_t* Y, const numeric_t* X, int L)
{
#ifdef _BUILD_FOR_CUDA
    if (L > SHORT_VECTOR)
    {
        if (cudaSuccess == cuda_forward_vectors_math_op(Y, Y, L, vector_forward_tanf))
            return;
    }
#endif // _BUILD_FOR_CUDA
    int l = 0;
    while (l < L)
    {
        Y[l] = X[l] / sqrt(1.0 + X[l] * X[l]);
        ++l;
    }
}

#if 0
//    Y = tanhf(X), dldY, Y, &dldX, length
void  tanf_backward(numeric_t* dldY, numeric_t* Y, numeric_t* dldX, int L, int td)
{
    int l = 0;
    numeric_t t;
    while ( l < L )
    {
        // TODO: test derivative (overfitting faster than tanh()?)
        t = (1.0 - Y[l] * Y[l]);
        if (td)
            dldX[l] = sqrt(t * t * t) * dldY[l];
        else
            dldX[l] = t * dldY[l];
        ++l;
    }
}
#endif // 0

void  tanf_backward(const numeric_t* dldY, const numeric_t* Y, numeric_t* dldX, int L)
{
#ifdef _BUILD_FOR_CUDA
    if (L > SHORT_VECTOR)
    {
        if (cudaSuccess == cuda_backward_vectors_math_op(dldX, Y, dldY, L, vector_backward_tanf))
            return;
    }
#endif // _BUILD_FOR_CUDA
    int l = 0;
    numeric_t t;
    while (l < L)
    {
        t = (1.0 - Y[l] * Y[l]);
        dldX[l] = sqrt(t * t * t) * dldY[l];
        ++l;
    }
}
